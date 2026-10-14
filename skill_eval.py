"""Skills selection and reference evaluation, isolated from user conversations."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

import anyio
import httpx2
from openai import APIConnectionError

import agent_service
import api_app
from context_memory import ContextPolicy
from eval_scenarios import tools
from eval_trajectory import criterion, tool_results
from multiturn_eval import ScriptedConversationModel


SKILL = 'return-handling'
REFERENCE = 'references/missing-information.md'
SKILL_CASES = [
    {'id': 'skill_required', 'kind': 'skills', 'title': '明确加载技能和引用',
     'prompt': '先加载 return-handling，再读取 references/missing-information.md。我想退 O1003，但还没说明原因，请按指引询问。', 'reference': True},
    {'id': 'skill_auto', 'kind': 'skills', 'title': '自主选择缺失信息指引',
     'prompt': '我想退 O1003，但还没说明原因。请使用适合的处理流程及缺失信息询问细则，告诉我需要补充什么。', 'reference': True, 'compare': True},
    {'id': 'skill_unneeded', 'kind': 'skills', 'title': '普通查询无需技能',
     'prompt': 'O1003 当前是什么订单状态？', 'no_skill': True, 'compare': True},
    {'id': 'skill_wrong_resource', 'kind': 'skills', 'title': '错误引用后按清单纠正',
     'prompt': '测试引用读取：先加载 return-handling，调用 read_skill_resource 读取 references/not-listed.md，收到错误后改读 references/missing-information.md。然后问我 O1003 退货还需要什么信息。',
     'reference': True, 'expected_error': 'skill_resource_not_found'},
    {'id': 'skill_repeat', 'kind': 'skills', 'title': '重复引用不重复注入正文',
     'prompt': '测试重复读取：先加载 return-handling，分两次调用 read_skill_resource 读取 references/missing-information.md，然后根据指引询问 O1003 的退货原因。',
     'reference': True, 'repeat': True},
    {'id': 'skill_budget', 'kind': 'skills', 'title': '引用超预算后正常结束',
     'prompt': '先加载 return-handling，再尝试读取 references/missing-information.md。若超出预算就使用已有指引询问 O1003 的退货原因，不要重试。',
     'expected_error': 'skill_resource_budget_exceeded', 'budget': True},
    {'id': 'skill_resume', 'kind': 'skills', 'title': '引用后中断并恢复旧版本',
     'prompt': '先加载 return-handling，再读取 references/missing-information.md。我想退 O1003，但还没说明原因，请按指引询问。',
     'reference': True, 'fault': 'after_reference'},
]


def scripted_skill(case, *, baseline=False):
    if case.get('no_skill'):
        return {'script': [tools(('d', 'discover_tools', {'names': ['get_order']})),
                           tools(('o', 'get_order', {'order_id': 'O1003'})), {'text': 'O1003 当前已签收。'}]}
    answer = {'text': '请补充 O1003 的具体退货原因，例如你实际遇到的情况；信息齐全后再核对条件。'}
    if baseline:
        return {'script': [answer]}
    steps = [tools(('s', 'load_skill', {'name': SKILL}))]
    if case.get('expected_error') == 'skill_resource_not_found':
        steps.append(tools(('bad', 'read_skill_resource', {'name': SKILL, 'path': 'references/not-listed.md'})))
    steps.append(tools(('r', 'read_skill_resource', {'name': SKILL, 'path': REFERENCE})))
    if case.get('repeat'):
        steps.append(tools(('r2', 'read_skill_resource', {'name': SKILL, 'path': REFERENCE})))
    return {'script': steps + [answer]}


def grade_skills(case, trace, answer, observation, *, baseline=False):
    events = (trace or {}).get('events', [])
    loads = [e['data'] for e in events if e['kind'] == 'skill_load']
    reads = [e['data'] for e in events if e['kind'] == 'skill_resource']
    loaded = [e.get('name') for e in loads if e['result'].get('status') == 'loaded']
    accepted = [(e.get('name'), e.get('path')) for e in reads if e['result'].get('status') == 'loaded']
    expected = [] if case.get('no_skill') or baseline else [SKILL]
    checks = [criterion('completed', '任务正常结束', (trace or {}).get('status') in ('completed', 'waiting_input'), (trace or {}).get('status')),
              criterion('skill_selection', '选择所需技能，无额外技能', loaded == expected, loaded),
              criterion('reference_selection', '只读取所需引用', accepted == ([(SKILL, REFERENCE)] if case.get('reference') and not baseline else []), accepted)]
    if not case.get('repeat'):
        repeats = [e.get('path') for e in reads if e['result'].get('status') == 'already_loaded']
        checks.append(criterion('no_repeated_reference', '没有重复读取引用', not repeats, repeats))
    else:
        checks.append(criterion('repeat_metadata_only', '重复读取不再次注入正文', len(reads) == 2 and reads[-1]['result'].get('status') == 'already_loaded'
                                and observation.get('body_copies') == 1, observation.get('body_copies')))
    if case.get('expected_error'):
        errors = [e['result'].get('error_code') for e in reads if 'error' in e['result']]
        checks.append(criterion('expected_error', '实际返回预期读取错误', case['expected_error'] in errors, errors))
    if case.get('fault'):
        checks.append(criterion('resume', '故障后恢复同一任务且未重复读取', observation.get('resumed_same_run')
                                and observation.get('resource_calls_after_resume') == 1 and observation.get('pinned_body_after_resume'), observation))
    if case.get('no_skill'):
        orders = [e.get('result', {}) for e in tool_results(trace) if e.get('name') == 'get_order' and e.get('status') == 'ok']
        checks.append(criterion('order_evidence', '实际查询正确订单状态', any(o.get('order_id') == 'O1003' and o.get('status') == '已签收' for o in orders), orders))
        checks.append(criterion('answer_order', '回答包含订单及真实状态', 'O1003' in answer and '已签收' in answer, answer))
    else:
        checks.append(criterion('asks_reason', '缺少原因时询问用户', '原因' in answer and any(word in answer for word in ('请', '补充', '提供', '告诉', '什么')), answer))
        writes = [e for e in tool_results(trace) if e.get('name') == 'prepare_return_request' and e.get('status') == 'ok']
        checks.append(criterion('no_premature_preview', '缺少原因时未生成预览', not writes, len(writes)))
    budgets = [e['data'] for e in events if e['kind'] == 'context_budget']
    checks.append(criterion('request_budget', '所有模型请求均在预算内', bool(budgets) and all(b['status'] != 'blocked' and b['estimated_input_tokens'] <= b['input_limit'] for b in budgets),
                            [{'role': b['role'], 'input': b['estimated_input_tokens'], 'limit': b['input_limit']} for b in budgets]))
    return checks


async def run_skill_case(client, app, case, *, baseline=False):
    started = monotonic()
    sid = (await client.post('/api/sessions')).json()['session_id']
    original = app.state.client.chat.completions.create
    observation, error = {'fault_injected': False}, None
    pinned_body = None

    async def model(**kwargs):
        nonlocal pinned_body
        if kwargs.get('stream'):
            cp = app.state.checkpoints.load(sid)
            state = (cp or {}).get('state', {}).get('skills', {})
            resources = state.get('resources', {})
            if resources and case.get('fault') and not observation['fault_injected']:
                pinned_body = next(iter(resources.values()))['content']
                observation['fault_injected'] = True
                raise APIConnectionError(request=httpx2.Request('POST', 'https://eval.invalid'))
            if pinned_body is not None:
                observation['pinned_body_after_resume'] = any(
                    json.loads(m['content']).get('content') == pinned_body
                    for m in kwargs['messages'] if m.get('role') == 'tool')
                observation['resource_calls_after_resume'] = state.get('resource_calls')
        return await original(**kwargs)

    try:
        with patch.object(app.state.client.chat.completions, 'create', side_effect=model), anyio.fail_after(180):
            response = await client.post('/api/chat', json={'session_id': sid, 'message': case['prompt']})
            if response.status_code >= 400 and observation['fault_injected']:
                cp = app.state.checkpoints.load(sid)
                # The fixture is private to this case. Existing tasks retain the saved content.
                (app.state.skills_dir / SKILL / REFERENCE).write_text('评测修改后的不同版本，不应进入恢复请求。', encoding='utf8')
                response = await client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
                observation['resumed_same_run'] = {r['run_id'] for r in app.state.traces.list_runs(sid)['runs']} == {cp['run_id']}
            if response.status_code != 200:
                error = response.json().get('detail', '请求失败')
    except Exception as exc:
        error = agent_service.describe_error(exc, app.state.api_key)
    history = app.state.store.load_session(sid)['history']
    answer = next((m.get('content') or '' for m in reversed(history) if m['role'] == 'assistant' and not m.get('tool_calls')), '')
    reference_ids = {c['id'] for m in history for c in m.get('tool_calls', []) if c['function']['name'] == 'read_skill_resource'}
    observation['body_copies'] = sum('content' in json.loads(m['content']) for m in history if m.get('tool_call_id') in reference_ids)
    runs = app.state.traces.list_runs(sid)['runs']
    trace = app.state.traces.load(sid, runs[0]['run_id']) if runs else None
    checks = grade_skills(case, trace, answer, observation, baseline=baseline)
    checks.append(criterion('api_success', 'API 完成且检查点清理', error is None and app.state.checkpoints.load(sid) is None, error))
    checks.append(criterion('result_verified', '回答通过独立结果校验', (trace or {}).get('metrics', {}).get('verification_status') == 'passed', (trace or {}).get('metrics', {}).get('verification_status')))
    return {'id': case['id'] + ('_baseline' if baseline else ''), 'kind': 'skills',
            'title': case['title'] + (' · 关闭 Skills 对照' if baseline else ''),
            'baseline': baseline, 'passed': all(c['passed'] for c in checks), 'answer': answer,
            'elapsed_ms': round((monotonic() - started) * 1000), 'checks': checks,
            'observation': observation, 'trace': trace}


async def evaluate_skills(cases, *, live=False):
    results, model = [], ScriptedConversationModel()
    with TemporaryDirectory(prefix='agent-skills-eval-') as directory, ExitStack() as patches:
        if not live:
            env = {k: v for k, v in os.environ.items() if k.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
            env.update(DEEPSEEK_API_KEY='offline-eval-key', DEEPSEEK_MODEL='offline-script', AGENT_HISTORY_SEARCH='keywords')
            patches.enter_context(patch.dict(os.environ, env, clear=True))
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=model.client))
        else:
            patches.enter_context(patch.dict(os.environ, {'AGENT_HISTORY_SEARCH': 'keywords'}))
        root = Path(directory)
        knowledge = root / 'knowledge'
        knowledge.mkdir()
        (knowledge / 'policy.md').write_text('# 测试政策\n质量问题需说明具体情况，预览不等于申请已提交。', encoding='utf8')
        app = api_app.create_app(db_path=root / 'chat.db', knowledge_dir=knowledge, tool_mode='mcp')
        async with app.router.lifespan_context(app):
            default_policy = app.state.context_policy
            if app.state.client is None:
                raise ValueError('真实评测需要配置模型。')
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local', trust_env=False) as client:
                for case in cases:
                    for baseline in ([False, True] if case.get('compare') else [False]):
                        fixture = root / (case['id'] + ('-baseline' if baseline else ''))
                        shutil.copytree(Path(__file__).parent / 'skills', fixture)
                        app.state.skills_dir = None if baseline else fixture
                        app.state.context_policy = ContextPolicy(max_tokens=12000) if case.get('budget') else default_policy
                        if case.get('budget'):
                            (fixture / SKILL / REFERENCE).write_text('引用细则' * 1800, encoding='utf8')
                        model.reset(scripted_skill(case, baseline=baseline))
                        item = await run_skill_case(client, app, case, baseline=baseline)
                        results.append(item)
                        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['id']} · {item['elapsed_ms']} ms", flush=True)
    return {'created_at': datetime.now(UTC).isoformat(), 'mode': 'live' if live else 'offline', 'suite': 'skills',
            'passed': sum(r['passed'] for r in results), 'total': len(results), 'results': results}
