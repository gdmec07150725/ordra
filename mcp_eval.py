"""Local/MCP comparison using isolated stores, actual tools and observed model HTTP."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
from html import escape
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

import anyio
import httpx2
from openai import APIConnectionError

import agent_service
import api_app
from context_memory import estimate_request
from eval_scenarios import tools, delegation_script, specialist_scripts
from eval_trajectory import criterion, grade_trajectory, score_checks, summarize_metrics, tool_results
from multiturn_eval import ScriptedConversationModel


BUSINESS_TOOLS = frozenset({'get_order', 'search_knowledge'})
POLICY_TEXT = '# 评测专用示例政策\n\n## 退货期限\n签收后7天内可申请模拟退货，需人工核实；预览不等于提交或退款。'
CASES = [
    {'id': 'order_query', 'title': '订单状态', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003']},
    {'id': 'knowledge_rag', 'title': '示例退货政策', 'prompt': '请查阅知识库，告诉我示例退货期限。', 'knowledge': True},
    {'id': 'order_and_policy', 'title': '订单与政策', 'prompt': '请直接查询 O1003 当前状态，并查阅知识库说明示例退货期限，再汇总。', 'orders': ['O1003'], 'knowledge': True},
    {'id': 'checkpoint_resume', 'title': '工具完成后故障恢复', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'], 'fault': 'model'},
    {'id': 'delegation_parallel', 'title': '订单与知识子任务', 'prompt': '请使用任务委派功能，让订单查询 Agent 和知识检索 Agent 分别查询 O1003 状态与示例退货期限，再汇总。', 'orders': ['O1003'], 'knowledge': True, 'delegation': True},
]


def scripted_case(case, mode):
    """Only model decisions are simulated; every business call reaches real code."""
    calls = []
    if case.get('orders'):
        calls.append(('order', 'get_order', {'order_id': 'O1003'}))
    if case.get('knowledge'):
        calls.append(('policy', 'search_knowledge', {'query': '退货期限'}))
    answer = 'O1003 已签收。' if case.get('orders') else ''
    if case.get('knowledge'):
        answer += '示例政策为签收后7天内申请退货，需人工核实。'
    if case.get('delegation'):
        return {**case, 'script': [delegation_script(), {'text': answer}], 'specialists': specialist_scripts()}
    discovery = [tools(('discover', 'discover_tools', {'names': [call[1] for call in calls]}))] if mode == 'mcp' else []
    return {**case, 'script': discovery + [tools(*calls), {'text': answer}]}


def grade_mode(case, mode, trace, *, requests=None):
    """Discovery must precede the request carrying a main-agent business schema."""
    events = (trace or {}).get('events', [])
    filtered = {**(trace or {}), 'events': [e for e in events if not (e['kind'] == 'tool_result' and e['data'].get('name') in ('discover_tools', 'load_skill', 'read_skill_resource'))]}
    checks = grade_trajectory(case, filtered)['checks']
    business = [c for c in tool_results(trace) if c.get('name') in BUSINESS_TOOLS and c.get('attempts', 0) > 0]
    checks.append(criterion('business_transport', '业务工具来自指定执行方式', bool(business) and all(c.get('source') == mode for c in business), [c.get('source') for c in business]))
    if 'O1003' in case.get('orders', []):
        checks.append(criterion('signed_order', 'O1003 返回已签收状态', any(c.get('name') == 'get_order' and c.get('status') == 'ok' and c.get('result', {}).get('order_id') == 'O1003' and c.get('result', {}).get('status') == '已签收' for c in business), [c.get('result') for c in business if c.get('name') == 'get_order']))
    if mode == 'mcp':
        core = {'discover_tools', 'update_plan', 'delegate_tasks', 'load_skill', 'read_skill_resource'}
        specialist_tools = {'orders': 'get_order', 'knowledge': 'search_knowledge'}
        loaded, remote_loaded, actor_available = set(), set(), set()
        specialists, child_schemas, violations, actor_leaks, child_leaks = {}, {}, [], [], []
        first_actor = None
        for event in events:
            data = event['data']
            if event['kind'] == 'tool_discovery' and data.get('status') == 'loaded':
                loaded.update(t['name'] for t in data.get('tools', []) if t.get('source') in ('local', 'mcp'))
                remote_loaded.update(t['name'] for t in data.get('tools', []) if t.get('source') == 'mcp')
            elif event['kind'] == 'context_budget':
                if data.get('role') == 'actor':
                    first_actor = data.get('tool_names', []) if first_actor is None else first_actor
                    names = set(data.get('tool_names', []))
                    leaked = names - core - loaded
                    if leaked:
                        actor_leaks.append({'sequence': data.get('sequence'), 'unloaded_names': sorted(leaked)})
                    actor_available = remote_loaded & names
                elif data.get('role') == 'specialist':
                    agent, names = data.get('agent'), data.get('tool_names', [])
                    permitted = specialist_tools.get(agent)
                    valid = permitted is not None and names == [permitted]
                    child_schemas[agent] = set(names) if valid else set()
                    if not valid:
                        child_leaks.append({'agent': agent, 'tool_names': names, 'expected': [permitted] if permitted else []})
            elif event['kind'] == 'delegated_task':
                specialists[data['id']] = data['agent']
            elif event['kind'] == 'tool_result' and data.get('name') in BUSINESS_TOOLS:
                task = str(data.get('call_id', '')).split(':')[0]
                agent = specialists.get(task)
                allowed_child = specialist_tools.get(agent)
                child = allowed_child == data['name'] and data['name'] in child_schemas.get(agent, set())
                if (task in specialists and not child) or (task not in specialists and data['name'] not in actor_available):
                    violations.append(data.get('call_id'))
        checks.extend([
            criterion('initial_schemas_deferred', '首个主管请求只加载核心工具', first_actor is not None and 'discover_tools' in first_actor and set(first_actor) <= core, first_actor),
            criterion('actor_schema_whitelist', '每次主管请求只包含核心工具和先前发现的定义', not actor_leaks, actor_leaks),
            criterion('specialist_schema_whitelist', '每次子任务请求只包含其职责工具', not child_leaks, child_leaks),
            criterion('discovery_before_business', '主管先发现再读取定义，子任务使用职责白名单', not violations, violations),
        ])
    if requests is not None:
        prepared, observed = {}, {}
        for event in events:
            if event['kind'] == 'context_budget':
                data = event['data']
                prepared.setdefault(data.get('role'), []).append(tuple(data.get('tool_names', [])))
        for request in requests:
            observed.setdefault(request.get('role'), []).append(tuple(request.get('tool_names', [])))
        # Child requests can race after their preparation has been recorded.
        # Preserve actor ordering while comparing parallel specialists as a bag.
        equal = set(prepared) == set(observed) and all(
            Counter(prepared[role]) == Counter(observed[role]) if role == 'specialist' else prepared[role] == observed[role]
            for role in prepared)
        checks.append(criterion('observed_request_schemas', '请求观察的工具名与准备记录一致', equal, {'prepared': prepared, 'observed': observed}))
    return score_checks(checks)


def grade_pair(local, mcp):
    return score_checks([
        criterion('both_modes_pass', '两种方式均完成实际业务目标', local['passed'] and mcp['passed'], {'local': local['passed'], 'mcp': mcp['passed']}),
        criterion('same_business_evidence', '两种方式返回相同业务证据', local['evidence'] == mcp['evidence'], {'local': local['evidence'], 'mcp': mcp['evidence']}),
    ])


def canonical_evidence(observed):
    orders, knowledge = [], []
    for item in observed:
        result = item['result']
        if result.get('error'):
            continue
        if item['name'] == 'get_order':
            orders.append({key: value for key, value in result.items() if not key.startswith('_')})
        elif item['name'] == 'search_knowledge':
            knowledge.extend({key: hit.get(key) for key in ('source', 'version', 'title', 'section', 'text')} for hit in result.get('results', []))
    stable = lambda rows: [json.loads(value) for value in sorted({json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows})]
    return {'orders': stable(orders), 'knowledge': stable(knowledge)}


def request_record(body, *, sent):
    role = 'actor' if body.get('stream') else 'verification' if body.get('response_format') == {'type': 'json_object'} else 'specialist' if body.get('tools') else 'summary'
    messages, schemas = body.get('messages', []), body.get('tools', [])
    return {'role': role, 'sent': sent, 'tool_names': [t['function']['name'] for t in schemas],
            'estimated_schema_tokens': estimate_request(messages, schemas) - estimate_request(messages)}


def metrics_for(trace, requests, attempts, *, live):
    sent = [r for r in requests if r['sent']]
    usage = summarize_metrics([trace], simulated=not live)
    reported = usage['reported_requests']
    coverage = 'unreported' if not reported else 'complete' if reported == len(sent) else 'partial'
    return {**usage, 'usage_status': coverage, 'usage_coverage_denominator': 'sent_requests', 'model_requests': len(sent),
            'request_count': len(sent), 'model_request_attempts': max(attempts, len(requests)),
            'estimated_schema_tokens': sum(r['estimated_schema_tokens'] for r in sent),
            'schema_estimator': 'utf8_bytes_conservative',
            'schema_tokens_by_request': [r['estimated_schema_tokens'] for r in sent],
            'request_counts_by_role': {role: sum(r['role'] == role for r in sent) for role in ('actor', 'specialist', 'verification', 'summary')}}


async def run_case(client, app, case, mode, *, live, timeout_seconds):
    started, error, failure_kind, requests, observed = monotonic(), None, None, [], []
    sid = (await client.post('/api/sessions')).json()['session_id']
    observation = {'actual_order_calls': 0, 'fault_injected': False, 'resumed_same_run': False}
    attempts = 0
    original_create = app.state.client.chat.completions.create
    original_order, original_search = agent_service.get_order, app.state.knowledge_base.search

    def capture(name, arguments, result):
        observed.append({'name': name, 'arguments': deepcopy(arguments), 'result': deepcopy(result)})
        observation['actual_order_calls'] += int(name == 'get_order')
        return result

    def order(order_id):
        return capture('get_order', {'order_id': order_id}, original_order(order_id))

    def search(query):
        return capture('search_knowledge', {'query': query}, original_search(query))

    async def on_http_request(request):
        requests.append(request_record(json.loads(request.content), sent=True))

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        if case.get('fault') == 'model' and kwargs.get('stream') and observation['actual_order_calls'] and not observation['fault_injected']:
            observation['fault_injected'] = True
            requests.append(request_record(kwargs, sent=False))
            raise APIConnectionError(request=httpx2.Request('POST', 'https://api.deepseek.com/chat/completions'))
        return await original_create(**kwargs)

    # The SDK's underlying HTTP client exposes public request hooks. Capture only
    # schema names/sizes; never retain request headers, credentials or prompts.
    hooks = app.state.client._client.event_hooks['request']
    hooks.append(on_http_request)
    try:
        with ExitStack() as patches:
            patches.enter_context(patch.object(app.state.client.chat.completions, 'create', side_effect=create))
            if mode == 'mcp':
                original_remote = app.state.mcp_client.call_tool

                async def remote(name, arguments, **kwargs):
                    return capture(name, arguments, await original_remote(name, arguments, **kwargs))

                patches.enter_context(patch.object(app.state.mcp_client, 'call_tool', side_effect=remote))
            else:
                patches.enter_context(patch.object(agent_service, 'get_order', side_effect=order))
                patches.enter_context(patch.object(app.state.knowledge_base, 'search', side_effect=search))
            with anyio.fail_after(timeout_seconds):
                response = await client.post('/api/chat', json={'session_id': sid, 'message': case['prompt']})
                if response.status_code >= 400 and observation['fault_injected']:
                    checkpoint = app.state.checkpoints.load(sid)
                    if checkpoint:
                        observation['resume_tool_selection'] = deepcopy(checkpoint.get('state', {}).get('tool_selection'))
                        response = await client.post(f'/api/sessions/{sid}/resume', json={'run_id': checkpoint['run_id']})
                        run_ids = {r['run_id'] for r in app.state.traces.list_runs(sid)['runs']}
                        observation['resumed_same_run'] = run_ids == {checkpoint['run_id']}
                observation['http_status'] = response.status_code
                if response.status_code != 200:
                    error = response.json().get('detail', '本地 API 请求失败')
                    failure_kind = 'api_error'
    except TimeoutError:
        error, failure_kind = f'场景在 {timeout_seconds:g} 秒内未完成。', 'timeout'
    except Exception as failure:
        error, failure_kind = agent_service.describe_error(failure, app.state.api_key), type(failure).__name__
    finally:
        hooks.remove(on_http_request)
    saved = app.state.store.load_session(sid)
    runs = app.state.traces.list_runs(sid)['runs']
    trace = app.state.traces.load(sid, runs[0]['run_id']) if runs else None
    evidence = canonical_evidence(observed)
    checks = grade_mode(case, mode, trace, requests=requests)['checks'] + [
        criterion('api_success', '本地 API 成功完成', observation.get('http_status') == 200 and error is None, error or observation.get('http_status')),
        criterion('committed_once', 'SQLite 仅提交一次本轮问题', sum(m['role'] == 'user' for m in saved['history']) == 1, sum(m['role'] == 'user' for m in saved['history'])),
        criterion('checkpoint_cleared', '成功后检查点已清除', app.state.checkpoints.load(sid) is None, app.state.checkpoints.load(sid) is None),
    ]
    if case.get('orders'):
        checks.append(criterion('actual_order_result', '底层工具实际返回目标订单', any(o.get('order_id') == 'O1003' and o.get('status') == '已签收' for o in evidence['orders']), evidence['orders']))
    if case.get('knowledge'):
        checks.append(criterion('actual_policy_text', '底层检索返回临时政策原文', any('签收后7天' in hit.get('text', '') for hit in evidence['knowledge']), evidence['knowledge']))
    if case.get('fault') == 'model':
        checks.extend([
            criterion('fault_resumed', '工具完成后注入一次故障并恢复同一执行', observation['fault_injected'] and observation['resumed_same_run'], observation['resumed_same_run']),
            criterion('actual_order_not_replayed', '已保存的订单工具没有重复执行', observation['actual_order_calls'] == 1, observation['actual_order_calls']),
        ])
    grade = score_checks(checks)
    return {'mode': mode, 'session_id': sid, 'passed': grade['passed'], 'elapsed_ms': round((monotonic() - started) * 1000),
            'error': error, 'failure_kind': failure_kind, 'answer': next((m.get('content') or '' for m in reversed(saved['history']) if m['role'] == 'assistant' and not m.get('tool_calls')), ''),
            'evidence': evidence, 'observation': observation, 'requests': requests, 'trace': trace, 'trajectory': grade,
            'metrics': metrics_for(trace, requests, attempts, live=live)}


async def evaluate_comparison(cases=None, *, live=False, timeout_seconds=90):
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds 必须是有限正数。')
    cases = deepcopy(CASES if cases is None else cases)
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('至少选择一个场景，且场景 id 不能重复。')
    outputs, startup = {case['id']: {} for case in cases}, {}
    model = ScriptedConversationModel()
    with TemporaryDirectory(prefix='agent-mcp-comparison-') as directory, ExitStack() as patches:
        if not live:
            # Keep Windows socket/process prerequisites while excluding user
            # credentials and environment-dependent configuration from scripts.
            environment = {key: value for key, value in os.environ.items() if key.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
            environment.update(DEEPSEEK_API_KEY='offline-eval-key', DEEPSEEK_MODEL='offline-script', AGENT_HISTORY_SEARCH='keywords')
            patches.enter_context(patch.dict(os.environ, environment, clear=True))
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=model.client))
        else:
            patches.enter_context(patch.dict(os.environ, {'AGENT_HISTORY_SEARCH': 'keywords'}))
        root = Path(directory)
        knowledge = root / 'knowledge'
        knowledge.mkdir()
        (knowledge / 'policy.md').write_text(POLICY_TEXT, encoding='utf-8')
        for mode in ('local', 'mcp'):
            app = api_app.create_app(db_path=root / f'{mode}.db', knowledge_dir=knowledge, tool_mode=mode)
            began = monotonic()
            try:
                # A bounded whole-mode guard includes startup/cleanup; individual
                # requests also receive their own deadline below.
                with anyio.fail_after(timeout_seconds * (len(cases) + 1)):
                    async with app.router.lifespan_context(app):
                        startup[mode] = round((monotonic() - began) * 1000)
                        if app.state.client is None:
                            raise ValueError('真实评测需要配置模型 API key。')
                        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local', trust_env=False) as client:
                            for case in cases:
                                model.reset(scripted_case(case, mode))
                                item = await run_case(client, app, case, mode, live=live, timeout_seconds=timeout_seconds)
                                outputs[case['id']][mode] = item
                                print(f"{'PASS' if item['passed'] else 'FAIL'} {case['id']} / {mode} · {item['metrics']['model_requests']} requests · {item['elapsed_ms']} ms", flush=True)
            except Exception as failure:
                timeout = isinstance(failure, TimeoutError)
                error = f'执行方式在限定时间内未完成（每场景 {timeout_seconds:g} 秒）。' if timeout else agent_service.describe_error(failure, getattr(app.state, 'api_key', ''))
                for case in cases:
                    existing = outputs[case['id']].get(mode)
                    if existing is not None:
                        check = criterion('lifecycle_completed', '执行方式正常关闭', False, error)
                        existing['trajectory'] = score_checks(existing['trajectory']['checks'] + [check])
                        existing.update(passed=False, error=error, failure_kind='timeout' if timeout else type(failure).__name__)
                    else:
                        outputs[case['id']][mode] = {'mode': mode, 'passed': False, 'elapsed_ms': round((monotonic() - began) * 1000),
                            'error': error, 'failure_kind': 'timeout' if timeout else type(failure).__name__, 'answer': '', 'evidence': {'orders': [], 'knowledge': []},
                            'requests': [], 'observation': {}, 'trace': None, 'trajectory': score_checks([criterion('startup_completed', '执行方式已启动', False, error)]),
                            'metrics': metrics_for(None, [], 0, live=live)}
    results = []
    for case in cases:
        pair = grade_pair(outputs[case['id']]['local'], outputs[case['id']]['mcp'])
        results.append({'id': case['id'], 'title': case['title'], 'prompt': case['prompt'], 'passed': pair['passed'], 'comparison': pair, 'modes': outputs[case['id']]})
    return {'created_at': datetime.now(UTC).isoformat(), 'suite': 'mcp_comparison', 'mode': 'live' if live else 'offline',
            'model_decisions_scripted': not live, 'startup_ms': startup, 'timeout_seconds': timeout_seconds,
            'method': 'Actual local API, temporary SQLite/knowledge fixtures and real MCP stdio. HTTP request hooks count sent model requests. Schema tokens are conservative estimates, not billing usage. Discovery can add a model request and does not guarantee lower total cost. Offline model decisions, reviews and provider usage are scripted.',
            'passed': sum(item['passed'] for item in results), 'total': len(results), 'results': results}


def report_html(report):
    rows = []
    for result in report['results']:
        for mode, item in result['modes'].items():
            metrics = item['metrics']
            detail = json.dumps({'answer': item.get('answer'), 'error': item.get('error'), 'comparison': result['comparison'],
                                 'checks': item['trajectory'], 'requests': item['requests'], 'observation': item['observation'], 'trace': item['trace']}, ensure_ascii=False, indent=2)
            usage = '未报告' if metrics['total_tokens'] is None else str(metrics['total_tokens'])
            if metrics['usage_status'] == 'partial':
                usage += '（部分请求）'
            rows.append(f"<tr><td>{escape(result['title'])}</td><td>{escape(mode)}</td><td>{'通过' if item['passed'] else '失败'}</td><td>{metrics['model_requests']}</td><td>{escape(usage)}</td><td>{metrics['estimated_schema_tokens']}</td><td>{item['elapsed_ms']}</td></tr><tr><td colspan='7'><details><summary>逐次请求、发现和实际业务证据</summary><pre>{escape(detail)}</pre></details></td></tr>")
    mode_label = '真实模型验收' if report['mode'] == 'live' else '离线模型决策、校验及 usage 为脚本模拟；MCP 进程、工具与临时数据库实际执行'
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MCP 工具发现对照</title><style>body{{max-width:1120px;margin:40px auto;padding:0 20px;font:15px/1.7 system-ui;background:#f7f9f5;color:#263f33}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dfe5db}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 monospace}}summary{{cursor:pointer}}</style><h1>MCP 工具发现对照</h1><p>{escape(mode_label)} · {report['passed']} / {report['total']} 场景通过</p><p>工具定义 tokens 是按实际模型 HTTP 请求估算的开销，不是计费用量。发现工具可能增加一轮模型请求，小目录不保证节省总成本。耗时包含真实本地执行，启动耗时单独记录。恢复场景明确注入一次模型连接故障。</p><p>{escape(report['created_at'])} · 启动耗时 ms：{escape(json.dumps(report['startup_ms']))}</p><table><thead><tr><th>场景</th><th>方式</th><th>结果</th><th>实际模型请求</th><th>已报告 tokens</th><th>定义 tokens 估算合计</th><th>耗时 ms</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='通过正常应用配置调用真实模型')
    parser.add_argument('--case', action='append', choices=[case['id'] for case in CASES])
    parser.add_argument('--timeout', type=float, default=90, help='每场景最大秒数')
    parser.add_argument('--output', type=Path, default=Path('.local/evals/mcp-comparison.json'))
    args = parser.parse_args()
    cases = [case for case in CASES if not args.case or case['id'] in args.case]
    report = anyio.run(lambda: evaluate_comparison(cases, live=args.live, timeout_seconds=args.timeout))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.html').write_text(report_html(report), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} passed · {args.output.resolve()}")
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
