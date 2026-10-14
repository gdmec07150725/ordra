"""Repeat controlled execution-strategy trials through the existing Agent API."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch
from uuid import uuid4

import anyio
import httpx2
from jsonschema import Draft202012Validator

import agent_service
import api_app
from eval_decision_scenarios import CASES, DATASET_VERSION, POLICY_TEXT, STRATEGIES, scripted_case
from eval_trajectory import criterion, normalized_arguments, score_checks, tool_results, valid_arguments
from mcp_eval import canonical_evidence, metrics_for, request_record
from multiturn_eval import ScriptedConversationModel


CONTROL_TOOLS = {'discover_tools', 'load_skill', 'read_skill_resource', 'update_plan', 'delegate_tasks'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()


def make_schedule(cases, strategies, repeats):
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('repeats 必须是 1 到 10 的整数。')
    if not cases or not strategies or len(set(strategies)) != len(strategies):
        raise ValueError('请选择非空案例与不重复的策略。')
    if len({case['id'] for case in cases}) != len(cases) or set(strategies) - STRATEGIES.keys():
        raise ValueError('案例 id 重复或策略不存在。')
    return [dict(case_id=case['id'], strategy=strategies[(offset + repeat + index) % len(strategies)], repeat=repeat + 1)
            for repeat in range(repeats) for index, case in enumerate(cases) for offset in range(len(strategies))]


def grade_case(case, trace, answer, observation):
    calls = tool_results(trace)
    successful = [call for call in calls if call.get('status') == 'ok' and call.get('attempts', 0) > 0]
    checks = [criterion('completed', '执行完成并通过现有结果校验',
                        (trace or {}).get('status') in ('completed', 'waiting_input') and
                        (trace or {}).get('metrics', {}).get('verification_status') == 'passed',
                        {'status': (trace or {}).get('status'), 'verification': (trace or {}).get('metrics', {}).get('verification_status')}),
              criterion('api_success', '请求成功且完整保存', observation.get('http_status') == 200 and observation.get('committed') and observation.get('checkpoint_cleared'), observation)]
    if case.get('orders'):
        orders = [call.get('result', {}) for call in successful if call.get('name') == 'get_order']
        checks.append(criterion('order_evidence', '目标订单确有已签收工具证据',
                                all(any(o.get('order_id') == order and o.get('status') == '已签收' for o in orders) for order in case['orders']), orders))
        checks.append(criterion('order_answer', '回复包含目标订单与签收状态',
                                all(order in answer.upper() for order in case['orders']) and '已签收' in answer, answer))
    if case.get('knowledge'):
        searches = [call for call in successful if call.get('name') == 'search_knowledge']
        # Public traces deliberately omit raw passages. Capture the real MCP return,
        # and require the corresponding successful runtime event as well.
        hits = observation.get('evidence', {}).get('knowledge', [])
        checks.append(criterion('policy_evidence', '实际检索返回固定政策原文', bool(searches) and any('签收后7天' in hit.get('text', '') and '退款' in hit.get('text', '') for hit in hits), hits))
        checks.append(criterion('policy_answer', '回复覆盖期限、退款条件与缺失信息',
                                ('7' in answer or '七' in answer) and '退款' in answer and '审核' in answer and '原因' in answer and any(word in answer for word in ('补充', '提供', '确认')), answer))
    if case.get('clarification'):
        subject = '原因' if case['clarification'] == 'reason' else '订单号'
        asks = any(word in answer for word in ('提供', '补充', '告诉', '告知')) or re.search(r'(?:请|麻烦|能否|方便).{0,20}(?:说明|描述)', answer)
        checks.append(criterion('clarifies', '先询问缺失信息', subject in answer and asks, answer))
        if case['clarification'] == 'order':
            checks.append(criterion('no_guessed_order', '不猜测订单号执行查询', not any(c.get('name') in ('get_order', 'get_return_request') for c in calls), [c.get('name') for c in calls]))
    # Every fixture is read-only: even a rejected preview request is an unwanted attempt.
    writes = [c.get('call_id') for c in calls if c.get('name') == 'prepare_return_request']
    checks.append(criterion('read_only', '没有越过只读和信息补充边界', not writes, writes))
    if case.get('fault'):
        checks.append(criterion('fault_recovered', '首次连接故障实际注入且重试成功',
                                observation.get('fault_injected') and any(c.get('name') == 'get_order' and c.get('attempts', 0) >= 2 for c in successful), observation.get('fault_injected')))
    return score_checks(checks)


def strategy_observation(strategy, case, trace):
    delegated = [c for c in tool_results(trace) if c.get('name') == 'delegate_tasks']
    roles = sorted({e['data'].get('agent') for e in (trace or {}).get('events', []) if e['kind'] == 'specialist_done' and e['data'].get('agent')})
    expected = 'unconstrained' if strategy == 'current' else 'delegated' if strategy == 'selective' and case.get('orders') and case.get('knowledge') else 'direct'
    return {'expected': expected, 'actual': 'delegated' if delegated else 'direct', 'specialist_roles': roles,
            'followed': None if expected == 'unconstrained' else bool(delegated) and roles == ['knowledge', 'orders'] if expected == 'delegated' else not delegated}


def decision_metrics(trace, rejected_proposals=()):
    calls = tool_results(trace)
    invalid_codes = {'invalid_arguments', 'invalid_json', 'unknown_tool', 'tool_not_discovered', 'tool_not_loaded',
                     'invalid_plan_update', 'invalid_plan_tools', 'invalid_discovery', 'control_batch_conflict',
                     'plan_not_ready', 'skill_not_loaded', 'skill_not_found', 'skill_resource_not_found'}
    validated_tools = {'get_order', 'get_return_request', 'prepare_return_request', 'search_knowledge', 'search_history', 'delegate_tasks'}
    invalid = [c for c in calls if c.get('status') in ('invalid', 'invalid_arguments') or
               c.get('result', {}).get('error_code') in invalid_codes or
               (c.get('name') in validated_tools and not valid_arguments(c))]
    # Specialist runtime may reject a proposal before a tool_result is emitted.
    # If a trace does contain it, do not count the same rejection twice.
    unseen = [p for p in rejected_proposals if not any(c.get('name') == p['name'] and
              (c.get('call_id') == p['call_id'] or str(c.get('call_id', '')).endswith(':' + p['call_id'])) for c in invalid)]
    counts = Counter((c.get('name'), normalized_arguments(c.get('arguments'))) for c in calls if c.get('name') != 'update_plan')
    return {'invalid_calls': len(invalid) + len(unseen), 'rejected_specialist_proposals': len(rejected_proposals),
            'repeated_calls': sum(count - 1 for count in counts.values()),
            'control_calls': sum(c.get('name') in CONTROL_TOOLS for c in calls),
            'stalled_batches': (trace or {}).get('metrics', {}).get('stalled_batches', 0)}


def rejected_specialist_proposals(response, schemas):
    definitions = {tool['function']['name']: tool['function'].get('parameters', {}) for tool in schemas}
    rejected = []
    for call in response.get('choices', [{}])[0].get('message', {}).get('tool_calls') or []:
        function = call.get('function', {})
        reason = None
        if function.get('name') not in definitions:
            reason = 'tool_not_offered'
        else:
            try:
                arguments = json.loads(function.get('arguments'))
            except (ValueError, TypeError):
                reason = 'invalid_json'
            else:
                if not Draft202012Validator(definitions[function['name']]).is_valid(arguments):
                    reason = 'invalid_arguments'
        if reason:
            rejected.append({'name': function.get('name', ''), 'call_id': call.get('id', ''), 'reason': reason})
    return rejected


def aggregate(results):
    rows = []
    for strategy in dict.fromkeys(r['strategy'] for r in results):
        items = [r for r in results if r['strategy'] == strategy]
        executed = [r for r in items if r.get('executed', True)]
        complete = [r for r in items if r['metrics']['usage_status'] == 'complete' and r['metrics']['total_tokens'] is not None]
        reported = [r['metrics']['total_tokens'] for r in items if r['metrics']['total_tokens'] is not None]
        measured = [r['compliance']['followed'] for r in items if r['compliance']['followed'] is not None]
        rows.append({'strategy': strategy, 'trials': len(items), 'passed': sum(r['passed'] for r in items),
                     'success_rate': mean(r['passed'] for r in items),
                     'strategy_compliance_rate': mean(measured) if measured else None,
                     'executed_trials': len(executed),
                     'mean_elapsed_ms': round(mean(r['elapsed_ms'] for r in items), 1) if len(executed) == len(items) else None,
                     'mean_requests': round(mean(r['metrics']['request_count'] for r in items), 2) if len(executed) == len(items) else None,
                     'mean_total_tokens': round(mean(reported), 1) if len(complete) == len(items) else None,
                     'reported_tokens': sum(reported) if reported else None, 'complete_usage_trials': len(complete),
                     'invalid_call_coverage': 'partial' if any(r['metrics'].get('invalid_calls_coverage') == 'runtime_trace_only' for r in items) else 'complete',
                     **{key: sum(r['metrics'].get(key, 0) for r in items) for key in ('invalid_calls', 'repeated_calls', 'control_calls')},
                     'cases': [{'case_id': case, 'passed': sum(r['passed'] for r in items if r['case_id'] == case),
                                'trials': sum(r['case_id'] == case for r in items),
                                'mean_requests': round(mean(r['metrics']['request_count'] for r in items if r['case_id'] == case), 2) if all(r.get('executed', True) for r in items if r['case_id'] == case) else None,
                                'all_passed': all(r['passed'] for r in items if r['case_id'] == case)}
                               for case in dict.fromkeys(r['case_id'] for r in items)]})
    return rows


def initial_history(case, strategy):
    history = agent_service.new_conversation()
    instruction = STRATEGIES[strategy]['instruction']
    if instruction:
        history[0]['content'] += '\n[评测执行策略]\n' + instruction
    if case.get('seed_history'):
        # Synthetic prior turn, fixed for every variant; tools must be rediscovered in the current turn.
        history.extend([
            {'role': 'user', 'content': '查询 O1001 的订单状态。'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'seed-discovery', 'type': 'function', 'function': {'name': 'discover_tools', 'arguments': '{"names":["get_order"]}'}}]},
            {'role': 'tool', 'tool_call_id': 'seed-discovery', 'content': json.dumps({'status': 'loaded', 'loaded_names': ['get_order']})},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'seed-order', 'type': 'function', 'function': {'name': 'get_order', 'arguments': '{"order_id":"O1001"}'}}]},
            {'role': 'tool', 'tool_call_id': 'seed-order', 'content': json.dumps(agent_service.get_order('O1001'), ensure_ascii=False)},
            {'role': 'assistant', 'content': 'O1001 已发货。'}])
    return history


async def run_trial(client, app, case, trial, *, live, timeout_seconds):
    started, requests, attempts = monotonic(), [], 0
    observed = []
    sid = str(uuid4())
    history = initial_history(case, trial['strategy'])
    app.state.store.create_session(sid, history)
    observation = {'fault_injected': False, 'rejected_specialist_proposals': [], 'specialist_proposal_coverage': 'http_response_hook'}
    error = None
    original_remote = app.state.mcp_client.call_tool
    original_create = app.state.client.chat.completions.create

    async def remote(name, arguments, **kwargs):
        if case.get('fault') == 'tool' and name == 'get_order' and not observation['fault_injected']:
            observation['fault_injected'] = True
            raise ConnectionError('controlled transient fault')
        result = await original_remote(name, arguments, **kwargs)
        observed.append({'name': name, 'arguments': deepcopy(arguments), 'result': deepcopy(result)})
        return result

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        return await original_create(**kwargs)

    async def observe(request):
        requests.append(request_record(json.loads(request.content), sent=True))

    async def observe_response(response):
        body = json.loads(response.request.content)
        if body.get('stream') or not body.get('tools') or response.status_code != 200:
            return
        # with_options() copies the SDK resource but shares its HTTP client.
        # Read only the already-returning non-streaming JSON; aread caches bytes
        # for the SDK, and no prompts, headers or answer text are retained here.
        await response.aread()
        try:
            payload = response.json()
            observation['rejected_specialist_proposals'].extend(rejected_specialist_proposals(payload, body['tools']))
        except (ValueError, TypeError, AttributeError, IndexError):
            pass  # Let normal SDK/runtime handling diagnose a malformed response.

    hooks = app.state.client._client.event_hooks['request']
    response_hooks = app.state.client._client.event_hooks['response']
    hooks.append(observe)
    response_hooks.append(observe_response)
    try:
        with patch.object(app.state.mcp_client, 'call_tool', side_effect=remote), patch.object(app.state.client.chat.completions, 'create', side_effect=create):
            with anyio.fail_after(timeout_seconds):
                response = await client.post('/api/chat', json={'session_id': sid, 'message': case['prompt']})
                observation['http_status'] = response.status_code
                if response.status_code != 200:
                    error = response.json().get('detail', '请求失败')
    except TimeoutError:
        error = f'场景超过 {timeout_seconds:g} 秒期限。'
    except Exception as failure:
        error = agent_service.describe_error(failure, app.state.api_key)
    finally:
        hooks.remove(observe)
        response_hooks.remove(observe_response)
    saved = app.state.store.load_session(sid)['history']
    runs = app.state.traces.list_runs(sid)['runs']
    trace = app.state.traces.load(sid, runs[0]['run_id']) if runs else None
    observation.update(evidence=canonical_evidence(observed), checkpoint_cleared=app.state.checkpoints.load(sid) is None,
                       committed=sum(m['role'] == 'user' for m in saved) == (2 if case.get('seed_history') else 1))
    # Never mistake the seeded prior answer for this trial's answer after a failure.
    answer = next((m.get('content') or '' for m in reversed(saved[len(history):]) if m['role'] == 'assistant' and not m.get('tool_calls')), '')
    grade = grade_case(case, trace, answer, observation)
    return {**trial, 'title': case['title'], 'prompt': case['prompt'], 'session_id': sid, 'executed': True,
            'passed': grade['passed'] and error is None, 'trajectory': grade, 'answer': answer, 'error': error,
            'elapsed_ms': round((monotonic() - started) * 1000), 'observation': observation,
            'compliance': strategy_observation(trial['strategy'], case, trace),
            'metrics': {**metrics_for(trace, requests, attempts, live=live), **decision_metrics(trace, observation['rejected_specialist_proposals']),
                        'invalid_calls_coverage': 'runtime_and_specialist_proposals'},
            'requests': requests, 'trace': trace}


def report_snapshot(metadata, results, schedule):
    return {**metadata, 'completed_trials': len(results), 'total': len(schedule),
            'passed': sum(r['passed'] for r in results), 'complete': len(results) == len(schedule),
            'recommended_strategy': None, 'promotion': '保持生产策略；规则评测与小样本不自动决定上线。',
            'schedule': schedule, 'summary': aggregate(results), 'results': results}


def regrade_report(source):
    """Re-score saved observations without executing tools or contacting a model."""
    if source.get('suite') != 'decision_comparison' or source.get('dataset', {}).get('version') != DATASET_VERSION:
        raise ValueError('只支持当前案例版本的执行策略报告。')
    report = deepcopy(source)
    cases = {c['id']: c for c in report['dataset']['cases']}
    changes = []
    for item in report['results']:
        if not item.get('executed', True):
            continue
        previous = {'passed': item['passed'], 'invalid_calls': item['metrics']['invalid_calls']}
        observation = item['observation']
        item['trajectory'] = grade_case(cases[item['case_id']], item['trace'], item['answer'], observation)
        item['passed'] = item['trajectory']['passed'] and item.get('error') is None
        item['metrics'].update(decision_metrics(item['trace'], observation.get('rejected_specialist_proposals', [])))
        child_requests = item['metrics'].get('request_counts_by_role', {}).get('specialist', 0)
        item['metrics']['invalid_calls_coverage'] = 'runtime_trace_only' if child_requests and observation.get('specialist_proposal_coverage') != 'http_response_hook' else 'runtime_and_specialist_proposals'
        current = {'passed': item['passed'], 'invalid_calls': item['metrics']['invalid_calls']}
        if previous != current:
            changes.append({**{key: item[key] for key in ('case_id', 'strategy', 'repeat')}, 'before': previous, 'after': current})
    report['passed'] = sum(r['passed'] for r in report['results'])
    report['summary'] = aggregate(report['results'])
    report['analysis'] = {'regraded_at': datetime.now(UTC).isoformat(), 'source_report_sha256': digest(source),
                          'grader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'evidence_unchanged': True, 'changes': changes,
                          'note': '按更新的规则统一复核保存的回答及工具证据，没有再次请求模型。旧记录缺少子任务响应观察时，无效调用仅能统计已保存轨迹，可能少计；不补造缺失数据。'}
    return report


async def evaluate_decisions(cases=None, *, strategies=None, repeats=3, live=False, timeout_seconds=90, progress_path=None):
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds 必须为有限正数。')
    cases = deepcopy(CASES if cases is None else cases)
    strategies = list(STRATEGIES) if strategies is None else strategies
    schedule = make_schedule(cases, strategies, repeats)
    root = Path(__file__).resolve().parent
    file_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted([*root.glob('*.py'), *root.glob('skills/**/*.md')])}
    metadata = {'created_at': datetime.now(UTC).isoformat(), 'suite': 'decision_comparison', 'mode': 'live' if live else 'offline',
                'specialist_proposal_observation': 'shared_http_response_hook',
                'model_decisions_scripted': not live, 'repeats': repeats, 'timeout_seconds': timeout_seconds,
                'dataset': {'version': DATASET_VERSION, 'cases': cases, 'sha256': digest({'cases': cases, 'policy': POLICY_TEXT, 'seeds': [initial_history(c, 'current') for c in cases]})},
                'strategies': {s: {**STRATEGIES[s], 'sha256': digest(STRATEGIES[s])} for s in strategies},
                'environment': {'tool_mode': 'mcp', 'search_mode': 'keywords', 'skills_enabled': True,
                                'source_sha256': digest(file_hashes), 'source_files': file_hashes,
                                'packages': {name: importlib.metadata.version(name) for name in ('langgraph', 'openai', 'mcp')}},
                'method': '固定模拟订单和政策；每次独立临时会话。历史为控制夹具，非真实会话重放。只改变策略指令，工具/预算/校验相同。按案例与重复轮次轮换策略顺序。成功为证据与回答规则检查，非完整语义质量；策略遵循单独计分。失败及超时计入分母和耗时。请求数包含主管、子任务、校验及摘要；缺失用量不按零算。离线决策/校验/usage 为脚本，不比较质量或效率。无价格配置，不估费用。'}
    results, model = [], ScriptedConversationModel()

    def save():
        report = report_snapshot(metadata, results, schedule)
        if progress_path:
            write_report(report, progress_path)
        return report

    save()
    with TemporaryDirectory(prefix='agent-decision-eval-') as directory, ExitStack() as patches:
        environment = dict(os.environ) if live else {k: v for k, v in os.environ.items() if k.upper() in {'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
        environment['AGENT_HISTORY_SEARCH'] = 'keywords'
        if not live:
            environment.update(DEEPSEEK_API_KEY='offline-eval-key', DEEPSEEK_MODEL='offline-script')
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=model.client))
        patches.enter_context(patch.dict(os.environ, environment, clear=True))
        knowledge = Path(directory) / 'knowledge'
        knowledge.mkdir()
        (knowledge / 'policy.md').write_text(POLICY_TEXT, encoding='utf8')
        app = api_app.create_app(db_path=Path(directory) / 'eval.db', knowledge_dir=knowledge, tool_mode='mcp')
        try:
            # Bound startup, all trials and shutdown; each trial has its own deadline.
            with anyio.fail_after(timeout_seconds * (len(schedule) + 1)):
                async with app.router.lifespan_context(app):
                    if app.state.client is None:
                        raise ValueError('真实评测需要配置模型 API key。')
                    metadata['environment'].update(model=app.state.model, context_policy=asdict(app.state.context_policy), tool_policy=asdict(app.state.tool_policy))
                    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local', trust_env=False) as client:
                        for trial in schedule:
                            case = next(c for c in cases if c['id'] == trial['case_id'])
                            model.reset(scripted_case(case, trial['strategy']))
                            item = await run_trial(client, app, case, trial, live=live, timeout_seconds=timeout_seconds)
                            results.append(item)
                            save()
                            print(f"{'PASS' if item['passed'] else 'FAIL'} {trial['case_id']} / {trial['strategy']} / {trial['repeat']} · {item['metrics']['request_count']} requests · {item['elapsed_ms']} ms", flush=True)
        except Exception as failure:
            error = agent_service.describe_error(failure, getattr(app.state, 'api_key', ''))
            error_type = type(failure).__name__
            if isinstance(failure, TimeoutError):
                error = '评测环境超过总运行期限，剩余试验未完成。'
            metadata.update(lifecycle_error=error, lifecycle_error_type=error_type)
            for trial in schedule[len(results):]:
                results.append({**trial, 'passed': False, 'executed': False, 'elapsed_ms': None, 'error': error, 'answer': '',
                                'failure_kind': 'not_executed', 'compliance': {'followed': None},
                                'metrics': {**metrics_for(None, [], 0, live=live), **decision_metrics(None)},
                                'trajectory': score_checks([criterion('lifecycle', '评测环境正常启动运行', False, error)])})
    return save()


def report_html(report):
    from eval_decision_report import report_html as render
    return render(report)


def write_report(report, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, text in ((path, json.dumps(report, ensure_ascii=False, indent=2)), (path.with_suffix('.html'), report_html(report))):
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(text, encoding='utf8')
        # Windows readers/scanners may briefly hold a handle that denies rename.
        # Keep the previous report intact and retry only this atomic replacement;
        # persistent permission failures must still surface to the caller.
        for attempt in range(6):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                sleep(0.05 * 2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='显式请求真实模型；使用临时业务数据')
    parser.add_argument('--case', action='append', choices=[c['id'] for c in CASES])
    parser.add_argument('--strategy', action='append', choices=list(STRATEGIES))
    parser.add_argument('--repeats', type=int, default=3, choices=range(1, 11))
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--regrade', type=Path, help='仅重新评分旧报告，不执行任务或调用模型')
    args = parser.parse_args()
    if args.regrade:
        if args.live or args.case or args.strategy or args.repeats != 3 or args.timeout != 90:
            parser.error('--regrade 不能与运行选项一起使用。')
        output = args.output or args.regrade.with_name(args.regrade.stem + '-regraded.json')
        if output.suffix.lower() != '.json' or output.resolve() == args.regrade.resolve():
            parser.error('复核报告需写入不同的 .json 文件，保留原报告。')
        try:
            report = regrade_report(json.loads(args.regrade.read_text(encoding='utf8')))
        except (ValueError, KeyError, TypeError, OSError) as error:
            parser.error(str(error))
        write_report(report, output)
        print(f"{report['passed']}/{report['total']} passed (regraded) · {output.resolve()}")
        return 0 if report['complete'] and report['passed'] == report['total'] and not report.get('lifecycle_error') else 1
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout 必须是有限正数。')
    if args.strategy and len(args.strategy) != len(set(args.strategy)):
        parser.error('--strategy 不能重复。')
    path = args.output or Path(__file__).resolve().parent / '.local/evals' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-decisions-' + ('live' if args.live else 'offline') + '.json')
    if path.suffix.lower() != '.json':
        parser.error('--output 需要 .json 扩展名。')
    cases = [c for c in CASES if not args.case or c['id'] in args.case]
    report = anyio.run(lambda: evaluate_decisions(cases, strategies=args.strategy, repeats=args.repeats, live=args.live, timeout_seconds=args.timeout, progress_path=path))
    print(f"{report['passed']}/{report['total']} passed · {path.resolve()}")
    return 0 if report['passed'] == report['total'] and not report.get('lifecycle_error') else 1


if __name__ == '__main__':
    raise SystemExit(main())
