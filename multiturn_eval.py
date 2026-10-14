"""Same-session evaluations against the actual local API and temporary SQLite stores."""
from contextlib import ExitStack
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

import anyio
import httpx2
from openai import APIConnectionError, AsyncOpenAI

import agent_service
import api_app
from context_memory import ContextPolicy, SUMMARY_PREFIX
from eval_multiturn_scenarios import seeded_history
from eval_trajectory import criterion, grade_trajectory, score_checks, summarize_metrics, tool_results


class ScriptedConversationModel:
    """Replace only external model HTTP; summaries/reviews are explicitly simulated."""
    def __init__(self):
        self.reset({})

    def reset(self, turn, summary_text='用户当前订单为 O1003，状态与政策需查询工具核实。'):
        self.turn, self.summary_text = turn, summary_text
        self.position, self.review_position, self.specialist_positions = 0, 0, {}

    @staticmethod
    def completion(message, *, tools=False):
        return httpx2.Response(200, json={
            'id': 'offline-multiturn', 'object': 'chat.completion', 'created': 0, 'model': 'offline-script',
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if tools else 'stop'}],
            'usage': {'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48}})

    async def transport(self, request):
        body = json.loads(request.content)
        if body.get('stream'):
            from agent_eval import ScriptStream
            if self.position >= len(self.turn.get('script', [])):
                raise ValueError('离线脚本已用完：出现额外主管模型请求。')
            response = self.turn['script'][self.position]
            self.position += 1
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=ScriptStream(response))
        if body.get('response_format') == {'type': 'json_object'}:
            payload = json.loads(body['messages'][-1]['content'])
            reviews = self.turn.get('reviews', [])
            verdict = reviews[self.review_position] if self.review_position < len(reviews) else 'pass'
            self.review_position += 1
            report = {'verdict': verdict, 'issues': [] if verdict == 'pass' else ['请依据当前证据修正答复。'],
                      'evidence_ids': payload['current_evidence_ids']}
            return self.completion({'role': 'assistant', 'content': json.dumps(report, ensure_ascii=False)})
        if not body.get('tools'):
            return self.completion({'role': 'assistant', 'content': self.summary_text})
        agent = 'orders' if body['tools'][0]['function']['name'] == 'get_order' else 'knowledge'
        position = self.specialist_positions.get(agent, 0)
        script = self.turn.get('specialists', {}).get(agent, [])
        if position >= len(script):
            raise ValueError('离线子任务脚本已用完：' + agent)
        response = script[position]
        self.specialist_positions[agent] = position + 1
        calls = response.get('calls', [])
        message = {'role': 'assistant', 'content': json.dumps(response, ensure_ascii=False)}
        if calls:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {**call, 'function': {**call['function'], 'arguments': json.dumps(call['function']['arguments'], ensure_ascii=False)}} for call in calls]}
        return self.completion(message, tools=bool(calls))

    def client(self, **kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.transport), trust_env=False))


async def run_turn(client, app, sid, turn, index, expected_user_turns):
    """Faults occur at the model boundary, with actual tool invocations observed."""
    before = app.state.store.load_session(sid)
    previous_prompts = [m['content'] for m in before['history'] if m['role'] == 'user']
    previous_runs = {run['run_id'] for run in app.state.traces.list_runs(sid)['runs']}
    observation = {'actual_order_calls': 0, 'model_request_attempts': 0, 'fault_injected': False,
                   'resumed_same_run': False, 'prior_context_visible': not bool(previous_prompts)}
    original_order, original_create = agent_service.get_order, app.state.client.chat.completions.create
    first_actor = True

    def observed_order(order_id):
        observation['actual_order_calls'] += 1
        return original_order(order_id)

    async def observed_model(**kwargs):
        nonlocal first_actor
        observation['model_request_attempts'] += 1
        if kwargs.get('stream'):
            if first_actor:
                contents = [m.get('content') or '' for m in kwargs.get('messages', [])]
                observation['prior_context_visible'] = not previous_prompts or any(previous_prompts[-1] in content or SUMMARY_PREFIX in content for content in contents)
                first_actor = False
            if turn.get('fault') == 'model' and observation['actual_order_calls'] and not observation['fault_injected']:
                observation['fault_injected'] = True
                raise APIConnectionError(request=httpx2.Request('POST', 'https://api.deepseek.com/chat/completions'))
        return await original_create(**kwargs)

    started, error = monotonic(), None
    try:
        with patch.object(agent_service, 'get_order', side_effect=observed_order), patch.object(app.state.client.chat.completions, 'create', side_effect=observed_model), anyio.fail_after(180):
            response = await client.post('/api/chat', json={'session_id': sid, 'message': turn['prompt']})
            if response.status_code >= 400 and observation['fault_injected']:
                checkpoint = app.state.checkpoints.load(sid)
                if checkpoint:
                    response = await client.post(f'/api/sessions/{sid}/resume', json={'run_id': checkpoint['run_id']})
                    current_runs = {run['run_id'] for run in app.state.traces.list_runs(sid)['runs']} - previous_runs
                    observation['resumed_same_run'] = current_runs == {checkpoint['run_id']}
            observation['http_status'] = response.status_code
            if response.status_code != 200:
                error = response.json().get('detail', '请求失败')
    except (Exception, TimeoutError) as failure:
        error = agent_service.describe_error(failure, app.state.api_key)
    elapsed = round((monotonic() - started) * 1000)
    saved = app.state.store.load_session(sid)
    runs = [run for run in app.state.traces.list_runs(sid)['runs'] if run['run_id'] not in previous_runs]
    trace = app.state.traces.load(sid, runs[0]['run_id']) if runs else None
    observation['checkpoint_cleared'] = app.state.checkpoints.load(sid) is None
    user_turns = sum(message['role'] == 'user' for message in saved['history'])
    trajectory = grade_trajectory(turn, trace)
    checks = trajectory['checks'] + [
        criterion('api_success', '本地API成功完成', observation.get('http_status') == 200 and error is None, error or observation.get('http_status')),
        criterion('one_committed_turn', '本轮只提交一次用户问题', user_turns == expected_user_turns, user_turns),
        criterion('checkpoint_cleared', '完成后清除检查点', observation['checkpoint_cleared'], observation['checkpoint_cleared']),
        criterion('prior_context_available', '请求能看到前文或滚动摘要', observation['prior_context_visible'], observation['prior_context_visible']),
    ]
    if turn.get('fault') == 'model':
        events = (trace or {}).get('events', [])
        checks.extend([
            criterion('fault_injected', '工具之后已注入模型连接故障', observation['fault_injected'], observation['fault_injected']),
            criterion('same_run_resumed', '故障后继续同一执行', observation['resumed_same_run'] and any(e['kind'] == 'resumed' for e in events), observation['resumed_same_run']),
            criterion('actual_tools_not_replayed', '底层订单工具也只执行一次', observation['actual_order_calls'] == 1, observation['actual_order_calls']),
        ])
    result_grade = score_checks(checks)
    answer = next((m.get('content') or '' for m in reversed(saved['history'][len(before['history']):]) if m['role'] == 'assistant' and not m.get('tool_calls')), '')
    return {'turn': index, 'prompt': turn['prompt'], 'answer': answer, 'session_id': sid,
            'passed': result_grade['passed'], 'elapsed_ms': elapsed, 'committed_user_turns': user_turns,
            'actual_orders': [(c.get('arguments') or {}).get('order_id') for c in tool_results(trace) if c.get('name') == 'get_order' and c.get('status') == 'ok' and c.get('attempts', 0) > 0],
            'trajectory': result_grade, 'checks': checks, 'observation': observation, 'trace': trace}


async def evaluate_multiturn(cases, *, live=False):
    results, model = [], ScriptedConversationModel()
    with TemporaryDirectory(prefix='agent-multiturn-eval-') as directory, ExitStack() as patches:
        patches.enter_context(patch.dict(os.environ, {'AGENT_HISTORY_SEARCH': 'keywords'}))
        if not live:
            patches.enter_context(patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-eval-key', 'DEEPSEEK_MODEL': 'offline-script', 'AGENT_HISTORY_SEARCH': 'keywords'}, clear=True))
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=model.client))
        root = Path(directory)
        knowledge = root / 'knowledge'
        knowledge.mkdir()
        (knowledge / 'policy.md').write_text('# 评测专用示例政策\n\n## 退货期限\n签收后7天内可申请模拟退货，需人工核实；预览不等于提交或退款。', encoding='utf-8')
        app = api_app.create_app(db_path=root / 'eval.db', knowledge_dir=knowledge, tool_mode='local')
        async with app.router.lifespan_context(app):
            if app.state.client is None:
                raise ValueError('真实评测需要配置 DEEPSEEK_API_KEY。')
            configured_policy = app.state.context_policy
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local') as client:
                for case in cases:
                    sid = (await client.post('/api/sessions')).json()['session_id']
                    fixture = case.get('fixture', {'kind': 'none', 'seeded_turns': 0, 'description': '所有对话轮次均通过本地API。'})
                    initial = seeded_history(case, agent_service.new_conversation())
                    if fixture.get('seeded_turns'):
                        app.state.store.replace_history(sid, initial)
                    # A reproducible stress budget applies only to this temporary
                    # scenario; normal cases retain the configured live policy.
                    app.state.context_policy = ContextPolicy(max_tokens=8192) if case.get('summary') else configured_policy
                    started, turns = monotonic(), []
                    for index, turn in enumerate(case['turns'], 1):
                        model.reset(turn, case.get('summary_text', '用户当前订单为 O1003，状态与政策需查询工具核实。'))
                        app.state.sessions.pop(sid, None)  # Require DB reload, not only the in-memory cache.
                        result = await run_turn(client, app, sid, turn, index, fixture.get('seeded_turns', 0) + index)
                        result['metrics'] = summarize_metrics([result['trace']], simulated=not live)
                        turns.append(result)
                        if not result['observation']['checkpoint_cleared']:
                            break  # Retain an unfinished checkpoint; do not send unrelated turns into it.
                    saved = app.state.store.load_session(sid)
                    prompts = [m['content'] for m in saved['history'] if m['role'] == 'user']
                    expected_prompts = [m['content'] for m in initial if m['role'] == 'user'] + [t['prompt'] for t in case['turns']]
                    checks = [
                        criterion('all_turns_executed', '全部场景轮次已执行', len(turns) == len(case['turns']), len(turns)),
                        criterion('same_session', '全部轮次复用同一会话', bool(turns) and all(t['session_id'] == sid for t in turns), sid),
                        criterion('persisted_conversation', 'SQLite保存完整有序对话且无重复提交', prompts == expected_prompts, {'expected': len(expected_prompts), 'actual': len(prompts)}),
                        criterion('fixture_history_preserved', '原始历史未被摘要替换', saved['history'][:len(initial)] == initial, len(initial)),
                    ]
                    traces = [turn['trace'] for turn in turns]
                    metrics = summarize_metrics(traces, simulated=not live)
                    if case.get('summary'):
                        checks.append(criterion('summary_persisted', '长历史产生并提交滚动摘要', bool(saved['memory'].text) and saved['memory'].covered_messages > 0 and metrics['summary_requests'] > 0,
                                                {'covered_messages': saved['memory'].covered_messages, 'summary_requests': metrics['summary_requests']}))
                    all_checks = checks + [{**check, 'id': f"turn_{turn['turn']}:{check['id']}"} for turn in turns for check in turn['checks']]
                    trajectory = score_checks(all_checks)
                    item = {'id': case['id'], 'title': case['title'], 'kind': 'multiturn', 'session_id': sid,
                            'passed': trajectory['passed'], 'elapsed_ms': round((monotonic() - started) * 1000),
                            'fixture': fixture, 'turns': turns, 'checks': checks, 'trajectory': trajectory, 'metrics': metrics,
                            'memory': {'covered_messages': saved['memory'].covered_messages, 'summary_chars': len(saved['memory'].text)},
                            'context_policy': {'max_tokens': app.state.context_policy.max_tokens, 'input_limit': app.state.context_policy.input_limit}}
                    results.append(item)
                    print(f"{'PASS' if item['passed'] else 'FAIL'} {case['id']} · {trajectory['score']}/100 · {item['elapsed_ms']} ms", flush=True)
    return {'created_at': datetime.now(UTC).isoformat(), 'mode': 'live' if live else 'offline', 'suite': 'multiturn',
            'score_method': 'Equal-weight, rule-based constraint satisfaction; not a semantic quality rating. Offline model decisions, reviews and token usage are scripted.',
            'passed': sum(item['passed'] for item in results), 'total': len(results), 'results': results}
