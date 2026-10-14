"""Agent 场景评测。默认离线；--live 才请求配置模型。只用临时业务数据库。"""
import argparse
from contextlib import ExitStack
from datetime import UTC, datetime
from html import escape
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

import anyio
import httpx2
from openai import AsyncOpenAI, APIConnectionError

import agent_service
import api_app
from eval_scenarios import CASES, grade_case
from eval_multiturn_scenarios import MULTITURN_CASES
from multiturn_eval import evaluate_multiturn


class CandidateFaultStream:
    """真实评测只替换最终文本，工具决策与独立校验仍来自实际模型。"""
    def __init__(self, stream, observation, *, repeated):
        self.stream, self.observation, self.repeated = stream, observation, repeated

    async def __aiter__(self):
        frames = [frame async for frame in self.stream]
        has_tools = any(choice.delta.tool_calls for frame in frames for choice in frame.choices)
        inject = not has_tools and (self.repeated or not self.observation.get('fault_injected'))
        changed = False
        for frame in frames:
            if inject:
                frame = frame.model_copy(deep=True)
                for choice in frame.choices:
                    if choice.delta.content:
                        choice.delta.content = 'O1003 已取消，退款已经到账。' if not changed else ''
                        changed = True
            yield frame
        if changed:
            self.observation['fault_injected'] = True

    async def close(self):
        await self.stream.close()


class ScriptStream(httpx2.AsyncByteStream):
    def __init__(self, response):
        self.response = response

    async def __aiter__(self):
        calls = self.response.get('calls', [])
        delta = {'tool_calls': [{**call, 'index': i, 'function': {**call['function'], 'arguments': json.dumps(call['function']['arguments'], ensure_ascii=False)}} for i, call in enumerate(calls)]} if calls else {'content': self.response['text']}
        frames = [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if calls else 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48}}]
        for frame in frames:
            yield ('data: ' + json.dumps({'id': 'offline-eval', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'offline-script', **frame}, ensure_ascii=False) + '\n\n').encode()
        yield b'data: [DONE]\n\n'


async def evaluate(cases, *, live=False):
    results = []
    script, position, review_script, review_position = [], 0, [], 0
    specialist_script, specialist_positions = {}, {}
    async def transport(request):
        nonlocal position, review_position
        body = json.loads(request.content)
        if body.get('response_format') == {'type': 'json_object'}:
            payload = json.loads(body['messages'][-1]['content'])
            verdict = review_script[review_position] if review_position < len(review_script) else 'pass'
            review_position += 1
            data = {'verdict': verdict, 'issues': ['订单状态与工具证据不一致，请修正。'] if verdict == 'revise' else [],
                    'evidence_ids': payload['current_evidence_ids']}
            return httpx2.Response(200, json={'id': 'offline-review', 'object': 'chat.completion', 'created': 0, 'model': 'offline-script',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(data, ensure_ascii=False)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48}})
        if not body.get('stream'):
            agent = 'orders' if body['tools'][0]['function']['name'] == 'get_order' else 'knowledge'
            number = specialist_positions.get(agent, 0)
            response = specialist_script[agent][number]
            specialist_positions[agent] = number + 1
            calls = response.get('calls')
            message = {'role': 'assistant', 'content': json.dumps(response, ensure_ascii=False)}
            if calls:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {**call, 'function': {**call['function'], 'arguments': json.dumps(call['function']['arguments'], ensure_ascii=False)}} for call in calls]}
            return httpx2.Response(200, json={'id': 'offline-specialist', 'object': 'chat.completion', 'created': 0, 'model': 'offline-script',
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
                'usage': {'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48}})
        if position >= len(script):
            raise ValueError('离线脚本已用完：Agent 出现额外模型请求。')
        response = script[position]; position += 1
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=ScriptStream(response))

    def offline_client(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport), trust_env=False))

    with TemporaryDirectory(prefix='agent-eval-') as directory, ExitStack() as patches:
        patches.enter_context(patch.dict(os.environ, {'AGENT_HISTORY_SEARCH': 'keywords'}))
        if not live:
            patches.enter_context(patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-eval-key', 'DEEPSEEK_MODEL': 'offline-script', 'AGENT_HISTORY_SEARCH': 'keywords'}, clear=True))
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=offline_client))
        root = Path(directory); knowledge = root / 'knowledge'; knowledge.mkdir()
        (knowledge / 'policy.md').write_text('# 评测专用示例政策\n\n## 退货期限\n签收后7天内可申请模拟退货，需人工核实；预览不等于提交或退款。', encoding='utf-8')
        app = api_app.create_app(db_path=root / 'eval.db', knowledge_dir=knowledge, tool_mode='local')
        async with app.router.lifespan_context(app):
            if app.state.client is None:
                raise ValueError('真实评测需要配置 DEEPSEEK_API_KEY。')
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local') as client:
                for case in cases:
                    script, position, review_script, review_position = case['script'], 0, case.get('reviews', []), 0
                    specialist_script, specialist_positions = case.get('specialists', {}), {}
                    sid = (await client.post('/api/sessions')).json()['session_id']
                    observation, started = {'preview_calls': 0, 'unapproved_preview_calls': 0}, monotonic()
                    original_order = agent_service.get_order
                    original_preview = app.state.returns.prepare_return_request
                    original_create = app.state.client.chat.completions.create
                    model_requests = 0

                    def flaky_order(order_id):
                        if case.get('fault') == 'specialist_tool':
                            observation['fault_injected'] = True
                            raise ConnectionError('评测注入的订单工具持续故障')
                        if case.get('fault') == 'tool' and not observation.get('fault_injected'):
                            observation['fault_injected'] = True
                            raise ConnectionError('评测注入的瞬时工具连接故障')
                        return original_order(order_id)

                    def observed_preview(session_id, order_id, reason):
                        # 调度器会计入预检拒绝尝试；这里只数真正进入业务函数的调用。
                        observation['preview_calls'] += 1
                        cp = app.state.checkpoints.load(session_id)
                        approvals = ((cp or {}).get('state') or {}).get('pending_batch', {}).get('approvals', [])
                        if not any(a['decision'] in ('approve', 'edit') and a['arguments'] == {'order_id': order_id, 'reason': reason} for a in approvals):
                            observation['unapproved_preview_calls'] += 1
                        return original_preview(session_id, order_id, reason)

                    async def flaky_model(**kwargs):
                        nonlocal model_requests
                        model_requests += int(kwargs.get('stream', False))
                        if case.get('fault') == 'model' and model_requests == 2:
                            observation['fault_injected'] = True
                            raise APIConnectionError(request=httpx2.Request('POST', 'https://api.deepseek.com/chat/completions'))
                        response = await original_create(**kwargs)
                        if live and kwargs.get('stream') and case.get('fault') in ('candidate', 'candidate_stuck'):
                            return CandidateFaultStream(response, observation, repeated=case['fault'] == 'candidate_stuck')
                        return response

                    error = None
                    try:
                        with patch.object(agent_service, 'get_order', side_effect=flaky_order), patch.object(app.state.returns, 'prepare_return_request', side_effect=observed_preview), patch.object(app.state.client.chat.completions, 'create', side_effect=flaky_model), anyio.fail_after(180):
                            response = await client.post('/api/chat', json={'session_id': sid, 'message': case['prompt']})
                            payload = response.json()
                            if payload.get('status') == 'awaiting_approval' and case.get('action'):
                                cp = payload['checkpoint']; observation['paused_before_action'] = True
                                observation['previews_before_action'] = len(app.state.returns.list_previews(sid))
                                for approval in cp['approvals']:
                                    body = {'run_id': cp['run_id'], 'approval_id': approval['approval_id'], 'decision': case['action']}
                                    if case['action'] == 'edit':
                                        body['arguments'] = {**approval['arguments'], 'reason': case['edited_reason']}
                                    decision = await client.post(f'/api/sessions/{sid}/checkpoint/decision', json=body)
                                    if decision.status_code != 200:
                                        raise ValueError('人工决定未保存：' + decision.text)
                                response = await client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
                            elif response.status_code >= 400 and case.get('fault') == 'model':
                                cp = app.state.checkpoints.load(sid)
                                if cp:
                                    response = await client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
                                    resumed_runs = app.state.traces.list_runs(sid)['runs']
                                    observation['resumed_same_run'] = len(resumed_runs) == 1 and resumed_runs[0]['run_id'] == cp['run_id']
                            observation['http_status'] = response.status_code
                            if response.status_code != 200:
                                error = response.json().get('detail', '请求失败')
                    except (Exception, TimeoutError) as failure:
                        error = agent_service.describe_error(failure, app.state.api_key)
                    saved = app.state.store.load_session(sid)
                    previews = app.state.returns.list_previews(sid)
                    observation['checkpoint_cleared'] = app.state.checkpoints.load(sid) is None
                    run_list = app.state.traces.list_runs(sid)['runs']
                    trace = app.state.traces.load(sid, run_list[0]['run_id']) if run_list else None
                    checks = grade_case(case, trace, saved['history'], previews, observation)
                    if error:
                        checks.append({'name': '场景执行无异常', 'passed': False, 'actual': error})
                    item = {'id': case['id'], 'title': case['title'], 'passed': all(c['passed'] for c in checks), 'elapsed_ms': round((monotonic() - started) * 1000),
                            'fault_injection': case.get('fault'), 'checks': checks, 'trace': trace}
                    results.append(item)
                    print(f"{'PASS' if item['passed'] else 'FAIL'} {case['id']} · {item['elapsed_ms']} ms", flush=True)
    return {'created_at': datetime.now(UTC).isoformat(), 'mode': 'live' if live else 'offline',
            'passed': sum(item['passed'] for item in results), 'total': len(results), 'results': results}


def report_html(report):
    mode = '真实模型：使用配置的模型，场景中的故障按定义注入。' if report['mode'] == 'live' else '离线流程评测：执行模型、校验模型及 token 用量均为脚本模拟，业务工具与数据库实际执行。'
    rows = []
    for result in report['results']:
        if result.get('kind') == 'multiturn':
            from eval_multiturn_report import multiturn_rows
            rows.extend(multiturn_rows(result))
            continue
        metrics = (result['trace'] or {}).get('metrics', {})
        details = ''.join(f"<li>{'通过' if c['passed'] else '失败'} · {escape(c['name'])}<pre>{escape(json.dumps(c['actual'], ensure_ascii=False, indent=2))}</pre></li>" for c in result['checks'])
        rows.append(f"<tr><td>{escape(result['title'])}</td><td>{'通过' if result['passed'] else '失败'}</td><td>{metrics.get('model_requests', 0)}</td><td>{metrics.get('total_tokens', 0) if metrics.get('reported_requests') else '未报告'}</td><td>{metrics.get('tool_attempts', 0)}</td><td>{result['elapsed_ms']}</td></tr><tr><td colspan='6'><details><summary>查看评测依据</summary><ul>{details}</ul></details></td></tr>")
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent 评测报告</title><style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:15px/1.7 system-ui;background:#f8faf5;color:#294337}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dfe5d7}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace}}summary{{cursor:pointer}}li{{margin:12px 0}}</style><h1>Agent 评测报告</h1><p>{escape(mode)}</p><p>{report['passed']} / {report['total']} 个场景通过 · {escape(report['created_at'])}</p><table><thead><tr><th>场景</th><th>结果</th><th>模型请求</th><th>已报告 tokens</th><th>工具调度尝试</th><th>耗时 ms</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>"""


def main():
    from skill_eval import SKILL_CASES, evaluate_skills
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--suite', choices=('single', 'multiturn', 'skills', 'all'), default=None, help='默认单轮；multiturn 连续对话；skills 技能选择及引用；all 全部')
    parser.add_argument('--case', action='append', choices=[case['id'] for case in CASES + MULTITURN_CASES + SKILL_CASES])
    parser.add_argument('--output', type=Path, help='JSON 报告路径；同目录生成同名 HTML')
    args = parser.parse_args()
    if args.output and args.output.suffix.lower() != '.json':
        parser.error('--output 请使用 .json 扩展名。')
    available = CASES + MULTITURN_CASES + SKILL_CASES if args.case and args.suite is None or args.suite == 'all' else SKILL_CASES if args.suite == 'skills' else MULTITURN_CASES if args.suite == 'multiturn' else CASES
    cases = [case for case in available if not args.case or case['id'] in args.case]
    if args.case and set(args.case) - {case['id'] for case in cases}:
        parser.error('--case 与 --suite 不匹配。')
    async def run_selected():
        single = [case for case in cases if 'turns' not in case and case.get('kind') != 'skills']
        multi = [case for case in cases if 'turns' in case]
        skills = [case for case in cases if case.get('kind') == 'skills']
        reports = []
        if single:
            reports.append(await evaluate(single, live=args.live))
        if multi:
            reports.append(await evaluate_multiturn(multi, live=args.live))
        if skills:
            reports.append(await evaluate_skills(skills, live=args.live))
        report = reports[0]
        for extra in reports[1:]:
            report['results'].extend(extra['results'])
            report['passed'] += extra['passed']
            report['total'] += extra['total']
        report['suite'] = 'all' if len(reports) > 1 else 'skills' if skills else 'multiturn' if multi else 'single'
        return report
    report = anyio.run(run_selected)
    path = args.output or Path(__file__).resolve().parent / '.local' / 'evals' / (datetime.now().strftime('%Y%m%d-%H%M%S') + ('-live' if args.live else '-offline') + '.json')
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    path.with_suffix('.html').write_text(report_html(report), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} passed; {path}")
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
