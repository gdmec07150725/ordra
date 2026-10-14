"""正式聊天图与单问题实验图的流程对照；默认离线，--live 使用配置模型。"""
import argparse
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
from openai import AsyncOpenAI, APIConnectionError

import agent_eval
import agent_service
import api_app
from eval_scenarios import CASES


COMPARISON_CASES = [case for case in CASES if case['id'] in (
    'order_query', 'knowledge_rag', 'approval_edit', 'checkpoint_resume')]


def completion(response):
    calls = response.get('calls', [])
    message = {'role': 'assistant', 'content': response.get('text')}
    if calls:
        message['tool_calls'] = [{**call, 'function': {**call['function'],
            'arguments': json.dumps(call['function']['arguments'], ensure_ascii=False)}} for call in calls]
    return {'id': 'graph-eval', 'object': 'chat.completion', 'created': 0, 'model': 'offline-script',
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
            'usage': {'prompt_tokens': 40, 'completion_tokens': 8, 'total_tokens': 48}}


def grade_graph(case, result, observed):
    checks = []
    def check(name, passed, actual):
        checks.append({'name': name, 'passed': bool(passed), 'actual': actual})
    check('结果通过校验并完成', result.get('status') == 'completed' and result.get('answer'), result.get('status'))
    calls = [event['tool'] for event in result.get('events', []) if event.get('tool')]
    successes = [call for call in calls if call.get('status') == 'ok' and call.get('attempts', 0) > 0]
    if case.get('orders'):
        orders = [call['arguments'].get('order_id') for call in successes if call['name'] == 'get_order']
        check('指定订单实际查询一次', orders == case['orders'], orders)
    if case.get('knowledge'):
        found = [call for call in successes if call['name'] == 'search_knowledge' and call.get('result', {}).get('returned_items', 0) > 0]
        check('真实检索得到知识片段', bool(found) and result.get('sources'), len(found))
    if case.get('action'):
        previews = result.get('return_previews', [])
        check('审批前未生成预览', observed.get('paused_before_action') and observed.get('preview_calls_before') == 0, observed.get('preview_calls_before'))
        check('修改后业务函数仅调用一次', observed.get('preview_calls') == 1, observed.get('preview_calls'))
        valid = len(previews) == 1 and previews[0].get('order_id') == case['preview_order'] and previews[0].get('reason') == case['edited_reason'] and previews[0].get('state') == 'pending'
        check('采用修改后的原因且没有提交', valid, previews)
    if case.get('fault') == 'model':
        check('模型故障后从原实验恢复', observed.get('fault_injected') and result.get('resume_count') == 1, result.get('resume_count'))
        check('失败请求计入额度', result.get('metrics', {}).get('model_requests', 0) >= 3, result.get('metrics'))
    check('校验请求在预算内', 1 <= result.get('metrics', {}).get('verification_requests', 0) <= 3, result.get('metrics', {}).get('verification_requests'))
    return checks


async def evaluate_graph(cases, *, live=False):
    script, position = [], 0
    async def transport(request):
        nonlocal position
        body = json.loads(request.content)
        if body.get('response_format'):
            payload = json.loads(body['messages'][-1]['content'])
            return httpx2.Response(200, json=completion({'text': json.dumps({
                'verdict': 'pass', 'issues': [], 'evidence_ids': payload['current_evidence_ids']})}))
        if position >= len(script):
            raise ValueError('图流程发出了离线脚本之外的模型请求。')
        response = script[position]; position += 1
        return httpx2.Response(200, json=completion(response))

    def offline_client(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport), trust_env=False))

    results = []
    with TemporaryDirectory(prefix='langgraph-eval-') as directory, ExitStack() as patches:
        patches.enter_context(patch.dict(os.environ, {'AGENT_HISTORY_SEARCH': 'keywords'}))
        if not live:
            patches.enter_context(patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-eval-key', 'DEEPSEEK_MODEL': 'offline-script', 'AGENT_HISTORY_SEARCH': 'keywords'}, clear=True))
            patches.enter_context(patch.object(api_app, 'load_dotenv', return_value=False))
            patches.enter_context(patch.object(api_app, 'AsyncOpenAI', side_effect=offline_client))
        root = Path(directory); knowledge = root / 'knowledge'; knowledge.mkdir()
        (knowledge / 'policy.md').write_text('# 评测专用示例政策\n\n## 退货期限\n签收后7天内可申请模拟退货，需人工核实；预览不等于提交或退款。', encoding='utf8')
        app = api_app.create_app(db_path=root / 'eval.db', knowledge_dir=knowledge, tool_mode='local')
        async with app.router.lifespan_context(app):
            if app.state.client is None:
                raise ValueError('真实评测需要配置 DEEPSEEK_API_KEY。')
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url='http://eval.local') as client:
                await client.get('/api/graph/runs')
                manager = app.state.graph_experiments
                for case in cases:
                    script, position = case['script'], 0
                    # 实验仅生成预览；两种引擎使用相同问题和真实业务函数。
                    if case.get('action'):
                        script = script[:-1] + [{'text': '已按修改后的原因生成实验退货预览，尚未提交。'}]
                    observed, started, result, failure = {'preview_calls': 0}, monotonic(), {}, None
                    original_model = app.state.client.chat.completions.create
                    original_options = app.state.client.with_options
                    original_preview = manager.returns.prepare_return_request
                    requests = 0
                    async def model(**kwargs):
                        nonlocal requests
                        if not kwargs.get('response_format'):
                            requests += 1
                            if case.get('fault') == 'model' and requests == 2:
                                observed['fault_injected'] = True
                                raise APIConnectionError(request=httpx2.Request('POST', 'https://api.deepseek.com/chat/completions'))
                        return await original_model(**kwargs)
                    def preview(*args):
                        observed['preview_calls'] += 1
                        return original_preview(*args)
                    def instrumented_client(**kwargs):
                        # GraphAgent 克隆 SDK 客户端以禁用内部重试，故障注入也必须
                        # 装到这个实际发送请求的克隆上。请求仍走同一个真实 HTTP 层。
                        copied = original_options(**kwargs)
                        copied.chat.completions.create = model
                        return copied
                    async def settle(run_id):
                        while True:
                            response = await client.get(f'/api/graph/runs/{run_id}')
                            response.raise_for_status()
                            state = response.json()
                            if not state['busy']:
                                return state
                            await anyio.sleep(.02)
                    try:
                        with patch.object(app.state.client, 'with_options', side_effect=instrumented_client), patch.object(manager.returns, 'prepare_return_request', side_effect=preview), anyio.fail_after(180):
                            response = await client.post('/api/graph/runs', json={'question': case['prompt']})
                            response.raise_for_status(); run_id = response.json()['run_id']
                            result = await settle(run_id)
                            if result['status'] == 'awaiting_approval' and case.get('action'):
                                observed.update(paused_before_action=True, preview_calls_before=observed['preview_calls'])
                                approval = result['approvals'][0]
                                body = {'approval_id': approval['approval_id'], 'decision': case['action'],
                                        'arguments': {**approval['arguments'], 'reason': case['edited_reason']}}
                                response = await client.post(f'/api/graph/runs/{run_id}/resume', json=body)
                                response.raise_for_status(); result = await settle(run_id)
                            elif result['status'] == 'failed' and observed.get('fault_injected'):
                                response = await client.post(f'/api/graph/runs/{run_id}/resume', json={})
                                response.raise_for_status(); result = await settle(run_id)
                    except Exception as error:
                        failure = agent_service.describe_error(error, app.state.api_key)
                    checks = grade_graph(case, result, observed)
                    if failure or result.get('error'):
                        checks.append({'name': '场景执行无异常', 'passed': False, 'actual': failure or result['error']})
                    # 复用 HTML 报告布局；原始公开节点状态另存 graph 字段。
                    metrics = result.get('metrics', {})
                    attempts = sum(event['tool']['attempts'] for event in result.get('events', []) if event.get('tool'))
                    trace = {'metrics': {**metrics, 'tool_attempts': attempts}}
                    item = {'id': 'experiment_graph:' + case['id'], 'engine': 'experiment_graph', 'title': '单问题实验图 · ' + case['title'],
                            'passed': all(check['passed'] for check in checks), 'checks': checks,
                            'elapsed_ms': round((monotonic() - started) * 1000), 'trace': trace, 'graph': result,
                            'fault_injection': case.get('fault')}
                    results.append(item)
                    print(f"{'PASS' if item['passed'] else 'FAIL'} {item['id']} · {item['elapsed_ms']} ms", flush=True)
    return results


async def compare(*, live=False, case_ids=None):
    cases = [case for case in COMPARISON_CASES if case_ids is None or case['id'] in case_ids]
    chat = await agent_eval.evaluate(cases, live=live)
    results = [{**row, 'id': 'chat_graph:' + row['id'], 'engine': 'chat_graph', 'title': '正式聊天图 · ' + row['title']} for row in chat['results']]
    results += await evaluate_graph(cases, live=live)
    return {'created_at': datetime.now(UTC).isoformat(), 'mode': 'live' if live else 'offline',
            'passed': sum(row['passed'] for row in results), 'total': len(results), 'results': results}


def report_html(report):
    return agent_eval.report_html(report).replace('Agent 评测报告', '正式聊天图 / 单问题实验图流程对照')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--case', action='append', choices=[case['id'] for case in COMPARISON_CASES])
    parser.add_argument('--output', type=Path, default=Path('.local/evals/langgraph-comparison.json'))
    args = parser.parse_args()
    if args.output.suffix.lower() != '.json':
        parser.error('--output 必须是 .json 文件')
    report = anyio.run(lambda: compare(live=args.live, case_ids=args.case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    args.output.with_suffix('.html').write_text(report_html(report), encoding='utf8')
    print(f"{report['passed']} / {report['total']} 通过；报告：{args.output.resolve()}")
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
