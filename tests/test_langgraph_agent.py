"""Real StateGraph + SQLite + SDK transport tests for isolated experiments."""
import asyncio
from contextlib import AsyncExitStack
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx2
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI, APIStatusError

from context_memory import ContextPolicy
from tool_runtime import ToolPolicy


def completion(text=None, calls=None, finish=None):
    message = {'role': 'assistant', 'content': text}
    if calls is not None:
        message['tool_calls'] = [{'id': ident, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}} for ident, name, args in calls]
    return {'id': 'graph-test', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
            'choices': [{'index': 0, 'finish_reason': finish or ('tool_calls' if calls is not None else 'stop'),
                         'message': message}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}}


class GraphAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(importlib.util.find_spec('langgraph_agent'), 'GraphAgent core has not been implemented')
        from langgraph_agent import GraphAgent
        self.GraphAgent = GraphAgent
        self.temp = TemporaryDirectory()
        self.stack = AsyncExitStack()
        self.responses, self.requests, self.return_calls, self.searches = [], [], [], []
        self.review_responses = []
        self.block_actor = None
        self.block_review = None
        self.entered = asyncio.Event()

        async def handle(request):
            body = json.loads(request.content)
            self.requests.append(body)
            reviewing = bool(body.get('response_format'))
            block = self.block_review if reviewing else self.block_actor
            if block is not None:
                self.entered.set()
                await block.wait()
            if reviewing:
                data = json.loads(body['messages'][-1]['content'])
                response = self.review_responses.pop(0) if self.review_responses else {
                    'verdict': 'pass', 'issues': [], 'evidence_ids': data['current_evidence_ids']}
                if response == 'error':
                    return httpx2.Response(503, json={'error': {'message': 'offline unavailable', 'type': 'server_error'}})
                return httpx2.Response(200, json=completion(json.dumps(response, ensure_ascii=False)))
            response = self.responses.pop(0)
            if response == 'error':
                return httpx2.Response(503, json={'error': {'message': 'offline unavailable', 'type': 'server_error'}})
            return httpx2.Response(200, json=response)

        self.client = AsyncOpenAI(api_key='offline-only', max_retries=2,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False))
        self.saver = await self.stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(Path(self.temp.name) / 'graph.db')))
        self.agent = self.new_agent()

    async def asyncTearDown(self):
        if hasattr(self, 'stack'):
            await self.stack.aclose()
            await self.client.close()
            self.temp.cleanup()

    def new_agent(self, **kwargs):
        return self.GraphAgent(self.client, 'offline', self.saver, search_knowledge=self.search,
                               return_handler=self.prepare, **kwargs)

    async def search(self, query):
        self.searches.append(query)
        return {'results': [{'source': 'kb:fixture:0', 'version': 'v1', 'title': '售后示例',
                             'section': '条件', 'text': '示例：商品质量问题可以申请退货。'}]}

    async def prepare(self, name, arguments):
        self.return_calls.append((name, arguments))
        return {'preview': {'preview_id': 'preview-local', **arguments, 'state': 'pending', 'request_id': None}}

    async def restart(self):
        await self.stack.aclose()
        self.stack = AsyncExitStack()
        self.saver = await self.stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(Path(self.temp.name) / 'graph.db')))
        self.agent = self.new_agent()

    async def test_order_query_uses_actual_tool_evidence_and_default_verification(self):
        self.responses = [completion(calls=[('order', 'get_order', {'order_id': 'O1003'})]), completion('O1003 已签收。')]
        result = await self.agent.run('order-run', question='查询 O1003')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], 'O1003 已签收。')
        self.assertEqual(result['metrics']['model_requests'], 2)
        self.assertEqual(result['metrics']['verification_requests'], 1)
        self.assertEqual(result['metrics']['tool_calls'], 1)
        self.assertEqual(result['metrics']['total_tokens'], 120)
        self.assertEqual(result['metrics']['reported_requests'], 3)
        evidence = json.loads(self.requests[-1]['messages'][-1]['content'])
        order = next(json.loads(m['content']) for m in evidence['messages'] if m['role'] == 'tool')
        self.assertEqual(order['order_id'], 'O1003')
        self.assertEqual(order['status'], '已签收')
        self.assertEqual(evidence['current_evidence_ids'], ['order'])
        self.assertTrue(all(request.get('thinking') == {'type': 'disabled'} for request in self.requests))
        self.assertEqual({tool['function']['name'] for tool in self.requests[0]['tools']},
                         {'get_order', 'search_knowledge', 'prepare_return_request'})
        with self.assertRaises(ValueError):
            await self.agent.run('order-run')
        with self.assertRaises(ValueError):
            await self.agent.run('order-run', question='覆盖旧问题')

    async def test_missing_order_is_a_verified_clarification(self):
        self.responses = [completion('请提供订单号。')]
        result = await self.agent.run('missing', question='订单现在怎么样')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], '请提供订单号。')
        self.assertEqual(result['metrics']['tool_calls'], 0)

    async def test_retrieval_preserves_source_evidence(self):
        self.responses = [completion(calls=[('policy', 'search_knowledge', {'query': '退货条件'})]),
                          completion('《售后示例》说明质量问题可以申请退货。[kb:fixture:0]')]
        result = await self.agent.run('rag', question='退货条件是什么')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['sources'][0]['source'], 'kb:fixture:0')
        self.assertEqual(result['metrics']['knowledge_calls'], 1)
        self.assertEqual(self.searches, ['退货条件'])

    async def test_approval_restart_edit_uses_actual_arguments_without_repeating_order_query(self):
        self.responses = [completion(calls=[('order', 'get_order', {'order_id': 'O1003'})]),
            completion(calls=[('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '左侧无声'})]),
            completion('已生成 O1004 的退货预览，尚未提交。')]
        paused = await self.agent.run('approval', question='O1003 左侧无声，请生成退货预览')
        self.assertEqual(paused['status'], 'awaiting_approval')
        self.assertEqual(paused['answer'], '')
        self.assertEqual(paused['return_previews'], [])
        self.assertEqual(self.return_calls, [])
        approval = paused['approvals'][0]
        with self.assertRaises((ValueError, LookupError)):
            await self.agent.run('approval', decision={'approval_id': 'stale', 'decision': 'approve'})
        with self.assertRaises(ValueError):
            await self.agent.run('approval', decision={'approval_id': approval['approval_id'], 'decision': 'edit',
                'arguments': {'order_id': 'O1004', 'reason': ''}})
        self.assertEqual((await self.agent.get('approval'))['status'], 'awaiting_approval')
        await self.restart()
        done = await self.agent.run('approval', decision={'approval_id': approval['approval_id'], 'decision': 'edit',
            'arguments': {'order_id': 'O1004', 'reason': '外壳开裂'}})
        self.assertEqual(done['status'], 'completed', done)
        self.assertEqual(done['return_previews'][0]['order_id'], 'O1004')
        self.assertEqual(self.return_calls, [('prepare_return_request', {'order_id': 'O1004', 'reason': '外壳开裂'})])
        self.assertEqual(done['metrics']['tool_calls'], 2)
        tool_events = [e for e in done['events'] if e.get('tool', {}).get('name') == 'prepare_return_request']
        self.assertEqual(tool_events[0]['tool']['arguments']['order_id'], 'O1004')

    async def test_approve_generates_only_one_preview(self):
        self.responses = [completion(calls=[('p1', 'prepare_return_request', {'order_id': 'O1003', 'reason': '无声'})]),
            completion(calls=[('p2', 'prepare_return_request', {'order_id': 'O1003', 'reason': '再试'})]),
            completion('预览已生成，未提交。')]
        paused = await self.agent.run('approve', question='O1003 无声，生成预览')
        done = await self.agent.run('approve', decision={'approval_id': paused['approvals'][0]['approval_id'], 'decision': 'approve'})
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(len(done['return_previews']), 1)
        self.assertEqual(len(self.return_calls), 1)

    async def test_rejection_blocks_retry_with_new_arguments(self):
        self.responses = [completion(calls=[('p1', 'prepare_return_request', {'order_id': 'O1003', 'reason': '无声'})]),
            completion(calls=[('p2', 'prepare_return_request', {'order_id': 'O1004', 'reason': '开裂'})]),
            completion('已取消生成预览，未提交申请。')]
        paused = await self.agent.run('reject', question='O1003 无声，生成预览')
        done = await self.agent.run('reject', decision={'approval_id': paused['approvals'][0]['approval_id'], 'decision': 'reject'})
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['return_previews'], [])
        self.assertEqual(self.return_calls, [])
        self.assertIn('human_rejected', json.dumps(self.requests, ensure_ascii=False))

    async def test_actor_errors_reserve_each_actual_retry_and_stop_at_budget(self):
        self.responses = ['error'] * 4
        for index in range(4):
            with self.assertRaises(APIStatusError):
                await self.agent.run('fail', **({'question': '查询订单'} if index == 0 else {}))
            self.assertEqual((await self.agent.get('fail'))['metrics']['model_requests'], index + 1)
            if index == 1:
                await self.restart()
        result = await self.agent.run('fail')
        self.assertEqual(result['status'], 'unverified')
        self.assertEqual(len(self.requests), 4)

    async def test_cancelled_actor_is_charged_on_resume(self):
        self.responses = [completion('请提供订单号。')]
        self.block_actor = asyncio.Event()
        task = asyncio.create_task(self.agent.run('cancel', question='查询订单'))
        await asyncio.wait_for(self.entered.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((await self.agent.get('cancel'))['metrics']['model_requests'], 1)
        self.block_actor = None
        await self.restart()
        done = await self.agent.run('cancel')
        self.assertEqual(done['metrics']['model_requests'], 2)
        self.assertEqual(done['status'], 'completed')

    async def test_review_error_resumes_candidate_without_publishing_or_actor_replay(self):
        self.responses = [completion('请提供订单号。')]
        self.review_responses = ['error']
        with self.assertRaises(APIStatusError):
            await self.agent.run('review-error', question='查询订单')
        paused = await self.agent.get('review-error')
        self.assertEqual(paused['answer'], '')
        self.assertNotIn('请提供订单号', json.dumps(paused, ensure_ascii=False))
        await self.restart()
        done = await self.agent.run('review-error')
        self.assertEqual(done['metrics']['model_requests'], 1)
        self.assertEqual(done['metrics']['verification_requests'], 2)
        self.assertEqual(done['answer'], '请提供订单号。')

    async def test_failed_reviews_hide_candidates_and_stop_after_two_corrections(self):
        self.responses = [completion('绝不能公开的错误草稿')] * 3
        self.review_responses = [{'verdict': 'revise', 'issues': ['绝不能公开的错误草稿缺少证据'], 'evidence_ids': []}] * 3
        done = await self.agent.run('revise', question='查询订单')
        self.assertEqual(done['status'], 'unverified')
        self.assertEqual(done['metrics']['corrections'], 2)
        self.assertEqual(done['metrics']['verification_requests'], 3)
        self.assertNotIn('绝不能公开', json.dumps(done, ensure_ascii=False))

    async def test_complete_current_evidence_over_budget_stops_before_next_model(self):
        async def enormous(query):
            return {'results': [{'source': 'large', 'version': '1', 'text': '事实' * 20000}]}
        self.agent = self.GraphAgent(self.client, 'offline', self.saver, search_knowledge=enormous, return_handler=self.prepare)
        self.responses = [completion(calls=[('policy', 'search_knowledge', {'query': '退货'})])]
        done = await self.agent.run('large', question='退货规则')
        self.assertEqual(done['status'], 'unverified')
        self.assertEqual(len(self.requests), 1)

    async def test_unsupported_and_empty_tool_batches_fail_closed(self):
        for ident, response in [('unknown', completion(calls=[('submit', 'submit_return_request', {})])),
                                ('empty', completion(calls=[])), ('truncated', completion('半截草稿', finish='length'))]:
            self.responses = [response]
            with self.assertRaises(ValueError):
                await self.agent.run(ident, question='生成退货申请')
            public = await self.agent.get(ident)
            self.assertEqual(public['answer'], '')
            self.assertEqual(public['metrics']['tool_calls'], 0)
        self.assertEqual(self.return_calls, [])

    async def test_later_tool_failure_does_not_replay_saved_earlier_tools(self):
        first = True
        async def search(query):
            nonlocal first
            if first:
                first = False
                raise RuntimeError('temporary test interruption')
            return await self.search(query)
        self.agent = self.GraphAgent(self.client, 'offline', self.saver, search_knowledge=search, return_handler=self.prepare)
        self.responses = [completion(calls=[('order', 'get_order', {'order_id': 'O1003'}),
                                           ('policy', 'search_knowledge', {'query': '退货'})]),
                          completion('O1003 已签收，商品质量问题可以申请退货。')]
        with self.assertRaisesRegex(RuntimeError, 'test interruption'):
            await self.agent.run('partial', question='查询 O1003 和退货规则')
        paused = await self.agent.get('partial')
        self.assertEqual(paused['metrics']['tool_calls'], 2)
        self.assertEqual([event['tool']['call_id'] for event in paused['events'] if 'tool' in event], ['order'])
        await self.restart()
        done = await self.agent.run('partial')
        self.assertEqual(done['status'], 'completed')
        self.assertEqual([event['tool']['call_id'] for event in done['events'] if 'tool' in event], ['order', 'policy'])
        self.assertEqual(done['metrics']['model_requests'], 2)
        self.assertEqual(done['metrics']['tool_calls'], 2)

    async def test_tool_and_knowledge_budgets_stop_without_excess_execution(self):
        self.agent = self.new_agent(tool_policy=ToolPolicy(max_calls=2))
        self.responses = [completion(calls=[('order1', 'get_order', {'order_id': 'O1003'}),
                                           ('order2', 'get_order', {'order_id': 'O1004'})]),
                          completion(calls=[('order3', 'get_order', {'order_id': 'O1001'})])]
        done = await self.agent.run('tool-limit', question='查询 O1001 O1003 O1004')
        self.assertEqual(done['status'], 'unverified')
        self.assertEqual(done['metrics']['tool_calls'], 2)
        self.assertEqual(len([event for event in done['events'] if 'tool' in event]), 2)
        self.agent = self.new_agent()
        self.responses = [completion(calls=[(f's{index}', 'search_knowledge', {'query': f'退货 {index}'}) for index in range(3)]),
                          completion(calls=[('excess', 'search_knowledge', {'query': '第四次'})])]
        done = await self.agent.run('knowledge-limit', question='查询退货规则')
        self.assertEqual(done['status'], 'unverified')
        self.assertEqual(done['metrics']['knowledge_calls'], 3)
        self.assertEqual(len(self.searches), 3)

    async def test_custom_verifier_cannot_bypass_evidence_and_preview_guards(self):
        async def careless(**kwargs):
            return {'verdict': 'pass', 'issues': [], 'evidence_ids': ['invented']}
        self.agent = self.new_agent(result_verifier=careless)
        self.responses = [completion(calls=[('order', 'get_order', {'order_id': 'O1003'})]),
                          completion('错误答复')] + [completion('错误答复')] * 2
        done = await self.agent.run('careless', question='查询 O1003')
        self.assertEqual(done['status'], 'unverified')
        self.assertNotIn('错误答复', json.dumps(done, ensure_ascii=False))

    async def test_reviewer_failure_budget_survives_restart(self):
        self.responses = [completion('请提供订单号。')]
        self.review_responses = ['error'] * 3
        for index in range(3):
            with self.assertRaises(APIStatusError):
                await self.agent.run('reviews', **({'question': '查询订单'} if index == 0 else {}))
            self.assertEqual((await self.agent.get('reviews'))['metrics']['verification_requests'], index + 1)
            await self.restart()
        done = await self.agent.run('reviews')
        self.assertEqual(done['status'], 'unverified')
        self.assertEqual(done['metrics']['model_requests'], 1)
        self.assertEqual(done['metrics']['verification_requests'], 3)

    async def test_empty_calls_cannot_be_disguised_as_a_normal_final_answer(self):
        self.responses = [completion('未经工具验证的答案', calls=[], finish='stop')]
        with self.assertRaises(ValueError):
            await self.agent.run('empty-stop', question='查询订单')
        self.assertEqual((await self.agent.get('empty-stop'))['answer'], '')

    async def test_saved_experiment_remains_readable_without_a_model_client(self):
        self.responses = [completion('请提供订单号。')]
        await self.agent.run('read-only', question='查询订单')
        reader = self.GraphAgent(None, 'offline', self.saver, search_knowledge=self.search, return_handler=self.prepare)
        result = await reader.get('read-only')
        self.assertEqual(result['answer'], '请提供订单号。')
        self.assertEqual(result['status'], 'completed')

    async def test_caller_tool_policy_cannot_raise_experiment_twelve_tool_ceiling(self):
        self.agent = self.new_agent(tool_policy=ToolPolicy(max_calls=20))
        self.responses = [completion(calls=[(f'order{index}', 'get_order', {'order_id': 'O1003'}) for index in range(13)]),
                          completion('O1003 已签收。')]
        result = await self.agent.run('absolute-tool-cap', question='查询 O1003')
        self.assertLessEqual(result['metrics']['tool_calls'], 12)
        self.assertEqual(result['status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
