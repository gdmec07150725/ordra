"""关闭生成器并通过JSON重建执行，验证真实Agent循环的恢复。"""
import json
import asyncio
import unittest
from unittest.mock import patch

import agent_service
import test_api_app as fixture
from test_agent_plan_api import steps, update
from test_tool_execution_api import tool_stream
from test_stream_agent import text_stream
from tool_runtime import ToolPolicy
from context_memory import ContextPolicy


class CheckpointAgentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def save(self, state):
        self.snapshot = json.loads(json.dumps(state))

    def run_agent(self, checkpoint=None, **kwargs):
        return agent_service.stream_agent(self.app.state.client, self.app.state.model,
            kwargs.pop('history', agent_service.new_conversation()), '请规划并查询 O1003', checkpoint=checkpoint, save_checkpoint=self.save, **kwargs)

    async def test_restores_plan_evidence_and_skips_saved_tool_result(self):
        plan = steps()[:1]
        self.responses = [update('plan-start', plan), tool_stream(('order-result', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order-result'])
        self.responses.extend([update('plan-finish', plan), text_stream('订单已签收。')])
        first = self.run_agent()
        async for event in first:
            if event['event'] == 'tool_result' and event['name'] == 'get_order':
                break
        await first.aclose()
        events = [event async for event in self.run_agent(self.snapshot)]
        completed = events[-1]
        self.assertEqual(completed['event'], 'complete')
        self.assertEqual(completed['plan']['status'], 'completed')
        self.assertEqual(completed['execution']['calls'], 3)
        self.assertEqual(sum(item['name'] == 'get_order' for item in completed['execution']['tools']), 1)
        resumed_messages = self.requests[2]['messages']
        result = next(m for m in resumed_messages if m.get('tool_call_id') == 'order-result')
        self.assertEqual(json.loads(result['content'])['status'], '已签收')
        self.assertEqual(sum(m['role'] == 'user' for m in completed['history']), 1)

    async def test_final_candidate_resumes_without_model_request(self):
        self.responses = [text_stream('你好。')]
        original = [event async for event in self.run_agent()]
        self.assertEqual(original[-1]['event'], 'complete')
        recovered = [event async for event in self.run_agent(self.snapshot)]
        self.assertEqual(recovered[-1]['history'], original[-1]['history'])
        self.assertEqual(len(self.requests), 1)

    async def test_interruption_after_blocked_plan_still_asks_user_without_model(self):
        plan = steps()[:1]
        self.responses = [update('start', plan)]
        plan[0]['status'] = 'blocked'
        self.responses.append(update('wait', plan, '请提供订单号。'))
        self.responses.append(text_stream('错误地再次请求了模型。'))
        first = self.run_agent()
        async for event in first:
            if event['event'] == 'plan' and event['plan']['status'] == 'blocked':
                break
        await first.aclose()
        events = [event async for event in self.run_agent(self.snapshot)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(events[-1]['answer'], '请提供订单号。')
        self.assertEqual(len(self.requests), 2)

    async def test_interruption_does_not_reset_model_round_budget(self):
        # 每次响应不包含完成标记，已开始的模型请求仍占额度。
        self.responses = [['[DONE]']] * 4
        checkpoint = None
        for count in range(4):
            with self.assertRaises(ValueError):
                _ = [event async for event in self.run_agent(checkpoint)]
            checkpoint = self.snapshot
            self.assertEqual(checkpoint['rounds'], count + 1)
        with self.assertRaisesRegex(ValueError, '上限'):
            _ = [event async for event in self.run_agent(checkpoint)]
        self.assertEqual(len(self.requests), 4)

    async def test_resume_preserves_failed_call_cache_and_tool_budget(self):
        self.responses = [tool_stream(('first', 'get_order', {'order_id': 'O1003'})),
                          tool_stream(('repeat', 'get_order', {'order_id': 'O1003'})), text_stream('查询暂时失败。')]
        with patch.object(agent_service, 'get_order', side_effect=ConnectionError('temporary failure')):
            first = self.run_agent()
            async for event in first:
                if event['event'] == 'tool_result':
                    break
            await first.aclose()
            restored = [event async for event in self.run_agent(self.snapshot)]
        execution = restored[-1]['execution']
        self.assertEqual(execution['attempts'], 2)
        self.assertEqual(execution['tools'][-1]['status'], 'skipped')
        self.assertEqual(execution['retries'], 1)

    async def test_interrupted_tool_reservation_still_consumes_budget_on_resume(self):
        self.responses = [tool_stream(('first', 'search_history', {'query': 'O1003'})),
                          tool_stream(('again', 'search_history', {'query': 'O1003'})), text_stream('本次查询未完成。')]
        entered = asyncio.Event()
        executions = []
        async def reader(query, order):
            executions.append(query)
            if len(executions) == 1:
                entered.set()
                await asyncio.Event().wait()
            return {'results': [], 'query': query}
        async def collect():
            return [event async for event in self.run_agent(search_history=reader, tool_policy=ToolPolicy(max_calls=1))]
        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), 10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        events = [event async for event in self.run_agent(self.snapshot, search_history=reader)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(executions, ['O1003'])
        self.assertEqual(events[-1]['execution']['tools'][-1]['status'], 'skipped')

    async def test_completed_summary_batch_is_not_repeated_after_interruption(self):
        history = agent_service.new_conversation()
        for index in range(30):
            history.extend([{'role': 'user', 'content': f'第{index}轮订单说明：' + '资料' * 100},
                            {'role': 'assistant', 'content': '已记录。'}])
        self.responses = [fixture.ApiAppTests.summary_response(self, '已保留前段订单信息。')] * 3
        policy = ContextPolicy(max_tokens=5306, recent_tokens=500, summary_tokens=512)
        first = self.run_agent(history=history, policy=policy)
        count = 0
        async for event in first:
            if event['event'] == 'summarizing':
                count += 1
                if count == 2:
                    break
        await first.aclose()
        self.assertTrue(hasattr(self, 'snapshot'), '第一批成功摘要必须生成检查点')
        self.assertGreater(self.snapshot['context']['memory']['covered_messages'], 0)
        restored = self.run_agent(self.snapshot, history=history)
        count = 0
        async for event in restored:
            if event['event'] == 'summarizing':
                count += 1
                if count == 2:
                    break
        await restored.aclose()
        original_prompt = json.loads(self.requests[0]['messages'][-1]['content'])
        restored_prompt = json.loads(self.requests[1]['messages'][-1]['content'])
        self.assertEqual(restored_prompt['previous_summary'], '已保留前段订单信息。')
        self.assertNotEqual(original_prompt['completed_turns'][0], restored_prompt['completed_turns'][0])
        self.assertEqual(self.snapshot['context']['summary_usage']['reported_rounds'], 2)

    async def test_partial_turn_summary_resumes_from_saved_fragment(self):
        history = agent_service.new_conversation() + [{'role': 'user', 'content': '订单历史🎧' * 2000}, {'role': 'assistant', 'content': '已记录。'}]
        self.responses = [fixture.ApiAppTests.summary_response(self, '已保留前段订单信息。')] * 20
        policy = ContextPolicy(max_tokens=5306, recent_tokens=500, summary_tokens=512)
        first = self.run_agent(history=history, policy=policy)
        count = 0
        async for event in first:
            if event['event'] == 'summarizing':
                count += 1
                if count == 2:
                    break
        await first.aclose()
        pending = self.snapshot['context']['pending_summary']
        self.assertGreater(pending['offset'], 0)
        self.assertEqual(self.snapshot['context']['memory']['covered_messages'], 0)
        restored = self.run_agent(self.snapshot, history=history)
        count = 0
        async for event in restored:
            if event['event'] == 'summarizing':
                count += 1
                if count == 2:
                    break
        await restored.aclose()
        resumed_prompt = json.loads(self.requests[1]['messages'][-1]['content'])
        self.assertEqual(resumed_prompt['turn_fragment']['offset'], pending['offset'])
        self.assertEqual(resumed_prompt['previous_summary'], '已保留前段订单信息。')
        self.assertEqual(self.snapshot['context']['summary_usage']['reported_rounds'], 2)
