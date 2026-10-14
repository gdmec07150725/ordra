"""Numeric request diagnostics must conserve estimates and never retain source text."""
import json
import unittest
import test_context_memory as memory_fixture
import test_result_verification as review_fixture
import test_api_app as api_fixture
from context_memory import ContextPolicy, estimate_request, summary_message


class RequestBudgetTests(unittest.TestCase):
    def test_components_conserve_request_estimate_without_content(self):
        from context_observation import request_budget
        messages = [{'role': 'system', 'content': '私有系统说明'}, *summary_message('私有历史摘要'),
                    {'role': 'user', 'content': '过去的问题'}, {'role': 'assistant', 'content': '过去的回答'},
                    {'role': 'user', 'content': '当前问题'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'a', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': '私有工具原文'}]
        tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': '工具描述'}}]
        value = request_budget(messages, tools, ContextPolicy(), role='actor', history_end=4, before_tokens=8000)
        self.assertEqual(sum(value['components'].values()), estimate_request(messages, tools))
        self.assertEqual(value['estimated_input_tokens'], estimate_request(messages, tools))
        self.assertEqual(value['saved_tokens'], 8000 - value['estimated_input_tokens'])
        for key in ('system', 'tools', 'history_summary', 'recent_messages', 'current_messages', 'tool_results'):
            self.assertGreater(value['components'][key], 0)
        # Discovery diagnostics expose capability names, never their descriptions or payloads.
        self.assertEqual(value['tool_names'], ['lookup'])
        for secret in ('私有', '过去的问题', '工具描述'):
            self.assertNotIn(secret, json.dumps(value, ensure_ascii=False))
        self.assertEqual(value['status'], 'prepared')

    def test_summary_output_reserve_and_blocked_request(self):
        from context_observation import request_budget
        policy = ContextPolicy(max_tokens=8192)
        value = request_budget([{'role': 'user', 'content': '长' * 7000}], [], policy,
                               role='summary', output_tokens=256)
        self.assertEqual(value['input_limit'], policy.max_tokens - 256 - policy.safety_tokens)
        self.assertEqual(value['status'], 'blocked')
        self.assertEqual(value['saved_tokens'], 0)
        self.assertGreater(value['components']['request_data'], 0)


class ObserverScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_scopes_do_not_mix_or_leak(self):
        import asyncio
        from context_observation import observe_request, request_observer
        first, second = [], []
        async def run(target, text):
            async def record(data):
                target.append(data)
            with request_observer(record):
                await asyncio.sleep(0)
                await observe_request([{'role': 'user', 'content': text}], [], ContextPolicy(), role='actor')
        await asyncio.gather(run(first, '短'), run(second, '更长的一段问题'))
        await observe_request([], [], ContextPolicy(), role='actor')
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0]['estimated_input_tokens'], second[0]['estimated_input_tokens'])


class SummaryObservationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = memory_fixture.ContextMemoryTests.asyncSetUp
    asyncTearDown = memory_fixture.ContextMemoryTests.asyncTearDown
    prepare = memory_fixture.ContextMemoryTests.prepare

    async def test_minimum_summary_fragment_block_is_recorded_without_reserving_request(self):
        from context_memory import RollingContext, ContextMemory
        from context_observation import request_observer
        policy = ContextPolicy(max_tokens=2048, recent_tokens=128, summary_tokens=1600, output_tokens=1, safety_tokens=1)
        history = [{'role': 'system', 'content': '客服'}, {'role': 'user', 'content': '长' * 10000},
                   {'role': 'assistant', 'content': '已记录'}]
        context = RollingContext(history, ContextMemory(), policy, [])
        records = []
        async def record(value):
            records.append(value)
        with request_observer(record), self.assertRaises(ValueError):
            await self.prepare(context)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['role'], 'summary')
        self.assertEqual(records[0]['status'], 'blocked')
        self.assertGreater(records[0]['estimated_input_tokens'], records[0]['input_limit'])
        self.assertEqual(context.summary_calls, 0)
        self.assertEqual(self.requests, [])

    async def test_summary_observes_actual_requests_and_oversize_actor_is_blocked(self):
        from context_memory import RollingContext, ContextMemory
        from context_observation import request_observer
        records = []
        async def record(value):
            records.append(value)
        context = RollingContext(memory_fixture.history_of(), ContextMemory(), self.policy, [])
        with request_observer(record):
            await self.prepare(context)
            self.assertEqual(len(records), len(self.requests))
            for data, request in zip(records, self.requests):
                self.assertEqual(data['role'], 'summary')
                self.assertEqual(data['estimated_input_tokens'], estimate_request(request['messages']))
                self.assertEqual(data['output_reserve'], request['max_tokens'])
            count = len(self.requests)
            with self.assertRaises(ValueError):
                await self.prepare(context, [{'role': 'user', 'content': '大' * 10000}])
        self.assertEqual(len(self.requests), count)
        self.assertEqual(records[-1]['status'], 'blocked')
        self.assertEqual(records[-1]['role'], 'actor')
        self.assertGreater(records[-1]['components']['recent_messages'], 0)
        self.assertGreater(records[-1]['components']['history_summary'], 0)


class ReviewObservationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = review_fixture.ResultVerificationTests.asyncSetUp
    asyncTearDown = review_fixture.ResultVerificationTests.asyncTearDown
    review = review_fixture.ResultVerificationTests.review

    async def test_reviewer_reports_request_and_budget_block_without_sending(self):
        from context_observation import request_observer
        from result_verification import ReviewBudgetError
        records = []
        async def record(value):
            records.append(value)
        with request_observer(record):
            await self.review()
            with self.assertRaises(ReviewBudgetError):
                await self.review(messages=[self.messages[0], {'role': 'tool', 'tool_call_id': 'order', 'content': '大' * 10000}])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([d['status'] for d in records], ['prepared', 'blocked'])
        self.assertTrue(all(d['role'] == 'verification' for d in records))
        self.assertNotIn('O1003', json.dumps(records))


class BlockedApiObservationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = api_fixture.ApiAppTests.asyncSetUp
    asyncTearDown = api_fixture.ApiAppTests.asyncTearDown
    new_session = api_fixture.ApiAppTests.new_session

    async def test_blocked_request_persists_numeric_diagnostic_without_model_call(self):
        self.app.state.context_policy = ContextPolicy(max_tokens=4096, recent_tokens=256, summary_tokens=256)
        sid = await self.new_session()
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '长' * 2000})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.requests, [])
        cp = self.app.state.checkpoints.load(sid)
        self.assertIsNotNone(cp)
        trace = self.app.state.traces.load(sid, cp['run_id'])
        values = [e['data'] for e in trace['events'] if e['kind'] == 'context_budget']
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['status'], 'blocked')
        self.assertEqual(trace['metrics']['model_requests'], 0)
        self.assertNotIn('长', json.dumps(values, ensure_ascii=False))
        self.app.state.checkpoints.save(sid, cp['run_id'], cp['state'])
        replay = self.app.state.traces.load(sid, cp['run_id'])
        self.assertEqual(sum(e['kind'] == 'context_budget' for e in replay['events']), 1)
