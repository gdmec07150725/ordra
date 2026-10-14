"""子任务协议与恢复：真实 AsyncOpenAI，仅替换 HTTP 传输和只读业务边界。"""
import asyncio
from copy import deepcopy
import importlib.util
import json
import sqlite3
import unittest

import httpx2
from openai import AsyncOpenAI

from context_memory import ContextPolicy, estimate_request


def call(name='get_order', call_id='lookup', arguments=None):
    return {'id': call_id, 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(arguments or {'order_id': 'O1003'})}}


def response(*, calls=None, status='completed', summary='查询结果已获得。', evidence=None,
             raw=None, finish=None, extra=None):
    content = raw if raw is not None else json.dumps({
        'status': status, 'summary': summary, 'evidence_ids': evidence or []}, ensure_ascii=False)
    message = {'role': 'assistant', 'content': None, 'tool_calls': calls} if calls is not None else {
        'role': 'assistant', 'content': content}
    return {'id': 'child', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
            'choices': [{'index': 0, 'message': {**message, **(extra or {})},
                         'finish_reason': finish or ('tool_calls' if calls is not None else 'stop')}],
            'usage': {'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60}}


TOOLS = {agent: [{'type': 'function', 'function': {'name': name, 'description': '只读查询',
          'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': True}}}]
         for agent, name in [('orders', 'get_order'), ('knowledge', 'search_knowledge')]}


class SubagentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(importlib.util.find_spec('subagent_runtime'), '独立子任务运行时尚未实现')
        import subagent_runtime
        self.runtime = subagent_runtime
        self.state = subagent_runtime.new_delegation_state()
        self.requests, self.saves, self.reservations, self.executions = [], [], [], []
        self.queue = {'orders': [], 'knowledge': []}
        self.policy = ContextPolicy()
        self.save_hook = self.execute_hook = self.http_hook = None
        self.running_saves = self.peak_saves = 0

        async def handle(request):
            body = json.loads(request.content)
            self.requests.append(body)
            agent = 'orders' if body['tools'][0]['function']['name'] == 'get_order' else 'knowledge'
            if self.http_hook:
                await self.http_hook(agent, body)
            item = self.queue[agent].pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, int):
                return httpx2.Response(item, json={'error': {'message': 'temporary failure', 'type': 'server_error'}})
            return httpx2.Response(200, json=item)

        self.client = AsyncOpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False))

    async def asyncTearDown(self):
        if hasattr(self, 'client'):
            await self.client.close()

    def register(self, *agents):
        return self.runtime.register_tasks(self.state, json.dumps({'tasks': [
            {'agent': agent, 'task': '查询 O1003' if agent == 'orders' else '检索退款政策', 'context': '自包含资料'}
            for agent in agents]}), set(TOOLS))

    def prepare(self, task, calls):
        self.reservations.extend(c['id'] for c in calls)
        return {'reserved': [True] * len(calls), 'preflight': {}}

    async def execute(self, task, pending):
        self.assertIn(pending, [snapshot['tasks'][int(task['id'][-1]) - 1]['pending'] for snapshot in self.saves])
        self.executions.extend(c['id'] for c in pending['message']['tool_calls'])
        if self.execute_hook:
            return await self.execute_hook(task, pending)
        return [{'call_id': c['id'], 'name': c['function']['name'],
                 'result': {'order_id': 'O1003', 'status': '已签收', '_tool_call_id': 'forged'},
                 'attempts': 1, 'status': 'ok', 'retries': [], 'elapsed_ms': 1}
                for c in pending['message']['tool_calls']]

    async def save(self):
        self.running_saves += 1
        self.peak_saves = max(self.peak_saves, self.running_saves)
        try:
            await asyncio.sleep(0)
            self.saves.append(json.loads(json.dumps(self.state)))
            if self.save_hook:
                await self.save_hook()
        finally:
            self.running_saves -= 1

    def stream(self, ids, **kwargs):
        return self.runtime.run_specialists(self.client, 'offline', ids, question='查询订单并检索政策',
            policy=self.policy, state=self.state, tools_by_agent=TOOLS,
            prepare_tools=self.prepare, execute_tools=self.execute, save_checkpoint=self.save, **kwargs)

    async def run_tasks(self, ids):
        return [event async for event in self.stream(ids)]

    async def test_registration_is_strict_atomic_and_bounded_across_batches(self):
        invalid = [[], {}, {'tasks': []}, {'tasks': [{'agent': 'admin', 'task': 'x'}]},
                   {'tasks': [{'agent': 'orders', 'task': ' '}]},
                   {'tasks': [{'agent': 'orders', 'task': 'x' * 501}]},
                   {'tasks': [{'agent': 'orders', 'task': 'x', 'context': 'x' * 801}]},
                   {'tasks': [{'agent': 'orders', 'task': 1}]},
                   {'tasks': [{'agent': 'orders', 'task': 'x', 'id': 'forged'}]},
                   {'tasks': [{'agent': 'orders', 'task': 'x'}], 'extra': True},
                   {'tasks': [{'agent': 'orders', 'task': 'x'}, {'agent': 'root', 'task': 'y'}]},
                   {'tasks': [{'agent': 'orders', 'task': 'x'}] * 3}]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                before = deepcopy(self.state)
                with self.assertRaises(ValueError):
                    self.runtime.register_tasks(self.state, json.dumps(arguments), set(TOOLS))
                self.assertEqual(self.state, before)
        for raw in ['{', '{"tasks":[],"tasks":[{"agent":"orders","task":"x"}]}']:
            with self.assertRaises(ValueError):
                self.runtime.register_tasks(self.state, raw, set(TOOLS))
        with self.assertRaises(ValueError):
            self.runtime.register_tasks(self.state, '{"tasks":[{"agent":"knowledge","task":"x"}]}', {'orders'})
        self.assertEqual(self.register('orders'), ['specialist_1'])
        self.assertEqual(self.register('knowledge'), ['specialist_2'])
        before = deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.register('orders')
        self.assertEqual(self.state, before)
        json.dumps(self.state, allow_nan=False)

    async def test_parallel_tasks_have_isolated_messages_real_evidence_and_serial_saves(self):
        ids = self.register('orders', 'knowledge')
        both_started, started = asyncio.Event(), set()

        async def gate(agent, body):
            started.add(agent)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 1)
        self.http_hook = gate
        self.queue['orders'] = [response(calls=[call()]), response(evidence=['specialist_1:r1:lookup'])]
        self.queue['knowledge'] = [response(calls=[call('search_knowledge')]), response(evidence=['specialist_2:r1:lookup'])]
        events = await self.run_tasks(ids)
        self.assertEqual(events[0]['event'], 'delegation_start')
        results = events[-1]['results']
        self.assertEqual([r['status'] for r in results], ['completed', 'completed'])
        self.assertEqual(self.state['requests'], 4)
        self.assertEqual(self.state['usage']['total_tokens'], 240)
        self.assertEqual(self.peak_saves, 1)
        for index, task in enumerate(self.state['tasks'], 1):
            self.assertEqual(task['usage']['total_tokens'], 120)
            self.assertIsNone(task['pending'])
            messages = task['messages']
            self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'tool', 'assistant'])
            self.assertEqual(set(json.loads(messages[1]['content'])), {'question', 'task', 'context'})
            actual_id = f'specialist_{index}:r1:lookup'
            self.assertEqual(messages[2]['tool_calls'][0]['id'], actual_id)
            self.assertEqual(messages[3]['tool_call_id'], actual_id)
            self.assertEqual(json.loads(messages[3]['content'])['_tool_call_id'], actual_id)
            self.assertEqual(results[index - 1]['evidence'][0]['call_id'], actual_id)
            self.assertEqual(results[index - 1]['evidence'][0]['result']['order_id'], 'O1003')
        for request in self.requests:
            self.assertFalse(request['stream'])
            self.assertEqual(request['tool_choice'], 'auto')
            self.assertEqual(request['thinking'], {'type': 'disabled'})
            self.assertLessEqual(estimate_request(request['messages'], request['tools']) + request['max_tokens'] + 256, 8192)
            self.assertEqual(len(request['tools']), 1)
        for event in events:
            if event['event'] in ('specialist_start', 'specialist_done'):
                self.assertNotIn('summary', event)
                self.assertNotIn('messages', event)

    async def test_unknown_or_cross_agent_tools_never_execute_or_reserve(self):
        for name in ['prepare_return_request', 'delegate_tasks', 'search_knowledge']:
            with self.subTest(name=name):
                self.state = self.runtime.new_delegation_state()
                ids = self.register('orders')
                self.queue['orders'] = [response(calls=[call(name)])]
                result = (await self.run_tasks(ids))[-1]['results'][0]
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['evidence'], [])
        self.assertEqual(self.executions, [])
        self.assertEqual(self.reservations, [])

    async def test_invalid_completion_or_unearned_evidence_cannot_pass(self):
        cases = [response(), response(evidence=['forged']), response(raw='{'),
                 response(status='approved'), response(status='needs_input', summary='x' * 801),
                 response(status='needs_input', evidence=['x'] * 13),
                 response(status='needs_input', finish='length'),
                 response(raw='{"status":"needs_input","status":"completed","summary":"x","evidence_ids":[]}'),
                 response(raw='{"status":"needs_input","summary":"x","evidence_ids":[],"approved":true}'),
                 response(raw='{"status":"needs_input","summary":"x","evidence_ids":[1]}'),
                 response(status='needs_input', extra={'refusal': '拒绝'}),
                 {**response(), 'choices': []}]
        for item in cases:
            with self.subTest(item=item):
                self.state = self.runtime.new_delegation_state()
                ids = self.register('orders')
                self.queue['orders'] = [item]
                self.assertEqual((await self.run_tasks(ids))[-1]['results'][0]['status'], 'failed')

    async def test_needs_input_without_tools_is_a_valid_terminal_result(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(status='needs_input', summary='请提供订单号。')]
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'needs_input')
        self.assertEqual(result['summary'], '请提供订单号。')
        self.assertEqual(result['evidence'], [])

    async def test_failed_tools_remain_in_actual_log_but_cannot_justify_completion(self):
        async def failure(task, pending):
            return [{'call_id': pending['message']['tool_calls'][0]['id'], 'name': 'get_order',
                     'result': {'error': '暂不可用'}, 'status': 'ok', 'attempts': 1, 'retries': [], 'elapsed_ms': 1}]
        self.execute_hook = failure
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call()]), response(evidence=['specialist_1:r1:lookup'])]
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.state['tasks'][0]['tool_results'][0]['result']['error'], '暂不可用')
        self.assertEqual(len(result['evidence']), 1)
        self.assertEqual(result['evidence'][0]['result']['error'], '暂不可用')
        self.assertEqual(result['cited_evidence_ids'], [])

    async def test_handoff_preserves_uncited_real_evidence_separately_from_citations(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call(call_id='one'), call(call_id='two')]),
                                response(evidence=['specialist_1:r1:one'])]
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([item['call_id'] for item in result['evidence']],
                         ['specialist_1:r1:one', 'specialist_1:r1:two'])
        self.assertEqual(result['cited_evidence_ids'], ['specialist_1:r1:one'])

    async def test_empty_summary_is_not_a_valid_handoff(self):
        for summary in ['', '  ']:
            with self.subTest(summary=summary):
                self.state = self.runtime.new_delegation_state()
                ids = self.register('orders')
                self.queue['orders'] = [response(status='needs_input', summary=summary)]
                self.assertEqual((await self.run_tasks(ids))[-1]['results'][0]['status'], 'failed')

    async def test_evidence_ids_match_exactly_without_whitespace_normalization(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call()]), response(evidence=[' specialist_1:r1:lookup '])]
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'failed')

    async def test_model_failure_does_not_cancel_other_task_and_disables_sdk_retries(self):
        self.client = self.client.with_options(max_retries=2)
        ids = self.register('orders', 'knowledge')
        self.queue['orders'] = [503]
        self.queue['knowledge'] = [response(status='needs_input')]
        results = (await self.run_tasks(ids))[-1]['results']
        self.assertEqual([r['status'] for r in results], ['failed', 'needs_input'])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.state['requests'], 2)

    async def test_no_more_than_three_model_decisions_or_six_shared_requests(self):
        ids = self.register('orders', 'knowledge')
        self.queue['orders'] = [response(calls=[call()])] * 3
        self.queue['knowledge'] = [response(calls=[call('search_knowledge')])] * 3
        results = (await self.run_tasks(ids))[-1]['results']
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(self.state['requests'], 6)
        self.assertEqual([r['status'] for r in results], ['failed', 'failed'])
        self.assertEqual([t['rounds'] for t in self.state['tasks']], [3, 3])
        self.assertEqual(len(set(self.executions)), 6)

    async def test_context_overflow_is_not_truncated_and_spends_no_request(self):
        ids = self.register('orders')
        self.state['tasks'][0]['context'] = '完整证据' * 10000
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.state['requests'], 0)
        self.assertEqual(self.requests, [])
        payload = json.loads(self.state['tasks'][0]['messages'][1]['content'])
        self.assertEqual(payload['context'], '完整证据' * 10000)

    async def test_pending_resumes_without_reservation_and_completed_tasks_do_not_rerun(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call()]), response(evidence=['specialist_1:r1:lookup'])]
        async def interrupt():
            if self.state['tasks'][0]['pending']:
                raise asyncio.CancelledError()
        self.save_hook = interrupt
        with self.assertRaises(asyncio.CancelledError):
            await self.run_tasks(ids)
        self.assertEqual(self.executions, [])
        self.assertEqual(self.reservations, ['specialist_1:r1:lookup'])
        self.state = deepcopy(self.saves[-1])
        self.save_hook = None
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.reservations, ['specialist_1:r1:lookup'])
        self.assertEqual(self.executions, ['specialist_1:r1:lookup'])
        before = deepcopy((self.requests, self.executions))
        await self.run_tasks(ids)
        self.assertEqual((self.requests, self.executions), before)

    async def test_completed_tool_batch_is_not_repeated_after_next_model_cancellation(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call()]), response(evidence=['specialist_1:r1:lookup'])]
        async def interrupt(agent, body):
            if len(body['messages']) > 2:
                raise asyncio.CancelledError()
        self.http_hook = interrupt
        with self.assertRaises(asyncio.CancelledError):
            await self.run_tasks(ids)
        self.assertIsNone(self.state['tasks'][0]['pending'])
        self.assertEqual(self.state['requests'], 2)
        self.state = deepcopy(self.saves[-1])
        self.http_hook = None
        result = (await self.run_tasks(ids))[-1]['results'][0]
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.executions, ['specialist_1:r1:lookup'])
        self.assertEqual(self.state['requests'], 3)

    async def test_tool_storage_errors_propagate_and_leave_recoverable_pending(self):
        ids = self.register('orders')
        self.queue['orders'] = [response(calls=[call()])]
        async def broken(task, pending):
            raise sqlite3.DatabaseError('offline simulated corruption')
        self.execute_hook = broken
        with self.assertRaises(sqlite3.DatabaseError):
            await self.run_tasks(ids)
        self.assertIsNotNone(self.state['tasks'][0]['pending'])
        self.assertIsNotNone(self.saves[-1]['tasks'][0]['pending'])

    async def test_closing_generator_cancels_inflight_model_workers(self):
        ids = self.register('orders', 'knowledge')
        entered, left = set(), set()
        ready = asyncio.Event()
        async def blocked(agent, body):
            entered.add(agent)
            if len(entered) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                left.add(agent)
        self.http_hook = blocked
        stream = self.stream(ids)
        self.assertEqual((await anext(stream))['event'], 'delegation_start')
        await anext(stream)
        await asyncio.wait_for(ready.wait(), 1)
        await stream.aclose()
        self.assertEqual(left, {'orders', 'knowledge'})
        self.assertEqual(self.state['requests'], 2)
