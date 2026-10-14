"""Formal graph persists phases, approvals and request reservations."""
import inspect
import json
import unittest

from langgraph.checkpoint.memory import InMemorySaver
import agent_service
import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class FullAgentGraphTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def save(self, state):
        self.snapshot = json.loads(json.dumps(state))

    def run_agent(self, saver, checkpoint=None, **kwargs):
        return agent_service.stream_agent(self.app.state.client, self.app.state.model,
            agent_service.new_conversation(), '查询 O1003', graph_checkpointer=saver,
            graph_run_id='graph-test', checkpoint=checkpoint, save_checkpoint=self.save, **kwargs)

    async def test_formal_entry_accepts_real_graph_checkpointer(self):
        self.assertIn('graph_checkpointer', inspect.signature(agent_service.stream_agent).parameters)
        saver = InMemorySaver()
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        events = [event async for event in self.run_agent(saver)]
        self.assertEqual(events[-1]['answer'], '已签收。')
        saved = [item async for item in saver.alist({'configurable': {'thread_id': 'graph-test'}})]
        phases = {item.checkpoint['channel_values'].get('route') for item in saved}
        self.assertTrue({'context', 'model', 'tools', 'verify', 'finish', 'end'} <= phases, phases)
        self.assertEqual(saved[0].checkpoint['channel_values']['snapshot']['answer'], '已签收。')

    async def test_approval_is_persisted_interrupt_and_resumes_original_call(self):
        self.assertIn('graph_checkpointer', inspect.signature(agent_service.stream_agent).parameters)
        saver = InMemorySaver()
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})), text_stream('已准备预览。')]
        executions = []
        async def handler(name, arguments):
            executions.append(arguments)
            return {'preview': {'preview_id': 'preview-one', **arguments}}
        kwargs = {'return_handler': handler, 'approval_tools': frozenset({'prepare_return_request'})}
        paused = [event async for event in self.run_agent(saver, **kwargs)]
        self.assertEqual(paused[-1]['event'], 'paused')
        saved = await saver.aget_tuple({'configurable': {'thread_id': 'graph-test'}})
        self.assertTrue(any(channel == '__interrupt__' for _, channel, _ in saved.pending_writes))
        self.assertEqual(executions, [])
        self.snapshot['pending_batch']['approvals'][0]['decision'] = 'approve'
        resumed = [event async for event in self.run_agent(saver, self.snapshot, **kwargs)]
        self.assertEqual(resumed[-1]['event'], 'complete')
        self.assertEqual(len(executions), 1)
        self.assertEqual(len(self.requests), 2)

    async def test_close_at_round_reserves_budget_without_starting_sdk(self):
        self.assertIn('graph_checkpointer', inspect.signature(agent_service.stream_agent).parameters)
        saver = InMemorySaver()
        self.responses = [text_stream('已恢复。')]
        stream = self.run_agent(saver)
        async for event in stream:
            if event['event'] == 'round':
                break
        await stream.aclose()
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(self.snapshot['rounds'], 1)
        events = [event async for event in self.run_agent(saver, self.snapshot)]
        self.assertEqual(events[-1]['answer'], '已恢复。')
        self.assertEqual(events[-1]['usage']['reported_rounds'], 0)
        self.assertEqual(self.snapshot['rounds'], 2)

    async def test_close_at_delta_closes_sdk_and_never_accepts_partial_answer(self):
        saver = InMemorySaver()
        self.responses = [text_stream('尚未完成的回答。'), text_stream('恢复后的回答。')]
        stream = self.run_agent(saver)
        async for event in stream:
            if event['event'] == 'delta':
                break
        await stream.aclose()
        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.snapshot['answer'])
        self.assertEqual(len(self.requests), 1)
        events = [event async for event in self.run_agent(saver, self.snapshot)]
        self.assertEqual(events[-1]['answer'], '恢复后的回答。')
        self.assertNotIn('尚未完成', json.dumps(events[-1]['history'], ensure_ascii=False))
        self.assertEqual(self.snapshot['rounds'], 2)

    async def test_latest_progress_overrides_root_before_tools_node_finishes(self):
        saver = InMemorySaver()
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})), text_stream('预览已准备好。')]
        executions = []
        async def handler(name, arguments):
            executions.append(arguments)
            return {'preview': {'preview_id': 'only-preview', **arguments}}
        kwargs = {'return_handler': handler, 'approval_tools': frozenset({'prepare_return_request'})}
        _ = [event async for event in self.run_agent(saver, **kwargs)]
        self.snapshot['pending_batch']['approvals'][0]['decision'] = 'approve'
        stream = self.run_agent(saver, self.snapshot, **kwargs)
        async for event in stream:
            if event['event'] == 'tool_result':
                break
        await stream.aclose()
        self.assertIsNone(self.snapshot['pending_batch'])
        root = await saver.aget_tuple({'configurable': {'thread_id': 'graph-test'}})
        self.assertIsNotNone(root.checkpoint['channel_values']['snapshot']['pending_batch'])
        events = [event async for event in self.run_agent(saver, self.snapshot, **kwargs)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(events[-1]['return_previews'][0]['preview_id'], 'only-preview')
        self.assertEqual(len(executions), 1)
        self.assertEqual(len(self.requests), 2)

    async def test_anyio_cancellation_drains_model_worker_before_returning(self):
        import anyio
        saver = InMemorySaver()
        self.responses = [text_stream('断连时不能保存。')]
        self.gate.clear()
        async def consume():
            stream = self.run_agent(saver)
            try:
                async for event in stream:
                    pass
            finally:
                await stream.aclose()
        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await self.entered.wait()
            group.cancel_scope.cancel()
        self.assertTrue(self.streams[0].closed)
        self.assertIsNone(self.snapshot['answer'])
        self.assertEqual(len(self.requests), 1)
