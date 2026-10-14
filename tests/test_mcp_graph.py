"""Real graph/SDK; the transport boundary is isolated from subprocess tests."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch
import time

import agent_service
from agent_plan import PLAN_TOOL
from knowledge_base import KNOWLEDGE_TOOL
from tool_discovery import ToolDiscovery
from tool_runtime import ToolPolicy
import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class InlineCatalog:
    """Transport double; execute actual order validation and fixture knowledge reader."""
    def __init__(self, reader):
        self.reader, self.calls = reader, []

    async def list_tools(self):
        return deepcopy(agent_service.TOOLS + [KNOWLEDGE_TOOL])

    async def call_tool(self, name, arguments, *, expected_schema=None):
        self.calls.append((name, deepcopy(arguments)))
        if name == 'search_knowledge':
            return await self.reader(arguments['query'])
        return agent_service.execute_tool(name, json.dumps(arguments))


class MCPGraphTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def run_graph(self, *, checkpoint=None, remote=None, **kwargs):
        self.assertIn('mcp_client', inspect.signature(agent_service.stream_agent).parameters)
        async def reader(query):
            return self.app.state.knowledge_base.search(query)
        async def save(state):
            self.snapshot = json.loads(json.dumps(state))
        self.remote = remote or InlineCatalog(reader)
        return agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
            '查询 O1003', search_knowledge=reader, mcp_client=self.remote,
            save_checkpoint=save, checkpoint=checkpoint, **kwargs)

    def request_names(self, index):
        return {tool['function']['name'] for tool in self.requests[index]['tools']}

    async def test_discover_then_call_uses_only_loaded_schema_and_actual_evidence(self):
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        events = [event async for event in await self.run_graph()]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertNotIn('get_order', self.request_names(0))
        self.assertIn('get_order', self.request_names(1))
        self.assertNotIn('search_knowledge', self.request_names(1))
        self.assertEqual(self.remote.calls, [('get_order', {'order_id': 'O1003'})])
        self.assertEqual(self.snapshot['runtime']['calls'], 1)
        observation = self.snapshot['context']['budget_observations'][1]
        self.assertIn('get_order', observation['tool_names'])
        self.assertNotIn('search_knowledge', observation['tool_names'])

    async def test_unloaded_tool_is_rejected_before_business_execution(self):
        self.responses = [tool_stream(('early', 'get_order', {'order_id': 'O1003'})),
                          tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        events = [event async for event in await self.run_graph()]
        errors = [e['result'] for e in events if e['event'] == 'tool_result' and 'error' in e['result']]
        self.assertEqual(errors[0]['error_code'], 'tool_not_loaded')
        self.assertEqual(len(self.remote.calls), 1)
        self.assertEqual(events[-1]['event'], 'complete')

    async def test_same_batch_discovery_cannot_authorize_unseen_arguments(self):
        self.responses = [tool_stream(('discover-early', 'discover_tools', {'names': ['get_order']}),
                                     ('order-early', 'get_order', {'order_id': 'O1003'})),
                          tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        events = [event async for event in await self.run_graph()]
        self.assertNotIn('get_order', self.request_names(1))
        self.assertEqual(len(self.remote.calls), 1)
        self.assertEqual(events[-1]['event'], 'complete')

    async def test_resume_retains_selection_without_replaying_order(self):
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        stream = await self.run_graph()
        async for event in stream:
            if event['event'] == 'tool_result' and event['name'] == 'get_order':
                break
        await stream.aclose()
        saved, remote = deepcopy(self.snapshot), self.remote
        self.assertIn('get_order', saved['tool_selection']['loaded'])
        events = [event async for event in await self.run_graph(checkpoint=saved, remote=remote)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(len(remote.calls), 1)
        self.assertIn('get_order', self.request_names(2))
        self.assertEqual(self.snapshot['tool_selection']['calls'], 1)

    async def test_legacy_checkpoint_keeps_original_tool_path(self):
        self.assertIn('mcp_client', inspect.signature(agent_service.stream_agent).parameters)
        async def save(state):
            self.snapshot = deepcopy(state)
        async def reader(query):
            return self.app.state.knowledge_base.search(query)
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        stream = agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
            '查询 O1003', search_knowledge=reader, save_checkpoint=save)
        async for event in stream:
            if event['event'] == 'tool_result':
                break
        await stream.aclose()
        saved = deepcopy(self.snapshot)
        self.assertNotIn('tool_selection', saved)
        events = [event async for event in await self.run_graph(checkpoint=saved)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(self.remote.calls, [])
        self.assertNotIn('discover_tools', self.request_names(1))

    async def test_timed_out_discovery_does_not_publish_loaded_tools(self):
        original = ToolDiscovery.discover
        def slow_discover(selection, arguments):
            result = original(selection, arguments)
            time.sleep(.03)
            return result
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          text_stream('工具发现未完成。')]
        with patch.object(ToolDiscovery, 'discover', slow_discover):
            events = [event async for event in await self.run_graph(tool_policy=ToolPolicy(timeout_seconds=.01))]
        outcome = next(e['result'] for e in events if e['event'] == 'tool_result')
        self.assertEqual(outcome['error_code'], 'tool_timeout')
        self.assertNotIn('get_order', self.request_names(1))
        self.assertEqual(self.snapshot['tool_selection']['loaded'], {})


if __name__ == '__main__':
    unittest.main()
