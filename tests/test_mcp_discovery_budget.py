"""Newly loaded schemas must participate in retrieval fitting before the next request."""
from copy import deepcopy
import json
import unittest

import agent_service
from context_memory import ContextPolicy, estimate_request
import test_api_app as fixture
from test_mcp_graph import InlineCatalog
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class MCPDiscoveryBudgetTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def test_discovery_keeps_fitting_evidence_and_full_source_history(self):
        # Under 8K, previously verbose recovery feedback evicted this chunk
        # after another schema was loaded. Optional feedback must give way first.
        hit = {'source': 'kb:abcdef123456:0', 'document_id': 'abcdef123456',
               'version': '123456abcdef', 'title': 'title', 'section': 'section',
               'text': 'x' * 7800, 'start': 0, 'end': 7800, 'score': 1.0}

        async def reader(query):
            return {'query': query, 'document_count': 1, 'matched_chunks': 1,
                    'results': [deepcopy(hit)], 'has_more': False, 'budget_limited': False,
                    'search_mode': 'keywords', 'semantic_status': 'disabled'}

        self.responses = [
            tool_stream(('d1', 'discover_tools', {'names': ['search_knowledge']})),
            tool_stream(('k', 'search_knowledge', {'query': 'return'})),
            tool_stream(('d2', 'discover_tools', {'names': ['get_order']})),
            text_stream('finished'),
        ]
        policy = ContextPolicy(max_tokens=8192)
        remote = InlineCatalog(reader)
        stream = agent_service.stream_agent(
            self.app.state.client, 'offline', agent_service.new_conversation(), 'lookup',
            policy=policy, search_knowledge=reader, mcp_client=remote)
        try:
            events = [event async for event in stream]
        except ValueError as error:
            self.fail(f'加载新工具后应重新裁剪请求视图，不能因旧 schema 估算中断：{error}')

        self.assertEqual(len(self.requests), 4)
        self.assertEqual(events[-1]['event'], 'complete')
        before = json.loads(next(message['content'] for message in self.requests[2]['messages']
                                 if message.get('tool_call_id') == 'k'))
        self.assertEqual(before['results'], [hit], '第一次请求确实已带入完整原始资料')
        final_request = self.requests[3]
        names = {tool['function']['name'] for tool in final_request['tools']}
        self.assertIn('get_order', names)
        self.assertIn('search_knowledge', names)
        for request in self.requests:
            self.assertLessEqual(estimate_request(request['messages'], request['tools']), policy.input_limit)

        projected = json.loads(next(message['content'] for message in final_request['messages']
                                    if message.get('tool_call_id') == 'k'))
        self.assertFalse(projected['budget_limited'])
        self.assertEqual(projected['results'], [hit])
        saved = json.loads(next(message['content'] for message in events[-1]['history']
                                if message.get('tool_call_id') == 'k'))
        self.assertEqual(saved['results'], [hit], '请求裁剪不得覆盖已查阅的原始资料')
        self.assertFalse(saved['budget_limited'])
        self.assertEqual(events[-1]['sources'], [hit])
        self.assertEqual(remote.calls, [('search_knowledge', {'query': 'return'})])


if __name__ == '__main__':
    unittest.main()
