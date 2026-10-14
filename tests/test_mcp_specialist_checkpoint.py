"""MCP child decisions keep their schema contract across persisted API recovery."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import test_api_app as fixture
from test_delegation_api import child_response, tool_message
from test_mcp_graph import InlineCatalog
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class ChangingCatalog(InlineCatalog):
    """Only the transport boundary is replaced; graph, checkpoints and traces are real."""
    changed = False

    async def list_tools(self):
        tools = await super().list_tools()
        if self.changed:
            order = next(tool for tool in tools if tool['function']['name'] == 'get_order')
            order['function']['parameters']['properties']['order_id']['maxLength'] = 16
        return tools

    async def close(self):
        pass


class MCPSpecialistCheckpointTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session

    async def asyncSetUp(self):
        await fixture.ApiAppTests.asyncSetUp(self)

        async def read_knowledge(query):
            return self.app.state.knowledge_base.search(query)

        self.remote = ChangingCatalog(read_knowledge)
        self.app.state.mcp_client = self.remote
        self.app.state.tool_mode = 'mcp'
        self.app.state.delegation_enabled = True
        self.responses = [
            tool_stream(('delegate', 'delegate_tasks', {'tasks': [
                {'agent': 'orders', 'task': '查询 O1003'}]})),
            child_response(calls=[tool_message('order', 'get_order', {'order_id': 'O1003'})]),
            child_response(ids=['specialist_1:r1:order']),
            text_stream('O1003 已签收。'),
        ]

    async def test_changed_child_schema_blocks_saved_pending_decision_before_execution(self):
        sid = await self.new_session()
        save = self.app.state.checkpoints.save

        def interrupt_after_pending(session_id, run_id, state):
            save(session_id, run_id, state)
            if any(task.get('pending') for task in state['delegation']['tasks']):
                raise ValueError('测试：已保存子任务工具决策后中断')

        with patch.object(self.app.state.checkpoints, 'save', side_effect=interrupt_after_pending):
            first = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        self.assertEqual(first.status_code, 502, first.text)
        saved = deepcopy(self.app.state.checkpoints.load(sid))
        self.assertEqual(saved['state']['tool_selection']['loaded'], {})
        self.assertTrue(saved['state']['delegation']['tasks'][0]['pending'])
        self.assertEqual(self.remote.calls, [])
        self.assertEqual(len(self.requests), 2)

        self.remote.changed = True
        resumed = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': saved['run_id']})
        self.assertEqual(resumed.status_code, 502, resumed.text)
        self.assertRegex(resumed.json()['detail'], '定义.*变化')
        self.assertEqual(self.remote.calls, [], '不能使用新定义执行旧的子任务参数')
        self.assertEqual(len(self.requests), 2, '定义变化必须在继续请求模型前拦截')
        self.assertTrue(self.app.state.checkpoints.load(sid)['state']['delegation']['tasks'][0]['pending'])

    async def test_specialist_tool_trace_records_mcp_source_with_actual_order_evidence(self):
        sid = await self.new_session()
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        run = self.app.state.traces.list_runs(sid)['runs'][0]
        trace = self.app.state.traces.load(sid, run['run_id'])
        orders = [event['data'] for event in trace['events']
                  if event['kind'] == 'tool_result' and event['data']['name'] == 'get_order']
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['call_id'], 'specialist_1:r1:order')
        self.assertEqual(orders[0]['result']['order_id'], 'O1003')
        self.assertEqual(orders[0].get('source'), 'mcp')
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(self.remote.calls, [('get_order', {'order_id': 'O1003'})])


if __name__ == '__main__':
    unittest.main()
