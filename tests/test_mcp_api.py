"""Temporary API plus actual local MCP subprocess; only model HTTP is scripted."""
import json
import unittest
from copy import deepcopy

import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class MCPApiTests(unittest.IsolatedAsyncioTestCase):
    tool_mode = 'mcp'
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    seed_knowledge = fixture.ApiAppTests.seed_knowledge

    async def test_same_session_followup_uses_current_selection_after_correcting_mixed_batch(self):
        self.seed_knowledge()
        sid = await self.new_session()
        self.responses = [tool_stream(('early1', 'get_order', {'order_id': 'O1001'})),
                          tool_stream(('d1', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('o1', 'get_order', {'order_id': 'O1001'})), text_stream('O1001 已发货。')]
        first = await self.client.post('/api/chat', json={'session_id': sid, 'message': '我的 O1001 订单到哪里了？'})
        self.assertEqual(first.status_code, 200, first.text)
        previous = deepcopy(self.app.state.store.load_session(sid)['history'])
        steps = [dict(id='order', title='查询 O1003 订单状态', tools=['get_order'], status='in_progress', depends_on=[], evidence=[]),
                 dict(id='policy', title='检索退货退款政策', tools=['search_knowledge'], status='pending', depends_on=[], evidence=[])]
        next_plan = deepcopy(steps)
        next_plan[0].update(status='completed', evidence=['o3'])
        next_plan[1]['status'] = 'in_progress'
        done_plan = deepcopy(next_plan)
        done_plan[1].update(status='completed', evidence=['kb'])
        self.responses += [tool_stream(('p1', 'update_plan', {'steps': steps, 'note': ''})),
                           tool_stream(('early3', 'get_order', {'order_id': 'O1003'}), ('mixed', 'discover_tools', {'names': ['search_knowledge']})),
                           tool_stream(('dk', 'discover_tools', {'names': ['search_knowledge']})),
                           tool_stream(('do', 'discover_tools', {'names': ['get_order']})),
                           tool_stream(('o3', 'get_order', {'order_id': 'O1003'})),
                           tool_stream(('p2', 'update_plan', {'steps': next_plan, 'note': ''})),
                           tool_stream(('kb', 'search_knowledge', {'query': '退款'})),
                           tool_stream(('p3', 'update_plan', {'steps': done_plan, 'note': ''})), text_stream('O1003 已签收，请补充签收日期和具体退货原因。')]
        second = await self.client.post('/api/chat', json={'session_id': sid, 'message': '帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。'})
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()['plan']['status'], 'completed')
        self.assertFalse(second.json()['recovery']['stopped'])
        initial_states = [m for m in self.requests[4]['messages'] if (m.get('content') or '').startswith('[本轮工具状态]')]
        self.assertEqual(len(initial_states), 1)
        self.assertEqual(json.loads(initial_states[0]['content'].split('\n', 1)[1])['loaded'], [])
        ready = self.requests[7]['messages']
        state = next(m for m in ready if (m.get('content') or '').startswith('[本轮工具状态]'))
        self.assertEqual(json.loads(state['content'].split('\n', 1)[1])['missing_for_current_step'], ['get_order'])
        self.assertFalse(any((m.get('content') or '').startswith('[执行恢复反馈') for m in ready))
        trace = self.app.state.traces.load(sid, self.app.state.traces.list_runs(sid)['runs'][0]['run_id'])
        results = [e['data'] for e in trace['events'] if e['kind'] == 'tool_result' and e['data'].get('status') == 'ok']
        self.assertEqual([r['result']['order_id'] for r in results if r['name'] == 'get_order'], ['O1003'])
        self.assertTrue(second.json()['sources'])
        self.assertEqual(self.app.state.store.load_session(sid)['history'][:len(previous)], previous)

    async def test_formal_api_uses_mcp_and_records_loaded_definitions(self):
        health = (await self.client.get('/api/health')).json()
        self.assertEqual(health.get('tool_mode'), 'mcp')
        sid = await self.new_session()
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        run = self.app.state.traces.list_runs(sid)['runs'][0]
        trace = self.app.state.traces.load(sid, run['run_id'])
        self.assertEqual(trace['status'], 'completed')
        discovered = [e['data'] for e in trace['events'] if e['kind'] == 'tool_discovery']
        self.assertEqual(discovered[0]['loaded_names'], ['get_order'])
        orders = [e['data'] for e in trace['events'] if e['kind'] == 'tool_result' and e['data']['name'] == 'get_order']
        self.assertEqual(orders[0]['source'], 'mcp')
        self.assertEqual(orders[0]['result']['order_id'], 'O1003')
        self.assertEqual(trace['metrics']['tool_calls'], 1)
        self.assertEqual(trace['metrics']['tool_discoveries'], 1)

    async def test_mcp_knowledge_preserves_real_source_and_new_session_selection(self):
        self.assertEqual((await self.client.get('/api/health')).json().get('tool_mode'), 'mcp')
        self.seed_knowledge()
        sid = await self.new_session()
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['search_knowledge']})),
                          tool_stream(('policy', 'search_knowledge', {'query': '退款'})), text_stream('这是示例退款政策。'),
                          tool_stream(('discover2', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '退款政策是什么'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['sources'])
        sid2 = await self.new_session()
        response = await self.client.post('/api/chat', json={'session_id': sid2, 'message': '查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        second_initial = {t['function']['name'] for t in self.requests[3]['tools']}
        self.assertNotIn('search_knowledge', second_initial)
        self.assertNotIn('get_order', second_initial)

    async def test_discovery_model_slot_is_available_to_api_resume(self):
        self.seed_knowledge()
        sid = await self.new_session()
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order', 'search_knowledge']})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'})),
                          tool_stream(('policy', 'search_knowledge', {'query': '退款'})),
                          ['[DONE]'], text_stream('订单已签收，示例退款需仓库验收。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003 和退款政策'})
        self.assertEqual(response.status_code, 502, response.text)
        checkpoint = (await self.client.get(f'/api/sessions/{sid}/checkpoint')).json()['checkpoint']
        self.assertEqual(checkpoint['rounds'], 4)
        self.assertTrue(checkpoint['can_resume'], checkpoint)
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': checkpoint['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        trace = self.app.state.traces.load(sid, checkpoint['run_id'])
        self.assertEqual(trace['metrics']['tool_calls'], 2)
        self.assertEqual(trace['status'], 'completed')

    async def test_invalid_mcp_arguments_return_to_model_for_correction(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('invalid', 'get_order', {'order_id': 'O1003', 'extra': 'invalid'})),
                          tool_stream(('corrected', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        error = json.loads(next(message['content'] for message in self.requests[2]['messages']
                                if message.get('tool_call_id') == 'invalid'))
        self.assertIn('error', error)
        evidence = next(message for message in self.requests[2]['messages'] if message.get('tool_call_id') == 'invalid')
        self.assertEqual(evidence['role'], 'tool')


if __name__ == '__main__':
    unittest.main()
