"""工具调度集成验收：临时 SQLite、真实工具循环、离线模型响应。"""
import asyncio
import json
import sqlite3
import threading
import unittest
from unittest.mock import patch

import anyio
import agent_service
import test_api_app as fixture
from test_stream_agent import chunk, tool_delta, text_stream
from tool_runtime import ToolPolicy


def tool_stream(*calls):
    return [chunk(calls=[tool_delta(i, call_id, name, json.dumps(args))
                         for i, (call_id, name, args) in enumerate(calls)]),
            chunk(finish='tool_calls'), '[DONE]']


class ToolExecutionApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session

    async def test_independent_order_queries_overlap_and_keep_tool_pairs(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('first', 'get_order', {'order_id': 'O1001'}),
                                      ('second', 'get_order', {'order_id': 'O1002'})),
                          text_stream('两笔订单已核实。')]
        barrier, overlaps = threading.Barrier(2), []
        actual = agent_service.get_order

        def read(order_id):
            try:
                barrier.wait(1)
                overlaps.append(True)
            except threading.BrokenBarrierError:
                overlaps.append(False)
            return actual(order_id)

        with patch.object(agent_service, 'get_order', side_effect=read):
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1001 和 O1002'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(overlaps, [True, True], '独立查询必须并行进入工具，而不是逐个等待')
        messages = self.requests[-1]['messages']
        self.assertEqual([m['tool_call_id'] for m in messages[-2:]], ['first', 'second'])
        self.assertEqual([json.loads(m['content'])['order_id'] for m in messages[-2:]], ['O1001', 'O1002'])
        self.assertEqual(response.json()['execution']['max_parallel'], 2)

    async def test_database_busy_is_retried_with_one_logical_tool_result(self):
        for endpoint in ('/api/chat', '/api/chat/stream'):
            with self.subTest(endpoint=endpoint):
                sid = await self.new_session()
                self.app.state.tool_policy = ToolPolicy(retry_delay_seconds=0)
                self.responses.extend([tool_stream(('history', 'search_history', {'query': '收货'})), text_stream('已查询历史。')])
                calls = 0
                actual = self.app.state.store.search_history
                def read(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise sqlite3.OperationalError('database is locked')
                    return actual(*args)
                with patch.object(self.app.state.store, 'search_history', side_effect=read):
                    response = await self.client.post(endpoint, json={'session_id': sid, 'message': '查询原话'})
                data = fixture.parse_sse(response.text)[-1][1] if endpoint.endswith('stream') else response.json()
                self.assertEqual(calls, 2)
                self.assertEqual(data['retrieval']['calls'], 1)
                self.assertEqual(data['execution']['retries'], 1)
                self.assertEqual(data['execution']['tools'][0]['attempts'], 2)
                self.assertEqual(len([m for m in self.requests[-1]['messages'] if m['role'] == 'tool']), 1)

    async def test_native_read_timeout_returns_control_without_retry_or_late_commit(self):
        sid = await self.new_session()
        # 给正常线程调度留出余量；慢线程一直等显式 release，仍然必然超时。
        self.app.state.tool_policy = ToolPolicy(timeout_seconds=.3)
        self.responses = [tool_stream(('slow', 'get_order', {'order_id': 'O1001'}),
                                      ('fast', 'get_order', {'order_id': 'O1002'})),
                          text_stream('O1001 暂时无法查询，O1002 正在备货。')]
        release, finished = threading.Event(), threading.Event()
        actual = agent_service.get_order
        def read(order_id):
            if order_id == 'O1001':
                try:
                    release.wait(30)
                    return actual(order_id)
                finally:
                    finished.set()
            return actual(order_id)
        try:
            with patch.object(agent_service, 'get_order', side_effect=read):
                # 冷加载图和磁盘检查点也占用请求时间；以仍被阻塞的慢线程
                # 验证工具超时会返回控制，避免把整条 HTTP 链路误限为 2 秒。
                response = await asyncio.wait_for(self.client.post('/api/chat', json={'session_id': sid, 'message': '查两笔订单'}), 10)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(finished.is_set(), '超时不应继续等待只读线程')
            messages = self.requests[-1]['messages']
            tool_messages = [message for message in messages if message.get('role') == 'tool']
            self.assertEqual(json.loads(tool_messages[-2]['content'])['error_code'], 'tool_timeout')
            self.assertEqual(json.loads(tool_messages[-1]['content'])['order_id'], 'O1002')
            slow_execution = next(item for item in response.json()['execution']['tools'] if item['call_id'] == 'slow')
            self.assertEqual(slow_execution['attempts'], 1)
            self.assertEqual(slow_execution['retries'], [])
            saved = (await self.client.get(f'/api/sessions/{sid}')).json()
            self.assertFalse(saved['busy'])
        finally:
            release.set()
            self.assertTrue(await anyio.to_thread.run_sync(finished.wait, 3))
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json(), saved)

    async def test_preview_is_serial_after_queries_and_only_one_is_prepared(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('policy', 'search_knowledge', {'query': '退货'}),
                                      ('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'}),
                                      ('duplicate', 'prepare_return_request', {'order_id': 'O1003', 'reason': '另一原因'})),
                          text_stream('请核对预览。')]
        policy_finished = threading.Event()
        actual_search, actual_prepare = self.app.state.knowledge_base.search, self.app.state.returns.prepare_return_request
        def search(query):
            result = actual_search(query)
            policy_finished.set()
            return result
        def prepare(*args):
            self.assertTrue(policy_finished.is_set(), '预览必须等待前面查询完成')
            return actual_prepare(*args)
        with patch.object(self.app.state.knowledge_base, 'search', side_effect=search), patch.object(self.app.state.returns, 'prepare_return_request', side_effect=prepare) as prepare_mock:
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '退货'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(prepare_mock.call_count, 1)
        self.assertEqual(len(response.json()['return_previews']), 1)
        self.assertIn('error', fixture.last_tool_result(self.requests[-1]))

    async def test_model_can_correct_parameters_on_next_decision(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('invalid', 'get_order', {'order_id': 1001})),
                          tool_stream(('corrected', 'get_order', {'order_id': 'O1001'})),
                          text_stream('订单已发货。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1001'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('error', fixture.last_tool_result(self.requests[1]))
        self.assertEqual(json.loads(self.requests[2]['messages'][-1]['content'])['status'], '已发货')
        self.assertEqual(response.json()['execution']['retries'], 0, '纠正参数是模型的新决策，不是执行器重试')

    async def test_late_preview_result_is_not_saved_after_tool_timeout(self):
        sid = await self.new_session()
        self.app.state.tool_policy = ToolPolicy(timeout_seconds=.02)
        preview = self.app.state.returns.prepare_return_request(sid, 'O1003', '质量问题')
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})),
                          text_stream('准备超时，请稍后重新申请。')]
        async def late_result(*args):
            # 清理阶段可能短暂屏蔽取消，超时后的返回值仍不能成为可保存预览。
            with anyio.CancelScope(shield=True):
                await anyio.sleep(.05)
            return preview
        with patch.object(agent_service, 'execute_return_tool', side_effect=late_result):
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '申请退货'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['return_previews'], [])
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json()['return_previews'], [])
