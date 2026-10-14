"""执行轨迹与业务状态使用同一真实临时数据库。"""
from contextlib import closing
import asyncio
import json
import sqlite3
import unittest

import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


class TraceApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart_app = fixture.ApiAppTests.restart_app

    async def runs(self, sid):
        response = await self.client.get(f'/api/sessions/{sid}/runs')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['runs']

    async def trace(self, sid, rid):
        response = await self.client.get(f'/api/sessions/{sid}/runs/{rid}')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_success_has_one_model_event_per_round_and_safe_tool_result(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('已签收。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        runs = await self.runs(sid)
        self.assertEqual(len(runs), 1)
        trace = await self.trace(sid, runs[0]['run_id'])
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(trace['metrics']['model_requests'], 2)
        self.assertEqual(trace['metrics']['tool_calls'], 1)
        self.assertEqual(len([e for e in trace['events'] if e['kind'] == 'model_request']), 2)
        results = [e for e in trace['events'] if e['kind'] == 'tool_result']
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['data']['arguments']['order_id'], 'O1003')
        self.assertEqual(results[0]['data']['result']['status'], '已签收')
        self.assertNotIn('offline-only-key', json.dumps(trace))
        self.assertNotIn('system', results[0]['data'])
        await self.restart_app()
        self.assertEqual(await self.trace(sid, runs[0]['run_id']), trace)

    async def test_failure_resume_reuses_run_and_does_not_duplicate_saved_tool(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), ['[DONE]'], text_stream('订单已签收。')]
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
        runs = await self.runs(sid)
        self.assertEqual(runs[0]['status'], 'failed')
        rid = runs[0]['run_id']
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': rid})
        self.assertEqual(response.status_code, 200, response.text)
        trace = await self.trace(sid, rid)
        self.assertEqual(len(await self.runs(sid)), 1)
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(trace['metrics']['model_requests'], 3)
        self.assertEqual(sum(e['kind'] == 'tool_result' for e in trace['events']), 1)
        self.assertIn('resumed', [e['kind'] for e in trace['events']])

    async def test_approval_and_discard_keep_inspectable_events(self):
        self.app.state.approval_tools = frozenset({'prepare_return_request'})
        sid = await self.new_session()
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '故障'}))]
        paused = await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1003 故障申请退货'})
        cp = paused.json()['checkpoint']
        trace = await self.trace(sid, cp['run_id'])
        self.assertEqual(trace['status'], 'awaiting_approval')
        self.assertFalse(any(e['kind'] == 'tool_result' for e in trace['events']))
        decision = await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={
            'run_id': cp['run_id'], 'approval_id': cp['approvals'][0]['approval_id'], 'decision': 'reject'})
        self.assertEqual(decision.status_code, 200)
        await self.client.post(f'/api/sessions/{sid}/checkpoint/discard', json={'run_id': cp['run_id']})
        trace = await self.trace(sid, cp['run_id'])
        self.assertEqual(trace['status'], 'discarded')
        self.assertEqual(next(e['data']['decision'] for e in trace['events'] if e['kind'] == 'human_decision'), 'reject')

    async def test_failed_history_commit_does_not_leave_a_completed_trace(self):
        sid = await self.new_session()
        self.responses = [text_stream('完整答复。')]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("CREATE TRIGGER fail_history BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT,'failed'); END")
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '你好'})
        self.assertEqual(response.status_code, 503)
        run = (await self.runs(sid))[0]
        trace = await self.trace(sid, run['run_id'])
        self.assertEqual(trace['status'], 'failed')
        self.assertFalse(any(e['kind'] == 'completed' for e in trace['events']))
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute('DROP TRIGGER fail_history')
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': run['run_id']})
        self.assertEqual(response.status_code, 200)
        trace = await self.trace(sid, run['run_id'])
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(sum(e['kind'] == 'completed' for e in trace['events']), 1)

    async def test_other_session_cannot_read_trace_and_reset_cleans_it(self):
        sid, other = await self.new_session(), await self.new_session()
        self.responses = [text_stream('你好。')]
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '你好'})
        rid = (await self.runs(sid))[0]['run_id']
        self.assertEqual((await self.client.get(f'/api/sessions/{other}/runs/{rid}')).status_code, 404)
        await self.client.post(f'/api/sessions/{sid}/reset')
        self.assertEqual(await self.runs(sid), [])

    async def test_response_start_failure_stops_trace_before_generator_runs(self):
        sid = await self.new_session()
        incoming = asyncio.Queue()
        await incoming.put({'type': 'http.request', 'body': json.dumps({'session_id': sid, 'message': '你好'}).encode(), 'more_body': False})
        async def send(event):
            if event['type'] == 'http.response.start':
                raise OSError('浏览器已断开')
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'}, 'http_version': '1.1', 'method': 'POST', 'scheme': 'http', 'path': '/api/chat/stream', 'raw_path': b'/api/chat/stream', 'query_string': b'', 'root_path': '', 'headers': [(b'content-type', b'application/json')], 'client': ('127.0.0.1', 12345), 'server': ('testserver', 80)}
        with self.assertRaises(OSError):
            await self.app(scope, incoming.get, send)
        self.assertEqual((await self.runs(sid))[0]['status'], 'interrupted')
        self.assertFalse(self.app.state.sessions[sid].busy)
        self.assertEqual(len(self.requests), 0)

    async def test_nonfinite_proposed_arguments_still_return_validation_to_model(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('bad-order', 'get_order', {'order_id': float('nan')})), text_stream('请提供正确订单号。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询订单'})
        self.assertEqual(response.status_code, 200, response.text)
        trace = await self.trace(sid, (await self.runs(sid))[0]['run_id'])
        self.assertEqual(trace['metrics']['model_requests'], 2)
        result = next(e['data'] for e in trace['events'] if e['kind'] == 'tool_result')
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['result']['error'])
        json.dumps(trace, allow_nan=False)

    async def test_invalid_tool_call_keeps_proposed_arguments(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('', 'get_order', {'order_id': 'O1003'}))]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询订单'})
        self.assertEqual(response.status_code, 502)
        trace = await self.trace(sid, (await self.runs(sid))[0]['run_id'])
        proposed = next(e['data'] for e in trace['events'] if e['kind'] == 'model_response')['tool_calls'][0]
        self.assertEqual(proposed['arguments']['order_id'], 'O1003')

    async def test_legacy_final_candidate_restores_metrics_without_historical_events(self):
        sid = await self.new_session()
        self.responses = [text_stream('已完成。')]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("CREATE TRIGGER fail_history BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT,'failed'); END")
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '你好'})
        cp = self.app.state.checkpoints.load(sid)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('DROP TRIGGER fail_history')
            db.execute('DELETE FROM agent_runs')
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        trace = await self.trace(sid, cp['run_id'])
        self.assertEqual(trace['metrics']['model_requests'], 1)
        self.assertTrue(trace['events'][0]['data']['legacy_checkpoint'])
        self.assertEqual([e['kind'] for e in trace['events']], ['started', 'resumed', 'completed'])
