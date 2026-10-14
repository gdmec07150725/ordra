"""模拟模型、真实 HTTP 和临时 SQLite 验证退货确认边界。"""
from contextlib import closing
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_api_app as fixture
from test_stream_agent import chunk, tool_delta, text_stream


class ReturnApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart_app = fixture.ApiAppTests.restart_app

    async def state(self, sid):
        return (await self.client.get(f'/api/sessions/{sid}')).json()

    def tool(self, name='prepare_return_request', arguments=None):
        arguments = arguments if arguments is not None else {'order_id': 'O1003', 'reason': '收到的商品无法正常使用'}
        return [chunk(calls=[tool_delta(0, 'return-call', name, json.dumps(arguments, ensure_ascii=False))]), chunk(finish='tool_calls'), '[DONE]']

    async def prepare(self, sid, endpoint='/api/chat'):
        self.responses.extend([self.tool(), text_stream('请核对预览，点击确认提交。')])
        response = await self.client.post(endpoint, json={'session_id': sid, 'message': 'O1003 收到的商品无法正常使用，我想申请退货。'})
        self.assertEqual(response.status_code, 200, response.text)
        payload = fixture.parse_sse(response.text)[-1][1] if endpoint.endswith('stream') else response.json()
        self.assertIn('return_previews', payload)
        self.assertEqual(len(payload['return_previews']), 1)
        preview = payload['return_previews'][0]
        self.assertEqual(preview['state'], 'pending')
        self.assertIsNone(preview['request_id'])
        return preview

    async def test_preview_requires_explicit_confirmation_in_both_routes(self):
        for endpoint in ('/api/chat', '/api/chat/stream'):
            sid = await self.new_session()
            preview = await self.prepare(sid, endpoint)
            url = f"/api/sessions/{sid}/return-previews/{preview['preview_id']}/confirm"
            for body in ({}, {'confirmed': False}, {'confirmed': 1}, {'confirmed': True, 'reason': '被替换的原因'}):
                self.assertEqual((await self.client.post(url, json=body)).status_code, 422)
            self.assertEqual((await self.state(sid))['return_previews'][0]['state'], 'pending')
            before = (await self.state(sid))['messages']
            first = await self.client.post(url, json={'confirmed': True})
            self.assertEqual(first.status_code, 200, first.text)
            request = first.json()['request']
            self.assertEqual(request['state'], 'submitted')
            self.assertEqual(request['reason'], preview['reason'])
            self.assertTrue(request['request_id'].startswith('RET-'))
            second = await self.client.post(url, json={'confirmed': True})
            self.assertEqual(second.json()['request'], request)
            saved = await self.state(sid)
            self.assertEqual(len(saved['messages']), len(before) + 2)
            self.assertIn(request['request_id'], saved['messages'][-1]['content'])
            self.assertEqual(saved['return_previews'][0]['state'], 'submitted')

    async def test_model_cannot_confirm_or_supply_other_session(self):
        sid = await self.new_session()
        for name, arguments in (
            ('create_return_request', {'order_id': 'O1003', 'reason': '损坏', 'confirmed': True}),
            ('prepare_return_request', {'order_id': 'O1003', 'reason': '损坏', 'confirmed': True}),
            ('prepare_return_request', {'order_id': 'O1003', 'reason': '损坏', 'session_id': 'someone-else'}),
            ('get_return_request', {'order_id': 'O1003', 'session_id': 'someone-else'}),
        ):
            self.responses.extend([self.tool(name, arguments), text_stream('无法执行。')])
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '请处理'})
            self.assertEqual(response.status_code, 200, response.text)
            tool_result = next(message for message in reversed(self.requests[-1]['messages']) if message['role'] == 'tool')
            self.assertIn('error', json.loads(tool_result['content']))
            self.assertEqual(response.json()['return_previews'], [])
        names = {tool['function']['name'] for tool in self.requests[-1]['tools']}
        self.assertIn('prepare_return_request', names)
        self.assertIn('get_return_request', names)
        self.assertNotIn('create_return_request', names)

    async def test_invalid_order_or_reason_does_not_create_preview(self):
        sid = await self.new_session()
        for args in ({'order_id': 'missing', 'reason': '坏了'}, {'order_id': 'O1003', 'reason': ' '}, {'order_id': 'O1003', 'reason': '长' * 501}):
            self.responses.extend([self.tool(arguments=args), text_stream('请补充有效信息。')])
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '退货'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['return_previews'], [])
            self.assertIn('error', fixture.last_tool_result(self.requests[-1]))

    async def test_followup_invalidates_old_preview_and_cancel_is_idempotent(self):
        sid = await self.new_session()
        first = await self.prepare(sid)
        second = await self.prepare(sid)
        first_url = f"/api/sessions/{sid}/return-previews/{first['preview_id']}/confirm"
        self.assertEqual((await self.client.post(first_url, json={'confirmed': True})).status_code, 409)
        url = f"/api/sessions/{sid}/return-previews/{second['preview_id']}"
        for _ in range(2):
            response = await self.client.post(url + '/cancel', json={})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['request']['state'], 'cancelled')
        self.assertEqual((await self.client.post(url + '/confirm', json={'confirmed': True})).status_code, 409)

    async def test_confirmation_rejects_wrong_session_and_busy_session(self):
        sid, other = await self.new_session(), await self.new_session()
        preview = await self.prepare(sid)
        suffix = f"/return-previews/{preview['preview_id']}/confirm"
        for bad_sid in (other, 'missing'):
            self.assertEqual((await self.client.post(f'/api/sessions/{bad_sid}' + suffix, json={'confirmed': True})).status_code, 404)
        self.app.state.sessions[sid].busy = True
        self.assertEqual((await self.client.post(f'/api/sessions/{sid}' + suffix, json={'confirmed': True})).status_code, 409)
        self.app.state.sessions[sid].busy = False
        self.assertEqual((await self.state(sid))['return_previews'][0]['state'], 'pending')

    async def test_query_and_receipt_survive_restart_without_another_model_confirmation(self):
        sid = await self.new_session()
        preview = await self.prepare(sid)
        response = await self.client.post(f"/api/sessions/{sid}/return-previews/{preview['preview_id']}/confirm", json={'confirmed': True})
        self.assertEqual(response.status_code, 200)
        request_id = response.json()['request']['request_id']
        self.assertEqual(len(self.requests), 2, '点击确认不需要模型调用')
        self.responses.extend([self.tool('get_return_request', {'order_id': 'O1003'}), text_stream('申请待人工审核。')])
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '刚才申请进度怎样？'})
        self.assertIn(request_id, json.dumps(self.requests[-2]['messages'], ensure_ascii=False))
        self.assertEqual(json.loads(self.requests[-1]['messages'][-1]['content'])['request']['request_id'], request_id)
        before = await self.state(sid)
        await self.restart_app()
        self.assertEqual(await self.state(sid), before)

    async def test_failed_turn_does_not_save_preview_or_invalidate_previous(self):
        sid = await self.new_session()
        await self.prepare(sid)
        before = await self.state(sid)
        self.responses.extend([self.tool(), [chunk(content='未完成'), chunk(finish='length'), '[DONE]']])
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '改为其他原因'})
        self.assertEqual(fixture.parse_sse(response.text)[-1][0], 'error')
        self.assertEqual(await self.state(sid), before)

    async def test_save_and_confirmation_failure_roll_back_atomically(self):
        sid = await self.new_session()
        preview = await self.prepare(sid)
        before = await self.state(sid)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_returns BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        self.responses.extend([self.tool(), text_stream('新的预览')])
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '修改原因'})
        self.assertEqual(failed.status_code, 503)
        url = f"/api/sessions/{sid}/return-previews/{preview['preview_id']}/confirm"
        self.assertEqual((await self.client.post(url, json={'confirmed': True})).status_code, 409)
        pending = (await self.client.get(f'/api/sessions/{sid}/checkpoint')).json()['checkpoint']
        discarded = await self.client.post(f'/api/sessions/{sid}/checkpoint/discard', json={'run_id': pending['run_id']})
        self.assertEqual(discarded.status_code, 204)
        self.assertEqual((await self.client.post(url, json={'confirmed': True})).status_code, 503)
        self.assertEqual(await self.state(sid), before)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('DROP TRIGGER fail_returns')
        self.assertEqual((await self.client.post(url, json={'confirmed': True})).status_code, 200)

    async def test_committed_receipt_survives_cache_reload_failure(self):
        sid = await self.new_session()
        preview = await self.prepare(sid)
        url = f"/api/sessions/{sid}/return-previews/{preview['preview_id']}/confirm"
        with patch.object(self.app.state.store, 'load_session', side_effect=sqlite3.OperationalError('read failed after commit')):
            self.assertEqual((await self.client.post(url, json={'confirmed': True})).status_code, 503)
        saved = await self.state(sid)
        request_id = saved['return_previews'][0]['request_id']
        self.assertIn(request_id, saved['messages'][-1]['content'])
        self.assertEqual((await self.client.post(url, json={'confirmed': True})).status_code, 200)
        self.assertEqual((await self.state(sid))['messages'], saved['messages'])
