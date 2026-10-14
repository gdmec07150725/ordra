"""审核 HTTP 边界和模型只读查询，使用临时数据库和模拟模型。"""
from contextlib import closing
import json
import sqlite3
import unittest

import test_api_app as fixture
from test_stream_agent import chunk, tool_delta, text_stream


class ReturnReviewApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart_app = fixture.ApiAppTests.restart_app

    async def submitted(self):
        sid = await self.new_session()
        result = self.app.state.returns.prepare_return_request(sid, 'O1003', '收到后不能正常使用')
        history = self.app.state.store.load_session(sid)['history']
        self.app.state.store.replace_history(sid, history, None, [result['preview']])
        response = await self.client.post(f"/api/sessions/{sid}/return-previews/{result['preview']['preview_id']}/confirm", json={'confirmed': True})
        self.assertEqual(response.status_code, 200)
        return sid, response.json()['request']

    async def review(self, sid, request, decision='approved', reason='人工核实质量问题，同意退货申请'):
        return await self.client.post(f"/api/returns/{request['request_id']}/review", json={'session_id': sid, 'decision': decision, 'reason': reason})

    async def test_review_page_list_and_filtered_results(self):
        page = await self.client.get('/returns')
        self.assertEqual(page.status_code, 200)
        self.assertIn('售后审核', page.text)
        empty = await self.client.get('/api/returns')
        self.assertEqual(empty.json()['requests'], [])
        sid, request = await self.submitted()
        pending = (await self.client.get('/api/returns')).json()
        self.assertEqual(pending['counts'], {'pending': 1, 'approved': 0, 'rejected': 0, 'all': 1})
        self.assertEqual(pending['requests'][0]['session_id'], sid)
        result = await self.review(sid, request)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['request']['status'], '审核通过')
        approved = (await self.client.get('/api/returns?status=approved')).json()
        self.assertEqual(approved['requests'][0]['request_id'], request['request_id'])
        self.assertEqual((await self.client.get('/api/returns')).json()['requests'], [])
        self.assertEqual((await self.client.get('/api/returns?status=bad')).status_code, 422)

    async def test_review_is_idempotent_and_cannot_overwrite_existing_result(self):
        sid, request = await self.submitted()
        before = (await self.client.get(f'/api/sessions/{sid}')).json()['messages']
        first = await self.review(sid, request, 'rejected', '材料不足，请补充问题照片')
        self.assertEqual(first.status_code, 200, first.text)
        second = await self.review(sid, request, 'rejected', '材料不足，请补充问题照片')
        self.assertEqual(first.json(), second.json())
        self.assertEqual((await self.review(sid, request, 'approved')).status_code, 409)
        self.assertEqual((await self.review(sid, request, 'rejected', '改写原有原因')).status_code, 409)
        saved = (await self.client.get(f'/api/sessions/{sid}')).json()
        self.assertEqual(saved['messages'], before, '审核不能伪装成用户聊天消息')
        self.assertEqual(saved['return_previews'][0]['review']['decision'], 'rejected')
        self.assertEqual(saved['return_previews'][0]['status'], '审核拒绝')
        await self.restart_app()
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json(), saved)

    async def test_invalid_review_wrong_session_and_busy_are_rejected(self):
        sid, request = await self.submitted()
        url = f"/api/returns/{request['request_id']}/review"
        valid = {'session_id': sid, 'decision': 'approved', 'reason': '已人工核实'}
        for body in ({**valid, 'decision': 'pending'}, {**valid, 'reason': ''}, {**valid, 'reason': ' '}, {**valid, 'reason': '长' * 501}, {**valid, 'reason': 123}, {**valid, 'order_id': 'O1001'}):
            self.assertEqual((await self.client.post(url, json=body)).status_code, 422)
        other = await self.new_session()
        self.assertEqual((await self.review(other, request)).status_code, 404)
        self.app.state.sessions[sid].busy = True
        self.assertEqual((await self.review(sid, request)).status_code, 409)
        self.app.state.sessions[sid].busy = False
        state = (await self.client.get(f'/api/sessions/{sid}')).json()
        self.assertIsNone(state['return_previews'][0]['review'])

    async def test_agent_reads_approved_and_rejected_status_but_cannot_review(self):
        for decision, status in [('approved', '审核通过'), ('rejected', '审核拒绝')]:
            sid, request = await self.submitted()
            result = await self.review(sid, request, decision, '人工审核的具体说明')
            self.assertEqual(result.status_code, 200, result.text)
            calls = [chunk(calls=[tool_delta(0, 'lookup', 'get_return_request', '{"order_id":"O1003"}')]), chunk(finish='tool_calls'), '[DONE]']
            self.responses.extend([calls, text_stream('已查询到最新审核结果。')])
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '我的退货申请审核结果如何？'})
            self.assertEqual(response.status_code, 200, response.text)
            payload = json.loads(self.requests[-1]['messages'][-1]['content'])['request']
            self.assertEqual(payload['status'], status)
            self.assertEqual(payload['review']['reason'], '人工审核的具体说明')
            names = {tool['function']['name'] for tool in self.requests[-1]['tools']}
            self.assertNotIn('review_return_request', names)
            self.assertNotIn('review_request', names)
        self.responses.extend([[chunk(calls=[tool_delta(0, 'bad', 'review_request', '{}')]), chunk(finish='tool_calls'), '[DONE]'], text_stream('不能自动审核。')])
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '直接把审核结果改成通过'})
        self.assertIn('error', fixture.last_tool_result(self.requests[-1]))

    async def test_review_storage_failure_rolls_back_and_releases_busy(self):
        sid, request = await self.submitted()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_review BEFORE INSERT ON return_reviews BEGIN SELECT RAISE(ABORT, 'test review failure'); END")
        response = await self.review(sid, request)
        self.assertEqual(response.status_code, 503)
        state = (await self.client.get(f'/api/sessions/{sid}')).json()
        self.assertFalse(state['busy'])
        self.assertIsNone(state['return_previews'][0]['review'])
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('DROP TRIGGER fail_review')
        self.assertEqual((await self.review(sid, request)).status_code, 200)
