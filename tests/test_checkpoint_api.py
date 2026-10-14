"""真实HTTP边界与临时数据库，外部模型使用SDK传输替身。"""
from contextlib import closing
import asyncio
import json
import sqlite3
import unittest
from unittest.mock import patch
import agent_service
from context_memory import estimate_request

import test_api_app as fixture
from test_agent_plan_api import steps, update
from test_tool_execution_api import tool_stream
from test_stream_agent import text_stream


class CheckpointApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart_app = fixture.ApiAppTests.restart_app
    seed_knowledge = fixture.ApiAppTests.seed_knowledge
    seed_retrievable_history = fixture.ApiAppTests.seed_retrievable_history

    async def checkpoint(self, sid):
        response = await self.client.get(f'/api/sessions/{sid}/checkpoint')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['checkpoint']

    async def test_restart_resumes_partial_summary_then_commits_history_atomically(self):
        sid = await self.new_session()
        history = agent_service.new_conversation() + [{'role': 'user', 'content': '历史订单资料' * 3000}, {'role': 'assistant', 'content': '已记录。'}]
        self.app.state.store.replace_history(sid, history)
        self.app.state.sessions.clear()
        summary = fixture.ApiAppTests.summary_response(self, '已记录 O1003 的历史订单资料。')
        self.responses = [summary, fixture.ApiAppTests.summary_response(self, '未完成摘要', finish='length')]
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '继续查询。'})
        self.assertEqual(failed.status_code, 502, failed.text)
        cp = self.app.state.checkpoints.load(sid)
        pending = cp['state']['context']['pending_summary']
        self.assertGreater(pending['offset'], 0)
        self.assertEqual(cp['state']['context']['memory']['covered_messages'], 0)
        saved = self.app.state.store.load_session(sid)
        self.assertEqual(saved['history'], history)
        self.assertEqual(saved['memory'].covered_messages, 0)
        await self.restart_app()
        original_create = self.app.state.client.chat.completions.create
        async def model(**kwargs):
            self.responses.append(text_stream('请提供需要查询的订单号。') if kwargs.get('stream') else summary)
            return await original_create(**kwargs)
        with patch.object(self.app.state.client.chat.completions, 'create', side_effect=model):
            response = await self.client.post(f'/api/sessions/{sid}/resume/stream', json={'run_id': cp['run_id']})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        resumed = json.loads(self.requests[2]['messages'][-1]['content'])
        self.assertEqual(resumed['turn_fragment']['offset'], pending['offset'])
        self.assertEqual(resumed['previous_summary'], '已记录 O1003 的历史订单资料。')
        saved = self.app.state.store.load_session(sid)
        self.assertEqual(saved['history'][:len(history)], history)
        self.assertEqual(len(saved['history']), len(history) + 2)
        self.assertEqual(saved['memory'].covered_messages, len(history))
        self.assertIsNone(self.app.state.checkpoints.load(sid))
        trace = self.app.state.traces.load(sid, cp['run_id'])
        self.assertEqual(trace['status'], 'completed')
        for request in self.requests:
            if not request.get('stream'):
                self.assertLessEqual(estimate_request(request['messages']) + request['max_tokens'] + 256, 8192)

    async def test_restart_resume_preserves_plan_evidence_and_saves_one_turn(self):
        sid = await self.new_session()
        plan = steps()[:1]
        self.responses = [update('start', plan), tool_stream(('order-result', 'get_order', {'order_id': 'O1003'})), ['[DONE]']]
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '规划查询 O1003'})
        self.assertEqual(failed.status_code, 502)
        saved = await self.checkpoint(sid)
        self.assertEqual(saved['plan']['status'], 'in_progress')
        await self.restart_app()
        self.assertEqual(await self.checkpoint(sid), saved)
        plan[0].update(status='completed', evidence=['order-result'])
        self.responses.extend([update('finish', plan), text_stream('订单已签收。')])
        response = await self.client.post(f'/api/sessions/{sid}/resume/stream', json={'run_id': saved['run_id']})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        self.assertIn('resumed', [name for name, _ in events])
        self.assertEqual(events[-1][1]['plan']['status'], 'completed')
        self.assertEqual(sum(t['name'] == 'get_order' for t in events[-1][1]['execution']['tools']), 1)
        history = (await self.client.get(f'/api/sessions/{sid}')).json()['messages']
        self.assertEqual(sum(m['role'] == 'user' for m in history), 1)
        self.assertIsNone(await self.checkpoint(sid))
        repeat = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': saved['run_id']})
        self.assertEqual(repeat.status_code, 409)
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json()['messages'], history)

    async def test_pending_run_blocks_new_question_and_foreign_run_id_then_can_discard(self):
        sid, other = await self.new_session(), await self.new_session()
        self.responses = [['[DONE]'], text_stream('新问题已处理。')]
        await self.client.post('/api/chat', json={'session_id': sid, 'message': '原问题'})
        saved = await self.checkpoint(sid)
        overlap = await self.client.post('/api/chat', json={'session_id': sid, 'message': '新问题'})
        self.assertEqual(overlap.status_code, 409)
        for target, run_id in [(other, saved['run_id']), (sid, 'wrong')]:
            response = await self.client.post(f'/api/sessions/{target}/resume', json={'run_id': run_id})
            self.assertEqual(response.status_code, 409)
        self.assertEqual(await self.checkpoint(sid), saved)
        discarded = await self.client.post(f'/api/sessions/{sid}/checkpoint/discard', json={'run_id': saved['run_id']})
        self.assertEqual(discarded.status_code, 204)
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '新问题'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['answer'], '新问题已处理。')

    async def test_commit_failure_resumes_final_candidate_without_regenerating(self):
        sid = await self.new_session()
        self.responses = [text_stream('完整回复。')]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("CREATE TRIGGER fail_message BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT,'disk failure'); END")
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '你好'})
        self.assertEqual(failed.status_code, 503)
        saved = await self.checkpoint(sid)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute('DROP TRIGGER fail_message')
        await self.restart_app()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': saved['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['answer'], '完整回复。')
        self.assertEqual(len(self.requests), 1)
        self.assertIsNone(await self.checkpoint(sid))

    async def test_resume_preserves_summary_and_knowledge_sources(self):
        sid = await self.new_session()
        self.seed_knowledge()
        history, memory = await self.seed_retrievable_history(sid)
        self.responses = [tool_stream(('policy', 'search_knowledge', {'query': '退款'})), ['[DONE]'], text_stream('示例退款规则已查到。')]
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询退款规则'})
        self.assertEqual(failed.status_code, 502)
        run = await self.checkpoint(sid)
        await self.restart_app()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': run['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['knowledge']['calls'], 1)
        self.assertIn('3个工作日', response.json()['sources'][0]['text'])
        saved = self.app.state.store.load_session(sid)
        self.assertEqual(saved['memory'], memory)
        self.assertEqual(saved['history'][:len(history)], history)
        self.assertTrue(any(memory.text in (m.get('content') or '') for m in self.requests[-1]['messages']))

    async def test_resume_keeps_preview_identity_and_confirmation_is_idempotent(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})), ['[DONE]'], text_stream('请核对退货预览。')]
        await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1003 质量问题，准备退货预览'})
        run = self.app.state.checkpoints.load(sid)
        preview_id = run['state']['return_previews'][0]['preview_id']
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json()['return_previews'], [])
        await self.restart_app()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': run['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['return_previews'][0]['preview_id'], preview_id)
        url = f'/api/sessions/{sid}/return-previews/{preview_id}/confirm'
        first = await self.client.post(url, json={'confirmed': True})
        repeated = await self.client.post(url, json={'confirmed': True})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(repeated.json()['request']['request_id'], first.json()['request']['request_id'])
        self.assertEqual(len((await self.client.get(f'/api/sessions/{sid}')).json()['return_previews']), 1)

    async def test_browser_disconnect_retains_saved_batch_and_releases_busy(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('断连后不应保存此答复。')]
        incoming = asyncio.Queue()
        await incoming.put({'type': 'http.request', 'body': json.dumps({'session_id': sid, 'message': '查 O1003'}).encode(), 'more_body': False})
        sent = []
        async def send(event):
            sent.append(event)
            if b'event: tool_result' in event.get('body', b''):
                self.gate.clear()
                await incoming.put({'type': 'http.disconnect'})
                await asyncio.sleep(0)
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'}, 'http_version': '1.1',
                 'method': 'POST', 'scheme': 'http', 'path': '/api/chat/stream', 'raw_path': b'/api/chat/stream',
                 'query_string': b'', 'root_path': '', 'headers': [(b'content-type', b'application/json')],
                 'client': ('127.0.0.1', 12345), 'server': ('testserver', 80)}
        try:
            await asyncio.wait_for(self.app(scope, incoming.get, send), 10)
        finally:
            self.gate.set()
        self.assertFalse(self.app.state.sessions[sid].busy)
        self.assertFalse(any(b'event: done' in event.get('body', b'') for event in sent))
        checkpoint = await self.checkpoint(sid)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(len((await self.client.get(f'/api/sessions/{sid}')).json()['messages']), 1)
        self.responses = self.responses[:len(self.requests)] + [text_stream('订单已签收。')]
        await self.restart_app()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': checkpoint['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['execution']['calls'], 1)
