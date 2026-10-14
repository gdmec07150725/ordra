"""真实 API、SDK 与临时数据库验证候选隔离、审批和重启恢复。"""
import json
import sqlite3
import unittest
from unittest.mock import patch

import api_app
import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream
from test_agent_plan_api import steps, update


def review(verdict='pass', ids=(), issues=None):
    payload = {'verdict': verdict, 'issues': issues or (['结论与实际证据不一致，请补查或修正。'] if verdict == 'revise' else []), 'evidence_ids': list(ids)}
    return {'id': 'review', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(payload, ensure_ascii=False)}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}}


class ResultVerificationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixture.ApiAppTests.asyncSetUp(self)
        self.app.state.result_verification = True

    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session

    async def restart(self):
        await fixture.ApiAppTests.restart_app(self)
        self.app.state.result_verification = True

    async def test_default_app_enabled_and_wrong_draft_hidden_from_sse_history_trace(self):
        app = api_app.create_app(db_path=self.db_path, knowledge_dir=self.knowledge_dir)
        async with app.router.lifespan_context(app):
            self.assertTrue(app.state.result_verification)
        sid = await self.new_session()
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('错误草稿：订单已取消。'), review('revise', ['order'], ['错误草稿：订单已取消。这句话与证据不符。']),
                          text_stream('O1003 已签收。'), review(ids=['order'])]
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '查询 O1003'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        self.assertNotIn('错误草稿', response.text)
        done = events[-1][1]
        self.assertEqual(done['verification']['checks'], 2)
        self.assertEqual(done['verification']['corrections'], 1)
        trace = self.app.state.traces.list_runs(sid)['runs'][0]
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(trace['metrics']['verification_requests'], 2)
        self.assertEqual(trace['metrics']['total_tokens'], 80)
        self.assertNotIn('错误草稿', json.dumps(self.app.state.store.load_session(sid)['history'], ensure_ascii=False))
        self.assertNotIn('错误草稿', json.dumps(self.app.state.traces.load(sid, trace['run_id']), ensure_ascii=False))

    async def test_review_failure_restart_and_final_commit_retry_do_not_repeat_actor_or_passed_review(self):
        sid = await self.new_session()
        malformed = review(); malformed['choices'][0]['message']['content'] = 'bad-json'
        self.responses = [text_stream('请提供订单号。'), malformed, review()]
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '帮我查订单'})
        self.assertEqual(failed.status_code, 502, failed.text)
        cp = self.app.state.checkpoints.load(sid)
        self.assertEqual(cp['state']['verification']['checks'], 1)
        self.assertEqual(len(self.app.state.store.load_session(sid)['history']), 1)
        await self.restart()
        with patch.object(self.app.state.store, 'replace_history', side_effect=sqlite3.OperationalError('模拟最终提交失败')):
            failed = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(failed.status_code, 503, failed.text)
        self.assertEqual(len(self.requests), 3)
        await self.restart()
        final = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(final.status_code, 200, final.text)
        self.assertEqual(final.json()['verification']['checks'], 2)
        self.assertEqual(len(self.requests), 3)
        self.assertIsNone(self.app.state.checkpoints.load(sid))
        self.assertEqual(len(self.app.state.store.load_session(sid)['history']), 3)

    async def test_wrong_preview_hard_rule_ends_unverified_without_publishing_card(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '耳机左侧无声'}))]
        for _ in range(3):
            self.responses.extend([text_stream('错误预览草稿'), review(ids=['preview'])])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1002 申请退货，屏幕损坏'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['verification']['status'], 'unverified')
        self.assertEqual(response.json()['verification']['checks'], 3)
        self.assertEqual(response.json()['verification']['corrections'], 2)
        self.assertEqual(response.json()['return_previews'], [])
        self.assertEqual(self.app.state.returns.list_previews(sid), [])
        self.assertEqual(self.app.state.traces.list_runs(sid)['runs'][0]['status'], 'unverified')

    async def test_last_correction_disconnect_can_resume_local_unverified_finish(self):
        sid = await self.new_session()
        self.responses = [tool_stream((f'order{i}', 'get_order', {'order_id': f'O100{i}'})) for i in (1, 2, 3)]
        self.responses.extend([text_stream('错误草稿'), review('revise', ['order3']), text_stream('仍有错误'), review('revise', ['order3']), ['[DONE]']])
        failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询订单'})
        self.assertEqual(failed.status_code, 502, failed.text)
        cp = (await self.client.get(f'/api/sessions/{sid}/checkpoint')).json()['checkpoint']
        self.assertEqual(cp['rounds'], 6)
        self.assertTrue(cp['can_resume'], cp['resume_reason'])
        result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['verification']['status'], 'unverified')
        self.assertEqual(len(self.requests), 8)

    async def test_blocked_note_waits_for_review_and_never_becomes_visible_answer_if_rejected(self):
        sid = await self.new_session()
        plan = steps()[:1]
        self.responses = [update('start', plan)]
        plan[0]['status'] = 'blocked'
        self.responses.extend([update('blocked', plan, '错误草稿：退款已到账，请提供账号。'), review('revise', ['blocked']),
                               text_stream('错误草稿：退款已到账。'), review('revise', ['blocked'])])
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '查询订单'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        self.assertEqual(events[-1][1]['verification']['status'], 'unverified')
        self.assertNotIn('错误草稿', response.text)
        run = self.app.state.traces.list_runs(sid)['runs'][0]
        self.assertNotIn('错误草稿', json.dumps(self.app.state.traces.load(sid, run['run_id']), ensure_ascii=False))

    async def test_correction_that_needs_preview_still_pauses_for_human(self):
        self.app.state.approval_tools = frozenset({'prepare_return_request'})
        sid = await self.new_session()
        self.responses = [text_stream('已经生成退货预览。'), review('revise'),
                          tool_stream(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '耳机左侧无声'})),
                          text_stream('预览已生成，请核对并确认提交。'), review(ids=['preview'])]
        with patch.object(self.app.state.returns, 'prepare_return_request', wraps=self.app.state.returns.prepare_return_request) as execute:
            paused = await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1003 耳机左侧无声，生成退货预览'})
            self.assertEqual(paused.status_code, 200, paused.text)
            self.assertEqual(paused.json().get('status'), 'awaiting_approval', paused.text)
            execute.assert_not_called()
            cp = paused.json()['checkpoint']
            decision = await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={'run_id': cp['run_id'], 'approval_id': cp['approvals'][0]['approval_id'], 'decision': 'approve'})
            self.assertEqual(decision.status_code, 200, decision.text)
            result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(result.json()['verification']['corrections'], 1)
        self.assertEqual(result.json()['return_previews'][0]['state'], 'pending')

    async def test_completed_plan_allows_readonly_correction_without_reopening_steps(self):
        sid = await self.new_session()
        plan = steps()[:1]
        self.responses = [update('start', plan), tool_stream(('order', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order'])
        self.responses.extend([update('finish', plan), text_stream('状态待确认。'), review('revise', ['order']),
                               tool_stream(('recheck', 'get_order', {'order_id': 'O1003'})), text_stream('O1003 已签收。'), review(ids=['recheck'])])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '规划查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['verification']['status'], 'passed')
        self.assertEqual(response.json()['plan']['steps'][0]['evidence'], ['order'])
        self.assertEqual(sum(t['name'] == 'get_order' and t['status'] == 'ok' for t in response.json()['execution']['tools']), 2)
