"""人工介入的 HTTP、数据库和真实 Agent 循环边界；模型传输为离线替身。"""
import json
import asyncio
import unittest
from unittest.mock import patch

import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream
from tool_runtime import ToolPolicy


class HumanApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixture.ApiAppTests.asyncSetUp(self)
        self.app.state.approval_tools = frozenset({'prepare_return_request'})

    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session

    async def restart(self):
        await fixture.ApiAppTests.restart_app(self)
        self.app.state.approval_tools = frozenset({'prepare_return_request'})

    async def pause(self, stream=False):
        sid = await self.new_session()
        self.responses.append(tool_stream(('preview-call', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})))
        self.responses.append(text_stream('尚未实现审批时错误地提前完成。'))
        with patch.object(self.app.state.returns, 'prepare_return_request', wraps=self.app.state.returns.prepare_return_request) as execute:
            response = await self.client.post('/api/chat' + ('/stream' if stream else ''), json={'session_id': sid, 'message': 'O1003 因为质量问题申请退货'})
            self.assertEqual(response.status_code, 200, response.text)
            if stream:
                self.assertEqual(fixture.parse_sse(response.text)[-1][0], 'paused')
            else:
                self.assertEqual(response.json().get('status'), 'awaiting_approval', response.text)
            execute.assert_not_called()
        self.responses.pop()
        cp = (await self.client.get(f'/api/sessions/{sid}/checkpoint')).json()['checkpoint']
        self.assertFalse(cp['can_resume'])
        self.assertEqual(len(cp['approvals']), 1)
        return sid, cp

    async def decide(self, sid, cp, decision='approve', **kwargs):
        return await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={
            'run_id': cp['run_id'], 'approval_id': cp['approvals'][0]['approval_id'], 'decision': decision, **kwargs})

    async def test_approve_after_restart_executes_once_and_keeps_confirmation_flow(self):
        sid, cp = await self.pause(stream=True)
        await self.restart()
        direct = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(direct.status_code, 409)
        self.assertEqual(self.app.state.checkpoints.load(sid)['resume_count'], 0)
        decision = await self.decide(sid, cp)
        self.assertEqual(decision.status_code, 200, decision.text)
        await self.restart()
        self.responses.append(text_stream('预览已生成，请核对后确认提交。'))
        result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(result.status_code, 200, result.text)
        preview = result.json()['return_previews'][0]
        self.assertEqual(preview['reason'], '质量问题')
        repeat = await self.decide(sid, cp)
        self.assertEqual(repeat.status_code, 409)
        first = await self.client.post(f'/api/sessions/{sid}/return-previews/{preview["preview_id"]}/confirm', json={'confirmed': True})
        second = await self.client.post(f'/api/sessions/{sid}/return-previews/{preview["preview_id"]}/confirm', json={'confirmed': True})
        self.assertEqual(first.json(), second.json())
        self.assertEqual(len(self.requests), 2)

    async def test_edit_uses_actual_arguments_and_retains_original_in_audit(self):
        sid, cp = await self.pause()
        edited = {'order_id': 'O1003', 'reason': '耳机左侧没有声音'}
        decision = await self.decide(sid, cp, 'edit', arguments=edited)
        self.assertEqual(decision.status_code, 200, decision.text)
        self.responses.append(text_stream('预览已经准备好。'))
        response = await self.client.post(f'/api/sessions/{sid}/resume/stream', json={'run_id': cp['run_id']})
        final = fixture.parse_sse(response.text)[-1]
        self.assertEqual(final[0], 'done', response.text)
        self.assertEqual(final[1]['return_previews'][0]['reason'], edited['reason'])
        history = self.app.state.store.load_session(sid)['history']
        tool = json.loads(next(m['content'] for m in history if m.get('tool_call_id') == 'preview-call'))
        self.assertEqual(tool['_human_decision']['original_arguments']['reason'], '质量问题')
        self.assertEqual(tool['_human_decision']['arguments'], edited)

    async def test_reject_feedback_cannot_be_retried_into_execution(self):
        sid, cp = await self.pause()
        self.assertEqual((await self.decide(sid, cp, 'reject', reason='暂时不要生成')).status_code, 200)
        self.responses.extend([tool_stream(('repeat', 'prepare_return_request', {'order_id': 'O1003', 'reason': '改成别的原因'})), text_stream('已取消生成预览。')])
        with patch.object(self.app.state.returns, 'prepare_return_request') as execute:
            response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
            self.assertEqual(response.status_code, 200, response.text)
            execute.assert_not_called()
        self.assertEqual(response.json()['return_previews'], [])
        result = next(m for m in self.requests[1]['messages'] if m.get('tool_call_id') == 'preview-call')
        self.assertEqual(json.loads(result['content'])['error_code'], 'human_rejected')

    async def test_wrong_session_duplicate_and_invalid_edit_do_not_change_decision(self):
        sid, cp = await self.pause()
        other = await self.new_session()
        self.assertEqual((await self.decide(other, cp)).status_code, 409)
        for arguments in ({'order_id': 'O1003', 'reason': ''}, {'order_id': 'O1003', 'reason': '好', 'confirmed': True}):
            self.assertEqual((await self.decide(sid, cp, 'edit', arguments=arguments)).status_code, 422)
        self.assertEqual((await self.decide(sid, cp)).status_code, 200)
        self.assertEqual((await self.decide(sid, cp, 'reject')).status_code, 409)
        self.assertEqual(self.app.state.checkpoints.load(sid)['resume_count'], 0)

    async def test_concurrent_decisions_accept_only_one_and_keep_the_winner(self):
        sid, cp = await self.pause()
        results = await asyncio.gather(self.decide(sid, cp, 'approve'), self.decide(sid, cp, 'reject'))
        self.assertEqual(sorted(result.status_code for result in results), [200, 409])
        winner = 'approve' if results[0].status_code == 200 else 'reject'
        pending = self.app.state.checkpoints.load(sid)['state']['pending_batch']
        self.assertEqual(pending['approvals'][0]['decision'], winner)

    async def test_default_application_enables_registered_approval_policy(self):
        import api_app
        app = api_app.create_app(db_path=self.db_path, knowledge_dir=self.knowledge_dir)
        async with app.router.lifespan_context(app):
            self.assertIn('prepare_return_request', app.state.approval_tools)

    async def test_disconnect_after_tool_result_resumes_without_reapproval_or_second_preview(self):
        sid, cp = await self.pause()
        await self.decide(sid, cp)
        self.responses.extend([['[DONE]'], text_stream('预览已准备好。')])
        first = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(first.status_code, 502)
        saved = self.app.state.checkpoints.load(sid)
        self.assertEqual(saved['resume_count'], 0)
        preview_id = saved['state']['return_previews'][0]['preview_id']
        await self.restart()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['return_previews'][0]['preview_id'], preview_id)
        self.assertEqual(response.json()['execution']['calls'], 1)
        self.assertEqual(response.json()['human_decisions'][0]['decision'], 'approve')

    async def test_pause_preserves_search_limit_and_reserved_business_budget(self):
        sid = await self.new_session()
        self.app.state.tool_policy = ToolPolicy(max_calls=5)
        self.responses = [tool_stream(
            ('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'}),
            *((f'search-{i}', 'search_history', {'query': f'订单{i}'}) for i in range(4))),
            text_stream('请核对预览。')]
        paused = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询后预览'})
        self.assertEqual(paused.json().get('status'), 'awaiting_approval', paused.text)
        cp = paused.json()['checkpoint']
        self.assertEqual(self.app.state.checkpoints.load(sid)['state']['runtime']['calls'], 5)
        await self.decide(sid, cp)
        with patch.object(self.app.state.store, 'search_history', wraps=self.app.state.store.search_history) as search:
            response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(search.call_count, 3)
        final = response.json()
        self.assertEqual(final['retrieval']['calls'], 3)
        self.assertEqual(final['execution']['calls'], 5)

    async def test_rejection_of_one_call_does_not_override_another_explicit_approval(self):
        for rejected_first in (False, True):
            sid = await self.new_session()
            self.responses.extend([tool_stream(
                ('one', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'}),
                ('two', 'prepare_return_request', {'order_id': 'O1003', 'reason': '质量问题'})), text_stream('请核对已批准的预览。')])
            paused = await self.client.post('/api/chat', json={'session_id': sid, 'message': '申请退货'})
            self.assertEqual(paused.json().get('status'), 'awaiting_approval', paused.text)
            cp = paused.json()['checkpoint']
            for index, approval in enumerate(cp['approvals']):
                decision = 'reject' if (index == 0) == rejected_first else 'approve'
                result = await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={
                    'run_id': cp['run_id'], 'approval_id': approval['approval_id'], 'decision': decision})
                self.assertEqual(result.status_code, 200, result.text)
            final = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
            self.assertEqual(final.status_code, 200, final.text)
            self.assertEqual(len(final.json()['return_previews']), 1)
