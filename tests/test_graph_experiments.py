"""独立实验 API：真实 LangGraph/SQLite，离线模型 HTTP 替身。"""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx2
from openai import AsyncOpenAI
import test_api_app as fixture


def completion(text=None, calls=None):
    message = {'role': 'assistant', 'content': text}
    if calls:
        message['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}} for call_id, name, arguments in calls]
    return {'id': 'graph-offline', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 8, 'total_tokens': 38}}


class GraphExperimentTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def asyncSetUp(self):
        await fixture.ApiAppTests.asyncSetUp(self)
        self.actor = []
        self.graph_requests = []
        self.block = None
        await self.install_client()

    async def install_client(self):
        await self.app.state.client.close()
        async def handle(request):
            body = json.loads(request.content)
            self.graph_requests.append(body)
            if self.block is not None:
                await self.block.wait()
            if body.get('response_format'):
                payload = json.loads(body['messages'][-1]['content'])
                if getattr(self, 'echo_candidate_error', False):
                    return httpx2.Response(400, json={'error': {'message': payload['candidate'], 'type': 'invalid_request_error'}})
                return httpx2.Response(200, json=completion(json.dumps({
                    'verdict': 'pass', 'issues': [], 'evidence_ids': payload['current_evidence_ids']})))
            response = self.actor.pop(0)
            if response == 'unavailable':
                return httpx2.Response(503, json={'error': {'message': 'offline model unavailable', 'type': 'server_error'}})
            return httpx2.Response(200, json=response)
        self.app.state.client = AsyncOpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False))

    async def create_run(self, question='查询 O1003 当前状态'):
        response = await self.client.post('/api/graph/runs', json={'question': question})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['run_id']

    async def settled(self, run_id):
        for _ in range(300):
            response = await self.client.get(f'/api/graph/runs/{run_id}')
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            if not result['busy']:
                return result
            await asyncio.sleep(.02)
        self.fail('实验没有及时结束')

    async def test_page_and_completed_graph_are_isolated_from_formal_chat(self):
        page = await self.client.get('/graph')
        self.assertEqual(page.status_code, 200)
        self.assertIn('LangGraph', page.text)
        self.actor = [completion(calls=[('order', 'get_order', {'order_id': 'O1003'})]), completion('O1003 已签收。')]
        run_id = await self.create_run()
        result = await self.settled(run_id)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['answer'], 'O1003 已签收。')
        self.assertEqual(result['metrics']['model_requests'], 2)
        self.assertTrue(result['events'])
        self.assertEqual((await self.client.get('/api/sessions')).json()['sessions'], [])
        self.assertEqual(len((await self.client.get('/api/graph/runs')).json()['runs']), 1)

    async def test_approval_survives_restart_and_rejects_stale_decisions(self):
        self.actor = [completion(calls=[('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '耳机左侧无声'})]),
                      completion('已生成实验预览，尚未提交申请。')]
        run_id = await self.create_run('O1003 耳机左侧无声，请生成退货预览')
        paused = await self.settled(run_id)
        self.assertEqual(paused['status'], 'awaiting_approval', paused)
        self.assertEqual(paused['return_previews'], [])
        self.assertEqual(paused['answer'], '')
        approval = paused['approvals'][0]
        invalid = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={'approval_id': 'wrong', 'decision': 'approve'})
        self.assertEqual(invalid.status_code, 409)
        await fixture.ApiAppTests.restart_app(self)
        await self.install_client()
        restored = (await self.client.get(f'/api/graph/runs/{run_id}')).json()
        self.assertEqual(restored['approvals'][0]['approval_id'], approval['approval_id'])
        approved = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={
            'approval_id': approval['approval_id'], 'decision': 'edit',
            'arguments': {'order_id': 'O1003', 'reason': '耳机右侧无声'}})
        self.assertEqual(approved.status_code, 202, approved.text)
        result = await self.settled(run_id)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['return_previews'][0]['reason'], '耳机右侧无声')
        self.assertEqual(result['return_previews'][0]['state'], 'pending')
        self.assertEqual((await self.client.post(f'/api/graph/runs/{run_id}/resume', json={})).status_code, 409)
        self.assertEqual((await self.client.get('/api/sessions')).json()['sessions'], [])

    async def test_failed_model_can_resume_after_restart_and_counts_new_request(self):
        self.actor = ['unavailable', completion(calls=[('order', 'get_order', {'order_id': 'O1003'})]), completion('O1003 已签收。')]
        run_id = await self.create_run()
        failed = await self.settled(run_id)
        self.assertEqual(failed['status'], 'failed', failed)
        self.assertNotIn('offline-only', failed['error'])
        await fixture.ApiAppTests.restart_app(self)
        await self.install_client()
        resumed = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={})
        self.assertEqual(resumed.status_code, 202, resumed.text)
        final = await self.settled(run_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(final['metrics']['model_requests'], 3)
        self.assertEqual(final['resume_count'], 1)

    async def test_running_experiment_cannot_be_started_twice_and_question_is_not_replaceable(self):
        self.block = asyncio.Event()
        self.actor = [completion('请提供需要查询的订单号。')]
        run_id = await self.create_run('请查询订单')
        try:
            running = await self.client.get(f'/api/graph/runs/{run_id}')
            self.assertTrue(running.json()['busy'])
            retry = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={})
            self.assertEqual(retry.status_code, 409)
            altered = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={'question': '换个问题'})
            self.assertEqual(altered.status_code, 422)
        finally:
            self.block.set()
        self.assertEqual((await self.settled(run_id))['status'], 'completed')

    async def test_invalid_requests_do_not_create_experiments(self):
        for body in [{'question': ' '}, {'question': '查询', 'model': 'override'}, {'question': 'x'*2001}]:
            self.assertEqual((await self.client.post('/api/graph/runs', json=body)).status_code, 422)
        unknown = await self.client.get('/api/graph/runs/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual((await self.client.get('/api/graph/runs')).json()['runs'], [])

    async def test_shutdown_cancels_model_and_preserves_resumable_checkpoint(self):
        self.block = asyncio.Event()
        self.actor = [completion('请提供订单号。')]
        run_id = await self.create_run('查询订单')
        for _ in range(300):
            if self.graph_requests:
                break
            await asyncio.sleep(.02)
        self.assertTrue(self.graph_requests)
        await fixture.ApiAppTests.restart_app(self)
        self.block = None
        await self.install_client()
        restored = (await self.client.get(f'/api/graph/runs/{run_id}')).json()
        self.assertEqual(restored['status'], 'interrupted', restored)
        self.assertTrue(restored['can_resume'])
        self.assertEqual(restored['answer'], '')
        response = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={})
        self.assertEqual(response.status_code, 202, response.text)
        final = await self.settled(run_id)
        self.assertEqual(final['status'], 'completed', final)
        self.assertEqual(final['metrics']['model_requests'], 2)

    async def test_fault_recovery_is_limited_to_three_attempts(self):
        self.actor = ['unavailable'] * 4
        run_id = await self.create_run()
        for count in range(4):
            failed = await self.settled(run_id)
            self.assertEqual(failed['status'], 'failed', failed)
            self.assertEqual(failed['resume_count'], count)
            self.assertEqual(failed['can_resume'], count < 3)
            resumed = await self.client.post(f'/api/graph/runs/{run_id}/resume', json={})
            self.assertEqual(resumed.status_code, 202 if count < 3 else 409, resumed.text)
        self.assertEqual(failed['metrics']['model_requests'], 4)

    async def test_saved_experiments_are_readable_without_model_configuration(self):
        self.actor = [completion('请提供订单号。')]
        run_id = await self.create_run('查询订单')
        self.assertEqual((await self.settled(run_id))['status'], 'completed')
        await self.app.state.client.close()
        self.app.state.client = None
        self.app.state.api_key = ''
        result = await self.client.get(f'/api/graph/runs/{run_id}')
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['answer'], '请提供订单号。')
        self.assertFalse((await self.client.get('/api/graph/runs')).json()['configured'])
        self.assertEqual((await self.client.post('/api/graph/runs', json={'question': '新问题'})).status_code, 503)

    async def test_provider_error_cannot_expose_unverified_candidate(self):
        self.echo_candidate_error = True
        self.actor = [completion('PRIVATE_UNVERIFIED_CANDIDATE')]
        run_id = await self.create_run('查询订单')
        result = await self.settled(run_id)
        self.assertEqual(result['status'], 'failed', result)
        self.assertNotIn('PRIVATE_UNVERIFIED_CANDIDATE', json.dumps(result))
        self.assertNotIn('PRIVATE_UNVERIFIED_CANDIDATE', (await self.client.get('/api/graph/runs')).text)
