"""主管/子 Agent 使用真实 API 和 SDK；模型 HTTP 为离线可控替身。"""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx2
from openai import AsyncOpenAI
import agent_service
import api_app
from agent_eval import ScriptStream
from context_memory import estimate_request
from tool_runtime import ToolPolicy
import test_api_app as fixture


def tool_message(call_id, name, arguments):
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}


def actor_call(call_id, name, arguments):
    call = tool_message(call_id, name, arguments)
    call['function']['arguments'] = arguments
    return {'calls': [call]}


def child_response(*, calls=None, summary='已取得实际查询结果。', ids=(), status='completed'):
    message = {'role': 'assistant', 'content': None, 'tool_calls': calls} if calls else {'role': 'assistant', 'content': json.dumps({'status': status, 'summary': summary, 'evidence_ids': list(ids)}, ensure_ascii=False)}
    return {'id': 'child', 'object': 'chat.completion', 'created': 0, 'model': 'offline', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}], 'usage': {'prompt_tokens': 30, 'completion_tokens': 5, 'total_tokens': 35}}


def delegate(*agents):
    return {'calls': [{'id': 'delegation', 'type': 'function', 'function': {'name': 'delegate_tasks', 'arguments': {'tasks': [
        {'agent': agent, 'task': '查询 O1003' if agent == 'orders' else '检索退款政策', 'context': ''} for agent in agents]}}}]}


class DelegationApiTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    seed_knowledge = fixture.ApiAppTests.seed_knowledge

    async def asyncSetUp(self):
        await fixture.ApiAppTests.asyncSetUp(self)
        self.actor, self.children = [], {'orders': [], 'knowledge': []}
        self.requests = []
        self.hold_knowledge = None
        await self.install_client()

    async def install_client(self):
        await self.app.state.client.close()
        async def handle(request):
            body = json.loads(request.content)
            self.requests.append(body)
            if body.get('stream'):
                response = self.actor.pop(0)
                return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=ScriptStream(response))
            if body.get('response_format'):
                payload = json.loads(body['messages'][-1]['content'])
                result = child_response(ids=payload['current_evidence_ids'])
                result['choices'][0]['message']['content'] = json.dumps({'verdict': 'pass', 'issues': [], 'evidence_ids': payload['current_evidence_ids']})
                return httpx2.Response(200, json=result)
            agent = 'orders' if body['tools'][0]['function']['name'] == 'get_order' else 'knowledge'
            if agent == 'knowledge' and self.hold_knowledge is not None:
                await self.hold_knowledge.wait()
            response = self.children[agent].pop(0)
            if response == 'unavailable':
                return httpx2.Response(503, json={'error': {'message': 'offline child unavailable', 'type': 'server_error'}})
            return httpx2.Response(200, json=response)
        self.app.state.client = AsyncOpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False))
        self.app.state.delegation_enabled = True

    def successful_children(self):
        self.children['orders'] = [child_response(calls=[tool_message('order', 'get_order', {'order_id': 'O1003'})]), child_response(ids=['specialist_1:r1:order'], summary='O1003 已签收。')]
        self.children['knowledge'] = [child_response(calls=[tool_message('policy', 'search_knowledge', {'query': '退款'})]), child_response(ids=['specialist_2:r1:policy'], summary='仓库验收后3个工作日原路退款。')]

    async def test_parallel_specialists_have_isolated_context_actual_evidence_and_sources(self):
        sid = await self.new_session(); self.seed_knowledge()
        history = agent_service.new_conversation() + [{'role': 'user', 'content': '这段旧聊天秘密不得整段复制给子 Agent'}, {'role': 'assistant', 'content': '已记录。'}]
        self.app.state.store.replace_history(sid, history); self.app.state.sessions.clear()
        self.actor = [delegate('orders', 'knowledge'), {'text': 'O1003 已签收，退款需仓库验收后处理。'}]
        self.successful_children()
        self.app.state.result_verification = True
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '查询 O1003，并核对退款政策'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        self.assertIn('delegate_tasks', [t['function']['name'] for t in self.requests[0]['tools']])
        done = events[-1][1]
        self.assertEqual(done['delegation']['requests'], 4)
        self.assertEqual([t['status'] for t in done['delegation']['tasks']], ['completed', 'completed'])
        child_requests = [r for r in self.requests if not r.get('stream') and not r.get('response_format')]
        self.assertEqual(len(child_requests), 4)
        for request in child_requests:
            self.assertNotIn('这段旧聊天秘密', json.dumps(request, ensure_ascii=False))
            self.assertEqual(len(request['tools']), 1)
            self.assertLessEqual(estimate_request(request['messages'], request['tools']) + request['max_tokens'] + 256, 8192)
        final_request = next(r for r in reversed(self.requests) if r.get('stream'))
        result = json.loads(next(m['content'] for m in final_request['messages'] if m.get('tool_call_id') == 'delegation'))
        self.assertEqual(len(result['delegations']), 2)
        self.assertEqual(result['delegations'][0]['evidence'][0]['result']['order_id'], 'O1003')
        self.assertTrue(done['sources'])
        self.assertEqual(done['verification']['status'], 'passed')
        trace = self.app.state.traces.list_runs(sid)['runs'][0]
        self.assertEqual(trace['metrics']['subagent_requests'], 4)
        self.assertEqual(trace['metrics']['tool_calls'], 2)
        observed = [e['data'] for e in self.app.state.traces.load(sid, trace['run_id'])['events'] if e['kind'] == 'context_budget']
        self.assertEqual(len(observed), len(self.requests))
        self.assertEqual([d['sequence'] for d in observed], list(range(1, len(observed) + 1)))
        self.assertEqual(sum(d['role'] == 'specialist' for d in observed), 4)
        self.assertIsNone(self.app.state.checkpoints.load(sid))

    async def test_one_model_failure_keeps_other_specialist_result(self):
        sid = await self.new_session(); self.seed_knowledge()
        self.actor = [delegate('orders', 'knowledge'), {'text': '订单暂时无法查询，已查到退款政策。'}]
        self.successful_children(); self.children['orders'] = ['unavailable']
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003，并核对退款政策'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([t['status'] for t in response.json()['delegation']['tasks']], ['failed', 'completed'])
        self.assertTrue(response.json()['sources'])

    async def test_specialist_cannot_request_write_tool_or_create_approval(self):
        sid = await self.new_session()
        self.app.state.approval_tools = frozenset({'prepare_return_request'})
        self.actor = [delegate('orders'), {'text': '子任务未能完成，未生成退货预览。'}]
        self.children['orders'] = [child_response(calls=[tool_message('write', 'prepare_return_request', {'order_id': 'O1003', 'reason': '故障'})])]
        with patch.object(self.app.state.returns, 'prepare_return_request') as execute:
            result = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003'})
            self.assertEqual(result.status_code, 200, result.text)
            execute.assert_not_called()
        self.assertEqual(result.json()['return_previews'], [])
        self.assertNotEqual(result.json()['delegation']['tasks'][0]['status'], 'completed')
        self.assertIsNone(self.app.state.checkpoints.load(sid))

    async def test_children_share_parent_business_call_budget(self):
        sid = await self.new_session(); self.seed_knowledge()
        self.app.state.tool_policy = ToolPolicy(max_calls=1)
        self.actor = [delegate('orders', 'knowledge'), {'text': '工具额度不足，部分结果无法核实。'}]
        self.successful_children()
        result = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003，并核对退款政策'})
        self.assertEqual(result.status_code, 200, result.text)
        successful = [t for t in result.json()['execution']['tools'] if t['name'] != 'delegate_tasks' and t['status'] == 'ok']
        self.assertEqual(len(successful), 1)
        self.assertTrue(any(t['status'] != 'completed' for t in result.json()['delegation']['tasks']))

    async def test_cancel_and_restart_reuses_completed_sibling(self):
        sid = await self.new_session(); self.seed_knowledge()
        self.actor = [delegate('orders', 'knowledge'), {'text': 'O1003 已签收，已查询退款政策。'}]
        self.successful_children(); self.hold_knowledge = asyncio.Event()
        running = asyncio.create_task(self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O1003，并核对退款政策'}))
        try:
            for _ in range(150):
                cp = self.app.state.checkpoints.load(sid)
                tasks = ((cp or {}).get('state') or {}).get('delegation', {}).get('tasks', [])
                if tasks and tasks[0]['status'] == 'completed':
                    break
                if running.done():
                    self.fail('请求在保存子任务前结束：' + (await running).text)
                await asyncio.sleep(.02)
            else:
                self.fail('已完成的订单子任务未及时保存')
        finally:
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
        saved = self.app.state.checkpoints.load(sid)
        self.assertEqual(saved['state']['delegation']['tasks'][0]['status'], 'completed')
        await fixture.ApiAppTests.restart_app(self)
        self.hold_knowledge = None
        self.children['knowledge'][-1] = child_response(ids=['specialist_2:r2:policy'])
        await self.install_client()
        response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': saved['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        done = response.json()
        self.assertEqual(done['delegation']['tasks'][0]['rounds'], 2)
        self.assertEqual(done['delegation']['tasks'][1]['rounds'], 3)
        self.assertEqual(sum(t['name'] == 'get_order' for t in done['execution']['tools']), 1)
        self.assertEqual(sum(m['role'] == 'user' for m in self.app.state.store.load_session(sid)['history']), 1)

    async def test_delegated_evidence_completes_declared_plan_step(self):
        sid = await self.new_session()
        plan = [{'id': 'query', 'title': '委派订单查询', 'tools': ['delegate_tasks'],
                 'status': 'in_progress', 'depends_on': [], 'evidence': []}]
        self.actor = [actor_call('start', 'update_plan', {'steps': plan, 'note': ''}), delegate('orders')]
        finished = [{**plan[0], 'status': 'completed', 'evidence': ['delegation']}]
        self.actor += [actor_call('finish', 'update_plan', {'steps': finished, 'note': ''}), {'text': 'O1003 已签收。'}]
        self.successful_children()
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '按计划委派查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['plan']['status'], 'completed')
        self.assertEqual(response.json()['plan']['steps'][0]['evidence'], ['delegation'])
        self.assertEqual(response.json()['delegation']['tasks'][0]['status'], 'completed')

    async def test_supervisor_preview_after_delegation_still_requires_approval(self):
        sid = await self.new_session()
        self.app.state.approval_tools = frozenset({'prepare_return_request'})
        self.app.state.result_verification = True
        self.actor = [delegate('orders'), actor_call('preview', 'prepare_return_request',
            {'order_id': 'O1003', 'reason': '耳机左侧无声'}), {'text': '退货预览已生成，请核对并确认提交。'}]
        self.successful_children()
        with patch.object(self.app.state.returns, 'prepare_return_request', wraps=self.app.state.returns.prepare_return_request) as execute:
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1003 耳机左侧无声，先查询再生成退货预览'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json().get('status'), 'awaiting_approval', response.text)
            execute.assert_not_called()
            cp = response.json()['checkpoint']
            decision = await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={
                'run_id': cp['run_id'], 'approval_id': cp['approvals'][0]['approval_id'], 'decision': 'approve'})
            self.assertEqual(decision.status_code, 200, decision.text)
            result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(result.json()['return_previews'][0]['state'], 'pending')
        self.assertEqual(result.json()['delegation']['requests'], 2)
        self.assertEqual(result.json()['verification']['status'], 'passed')

    async def test_legacy_checkpoint_resumes_with_original_tool_set(self):
        sid = await self.new_session()
        self.app.state.delegation_enabled = False
        self.app.state.approval_tools = frozenset({'prepare_return_request'})
        self.actor = [actor_call('preview', 'prepare_return_request',
            {'order_id': 'O1003', 'reason': '耳机左侧无声'}), {'text': '预览已生成，请核对。'}]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': 'O1003 耳机左侧无声，生成退货预览'})
        self.assertEqual(response.status_code, 200, response.text)
        cp = response.json()['checkpoint']
        await fixture.ApiAppTests.restart_app(self)
        await self.install_client()
        decision = await self.client.post(f'/api/sessions/{sid}/checkpoint/decision', json={
            'run_id': cp['run_id'], 'approval_id': cp['approvals'][0]['approval_id'], 'decision': 'approve'})
        self.assertEqual(decision.status_code, 200, decision.text)
        result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(result.status_code, 200, result.text)
        resumed = next(r for r in reversed(self.requests) if r.get('stream'))
        self.assertNotIn('delegate_tasks', [t['function']['name'] for t in resumed['tools']])
        self.assertEqual(result.json()['delegation']['tasks'], [])

    async def test_handoff_context_keeps_facts_without_duplicate_plan_and_runtime_metadata(self):
        sid = await self.new_session()
        plan = [{'id': 'query', 'title': '委派订单查询', 'tools': ['delegate_tasks'],
                 'status': 'in_progress', 'depends_on': [], 'evidence': []}]
        self.actor = [actor_call('start', 'update_plan', {'steps': plan, 'note': ''}), delegate('orders')]
        finished = [{**plan[0], 'status': 'completed', 'evidence': ['delegation']}]
        self.actor += [actor_call('finish', 'update_plan', {'steps': finished, 'note': ''}), {'text': 'O1003 已签收。'}]
        self.successful_children()
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '按计划委派查询 O1003'})
        self.assertEqual(response.status_code, 200, response.text)
        projected = next(r for r in reversed(self.requests) if r.get('stream'))['messages']
        handoff = json.loads(next(m['content'] for m in projected if m.get('tool_call_id') == 'delegation'))['delegations'][0]
        self.assertNotIn('usage', handoff)
        self.assertNotIn('elapsed_ms', handoff['evidence'][0])
        self.assertEqual(handoff['evidence'][0]['result']['order_id'], 'O1003')
        latest_plan = next(m['tool_calls'][0] for m in projected if m.get('tool_calls', [{}])[0].get('id') == 'finish')
        self.assertEqual(json.loads(latest_plan['function']['arguments']), {})
        original = self.app.state.store.load_session(sid)['history']
        raw = json.loads(next(m['content'] for m in original if m.get('tool_call_id') == 'delegation'))['delegations'][0]
        self.assertIn('usage', raw)
        self.assertEqual(raw['evidence'][0]['result'], handoff['evidence'][0]['result'])
