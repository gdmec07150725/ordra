"""Exercise actual graph/API/checkpoints with only the model HTTP replies scripted."""
from copy import deepcopy
import json
import unittest

import agent_service
import test_api_app as fixture
from test_agent_plan_api import steps, update
from test_tool_execution_api import tool_stream
from test_stream_agent import text_stream
from context_memory import ContextPolicy, estimate_request


class RecoveryGraphTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    seed_knowledge = fixture.ApiAppTests.seed_knowledge
    restart_app = fixture.ApiAppTests.restart_app

    async def test_empty_search_replans_then_finishes_without_repeating_completed_order(self):
        sid = await self.new_session()
        self.seed_knowledge()
        p = steps()
        self.responses = [update('start', p), tool_stream(('order', 'get_order', {'order_id': 'O1003'}))]
        p[0].update(status='completed', evidence=['order'])
        p[1]['status'] = 'in_progress'
        self.responses += [update('next', p), tool_stream(('empty', 'search_knowledge', {'query': 'zzzznomatchzzzz'}))]
        p[1]['title'] = '改查退款政策'
        self.responses += [update('repair', p, '没有匹配资料，改查退款政策。'),
                           tool_stream(('policy', 'search_knowledge', {'query': '退款'}))]
        p[1].update(status='completed', evidence=['policy'])
        self.responses += [update('finish', p), text_stream('已核对订单并查到退款政策。')]
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '核对 O1003 后查询售后'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        feedback = '\n'.join(m.get('content') or '' for m in self.requests[4]['messages'])
        self.assertIn('[执行恢复反馈', feedback)
        self.assertIn('empty_result', feedback)
        self.assertEqual(events[-1][1]['plan']['steps'][0]['evidence'], ['order'])
        self.assertEqual(events[-1][1]['recovery']['replans'], 1)
        tools = events[-1][1]['execution']['tools']
        self.assertEqual(sum(t['name'] == 'get_order' for t in tools), 1)
        runs = (await self.client.get(f'/api/sessions/{sid}/runs')).json()['runs']
        trace = (await self.client.get(f'/api/sessions/{sid}/runs/{runs[0]["run_id"]}')).json()
        replans = [e for e in trace['events'] if e['kind'] == 'recovery' and e['data']['kind'] == 'replanned']
        self.assertEqual(len(replans), 1)
        self.assertEqual(replans[0]['data']['changed_steps'], ['policy'])

    async def test_plan_wording_loop_stops_saves_history_and_accepts_new_input(self):
        sid = await self.new_session()
        p = steps()[:1]
        self.responses = [update('start', p)] + [update(f'note{i}', p, f'继续查询{i}') for i in range(3)]
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '查询 O1003'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        self.assertTrue(events[-1][1]['recovery']['stopped'])
        self.assertEqual(events[-1][1]['plan']['status'], 'blocked')
        self.assertEqual(len(self.requests), 4)
        saved = (await self.client.get(f'/api/sessions/{sid}')).json()
        self.assertIsNone(saved.get('checkpoint'))
        metadata = [json.loads(m['content']).get('_recovery') for m in saved['messages'] if m['role'] == 'tool']
        self.assertEqual([item for item in metadata if item][-1]['plan']['status'], 'blocked')
        trace = (await self.client.get(f'/api/sessions/{sid}/runs')).json()['runs'][0]
        self.assertEqual(trace['status'], 'stalled')
        await self.restart_app()
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json()['messages'], saved['messages'])
        self.responses.append(text_stream('请补充你需要查询的具体事项。'))
        next_reply = await self.client.post('/api/chat', json={'session_id': sid, 'message': '帮我核对物流'})
        self.assertEqual(next_reply.status_code, 200, next_reply.text)

    async def test_three_failed_searches_stop_without_verifier_or_extra_actor_call(self):
        sid = await self.new_session()
        p = steps()[1:]
        p[0].update(status='in_progress', depends_on=[])
        self.responses = [update('start', p)] + [tool_stream((f's{i}', 'search_knowledge', {'query': f'zzzznomatch{i}'})) for i in range(3)]
        self.app.state.result_verification = True
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '请查询不存在的条款'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['recovery']['stopped'])
        self.assertEqual(response.json()['verification']['status'], 'not_applicable')
        self.assertEqual(len(self.requests), 4)

    async def test_checkpoint_resume_keeps_streak_and_finishes_without_replaying_queries(self):
        p = steps()[1:]
        p[0].update(status='in_progress', depends_on=[])
        self.responses = [update('start', p)] + [tool_stream((f's{i}', 'search_knowledge', {'query': f'zzzz{i}'})) for i in range(3)]
        captured, queried = {}, []
        async def save(state):
            captured['state'] = json.loads(json.dumps(state))
        async def search(query):
            queried.append(query)
            return {'query': query, 'results': []}
        def run(checkpoint=None):
            return agent_service.stream_agent(self.app.state.client, self.app.state.model, agent_service.new_conversation(),
                '查询条款', search_knowledge=search, checkpoint=checkpoint, save_checkpoint=save)
        first = run()
        async for event in first:
            if event['event'] == 'tool_result' and len(queried) == 2:
                break
        await first.aclose()
        self.assertEqual(captured['state']['recovery']['stalled_batches'], 2)
        events = [event async for event in run(captured['state'])]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertTrue(events[-1]['recovery']['stopped'])
        self.assertEqual(queried, ['zzzz0', 'zzzz1', 'zzzz2'])
        requests_before = len(self.requests)
        again = [event async for event in run(captured['state'])]
        self.assertEqual(again[-1]['answer'], events[-1]['answer'])
        self.assertEqual(len(self.requests), requests_before)

    async def test_missing_information_still_uses_existing_blocked_flow(self):
        sid = await self.new_session()
        p = steps()[:1]
        self.responses = [update('start', p)]
        p[0]['status'] = 'blocked'
        self.responses.append(update('ask', p, '请提供订单号。'))
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查一下订单'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['recovery']['stopped'])
        self.assertEqual(response.json()['answer'], '请提供订单号。')
        self.assertEqual(len(self.requests), 2)

    async def test_optional_feedback_does_not_evict_fitting_business_evidence(self):
        policy = ContextPolicy(max_tokens=8192)
        calls = [
            {'id': 'bad', 'type': 'function', 'function': {'name': 'get_order', 'arguments': json.dumps({'order_id': 'O9999'})}},
            {'id': 'kb', 'type': 'function', 'function': {'name': 'search_knowledge', 'arguments': json.dumps({'query': '政策'})}},
        ]
        original = {}
        async def search(query):
            result = {'query': query, 'results': [], 'has_more': False, 'budget_limited': False}
            hit = {'source': 'kb:abcdef:0', 'version': 'v1', 'title': '政策', 'text': ''}
            base = self.requests[-1]['messages'] + [{'role': 'assistant', 'content': None, 'tool_calls': calls},
                    {'role': 'tool', 'tool_call_id': 'bad', 'content': json.dumps(agent_service.execute_tool('get_order', calls[0]['function']['arguments']), ensure_ascii=False)}]
            low, high = 0, 20000
            while low < high:
                size = (low + high + 1) // 2
                hit['text'] = 'x' * size
                result['results'] = [hit]
                messages = base + [{'role': 'tool', 'tool_call_id': 'kb', 'content': json.dumps(result, ensure_ascii=False)}]
                if estimate_request(messages, self.requests[-1]['tools']) <= policy.input_limit - 20:
                    low = size
                else:
                    high = size - 1
            hit['text'] = 'x' * low
            original.update(deepcopy(result))
            return result
        self.responses = [tool_stream(('bad', 'get_order', {'order_id': 'O9999'}), ('kb', 'search_knowledge', {'query': '政策'})),
                          text_stream('订单未找到；已检索资料，部分资料受预算限制。')]
        events = [event async for event in agent_service.stream_agent(self.app.state.client, self.app.state.model,
            agent_service.new_conversation(), '查 O9999 和政策', policy=policy, search_knowledge=search)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            self.assertLessEqual(estimate_request(request['messages'], request['tools']), policy.input_limit)
        saved = next(json.loads(m['content']) for m in events[-1]['history'] if m.get('tool_call_id') == 'kb')
        self.assertEqual(saved['results'], original['results'])
        projected = next(json.loads(m['content']) for m in self.requests[-1]['messages'] if m.get('tool_call_id') == 'kb')
        self.assertFalse(projected['budget_limited'])
        self.assertEqual(projected['results'], original['results'])
        self.assertEqual(events[-1]['recovery']['stalled_batches'], 0)

    async def test_stop_checkpoint_can_commit_after_disconnect_without_reset_or_model(self):
        from checkpoint_store import resume_reason
        p = steps()[:1]
        self.responses = [update('start', p)] + [update(f'n{i}', p, f'仍在查询{i}') for i in range(3)]
        captured = {}
        async def save(state):
            captured['state'] = json.loads(json.dumps(state))
        def run(checkpoint=None):
            return agent_service.stream_agent(self.app.state.client, self.app.state.model, agent_service.new_conversation(),
                '查询 O1003', checkpoint=checkpoint, save_checkpoint=save)
        first = run()
        async for event in first:
            if event['event'] == 'tool_result' and event['result'].get('_recovery', {}).get('stopped'):
                break
        await first.aclose()
        checkpoint = captured['state']
        self.assertIsNone(checkpoint['answer'])
        self.assertEqual(resume_reason({'state': checkpoint, 'resume_count': 3}), '')
        events = [event async for event in run(checkpoint)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertTrue(events[-1]['recovery']['stopped'])
        self.assertEqual(len(self.requests), 4)


class MCPRecoveryTests(unittest.IsolatedAsyncioTestCase):
    tool_mode = 'mcp'
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session

    async def test_real_mcp_failures_stop_and_do_not_rediscover_or_repeat_remote_execution(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order']}))] + [
            tool_stream((f'bad{i}', 'get_order', {'order_id': 'O9999'})) for i in range(3)]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询 O9999'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['recovery']['stopped'])
        tools = [item for item in response.json()['execution']['tools'] if item['name'] == 'get_order']
        self.assertEqual(sum(item['attempts'] for item in tools), 1)
        self.assertTrue(all(item['source'] == 'mcp' for item in tools))
        self.assertEqual(len(self.requests), 4)
