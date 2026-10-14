"""计划 → 工具 → 更新计划 → 完成；只使用临时数据库与离线模型。"""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch

import agent_service
import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream
from tool_runtime import ToolPolicy


def steps():
    return [
        {'id': 'order', 'title': '核对订单', 'tools': ['get_order'], 'status': 'in_progress', 'depends_on': [], 'evidence': []},
        {'id': 'policy', 'title': '查询售后资料', 'tools': ['search_knowledge'], 'status': 'pending', 'depends_on': ['order'], 'evidence': []},
    ]


def update(call_id, plan, note=''):
    return tool_stream((call_id, 'update_plan', {'steps': deepcopy(plan), 'note': note}))


class AgentPlanApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart_app = fixture.ApiAppTests.restart_app
    seed_knowledge = fixture.ApiAppTests.seed_knowledge

    async def test_plan_executes_dependencies_and_finishes_after_more_than_four_decisions(self):
        sid = await self.new_session()
        self.seed_knowledge()
        plan = steps()
        self.responses = [update('plan-1', plan), tool_stream(('order-result', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order-result'])
        plan[1]['status'] = 'in_progress'
        self.responses.extend([update('plan-2', plan), tool_stream(('policy-result', 'search_knowledge', {'query': '退款'}))])
        plan[1].update(status='completed', evidence=['policy-result'])
        self.responses.extend([update('plan-3', plan), text_stream('订单已签收，已查阅示例退款政策。')])
        response = await self.client.post('/api/chat/stream', json={'session_id': sid, 'message': '先核对订单，再查政策'})
        events = fixture.parse_sse(response.text)
        self.assertEqual(events[-1][0], 'done', response.text)
        final = events[-1][1]
        self.assertEqual(final['plan']['status'], 'completed')
        self.assertEqual(final['plan']['revision'], 3)
        self.assertEqual(len([event for event in events if event[0] == 'plan']), 3)
        saved = (await self.client.get(f'/api/sessions/{sid}')).json()
        plan_results = [json.loads(m['content'])['plan'] for m in saved['messages'] if m['role'] == 'tool' and '"plan"' in m['content']]
        self.assertEqual(plan_results[-1], final['plan'])
        await self.restart_app()
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json(), saved)

    async def test_unfinished_plan_gets_one_correction_before_answer_is_accepted(self):
        sid = await self.new_session()
        plan = steps()[:1]
        self.responses = [update('start', plan), text_stream('已经完成。'),
                          tool_stream(('order-result', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order-result'])
        self.responses.extend([update('finish', plan), text_stream('查询确认订单已签收。')])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询订单'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['plan']['status'], 'completed')
        saved = (await self.client.get(f'/api/sessions/{sid}')).json()['messages']
        self.assertFalse(any(m.get('content') == '已经完成。' for m in saved))
        self.assertIn('上次答复提前结束', self.requests[2]['messages'][0]['content'])
        self.assertNotIn('上次答复提前结束', saved[0]['content'])

    async def test_repeated_premature_finish_keeps_original_history(self):
        sid = await self.new_session()
        before = (await self.client.get(f'/api/sessions/{sid}')).json()
        self.responses = [update('start', steps()), text_stream('已完成。'), text_stream('还是已完成。')]
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '复杂任务'})
        self.assertEqual(response.status_code, 502)
        self.assertEqual((await self.client.get(f'/api/sessions/{sid}')).json(), before)

    async def test_blocked_plan_reports_missing_input_and_can_be_saved(self):
        sid = await self.new_session()
        plan = steps()
        self.responses = [update('start', plan)]
        plan[0]['status'] = 'blocked'
        self.responses.extend([update('blocked', plan, '请提供订单号。'), text_stream('请先提供订单号，我再继续查询。')])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '核对后查售后'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['plan']['status'], 'blocked')

    async def test_missing_information_ends_turn_without_another_model_call_and_accepts_reply(self):
        sid = await self.new_session()
        plan = steps()
        self.responses = [update('start', plan), tool_stream(('order-result', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order-result'])
        plan[1]['status'] = 'blocked'
        self.responses.extend([update('wait', plan, '请补充具体退货原因。'), text_stream('收到补充信息。')])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询后申请退货'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['answer'], '请补充具体退货原因。')
        self.assertEqual(len(self.requests), 3)
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '耳机左侧没有声音'})
        self.assertEqual(response.status_code, 200, response.text)

    async def test_mixed_plan_batch_does_not_execute_business_tools(self):
        sid = await self.new_session()
        self.responses = [tool_stream(('plan', 'update_plan', {'steps': steps(), 'note': ''}),
                                      ('order', 'get_order', {'order_id': 'O1001'})), text_stream('需要先安排步骤。')]
        with patch.object(agent_service, 'get_order') as read:
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '复杂任务'})
        self.assertEqual(response.status_code, 200, response.text)
        read.assert_not_called()
        self.assertIsNone(response.json()['plan'])

    async def test_tool_for_later_step_is_rejected_without_consuming_search_budget(self):
        sid = await self.new_session()
        self.responses = [update('start', steps()), tool_stream(('later', 'search_knowledge', {'query': '退货'}))]
        plan = steps()
        plan[0]['status'] = 'blocked'
        self.responses.extend([update('blocked', plan, '缺少订单号。'), text_stream('请补充订单号。')])
        with patch.object(self.app.state.knowledge_base, 'search') as read:
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '复杂任务'})
        self.assertEqual(response.status_code, 200, response.text)
        read.assert_not_called()
        self.assertEqual(response.json()['knowledge']['calls'], 0)

    async def test_timed_out_plan_update_does_not_change_accepted_state(self):
        from agent_plan import AgentPlan
        sid = await self.new_session()
        self.app.state.tool_policy = ToolPolicy(timeout_seconds=.02)
        self.responses = [update('start', steps()), text_stream('安排步骤超时，请重试。')]
        original = AgentPlan.update
        def slow_update(instance, arguments):
            time.sleep(.05)
            return original(instance, arguments)
        with patch.object(AgentPlan, 'update', slow_update):
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '复杂任务'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()['plan'])

    async def test_plan_updates_do_not_consume_business_tool_budget(self):
        sid = await self.new_session()
        self.app.state.tool_policy = ToolPolicy(max_calls=1)
        plan = steps()[:1]
        self.responses = [update('start', plan), tool_stream(('order-result', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order-result'])
        self.responses.extend([update('finish', plan), text_stream('已确认签收。')])
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '安排并查询订单'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['plan']['status'], 'completed')
        tool_result = next(m for m in self.requests[2]['messages'] if m.get('tool_call_id') == 'order-result')
        self.assertEqual(json.loads(tool_result['content'])['_tool_call_id'], 'order-result')

    async def test_four_steps_fit_default_budget_but_keep_all_saved_plan_revisions(self):
        sid = await self.new_session()
        plan = [dict(steps()[0], id=f'step{i}', title=f'核对第{i + 1}笔订单状态',
                     status='in_progress' if i == 0 else 'pending',
                     depends_on=[f'step{i - 1}'] if i else []) for i in range(4)]
        self.responses = [update('plan-0', plan)]
        for index in range(4):
            self.responses.append(tool_stream((f'order-{index}', 'get_order', {'order_id': 'O1003'})))
            plan[index].update(status='completed', evidence=[f'order-{index}'])
            if index < 3:
                plan[index + 1]['status'] = 'in_progress'
            self.responses.append(update(f'plan-{index + 1}', plan))
        self.responses.append(text_stream('四项查询已经完成。'))
        response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '请分四步依次核对四笔订单'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.requests), 10)
        self.assertEqual(response.json()['plan']['status'], 'completed')
        saved = (await self.client.get(f'/api/sessions/{sid}')).json()['messages']
        self.assertEqual(sum(m.get('tool_call_id', '').startswith('plan-') for m in saved), 5)
        last = self.requests[-1]['messages']
        self.assertEqual(sum(m.get('tool_call_id', '').startswith('plan-') for m in last), 1)
        self.assertEqual(sum(m.get('tool_call_id', '').startswith('order-') for m in last), 4)
