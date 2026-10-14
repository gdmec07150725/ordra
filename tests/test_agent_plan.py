"""计划状态机不依赖模型、数据库或网页。"""
from copy import deepcopy
import json
import unittest

from agent_plan import AgentPlan
from agent_service import plan_context
from test_agent_plan_api import steps


class AgentPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = AgentPlan(['get_order', 'search_knowledge', 'prepare_return_request'])
        self.initial = steps()
        self.assertIn('plan', self.update(self.initial))

    def update(self, value, note=''):
        return self.plan.update(json.dumps({'steps': value, 'note': note}))

    def observe(self, call_id='order-result', name='get_order', result=None):
        self.plan.observe({'call_id': call_id, 'name': name, 'result': result or {'ok': True}, 'attempts': 1})

    def next_step(self):
        self.observe()
        changed = deepcopy(self.initial)
        changed[0].update(status='completed', evidence=['order-result'])
        changed[1]['status'] = 'in_progress'
        return changed

    def test_only_current_step_tools_can_execute(self):
        self.assertTrue(self.plan.can_call('get_order'))
        self.assertFalse(self.plan.can_call('search_knowledge'))
        self.assertFalse(self.plan.can_call('create_return_request'))

    def test_blocked_step_evidence_error_identifies_field_without_mutating_state(self):
        self.observe()
        before = self.plan.snapshot()
        proposed = deepcopy(self.initial)
        proposed[0].update(status='blocked', evidence=['order-result'])
        result = self.update(proposed, '请提供退货原因。')
        self.assertEqual(result['error_code'], 'invalid_plan_update')
        self.assertEqual(result['current_plan'], before)
        self.assertEqual(self.plan.snapshot(), before)
        self.assertIn('steps[0].evidence', result['repair_hint'])
        self.assertIn('[]', result['repair_hint'])
        self.assertEqual(result['available_evidence'], {'order': {'order-result': 'get_order'}})
        proposed[0]['evidence'] = []
        self.assertEqual(self.update(proposed, '请提供退货原因。')['plan']['status'], 'blocked')


    def test_empty_tool_step_error_identifies_field_and_explains_how_to_ask_user(self):
        invalid = deepcopy(self.initial)
        invalid[1]['tools'] = []
        error = self.update(invalid)['error']
        self.assertIn('steps.1.tools', error)
        self.assertIn('blocked', error)

    def test_changing_execution_order_requires_note(self):
        independent = deepcopy(self.initial)
        independent[1]['depends_on'] = []
        self.update(independent, '两项查询互不依赖。')
        independent[0]['status'] = 'pending'
        independent[1]['status'] = 'in_progress'
        before = self.plan.snapshot()
        self.assertIn('error', self.update(independent))
        self.assertEqual(self.plan.snapshot(), before)
        self.assertIn('plan', self.update(independent, '先补充政策依据。'))

    def test_successful_evidence_allows_dependency_to_start(self):
        result = self.update(self.next_step())
        self.assertEqual(result['plan']['steps'][0]['status'], 'completed')
        self.assertTrue(self.plan.can_call('search_knowledge'))
        self.assertFalse(self.plan.can_call('get_order'))

    def test_missing_failed_or_other_step_evidence_cannot_complete(self):
        for evidence in ['invented', 'failed', 'unrelated']:
            with self.subTest(evidence=evidence):
                self.observe('failed', result={'error': '未找到订单'})
                self.observe('unrelated', name='search_knowledge')
                changed = deepcopy(self.initial)
                changed[0].update(status='completed', evidence=[evidence])
                changed[1]['status'] = 'in_progress'
                before = self.plan.snapshot()
                self.assertIn('error', self.update(changed))
                self.assertEqual(self.plan.snapshot(), before)

    def test_block_requires_reason_and_is_a_terminal_pause(self):
        changed = deepcopy(self.initial)
        changed[0]['status'] = 'blocked'
        self.assertIn('error', self.update(changed))
        result = self.update(changed, '请补充正确订单号。')
        self.assertEqual(result['plan']['status'], 'blocked')
        self.assertFalse(self.plan.can_call('get_order'))

    def test_can_complete_current_and_block_next_for_missing_input_in_one_update(self):
        changed = self.next_step()
        changed[1]['status'] = 'blocked'
        result = self.update(changed, '请补充具体退货原因。')
        self.assertIn('plan', result)
        self.assertEqual(result['plan']['status'], 'blocked')
        self.assertEqual(result['plan']['steps'][0]['status'], 'completed')

    def test_terminal_block_remains_available_after_plan_update_budget(self):
        self.plan.revision = 6
        changed = deepcopy(self.initial)
        changed[0]['status'] = 'blocked'
        self.assertIn('plan', self.update(changed, '请提供订单号。'))

    def test_resuming_blocked_step_requires_note(self):
        changed = deepcopy(self.initial)
        changed[0]['status'] = 'blocked'
        self.update(changed, '缺少订单号。')
        changed[0]['status'] = 'in_progress'
        self.assertIn('error', self.update(changed))
        self.assertIn('plan', self.update(changed, '已经从本轮结果确认订单号。'))

    def test_cannot_drop_steps_or_rewrite_completed_work(self):
        self.assertIn('error', self.update(self.initial[:1], '删掉后续步骤'))
        changed = self.next_step()
        self.assertIn('plan', self.update(changed))
        for key, value in [('title', '编造新步骤'), ('evidence', ['another']), ('status', 'pending')]:
            attempt = deepcopy(changed)
            attempt[0][key] = value
            self.assertIn('error', self.update(attempt, '修改已完成步骤'))

    def test_rejected_plan_returns_authoritative_ids_and_successful_evidence_for_repair(self):
        self.observe('order-result')
        self.observe('failed', result={'error': '失败'})
        changed = self.next_step()
        changed[1].update(id='answer', status='blocked')
        before = self.plan.snapshot()
        result = self.update(changed, '请补充退货原因。')
        self.assertEqual(result.get('error_code'), 'invalid_plan_update')
        self.assertEqual(result.get('current_plan'), before)
        self.assertEqual(result.get('available_evidence'), {'order': {'order-result': 'get_order'}})
        self.assertIn('policy', result['error'])
        self.assertEqual(self.plan.snapshot(), before)
        repaired = deepcopy(result['current_plan']['steps'])
        repaired[0].update(status='completed', evidence=['order-result'])
        repaired[1]['status'] = 'blocked'
        self.assertIn('plan', self.update(repaired, '请补充退货原因。'))

    def test_pending_summary_cannot_borrow_evidence_but_receives_actionable_state(self):
        changed = self.next_step()
        self.update(changed)
        self.observe('policy-result', 'search_knowledge')
        extra = dict(id='summary', title='汇总', tools=['search_knowledge'], status='pending', depends_on=['policy'], evidence=[])
        changed.append(extra)
        self.update(changed, '增加一步。')
        proposed = deepcopy(changed)
        proposed[1].update(status='completed', evidence=['policy-result'])
        proposed[2].update(status='completed', evidence=['policy-result'])
        result = self.update(proposed)
        self.assertEqual(result.get('error_code'), 'invalid_plan_update')
        self.assertEqual(result['current_plan']['steps'][2]['status'], 'pending')
        self.assertEqual(result['available_evidence']['policy'], {'policy-result': 'search_knowledge'})
        self.assertNotIn('summary', result['available_evidence'])

    def test_revise_pending_steps_requires_note_and_preserves_completed_work(self):
        changed = self.next_step()
        self.assertIn('plan', self.update(changed))
        revised = deepcopy(changed)
        revised[1]['title'] = '重新查询缺失的售后依据'
        self.assertIn('error', self.update(revised))
        result = self.update(revised, '发现尚未获得退货条件，需要补充检索。')
        self.assertEqual(result['plan']['steps'][0], changed[0])
        self.assertEqual(result['plan']['revision'], 3)

    def test_revised_step_cannot_reuse_its_old_evidence(self):
        self.observe()
        changed = deepcopy(self.initial)
        changed[0]['title'] = '核对另一笔订单'
        self.assertIn('plan', self.update(changed, '需要核对另一笔订单。'))
        changed[0].update(status='completed', evidence=['order-result'])
        changed[1]['status'] = 'in_progress'
        self.assertIn('error', self.update(changed))

    def test_invalid_dependencies_statuses_tools_and_payload_leave_state_unchanged(self):
        attempts = []
        for mutate in [lambda p: p[0].update(depends_on=['policy']),
                       lambda p: p[1].update(id='order'),
                       lambda p: p[1].update(status='in_progress'),
                       lambda p: p[0].update(tools=['create_return_request']),
                       lambda p: p[0].update(evidence=['fake']),
                       lambda p: p[0].update(title='\x00')]:
            attempt = deepcopy(self.initial)
            mutate(attempt)
            attempts.append(attempt)
        before = self.plan.snapshot()
        for attempt in attempts:
            self.assertIn('error', self.update(attempt))
            self.assertEqual(self.plan.snapshot(), before)
        for payload in ('{}', 'not json', '{"steps":[],"session_id":"other"}'):
            self.assertIn('error', self.plan.update(payload))

    def test_completed_plan_is_immutable_and_snapshots_are_detached(self):
        changed = self.next_step()
        self.update(changed)
        self.observe('policy-result', 'search_knowledge')
        changed[1].update(status='completed', evidence=['policy-result'])
        result = self.update(changed)
        self.assertEqual(result['plan']['status'], 'completed')
        result['plan']['steps'].clear()
        self.assertEqual(len(self.plan.steps), 2)
        self.assertIn('error', self.update(changed, '继续修改'))

    def test_completion_requires_every_declared_tool(self):
        plan = AgentPlan(['get_order', 'search_knowledge'])
        value = [dict(self.initial[0], tools=['get_order', 'search_knowledge'])]
        plan.update(json.dumps({'steps': value}))
        plan.observe({'call_id': 'one', 'name': 'get_order', 'attempts': 1, 'result': {}})
        value[0].update(status='completed', evidence=['one'])
        self.assertIn('error', plan.update(json.dumps({'steps': value})))

    def test_context_keeps_latest_plan_and_latest_rejected_mixed_batch_as_complete_pairs(self):
        def batch(prefix, names, results):
            calls = [{'id': prefix + str(i), 'function': {'name': name, 'arguments': '{}'}} for i, name in enumerate(names)]
            return [{'role': 'assistant', 'tool_calls': calls}] + [
                {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}
                for call, result in zip(calls, results)]
        question = [{'role': 'user', 'content': '查询订单'}]
        valid = batch('valid', ['update_plan'], [{'plan': self.plan.snapshot()}])
        business = batch('business', ['get_order'], [{'status': '已签收'}])
        rejected = {'error': '请单独更新计划', 'error_code': 'plan_not_ready'}
        first = batch('mixed1', ['update_plan', 'get_order'], [rejected, rejected])
        latest = batch('mixed2', ['get_order', 'update_plan'], [rejected, rejected])
        active = question + valid + business + first + latest
        original = deepcopy(active)
        self.assertEqual(plan_context(active), question + valid + business + latest)
        self.assertEqual(active, original)

    def test_repair_snapshot_replaces_duplicate_plan_only_in_request_view(self):
        from agent_service import model_context
        def batch(call_id, arguments, payload):
            return [{'role': 'assistant', 'tool_calls': [{'id': call_id, 'function': {'name': 'update_plan', 'arguments': arguments}}]},
                    {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(payload)}]
        snapshot = self.plan.snapshot()
        original = batch('accepted', json.dumps({'steps': self.initial}), {'plan': snapshot}) + batch(
            'rejected', '{bad json', {'error': 'Invalid JSON', 'error_code': 'invalid_plan_update', 'current_plan': snapshot, 'available_evidence': {}})
        saved = deepcopy(original)
        projected = model_context(original)
        self.assertEqual([m.get('tool_call_id') for m in projected if m['role'] == 'tool'], ['rejected'])
        self.assertEqual(json.loads(projected[-1]['content'])['current_plan'], snapshot)
        self.assertEqual(projected[0]['tool_calls'][0]['function']['arguments'], '{}')
        self.assertEqual(original, saved)
