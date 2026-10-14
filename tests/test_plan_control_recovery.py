from copy import deepcopy
import json
from pathlib import Path
import unittest

import agent_service
from agent_plan import AgentPlan
from agent_recovery import RecoveryTracker
from context_memory import ContextPolicy, estimate_request
import test_api_app as fixture
from test_mcp_graph import InlineCatalog
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


def control_plan(tool='discover_tools'):
    return [{'id': 'discover', 'title': '加载工具', 'tools': [tool], 'status': 'in_progress', 'depends_on': [], 'evidence': []},
            {'id': 'order', 'title': '查询订单', 'tools': ['get_order'], 'status': 'pending', 'depends_on': ['discover'], 'evidence': []}]


class PlanControlValidationTests(unittest.TestCase):
    def test_obsolete_rejected_control_batch_is_removed_only_from_request_view(self):
        def batch(prefix, names, payloads):
            calls = [{'id': f'{prefix}{i}', 'function': {'name': name, 'arguments': '{}'}} for i, name in enumerate(names)]
            return [{'role': 'assistant', 'tool_calls': calls}] + [
                {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(payload)}
                for call, payload in zip(calls, payloads)]
        rejected = {'error': 'separate calls', 'error_code': 'control_batch_conflict'}
        bad = batch('bad', ['load_skill', 'discover_tools'], [rejected, rejected])
        loaded = batch('loaded', ['load_skill'], [{'status': 'loaded', 'instructions': '完整指引', 'version': 'pinned'}])
        business = batch('business', ['get_order'], [{'order_id': 'O1003', 'status': '已签收'}])
        latest = batch('latest', ['load_skill', 'discover_tools'], [rejected, rejected])
        active = [{'role': 'user', 'content': '查订单'}] + bad + loaded + business + latest
        original = deepcopy(active)
        projected = agent_service.model_context(active)
        self.assertEqual([m.get('tool_call_id') for m in projected if m['role'] == 'tool'], ['loaded0', 'business0', 'latest0', 'latest1'])
        body = next(json.loads(m['content']) for m in projected if m.get('tool_call_id') == 'loaded0')
        self.assertEqual(body['instructions'], '完整指引')
        self.assertEqual(active, original)
        # An incomplete or partly executed batch cannot be discarded.
        self.assertEqual(agent_service.model_context(bad[:-1]), bad[:-1])
        mixed = batch('partial', ['get_order', 'discover_tools'], [{'order_id': 'O1003'}, rejected])
        view = agent_service.model_context(mixed + loaded)
        self.assertIn('partial0', [m.get('tool_call_id') for m in view])

    def test_control_names_are_rejected_with_targeted_repair_and_no_partial_plan(self):
        plan = AgentPlan(['get_order', 'search_knowledge', 'delegate_tasks', 'discover_tools', 'load_skill', 'update_plan'])
        for name in ('discover_tools', 'load_skill', 'update_plan'):
            with self.subTest(name=name):
                result = plan.update(json.dumps({'steps': control_plan(name)}))
                self.assertEqual(result.get('error_code'), 'invalid_plan_tools')
                self.assertEqual(result['invalid_tools'], [name])
                self.assertEqual(result['allowed_tools'], ['delegate_tasks', 'get_order', 'search_knowledge'])
                self.assertIn('depends_on', result['error'])
                self.assertIn('evidence', result['error'])
                self.assertIsNone(plan.snapshot())

    def test_specific_protocol_feedback_is_not_generic_business_failure(self):
        for code in ('control_batch_conflict', 'invalid_plan_tools'):
            with self.subTest(code=code):
                tracker = RecoveryTracker()
                tracker.observe([{'name': 'update_plan', 'call_id': 'p', 'result': {'error': 'bad', 'error_code': code}}], round_number=1)
                self.assertEqual(tracker.snapshot()['feedback']['category'], code)
                feedback = tracker.feedback_messages(None)[0]['content']
                self.assertIn('discover_tools', feedback)
                self.assertNotIn('先单独更新计划，再执行该步骤', feedback)

    def test_no_fact_stop_does_not_claim_query_results_exist(self):
        tracker = RecoveryTracker()
        for i in range(3):
            tracker.observe([{'name': 'update_plan', 'call_id': str(i), 'result': {'error': 'bad', 'error_code': 'invalid_plan_tools'}}], round_number=i + 1)
        self.assertTrue(tracker.stopped)
        reason = tracker.snapshot()['events'][-1]['reason']
        self.assertIn('计划', reason)
        self.assertIn('尚未', reason)
        self.assertNotIn('已有查询结果已保留', reason)


class PlanControlGraphTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    def stream(self, **kwargs):
        async def reader(query):
            return self.app.state.knowledge_base.search(query)
        async def save(state):
            self.snapshot = deepcopy(state)
        self.remote = InlineCatalog(reader)
        return agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
            '帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。',
            search_knowledge=reader, mcp_client=self.remote, save_checkpoint=save, **kwargs)

    async def test_plan_schema_lists_business_tools_before_they_are_discovered(self):
        self.responses = [text_stream('请说明具体原因。')]
        await self.collect(self.stream(skills_dir=Path(agent_service.__file__).parent / 'skills', delegation_enabled=True))
        tools = {t['function']['name']: t['function'] for t in self.requests[0]['tools']}
        self.assertNotIn('get_order', tools)
        allowed = tools['update_plan']['parameters']['properties']['steps']['items']['properties']['tools']['items'].get('enum')
        self.assertEqual(set(allowed or []), {'get_order', 'search_knowledge', 'delegate_tasks'})
        self.assertNotIn('load_skill', allowed)
        self.assertNotIn('discover_tools', allowed)

    async def test_each_model_request_receives_current_tool_contract_after_discovery(self):
        self.responses = [tool_stream(('d', 'discover_tools', {'names': ['get_order']})),
                          tool_stream(('o', 'get_order', {'order_id': 'O1003'})),
                          text_stream('O1003 已签收。')]
        await self.collect(self.stream())
        self.assertEqual(len(self.requests), 3)
        for index, request in enumerate(self.requests):
            contracts = [message for message in request['messages']
                         if message['role'] == 'system' and message['content'].startswith('[本轮工具状态]')]
            self.assertEqual(len(contracts), 1)
            state = json.loads(contracts[0]['content'].split('\n', 1)[1])
            self.assertEqual(set(state['available_tools']), {t['function']['name'] for t in request['tools']})
            self.assertEqual('get_order' in state['available_tools'], index > 0)

    async def collect(self, stream):
        return [event async for event in stream]

    async def test_screenshot_prefix_receives_repair_and_completes_queries(self):
        self.knowledge_dir.mkdir(exist_ok=True)
        (self.knowledge_dir / 'policy.md').write_text('# 退货退款政策\n\n质量问题可申请退货，需说明具体故障。', encoding='utf8')
        bad = control_plan()
        valid = [{'id': 'check', 'title': '核对订单和政策', 'tools': ['get_order', 'search_knowledge'], 'status': 'in_progress', 'depends_on': [], 'evidence': []}]
        done = deepcopy(valid)
        done[0].update(status='completed', evidence=['order', 'policy'])
        self.responses = [tool_stream(('p1', 'update_plan', {'steps': bad, 'note': ''}), ('d1', 'discover_tools', {'names': ['get_order', 'search_knowledge']})),
                          tool_stream(('p2', 'update_plan', {'steps': bad, 'note': ''})),
                          tool_stream(('d2', 'discover_tools', {'names': ['get_order', 'search_knowledge']})),
                          tool_stream(('p3', 'update_plan', {'steps': valid, 'note': '移除准备步骤，只核对业务事实。'})),
                          tool_stream(('order', 'get_order', {'order_id': 'O1003'}), ('policy', 'search_knowledge', {'query': '退货退款'})),
                          tool_stream(('p4', 'update_plan', {'steps': done, 'note': ''})), text_stream('订单已签收，请补充具体故障。')]
        events = await self.collect(self.stream(skills_dir=Path(agent_service.__file__).parent / 'skills'))
        results = [event['result'] for event in events if event['event'] == 'tool_result']
        self.assertEqual(results[0].get('error_code'), 'control_batch_conflict')
        self.assertEqual(results[2].get('error_code'), 'invalid_plan_tools')
        next_input = json.dumps(self.requests[2], ensure_ascii=False)
        self.assertIn('allowed_tools', next_input)
        self.assertIn('depends_on', next_input)
        self.assertEqual([name for name, _ in self.remote.calls], ['get_order', 'search_knowledge'])
        self.assertEqual(events[-1]['plan']['status'], 'completed')
        self.assertFalse(events[-1]['recovery']['stopped'])
        for request in self.requests:
            self.assertLessEqual(estimate_request(request['messages'], request['tools']), ContextPolicy().input_limit)
