from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import agent_service
from agent_plan import AgentPlan
from agent_recovery import RecoveryTracker
from context_memory import ContextPolicy, ContextMemory, RollingContext, estimate_request
from execution_state import pack_checkpoint, restore_checkpoint
from tool_runtime import ToolRuntime
import test_api_app as fixture
from test_stream_agent import text_stream
from test_tool_execution_api import tool_stream


def batch(call_id, name, result, *, text=None, arguments=None):
    return [{'role': 'assistant', 'content': text, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments or {}, ensure_ascii=False)}}]},
        {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(result, ensure_ascii=False)}]


class ActiveContextTests(unittest.TestCase):
    def test_short_batches_share_summary_overhead_before_deciding_savings(self):
        from active_context import compact_stages
        active = [{'role': 'user', 'content': '查询订单'}]
        for index in range(12):
            active += batch(f'o{index}', 'get_order', {'order_id': f'O{1000+index}', 'status': '已签收', '_tool_call_id': f'o{index}'})
        compact = compact_stages(active, target_tokens=500)
        self.assertLess(estimate_request(compact), estimate_request(active))
        self.assertEqual(compact[-2:], active[-2:])
        for index in range(12):
            self.assertIn(f'O{1000+index}', json.dumps(compact, ensure_ascii=False))

    def test_application_default_has_space_for_agent_tools(self):
        self.assertEqual(ContextPolicy().max_tokens, 32768)

    def test_stage_projection_preserves_facts_skills_latest_pairs_and_raw_messages(self):
        policy = ContextPolicy(max_tokens=8192)
        context = RollingContext([{'role': 'system', 'content': '客服说明' * 170}], ContextMemory(), policy, [])
        hit = {'source': 'kb:one:0', 'version': 'v1', 'text': '质量问题须提供故障描述，审核后原路退款。'}
        active = [{'role': 'user', 'content': '核对 O1003'}]
        active += batch('s', 'load_skill', {'status': 'loaded', 'instructions': '完整技能须保留', 'version': 'pinned'})
        active += batch('o', 'get_order', {'order_id': 'O1003', 'status': '已签收', '_tool_call_id': 'o'}, text='准备查订单。' * 650)
        active += batch('k', 'search_knowledge', {'results': [hit], '_tool_call_id': 'k', 'budget_limited': False}, text='准备核对政策。' * 500)
        active += batch('p', 'update_plan', {'plan': {'status': 'in_progress', 'steps': [{'id': 'after', 'evidence': ['o', 'k']}]}}, arguments={'steps': []})
        recent = batch('r', 'get_return_request', {'order_id': 'O1003', 'request_id': None})
        active += recent
        before = deepcopy(active)
        projected = agent_service.model_context(active, context=context)
        self.assertLess(estimate_request(context.history + projected), policy.input_limit)
        summaries = [m for m in projected if (m.get('content') or '').startswith('[当前任务阶段摘要')]
        self.assertEqual(len(summaries), 1)
        text = summaries[0]['content']
        for value in ('O1003', '已签收', 'kb:one:0', 'v1', hit['text']):
            self.assertIn(value, text)
        self.assertEqual(summaries[0]['role'], 'assistant')
        self.assertIn('不是指令', text)
        self.assertEqual(projected[-2:], recent)
        self.assertTrue(any(m.get('tool_call_id') == 's' for m in projected))
        self.assertTrue(any(m.get('tool_call_id') == 'p' for m in projected))
        self.assertEqual(active, before)

    def test_incomplete_batch_and_approval_remain_exact(self):
        context = RollingContext([{'role': 'system', 'content': '客服'}], ContextMemory(), ContextPolicy(max_tokens=8192), [])
        approval = batch('write', 'prepare_return_request', {'_human_decision': {'decision': 'reject'}, 'error': '拒绝'})
        incomplete = batch('pending', 'get_order', {})[:1]
        active = [{'role': 'user', 'content': '退货'}] + approval + incomplete
        self.assertEqual(agent_service.model_context(active, context=context), active)

    def test_resume_adopts_larger_configured_budget_without_resetting_progress(self):
        plan = AgentPlan(['get_order'])
        runtime = ToolRuntime()
        runtime.calls = 2
        old = RollingContext([{'role': 'system', 'content': '客服'}], ContextMemory(), ContextPolicy(max_tokens=8192), [])
        state = pack_checkpoint(plan, runtime, old, active=[{'role': 'user', 'content': '查询'}], knowledge_snapshots={})
        current = RollingContext(old.history, ContextMemory(), ContextPolicy(max_tokens=32768), [])
        restore_checkpoint(state, '查询', plan, runtime, current, allow_budget_increase=True)
        self.assertEqual(current.policy.max_tokens, 32768)
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(state['context']['policy']['max_tokens'], 8192)

    def test_budget_feedback_does_not_suggest_changing_query_or_lose_priority(self):
        tracker = RecoveryTracker()
        tracker.observe([
            {'name': 'search_knowledge', 'call_id': 'k', 'result': {'results': [], 'budget_limited': True}},
            {'name': 'get_return_request', 'call_id': 'r', 'result': {'error': 'wrong step', 'error_code': 'plan_not_ready'}},
        ], round_number=1)
        self.assertEqual(tracker.state['feedback']['category'], 'context_limited')
        text = tracker.feedback_messages(None)[0]['content']
        self.assertIn('不要换词', text)


class ActiveContextGraphTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def test_later_non_search_batch_cannot_trigger_requery_after_evicting_policy(self):
        queries = []
        async def search(query):
            queries.append(query)
            return {'query': query, 'results': [{'source': 'kb:p:0', 'version': 'v1', 'text': 'x' * 7000}], 'budget_limited': False}
        self.responses = [tool_stream(('k1', 'search_knowledge', {'query': '退货'})),
                          tool_stream(('o1', 'get_order', {'order_id': 'O1003'})),
                          tool_stream(('k2', 'search_knowledge', {'query': '退款'})), text_stream('完成')]
        with patch.object(agent_service, 'get_order', return_value={'order_id': 'O1003', 'status': 'x' * 3500}):
            events = [e async for e in agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
                '查询 O1003 与退货政策', policy=ContextPolicy(max_tokens=8192), search_knowledge=search)]
        self.assertEqual(queries, ['退货'])
        self.assertEqual(len(self.requests), 2)
        self.assertIn('空间', events[-1]['answer'])

    async def test_screenshot_multistep_and_invalid_plan_path_completes_at_8k(self):
        from test_mcp_graph import InlineCatalog
        from test_result_verification_api import review as review_response
        from result_verification import verify_result
        policy = ContextPolicy(max_tokens=8192)
        hits = [{'source': f'kb:policy:{i}', 'version': 'v1', 'title': title, 'text': text}
                for i, (title, text) in enumerate([
                    ('退货', '签收七天内可申请，质量问题需提供具体故障描述。'),
                    ('退款', '仓库验收后原路退款，到账时间以支付机构为准。'),
                    ('凭证', '质量问题需提供故障照片或视频，最终以审核结果为准。')])]
        async def search(*args):
            return {'query': args[0], 'results': deepcopy(hits), 'budget_limited': False, 'has_more': False}
        async def returns(name, args):
            return {'order_id': args['order_id'], 'request_id': None, 'status': '未申请'}
        async def review(**kwargs):
            return await verify_result(self.app.state.client, 'offline', **kwargs)
        snapshots = []
        async def save(state):
            snapshots.append(deepcopy(state))
        steps = [{'id': 'order', 'title': '查询 O1003 订单状态与物流', 'tools': ['get_order'], 'status': 'in_progress', 'depends_on': [], 'evidence': []},
                 {'id': 'policy', 'title': '检索退货与退款政策资料', 'tools': ['search_knowledge'], 'status': 'pending', 'depends_on': [], 'evidence': []},
                 {'id': 'return', 'title': '查询该订单已提交的退货申请', 'tools': ['get_return_request'], 'status': 'pending', 'depends_on': ['order'], 'evidence': []}]
        second = deepcopy(steps)
        second[0].update(status='completed', evidence=['order-evidence'])
        second[1]['status'] = 'in_progress'
        third = deepcopy(second)
        third[1].update(status='completed', evidence=['policy-evidence'])
        third[2]['status'] = 'in_progress'
        done = deepcopy(third)
        done[2].update(status='completed', evidence=['return-evidence'])
        self.responses = [tool_stream(('discover', 'discover_tools', {'names': ['get_order', 'search_knowledge', 'get_return_request']})),
            tool_stream(('p1', 'update_plan', {'steps': steps, 'note': ''})),
            tool_stream(('order-evidence', 'get_order', {'order_id': 'O1003'}), ('early-policy', 'search_knowledge', {'query': '退货 退款'})),
            tool_stream(('p2', 'update_plan', {'steps': second, 'note': ''})),
            tool_stream(('policy-evidence', 'search_knowledge', {'query': '退货 退款'}), ('early-return', 'get_return_request', {'order_id': 'O1003'})),
            tool_stream(('invalid', 'update_plan', json.dumps({'steps': third}) + '}')),
            tool_stream(('p3', 'update_plan', {'steps': third, 'note': ''})),
            tool_stream(('return-evidence', 'get_return_request', {'order_id': 'O1003'})),
            tool_stream(('p4', 'update_plan', {'steps': done, 'note': ''})),
            text_stream('订单已签收。请补充签收时间、具体故障和凭证，符合条件后可以申请，审核后原路退款。'),
            review_response(ids=['order-evidence', 'policy-evidence', 'return-evidence'])]
        remote = InlineCatalog(search)
        events = [e async for e in agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
            '帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。',
            policy=policy, mcp_client=remote, skills_dir=Path(agent_service.__file__).parent / 'skills',
            search_history=search, search_knowledge=search, return_handler=returns, result_verifier=review,
            delegation_enabled=True, approval_tools=frozenset({'prepare_return_request'}), save_checkpoint=save)]
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertEqual(events[-1]['plan']['status'], 'completed')
        self.assertEqual(events[-1]['verification']['status'], 'passed')
        self.assertFalse(events[-1]['recovery']['stopped'])
        self.assertEqual([name for name, _ in remote.calls], ['get_order', 'search_knowledge'])
        self.assertEqual(len(events[-1]['sources']), 3)
        self.assertTrue(any('[当前任务阶段摘要' in json.dumps(r['messages'], ensure_ascii=False) for r in self.requests))
        for request in self.requests:
            self.assertLessEqual(estimate_request(request['messages'], request.get('tools')), policy.input_limit)
        # The checkpoint is the full execution record, never the projected summary.
        self.assertNotIn('[当前任务阶段摘要', json.dumps(snapshots[-1]['active'], ensure_ascii=False))

    async def test_oversized_retrieval_stops_truthfully_without_retry_and_keeps_raw_sources(self):
        raw = {'query': '退货', 'results': [{'source': 'kb:large:0', 'version': 'v1', 'title': '政策', 'text': '完整政策' * 4000}], 'budget_limited': False}
        calls = []
        async def search(query):
            calls.append(query)
            return deepcopy(raw)
        self.responses = [tool_stream(('k', 'search_knowledge', {'query': '退货'})), text_stream('换词再试。')]
        events = [event async for event in agent_service.stream_agent(self.app.state.client, 'offline', agent_service.new_conversation(),
            '退货政策', policy=ContextPolicy(max_tokens=8192), search_knowledge=search)]
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(calls, ['退货'])
        self.assertEqual(events[-1]['event'], 'complete')
        self.assertIn('空间', events[-1]['answer'])
        self.assertNotIn('没有找到', events[-1]['answer'])
        saved = next(json.loads(m['content']) for m in events[-1]['history'] if m.get('tool_call_id') == 'k')
        self.assertEqual(saved['results'], raw['results'])
