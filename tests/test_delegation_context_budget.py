"""大批子任务证据回传主管：预算、原文保留与旧进度恢复。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import test_api_app as api_fixture
import test_delegation_api as fixture
from context_memory import estimate_request
from delegation_context import compact_delegated_knowledge


class KnowledgeProjectionTests(unittest.TestCase):
    def test_versions_failures_and_incomplete_summaries_remain_distinct(self):
        hit = {'source': 'kb:doc:0', 'version': 'version-one', 'title': '政策', 'section': '退款', 'text': '退款需核实。'}
        evidence = [{'call_id': 'q1', 'name': 'search_knowledge', 'status': 'ok',
                     'result': {'results': [hit, {**hit, 'version': 'version-two'}], 'has_more': True, 'budget_limited': True}},
                    {'call_id': 'q2', 'name': 'search_knowledge', 'status': 'error',
                     'result': {'error': '资料读取失败', 'error_code': 'unavailable'}},
                    {'call_id': 'q3', 'name': 'search_knowledge', 'status': 'ok',
                     'arguments': {'query': '某项缺失政策'},
                     'result': {'query': '某项缺失政策', 'results': [], 'document_count': 1,
                                'has_more': False, 'budget_limited': False}}]
        raw = {'status': 'partial', 'delegations': [{'task_id': 'child', 'agent': 'knowledge', 'status': 'needs_input',
            'summary': '需要补充适用商品信息。', 'cited_evidence_ids': ['q1'], 'evidence': evidence}]}
        messages = [{'role': 'assistant', 'content': None, 'tool_calls': [fixture.tool_message('delegate', 'delegate_tasks', {})]},
                    {'role': 'tool', 'tool_call_id': 'delegate', 'content': json.dumps(raw, ensure_ascii=False)}]
        original = deepcopy(messages)
        projected = deepcopy(messages)
        compact_delegated_knowledge(projected)
        result = json.loads(projected[-1]['content'])
        self.assertEqual(len(result['knowledge_sources']), 2)
        self.assertEqual({item['version'] for item in result['knowledge_sources'].values()}, {'version-one', 'version-two'})
        task = result['delegations'][0]
        self.assertEqual(task['summary'], raw['delegations'][0]['summary'])
        self.assertEqual(task['evidence'][1]['result'], evidence[1]['result'])
        self.assertTrue(task['evidence'][0]['result']['has_more'])
        self.assertTrue(task['evidence'][0]['result']['budget_limited'])
        self.assertTrue(task['evidence'][0]['cited'])
        self.assertFalse(task['evidence'][1]['cited'])
        self.assertEqual(task['evidence'][2]['arguments']['query'], '某项缺失政策')
        self.assertEqual(task['evidence'][2]['result']['source_refs'], [])
        self.assertNotIn('error', task['evidence'][2]['result'])
        self.assertEqual(messages, original)


class DelegationContextBudgetTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.DelegationApiTests.asyncSetUp
    asyncTearDown = api_fixture.ApiAppTests.asyncTearDown
    install_client = fixture.DelegationApiTests.install_client
    new_session = api_fixture.ApiAppTests.new_session

    def prepare_case(self):
        self.app.state.result_verification = True
        self.chunks = [{
            'source': f'kb:policy123456:{index * 200}', 'document_id': 'policy123456',
            'version': 'a' * 64, 'title': '示例退货退款政策说明', 'section': title,
            'text': text * 5, 'start': index * 200, 'end': index * 200 + 180, 'score': index + .3,
        } for index, (title, text) in enumerate([
            ('退货期限', '签收七天内可以提出退货申请，具体情况需要人工核实。'),
            ('退款流程', '仓库验收后原路退款，到账时间取决于支付机构处理。'),
            ('质量问题', '商品质量问题需提供故障描述和必要凭证供人工核实。'),
            ('运费说明', '退货运费的承担方式需要结合售后政策和审核结果确认。'),
        ])]
        self.results = {query: {'query': query, 'document_count': 1, 'matched_chunks': len(indices),
            'results': [deepcopy(self.chunks[i]) for i in indices], 'has_more': False,
            'budget_limited': False, 'search_mode': 'hybrid', 'semantic_status': 'ready'}
            for query, indices in [('退货', [0, 1, 2]), ('退款', [1, 0, 2]), ('质量问题', [2, 3])]}
        plan = [{'id': 'policy', 'title': '委派查询退货退款政策', 'tools': ['delegate_tasks'],
                 'status': 'in_progress', 'depends_on': [], 'evidence': []}]
        self.actor = [fixture.actor_call('start', 'update_plan', {'steps': plan, 'note': ''}),
                      fixture.delegate('knowledge'),
                      fixture.actor_call('finish', 'update_plan', {'steps': [{**plan[0], 'status': 'completed', 'evidence': ['delegation']}], 'note': ''}),
                      {'text': '已查到退货期限和退款流程，具体申请需要人工审核。'}]
        self.children['knowledge'] = [fixture.child_response(calls=[
            fixture.tool_message('q' + str(i), 'search_knowledge', {'query': query})
            for i, query in enumerate(self.results)]), fixture.child_response(
                summary='已查到退货期限、退款流程及质量问题处理说明，具体申请需要人工核实。' * 7,
                ids=['specialist_1:r1:q' + str(i) for i in range(3)])]

    async def seed_history(self, sid):
        history = self.app.state.store.load_session(sid)['history'] + [
            {'role': 'user', 'content': 'LangGraph 是什么？'},
            {'role': 'assistant', 'content': '它用于组织 Agent 的执行流程。'}]
        self.app.state.store.replace_history(sid, history)
        self.app.state.sessions.clear()

    def assert_complete_evidence(self, sid, done):
        self.assertEqual(done['delegation']['requests'], 2)
        self.assertEqual(done['verification']['status'], 'passed')
        self.assertEqual(len(done['sources']), 4)
        history = self.app.state.store.load_session(sid)['history']
        raw = json.loads(next(m['content'] for m in history if m.get('tool_call_id') == 'delegation'))
        evidence = raw['delegations'][0]['evidence']
        self.assertEqual(sum(len(item['result']['results']) for item in evidence), 8)
        self.assertEqual([item['result']['results'] for item in evidence],
                         [result['results'] for result in self.results.values()])
        actors = [request for request in self.requests if request.get('stream')]
        handoff_request = next(request for request in actors if any(m.get('tool_call_id') == 'delegation' for m in request['messages']))
        projected = json.loads(next(m['content'] for m in handoff_request['messages'] if m.get('tool_call_id') == 'delegation'))
        self.assertEqual(len(projected['knowledge_sources']), 4)
        for chunk in self.chunks:
            self.assertTrue(any(source['source'] == chunk['source'] and source['text'] == chunk['text']
                                and source['version'] == chunk['version'] for source in projected['knowledge_sources'].values()))
        review = next(request for request in self.requests if request.get('response_format'))
        review_payload = json.loads(review['messages'][-1]['content'])
        review_raw = json.loads(next(m['content'] for m in review_payload['messages'] if m.get('tool_call_id') == 'delegation'))
        self.assertEqual(review_raw, raw, '校验器仍核对完整原始工具证据')
        for request in self.requests:
            self.assertLessEqual(estimate_request(request['messages'], request.get('tools')),
                                 self.app.state.context_policy.input_limit)
        self.assertIsNone(self.app.state.checkpoints.load(sid))
        run = self.app.state.traces.list_runs(sid)['runs'][0]
        trace = self.app.state.traces.load(sid, run['run_id'])
        budgets = [e['data'] for e in trace['events'] if e['kind'] == 'context_budget']
        self.assertEqual({item['role'] for item in budgets}, {'actor', 'specialist', 'verification'})
        sequences = [item['sequence'] for item in budgets]
        self.assertEqual(sequences, list(range(1, len(sequences) + 1)))
        self.assertTrue(any(item['saved_tokens'] > 0 for item in budgets if item['role'] == 'actor'))
        self.assertEqual(len(budgets), len(self.requests))
        for item in budgets:
            self.assertEqual(sum(item['components'].values()), item['estimated_input_tokens'])
            self.assertEqual(item['status'], 'prepared')

    async def test_three_child_searches_fit_parent_budget_and_keep_full_history(self):
        self.prepare_case()
        sid = await self.new_session()
        await self.seed_history(sid)
        with patch.object(self.app.state.knowledge_base, 'search', side_effect=lambda query: deepcopy(self.results[query])):
            response = await self.client.post('/api/chat', json={'session_id': sid, 'message': '帮我使用子agent的方式查询退货、退款政策'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_complete_evidence(sid, response.json())

    async def test_resume_raw_oversize_handoff_does_not_repeat_specialist(self):
        self.prepare_case()
        sid = await self.new_session()
        await self.seed_history(sid)
        original_save = self.app.state.checkpoints.save

        def stop_after_handoff(session_id, run_id, state):
            original_save(session_id, run_id, state)
            if state['active'][-1].get('tool_call_id') == 'delegation':
                raise ValueError('测试：交接保存后中断')

        with patch.object(self.app.state.knowledge_base, 'search', side_effect=lambda query: deepcopy(self.results[query])), \
                patch.object(self.app.state.checkpoints, 'save', side_effect=stop_after_handoff):
            failed = await self.client.post('/api/chat', json={'session_id': sid, 'message': '查询退货退款政策'})
        self.assertEqual(failed.status_code, 502)
        cp = self.app.state.checkpoints.load(sid)
        self.assertEqual(cp['state']['delegation']['requests'], 2)
        self.assertIsNone(cp['state']['pending_delegation'])
        await api_fixture.ApiAppTests.restart_app(self)
        await self.install_client()
        self.app.state.result_verification = True
        with patch.object(self.app.state.knowledge_base, 'search', side_effect=AssertionError('不得重做已保存的子任务检索')):
            response = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_complete_evidence(sid, response.json())
