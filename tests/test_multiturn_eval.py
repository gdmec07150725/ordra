"""Conversation evaluation uses real temporary API/storage and checks actual evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import agent_eval


class MultiturnEvalTests(unittest.TestCase):
    def runner(self):
        self.assertTrue(hasattr(agent_eval, 'evaluate_multiturn'), 'same-session evaluation runner is missing')
        return agent_eval.evaluate_multiturn

    def scorer(self):
        self.assertIsNotNone(importlib.util.find_spec('eval_trajectory'), 'trajectory scoring is missing')
        return importlib.import_module('eval_trajectory')

    def test_followup_and_correction_commit_each_turn_in_same_session(self):
        run = self.runner()
        from eval_multiturn_scenarios import MULTITURN_CASES
        cases = [c for c in MULTITURN_CASES if c['id'] in ('mt_followup', 'mt_correction')]
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: run(cases))
        self.assertEqual(report['passed'], 2, json.dumps(report, ensure_ascii=False))
        for result in report['results']:
            turns = result['turns']
            self.assertEqual(len(turns), 3)
            self.assertEqual([t['committed_user_turns'] for t in turns], [1, 2, 3])
            self.assertEqual({t['session_id'] for t in turns}, {result['session_id']})
            self.assertEqual(len({t['trace']['run_id'] for t in turns}), 3)
            self.assertEqual(result['trajectory']['score'], 100)
        corrections = report['results'][1]['turns']
        self.assertEqual([t['actual_orders'] for t in corrections], [['O1001'], ['O1003'], ['O1003']])

    def test_long_history_compacts_and_delegates_and_recovery_keeps_tools_once(self):
        run = self.runner()
        from eval_multiturn_scenarios import MULTITURN_CASES
        cases = [c for c in MULTITURN_CASES if c['id'] in ('mt_long_history', 'mt_recovery')]
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: run(cases))
        self.assertEqual(report['passed'], 2, json.dumps(report, ensure_ascii=False))
        long_case, recovery = report['results']
        self.assertGreater(long_case['fixture']['seeded_turns'], 0)
        self.assertTrue(long_case['memory']['covered_messages'] > 0)
        self.assertGreater(long_case['metrics']['summary_requests'], 0)
        self.assertGreater(long_case['metrics']['subagent_requests'], 0)
        budgets = [e['data'] for t in long_case['turns'] for e in t['trace']['events'] if e['kind'] == 'context_budget']
        self.assertTrue({'actor', 'specialist', 'summary', 'verification'}.issubset({b['role'] for b in budgets}))
        self.assertTrue(any(b['saved_tokens'] > 0 for b in budgets))
        recovered = recovery['turns'][1]
        self.assertTrue(recovered['observation']['resumed_same_run'])
        self.assertEqual(recovered['actual_orders'], ['O1003'])
        self.assertEqual(recovered['observation']['actual_order_calls'], 1)

    def test_wrong_order_and_missing_evidence_fail_without_a_required_tool_order(self):
        scorer = self.scorer()
        turn = {'orders': ['O1001', 'O1003'], 'knowledge': True}
        def event(order):
            return {'kind': 'tool_result', 'data': {'call_id': order, 'name': 'get_order', 'arguments': {'order_id': order}, 'status': 'ok', 'attempts': 1, 'result': {'order_id': order, 'status': '已签收'}}}
        policy = {'kind': 'tool_result', 'data': {'call_id': 'policy', 'name': 'search_knowledge', 'arguments': {'query': '退货期限'}, 'status': 'ok', 'attempts': 1, 'result': {'returned_items': 1}}}
        trace = {'status': 'completed', 'events': [event('O1003'), policy, event('O1001')]}
        self.assertEqual(scorer.grade_trajectory(turn, trace)['score'], 100)
        bad = deepcopy(trace)
        bad['events'][0]['data']['arguments']['order_id'] = 'O9999'
        self.assertLess(scorer.grade_trajectory(turn, bad)['score'], 100)
        missing = {'status': 'completed', 'events': [event('O1001'), event('O1003')]}
        failed = scorer.grade_trajectory(turn, missing)['failed_criteria']
        self.assertIn('knowledge_evidence', failed)

    def test_same_turn_equivalent_searches_fail_but_retry_or_refresh_is_allowed(self):
        scorer = self.scorer()
        first = {'kind': 'tool_result', 'data': {'call_id': 'a', 'name': 'search_knowledge', 'arguments': {'query': '  Return   Policy '}, 'status': 'ok', 'attempts': 2, 'result': {'returned_items': 1}}}
        second = deepcopy(first)
        second['data']['call_id'] = 'b'
        second['data']['arguments']['query'] = 'return policy'
        trace = {'status': 'completed', 'events': [first, second]}
        grade = scorer.grade_trajectory({'knowledge': True}, trace)
        self.assertIn('no_repeated_searches', grade['failed_criteria'])
        for event in (first, second):
            self.assertEqual(scorer.grade_trajectory({'knowledge': True}, {'status': 'completed', 'events': [event]})['score'], 100)

    def test_history_search_default_order_is_equivalent_to_explicit_relevance(self):
        scorer = self.scorer()
        events = [{'kind': 'tool_result', 'data': {'call_id': str(index), 'name': 'search_history',
                   'arguments': arguments, 'status': 'ok', 'attempts': 1, 'result': {'returned_items': 1}}}
                  for index, arguments in enumerate([{'query': 'O1003'}, {'query': 'o1003', 'order': 'relevance'}])]
        result = scorer.grade_trajectory({}, {'status': 'completed', 'events': events})
        self.assertIn('no_repeated_searches', result['failed_criteria'])

    def test_unknown_or_partial_usage_never_becomes_zero_cost(self):
        scorer = self.scorer()
        unknown = scorer.summarize_metrics([{'metrics': {'model_requests': 2, 'reported_requests': 0, 'total_tokens': 0}}])
        self.assertIsNone(unknown['total_tokens'])
        self.assertIsNone(unknown['estimated_cost'])
        self.assertEqual(unknown['usage_status'], 'unreported')
        partial = scorer.summarize_metrics([{'metrics': {'model_requests': 2, 'reported_requests': 1, 'total_tokens': 48}}])
        self.assertEqual(partial['total_tokens'], 48)
        self.assertEqual(partial['usage_status'], 'partial')

    def test_invalid_search_then_valid_evidence_still_fails_actual_argument_check(self):
        scorer = self.scorer()
        from eval_scenarios import tools
        run = self.runner()
        case = {'id': 'bad_search', 'title': '无效检索后再次查询', 'turns': [
            {'prompt': '请检索示例退货期限。', 'knowledge': True,
             'script': [tools(('bad', 'search_knowledge', {'query': ''})),
                        tools(('good', 'search_knowledge', {'query': '退货期限'})),
                        {'text': '示例政策是签收后7天内申请。'}]}]}
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: run([case]))
        self.assertEqual(report['passed'], 0)
        self.assertIn('valid_tool_arguments', report['results'][0]['turns'][0]['trajectory']['failed_criteria'])
        # A service connection failure has valid arguments and must not be labelled a parameter error.
        trace = {'status': 'completed', 'events': [{'kind': 'tool_result', 'data': {
            'call_id': 'network', 'name': 'search_knowledge', 'arguments': {'query': '退货期限'},
            'status': 'error', 'attempts': 1, 'result': {'error': '连接失败'}}}]}
        self.assertNotIn('valid_tool_arguments', scorer.grade_trajectory({}, trace)['failed_criteria'])

    def test_server_rejected_plan_then_successful_order_still_fails(self):
        run = self.runner()
        from eval_scenarios import tools
        plan = {'steps': [{'id': 'first', 'title': '查询订单', 'tools': ['nonexistent_tool'],
                           'status': 'in_progress', 'depends_on': [], 'evidence': []}], 'note': ''}
        case = {'id': 'bad_plan', 'title': '服务器拒绝非法计划', 'turns': [
            {'prompt': '请查询 O1003 当前状态。', 'orders': ['O1003'],
             'script': [tools(('plan', 'update_plan', plan)), tools(('order', 'get_order', {'order_id': 'O1003'})),
                        {'text': 'O1003 已签收。'}]}]}
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: run([case]))
        self.assertEqual(report['passed'], 0)
        turn = report['results'][0]['turns'][0]
        self.assertIn('valid_tool_arguments', turn['trajectory']['failed_criteria'])
        self.assertEqual(turn['actual_orders'], ['O1003'])

    def test_wrong_order_return_lookup_is_not_hidden_by_correct_order_query(self):
        run = self.runner()
        from eval_scenarios import tools
        cases = [{'id': order, 'title': '附加退货查询参数', 'turns': [
            {'prompt': '请查询 O1003 当前状态和这笔订单的退货申请。', 'orders': ['O1003'],
             'script': [tools(('order', 'get_order', {'order_id': 'O1003'}),
                              ('return', 'get_return_request', {'order_id': order})),
                        {'text': 'O1003 已签收，暂无退货申请。'}]}]}
                 for order in ('O1001', 'O1003')]
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: run(cases))
        wrong, correct = report['results']
        self.assertFalse(wrong['passed'])
        self.assertIn('valid_tool_arguments', wrong['turns'][0]['trajectory']['failed_criteria'])
        self.assertTrue(correct['passed'], json.dumps(correct, ensure_ascii=False))

    def test_multiturn_cli_and_report_escape_text(self):
        self.runner()
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'multi.json'
            result = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'agent_eval.py'), '--case', 'mt_followup', '--output', str(output)], cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(output.read_text(encoding='utf-8'))
            report['results'][0]['turns'][0]['answer'] = '<script>alert(1)</script>'
            html = agent_eval.report_html(report)
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('context_budget', html)
            self.assertIn('trajectory', html)
            self.assertIn('<th>角色</th>', html)
            self.assertIn('<th>输入 / 上限</th>', html)
            self.assertIn('<th>整理收益</th>', html)
            self.assertTrue(output.with_suffix('.html').exists())
