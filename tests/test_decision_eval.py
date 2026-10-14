"""Decision experiments grade real evidence, keep failures, and count every role."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import anyio


class DecisionEvalTests(unittest.TestCase):
    def evaluator(self):
        return importlib.import_module('decision_eval')

    def test_report_retries_transient_file_lock_and_keeps_permanent_errors_visible(self):
        evaluator = self.evaluator()
        real_replace = Path.replace
        attempts = []

        def locked_once(source, target):
            attempts.append(target.suffix)
            if target.suffix == '.html' and attempts.count('.html') == 1:
                raise PermissionError('sharing violation')
            return real_replace(source, target)

        with TemporaryDirectory() as directory, patch.object(evaluator, 'report_html', return_value='<p>new report</p>'):
            target = Path(directory) / 'report.json'
            target.with_suffix('.html').write_text('<p>previous report</p>', encoding='utf8')
            with patch.object(Path, 'replace', new=locked_once):
                evaluator.write_report({'passed': 2}, target)
            self.assertEqual(json.loads(target.read_text(encoding='utf8')), {'passed': 2})
            self.assertEqual(target.with_suffix('.html').read_text(encoding='utf8'), '<p>new report</p>')
            self.assertEqual(attempts.count('.html'), 2)
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace:
                with self.assertRaises(PermissionError):
                    evaluator.write_report({'passed': 3}, target)
            self.assertGreater(replace.call_count, 1)
            self.assertLessEqual(replace.call_count, 8)
            self.assertEqual(json.loads(target.read_text(encoding='utf8')), {'passed': 2})

    def observation(self):
        return {'http_status': 200, 'checkpoint_cleared': True, 'committed': True,
                'fault_injected': False}

    def trace(self):
        return {'status': 'completed', 'metrics': {'verification_status': 'passed'}, 'events': [
            {'kind': 'tool_result', 'data': {'name': 'get_order', 'call_id': 'order',
             'arguments': {'order_id': 'O1003'}, 'status': 'ok', 'attempts': 1,
             'result': {'order_id': 'O1003', 'status': '已签收'}}}]}

    def test_grader_rejects_fabrication_even_when_answer_and_reviewer_claim_success(self):
        evaluate = self.evaluator()
        case = {'orders': ['O1003']}
        trace = self.trace()
        answer = 'O1003 已签收。'
        self.assertTrue(evaluate.grade_case(case, trace, answer, self.observation())['passed'])
        for mutation in ('missing', 'wrong_order', 'wrong_status', 'wrong_answer', 'unverified'):
            bad, text = deepcopy(trace), answer
            if mutation == 'missing':
                bad['events'] = []
            elif mutation == 'wrong_order':
                bad['events'][0]['data']['result']['order_id'] = 'O1001'
            elif mutation == 'wrong_status':
                bad['events'][0]['data']['result']['status'] = '已取消'
            elif mutation == 'wrong_answer':
                text = 'O1003 已取消。'
            else:
                bad['metrics']['verification_status'] = 'unverified'
            with self.subTest(mutation=mutation):
                self.assertFalse(evaluate.grade_case(case, bad, text, self.observation())['passed'])

    def test_missing_reason_forbids_even_a_rejected_preview_attempt(self):
        evaluate = self.evaluator()
        case = {'clarification': 'reason'}
        trace = self.trace()
        text = '请提供具体退货原因。'
        self.assertTrue(evaluate.grade_case(case, trace, text, self.observation())['passed'])
        trace['events'].append({'kind': 'tool_result', 'data': {
            'name': 'prepare_return_request', 'attempts': 0, 'status': 'error', 'result': {}}})
        self.assertFalse(evaluate.grade_case(case, trace, text, self.observation())['passed'])

    def test_clarification_accepts_a_polite_equivalent_request(self):
        self.assertTrue(self.evaluator().grade_case({'clarification': 'reason'}, self.trace(),
                        '目前缺少具体退货原因，麻烦您说明一下。', self.observation())['passed'])

    def test_schedule_rotates_strategy_order_without_dropping_or_duplicating_trials(self):
        schedule = self.evaluator().make_schedule([{'id': 'a'}, {'id': 'b'}], ['current', 'direct', 'selective'], 3)
        self.assertEqual(len(schedule), 18)
        self.assertEqual(len({(x['case_id'], x['strategy'], x['repeat']) for x in schedule}), 18)
        for case in ('a', 'b'):
            first = [next(x['strategy'] for x in schedule if x['case_id'] == case and x['repeat'] == r)
                     for r in range(1, 4)]
            self.assertEqual(set(first), {'current', 'direct', 'selective'})

    def test_summary_includes_failed_runs_and_does_not_fill_unknown_usage_with_zero(self):
        evaluate = self.evaluator()
        results = [dict(strategy='direct', case_id='a', passed=passed, elapsed_ms=elapsed,
                        compliance={'followed': True}, metrics={'request_count': requests,
                        'total_tokens': tokens, 'usage_status': usage, 'invalid_calls': 1,
                        'repeated_calls': 0, 'control_calls': 2})
                   for passed, elapsed, requests, tokens, usage in
                   [(True, 10, 3, 100, 'complete'), (False, 90, 5, None, 'unreported')]]
        row = evaluate.aggregate(results)[0]
        self.assertEqual((row['trials'], row['passed'], row['success_rate']), (2, 1, 0.5))
        self.assertEqual(row['mean_requests'], 4)
        self.assertEqual(row['mean_elapsed_ms'], 50)
        self.assertIsNone(row['mean_total_tokens'])
        self.assertEqual(row['reported_tokens'], 100)
        self.assertEqual(row['complete_usage_trials'], 1)
        self.assertFalse(row['cases'][0]['all_passed'])

    def test_compliance_is_separate_from_business_success(self):
        evaluate = self.evaluator()
        trace = self.trace()
        self.assertTrue(evaluate.grade_case({'orders': ['O1003']}, trace, 'O1003 已签收。', self.observation())['passed'])
        self.assertFalse(evaluate.strategy_observation('selective', {'knowledge': True, 'orders': ['O1003']}, trace)['followed'])

    def test_unexecuted_lifecycle_failure_cannot_look_fast_or_cheap(self):
        row = self.evaluator().aggregate([dict(strategy='direct', case_id='a', passed=False,
             executed=False, elapsed_ms=None, compliance={'followed': None},
             metrics={'request_count': 0, 'usage_status': 'unreported', 'total_tokens': None})])[0]
        self.assertEqual(row['success_rate'], 0)
        self.assertEqual(row['executed_trials'], 0)
        self.assertIsNone(row['mean_elapsed_ms'])
        self.assertIsNone(row['mean_requests'])
        self.assertIsNone(row['mean_total_tokens'])

    def test_policy_requires_actual_passage_as_well_as_a_successful_trace(self):
        evaluate = self.evaluator()
        trace, observation = self.trace(), self.observation()
        text = 'O1003 已签收，退货7天内申请，退款需要审核，请补充原因。'
        case = {'orders': ['O1003'], 'knowledge': True}
        trace['events'].append({'kind': 'tool_result', 'data': {'name': 'search_knowledge',
                                'status': 'ok', 'attempts': 1, 'result': {'returned_items': 1}}})
        self.assertFalse(evaluate.grade_case(case, trace, text, observation)['passed'])
        observation['evidence'] = {'knowledge': [{'text': '签收后7天可申请退货，退款需要审核。'}]}
        self.assertTrue(evaluate.grade_case(case, trace, text, observation)['passed'])
        trace['events'].pop()
        self.assertFalse(evaluate.grade_case(case, trace, text, observation)['passed'])

    def test_not_loaded_tool_is_invalid_even_when_runtime_records_one_attempt(self):
        trace = self.trace()
        trace['events'][0]['data'].update(status='error', attempts=1,
                                         result={'error_code': 'tool_not_loaded', 'error': '尚未加载'})
        self.assertEqual(self.evaluator().decision_metrics(trace)['invalid_calls'], 1)

    def test_valid_delegation_with_child_service_failure_is_not_an_invalid_call(self):
        trace = self.trace()
        trace['events'][0]['data'].update(name='delegate_tasks', arguments={'tasks': [
            {'agent': 'orders', 'task': '查询 O1003'}]}, status='error', attempts=1,
            result={'status': 'partial', 'error': '子任务失败'})
        self.assertEqual(self.evaluator().decision_metrics(trace)['invalid_calls'], 0)

    def test_specialist_proposals_rejected_before_runtime_are_counted_once(self):
        evaluate = self.evaluator()
        rejected = [{'name': 'get_order', 'call_id': 'call_x', 'reason': 'invalid_arguments'}]
        self.assertEqual(evaluate.decision_metrics(None, rejected)['invalid_calls'], 1)
        trace = self.trace()
        trace['events'][0]['data'].update(call_id='specialist_1:r1:call_x', arguments={},
                                         status='error', result={'error_code': 'invalid_arguments'})
        self.assertEqual(evaluate.decision_metrics(trace, rejected)['invalid_calls'], 1)

    def test_copied_specialist_client_rejection_is_observed_through_real_runner(self):
        evaluate = self.evaluator()
        case = next(c for c in evaluate.CASES if c['id'] == 'order_policy')
        script = evaluate.scripted_case(case, 'selective')
        script['specialists']['orders'][0]['calls'][0]['function']['name'] = 'unknown_order_tool'
        with patch.object(evaluate, 'scripted_case', return_value=script):
            report = anyio.run(lambda: evaluate.evaluate_decisions([case], strategies=['selective'], repeats=1))
        item = report['results'][0]
        self.assertFalse(item['passed'])
        self.assertEqual(item['metrics']['invalid_calls'], 1)
        self.assertEqual(item['observation']['rejected_specialist_proposals'][0]['reason'], 'tool_not_offered')

    def test_cli_runs_actual_tools_isolated_with_all_strategies_and_escaped_report(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'decisions.json'
            process = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'decision_eval.py'),
                                      '--repeats', '1', '--output', str(output)], cwd=root,
                                     capture_output=True, encoding='utf8', timeout=180)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(report['passed'], report['total'])
            self.assertEqual(report['total'], len(report['dataset']['cases']) * 3)
            self.assertEqual(report['mode'], 'offline')
            delegated = next(r for r in report['results'] if r['strategy'] == 'selective' and r['case_id'] == 'order_policy')
            self.assertEqual(delegated['metrics']['request_counts_by_role']['specialist'], 4)
            self.assertGreater(delegated['metrics']['request_count'], delegated['metrics']['request_counts_by_role']['actor'])
            self.assertTrue(delegated['compliance']['followed'])
            self.assertTrue(delegated['metrics']['usage_simulated'])
            self.assertIsNone(report['recommended_strategy'])
            fault = next(r for r in report['results'] if r['case_id'] == 'tool_transient')
            self.assertTrue(fault['observation']['fault_injected'])
            self.assertEqual(fault['metrics']['tool_retries'], 1)
            old = deepcopy(report)
            clarification = next(r for r in old['results'] if r['case_id'] == 'missing_reason')
            clarification.update(answer='缺少具体退货原因，麻烦您说明一下。', passed=False)
            before = deepcopy(old)
            regraded = self.evaluator().regrade_report(old)
            self.assertEqual(old, before)
            self.assertTrue(regraded['analysis']['evidence_unchanged'])
            self.assertEqual(regraded['passed'], regraded['total'])
            self.assertEqual([r['trace'] for r in regraded['results']], [r['trace'] for r in old['results']])
            refused = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'decision_eval.py'),
                                      '--regrade', str(output), '--output', str(output)], cwd=root,
                                     capture_output=True, encoding='utf8', timeout=30)
            self.assertEqual(refused.returncode, 2)
            self.assertEqual(json.loads(output.read_text(encoding='utf8')), report)
            report['results'][0]['answer'] = '<script>alert(1)</script>'
            html = self.evaluator().report_html(report)
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertTrue(output.with_suffix('.html').exists())


if __name__ == '__main__':
    unittest.main()
