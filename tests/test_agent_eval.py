"""评测入口必须运行真实工具并输出可核对的报告。"""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
import io
from unittest.mock import patch

import agent_eval
from eval_scenarios import CASES, grade_case


class AgentEvalTests(unittest.TestCase):
    def test_delegated_queries_and_isolated_failure_scenarios(self):
        cases = [c for c in CASES if c['id'] in ('delegation_parallel', 'delegation_partial')]
        self.assertEqual(len(cases), 2)
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: agent_eval.evaluate(cases))
        self.assertEqual(report['passed'], 2, json.dumps(report, ensure_ascii=False))

    def test_verification_correction_and_exhaustion_scenarios(self):
        cases = [c for c in CASES if c['id'] in ('result_correction', 'result_unverified')]
        self.assertEqual(len(cases), 2)
        with redirect_stdout(io.StringIO()):
            report = agent_eval.anyio.run(lambda: agent_eval.evaluate(cases))
        self.assertEqual(report['passed'], 2, json.dumps(report, ensure_ascii=False))
        self.assertEqual([r['trace']['metrics']['corrections'] for r in report['results']], [1, 2])

    def test_offline_cli_outputs_report_with_actual_tool_evidence(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'report.json'
            result = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'agent_eval.py'), '--case', 'order_query', '--output', str(target)],
                                    cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(report['mode'], 'offline')
            self.assertTrue(report['results'][0]['passed'])
            evidence = [e for e in report['results'][0]['trace']['events'] if e['kind'] == 'tool_result']
            self.assertEqual(evidence[0]['data']['arguments']['order_id'], 'O1003')
            self.assertTrue(target.with_suffix('.html').exists())

    def test_bad_actual_order_fails_cli_and_report(self):
        case = deepcopy(CASES[0])
        case['script'][0]['calls'][0]['function']['arguments']['order_id'] = 'O1001'
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'failed.json'
            with patch.object(agent_eval, 'CASES', [case]), patch.object(sys, 'argv', ['agent_eval.py', '--output', str(target)]), redirect_stdout(io.StringIO()):
                self.assertEqual(agent_eval.main(), 1)
            report = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(report['passed'], 0)
            self.assertTrue(any(not c['passed'] and c['actual'] == ['O1001'] for c in report['results'][0]['checks']))

    def test_completed_plan_without_dependencies_or_real_evidence_fails(self):
        case = {'plan': True}
        trace = {'status': 'completed', 'events': [{'kind': 'plan', 'data': {'status': 'completed', 'steps': [
            {'id': 'one', 'status': 'completed', 'tools': ['get_order'], 'depends_on': [], 'evidence': ['invented']},
            {'id': 'two', 'status': 'completed', 'tools': ['get_order'], 'depends_on': [], 'evidence': ['invented2']} ]}}]}
        checks = grade_case(case, trace, [{'role': 'user'}], [], {'http_status': 200, 'checkpoint_cleared': True})
        self.assertFalse(all(c['passed'] for c in checks))

    def test_tool_execution_before_human_decision_fails(self):
        case = {'action': 'edit', 'edited_reason': '故障'}
        tool = {'kind': 'tool_result', 'data': {'name': 'prepare_return_request', 'call_id': 'preview', 'status': 'ok', 'attempts': 1}}
        decision = {'kind': 'human_decision', 'data': {'decision': 'edit', 'call_id': 'preview'}}
        trace = {'status': 'completed', 'events': [tool, decision]}
        checks = grade_case(case, trace, [{'role': 'user'}], [{'reason': '故障'}], {'http_status': 200, 'checkpoint_cleared': True, 'paused_before_action': True, 'previews_before_action': 0, 'unapproved_preview_calls': 0})
        self.assertFalse(all(c['passed'] for c in checks))

    def test_report_escapes_untrusted_text(self):
        text = '<script>alert(1)</script>'
        report = {'mode': 'offline', 'passed': 0, 'total': 1, 'created_at': 'today', 'results': [{'trace': None, 'title': text, 'passed': False, 'elapsed_ms': 0, 'checks': [{'name': text, 'passed': False, 'actual': text}]}]}
        html = agent_eval.report_html(report)
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)

    def test_preview_for_wrong_order_or_submitted_state_fails(self):
        case = next(c for c in CASES if c['id'] == 'approval_edit')
        events = [{'kind': 'human_decision', 'data': {'decision': 'edit', 'call_id': 'preview'}}]
        observation = {'http_status': 200, 'checkpoint_cleared': True, 'paused_before_action': True, 'previews_before_action': 0, 'unapproved_preview_calls': 0}
        for order_id, state in [('O1001', 'pending'), ('O1003', 'submitted')]:
            with self.subTest(order_id=order_id, state=state):
                previews = [{'order_id': order_id, 'reason': case['edited_reason'], 'state': state}]
                checks = grade_case(case, {'status': 'completed', 'events': events}, [{'role': 'user'}], previews, observation)
                self.assertFalse(all(c['passed'] for c in checks))
