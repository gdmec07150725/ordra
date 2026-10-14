"""同一场景分别走正式聊天图和实验图，检查实际工具与审批结果。"""
from contextlib import redirect_stdout
import io
import unittest

import anyio
import langgraph_eval


class LangGraphEvalTests(unittest.TestCase):
    def test_offline_comparison_runs_both_engines(self):
        with redirect_stdout(io.StringIO()):
            report = anyio.run(langgraph_eval.compare)
        self.assertEqual(report['total'], 8)
        failures = [{'id': row['id'], 'checks': row['checks']} for row in report['results'] if not row['passed']]
        self.assertEqual(report['passed'], 8, failures)
        self.assertEqual({row['engine'] for row in report['results']}, {'chat_graph', 'experiment_graph'})

    def test_reply_without_actual_order_evidence_fails(self):
        case = next(c for c in langgraph_eval.COMPARISON_CASES if c['id'] == 'order_query')
        checks = langgraph_eval.grade_graph(case, {'status': 'completed', 'answer': 'O1003 已签收', 'events': [], 'metrics': {}}, {})
        self.assertFalse(all(c['passed'] for c in checks))

    def test_report_escapes_external_text(self):
        report = {'mode': 'offline', 'passed': 0, 'total': 1, 'created_at': 'today', 'results': [
            {'engine': 'langgraph', 'title': '<script>alert(1)</script>', 'trace': None, 'passed': False, 'checks': [], 'elapsed_ms': 0}]}
        self.assertNotIn('<script>', langgraph_eval.report_html(report))
