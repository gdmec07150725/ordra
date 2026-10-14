"""The Skills suite runs through the real API and grades actual observations."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class SkillEvalTests(unittest.TestCase):
    def test_skills_cli_reports_selection_budget_recovery_and_comparison(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'skills.json'
            result = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'agent_eval.py'),
                                     '--suite', 'skills', '--output', str(output)],
                                    cwd=root, capture_output=True, text=True, encoding='utf8', timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(report['mode'], 'offline')
            self.assertEqual(report['passed'], report['total'])
            self.assertEqual(report['total'], 9)
            by_id = {item['id']: item for item in report['results']}
            self.assertEqual(by_id['skill_required']['trace']['metrics']['skill_resources_loaded'], 1)
            self.assertEqual(by_id['skill_unneeded']['trace']['metrics']['skills_loaded'], 0)
            self.assertEqual(by_id['skill_resume']['observation']['resource_calls_after_resume'], 1)
            self.assertIn('skill_auto_baseline', by_id)
            self.assertIn('skill_unneeded_baseline', by_id)
            self.assertTrue(output.with_suffix('.html').exists())

    def test_grader_rejects_wrong_skill_or_repeated_reference_for_normal_selection(self):
        # Imports happen here so the CLI test still records the missing suite first.
        from skill_eval import grade_skills, SKILL_CASES
        case = next(case for case in SKILL_CASES if case['id'] == 'skill_required')
        trace = {'status': 'completed', 'metrics': {}, 'events': [
            {'kind': 'skill_load', 'data': {'name': 'order-policy-check', 'result': {'status': 'loaded'}}},
            {'kind': 'skill_resource', 'data': {'name': 'return-handling', 'path': 'references/missing-information.md', 'result': {'status': 'loaded'}}},
            {'kind': 'skill_resource', 'data': {'name': 'return-handling', 'path': 'references/missing-information.md', 'result': {'status': 'already_loaded'}}}]}
        checks = grade_skills(case, trace, '请补充具体退货原因。', {})
        by_id = {item['id']: item['passed'] for item in checks}
        self.assertFalse(by_id['skill_selection'])
        self.assertFalse(by_id['no_repeated_reference'])
