import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skill_runtime import SkillRuntime, historical_skill_view


class SkillRuntimeTests(unittest.TestCase):
    def test_resume_budget_counts_accepted_skill_attempts(self):
        from checkpoint_store import resume_reason
        run = {'state': {'rounds': 4, 'skills': {'version': 1, 'calls': 1}}, 'resume_count': 0}
        self.assertEqual(resume_reason(run), '')
        run['state']['rounds'] = 5
        self.assertIn('模型请求上限', resume_reason(run))

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.write_skill('returns', '先读取订单，再检索政策，确认原因后生成预览。')

    def write_skill(self, name, body, description='处理退货申请'):
        path = self.root / name / 'SKILL.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\nname: {name}\ndescription: {description}\n---\n{body}\n', encoding='utf-8')
        return path

    def load(self, runtime, name='returns', accept=lambda result: True):
        return runtime.load(json.dumps({'name': name}), accept=accept)

    def test_metadata_only_then_load_complete_body(self):
        runtime = SkillRuntime(self.root)
        self.assertIn('处理退货申请', runtime.instructions())
        self.assertNotIn('先读取订单', runtime.instructions())
        result = self.load(runtime)
        self.assertEqual(result['status'], 'loaded')
        self.assertIn('先读取订单', result['instructions'])
        self.assertEqual(result['version'], runtime.catalog[0]['version'])

    def test_budget_rejection_is_not_saved_as_loaded(self):
        runtime = SkillRuntime(self.root)
        result = self.load(runtime, accept=lambda result: False)
        self.assertEqual(result['error_code'], 'skill_budget_exceeded')
        self.assertEqual(runtime.loaded, {})
        self.assertNotIn('instructions', runtime.snapshot()['events'][0]['result'])

    def test_repeat_is_metadata_only_and_attempts_bounded(self):
        runtime = SkillRuntime(self.root)
        self.load(runtime)
        again = self.load(runtime)
        self.assertEqual(again['status'], 'already_loaded')
        self.assertNotIn('instructions', again)
        self.load(runtime)
        self.assertEqual(self.load(runtime)['error_code'], 'skill_load_limit')
        self.assertEqual(runtime.calls, 3)

    def test_snapshot_pins_loaded_body_despite_file_edit_or_deletion(self):
        runtime = SkillRuntime(self.root)
        original = self.load(runtime)
        self.path.unlink()
        restored = SkillRuntime(self.root, state=json.loads(json.dumps(runtime.snapshot())))
        self.assertEqual(restored.loaded['returns'], original)
        self.assertEqual(self.load(restored)['status'], 'already_loaded')

    def test_unloaded_version_change_rejected_new_run_reads_new_body(self):
        old = SkillRuntime(self.root).snapshot()
        self.write_skill('returns', '新版处理流程')
        result = self.load(SkillRuntime(self.root, state=old))
        self.assertEqual(result['error_code'], 'skill_version_changed')
        self.assertIn('新版处理流程', self.load(SkillRuntime(self.root))['instructions'])

    def test_paths_invalid_arguments_and_unknown_names_rejected(self):
        for payload in ('[]', '{', '{"name":"../returns"}', '{"name":"returns","path":"x"}'):
            runtime = SkillRuntime(self.root)
            self.assertIn('error', runtime.load(payload, accept=lambda result: True))
        self.assertEqual(self.load(SkillRuntime(self.root), 'missing')['error_code'], 'skill_not_found')

    def test_invalid_and_oversized_files_excluded(self):
        self.write_skill('bad', 'x' * 25000)
        wrong = self.write_skill('wrong', 'hello')
        wrong.write_text('---\nname: mismatch\ndescription: demo\n---\nx', encoding='utf8')
        runtime = SkillRuntime(self.root)
        self.assertEqual([item['name'] for item in runtime.catalog], ['returns'])
        self.assertEqual(len(runtime.errors), 2)

    def test_past_run_body_removed_only_from_request_view(self):
        history = [{'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'load_skill'}}]},
                   {'role': 'tool', 'tool_call_id': 'c1', 'content': json.dumps({'status': 'loaded', 'name': 'returns', 'instructions': 'old steps'})}]
        view = historical_skill_view(history)
        self.assertNotIn('old steps', view[-1]['content'])
        self.assertIn('old steps', history[-1]['content'])
