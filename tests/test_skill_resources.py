"""Contracts for opt-in, versioned local reference reads."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
import subprocess

from skill_runtime import SkillRuntime, historical_skill_view, skill_allowance


class SkillResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / 'returns'
        (self.skill / 'references').mkdir(parents=True)
        self.path = self.skill / 'references' / 'missing.md'
        self.path.write_text('询问实际故障，不能编造原因。', encoding='utf8')
        (self.skill / 'SKILL.md').write_text('---\nname: returns\ndescription: 办理退货\n---\n缺信息时读取[询问指引](references/missing.md)。', encoding='utf8')

    def runtime(self, *, loaded=True):
        runtime = SkillRuntime(self.root)
        if loaded:
            runtime.load('{"name":"returns"}', accept=lambda value: True)
        return runtime

    def read(self, runtime, path='references/missing.md', *, accept=lambda value: True):
        self.assertTrue(callable(getattr(runtime, 'read_resource', None)), 'Skill runtime must support reference reads')
        return runtime.read_resource(json.dumps({'name': 'returns', 'path': path}), accept=accept)

    def test_catalog_and_skill_body_do_not_eagerly_include_reference_text(self):
        runtime = self.runtime(loaded=False)
        self.assertNotIn('询问实际故障', runtime.instructions())
        loaded = runtime.load('{"name":"returns"}', accept=lambda value: True)
        self.assertIn('resources', loaded)
        self.assertEqual(loaded['resources'][0]['path'], 'references/missing.md')
        self.assertNotIn('询问实际故障', json.dumps(loaded, ensure_ascii=False))
        result = self.read(runtime)
        self.assertEqual(result['content'], '询问实际故障，不能编造原因。')
        self.assertEqual(result['status'], 'loaded')
        self.assertGreater(result['estimated_tokens'], 0)

    def test_must_load_skill_and_use_declared_reference(self):
        self.assertEqual(self.read(self.runtime(loaded=False))['error_code'], 'skill_not_loaded')
        (self.skill / 'references' / 'private.md').write_text('not declared', encoding='utf8')
        self.assertEqual(self.read(self.runtime(), 'references/private.md')['error_code'], 'skill_resource_not_found')

    def test_rejects_traversal_absolute_paths_and_windows_aliases(self):
        for path in ('../SKILL.md', '/etc/passwd', 'C:/secret.md', 'references/../SKILL.md',
                     'references\\missing.md', 'references/missing.md:secret', 'references//missing.md'):
            with self.subTest(path=path):
                self.assertIn('error', self.read(self.runtime(), path))

    def test_repeated_read_has_no_body_and_attempts_remain_bounded_on_resume(self):
        runtime = self.runtime()
        self.read(runtime)
        for _ in range(3):
            result = self.read(runtime)
            self.assertEqual(result['status'], 'already_loaded')
            self.assertNotIn('content', result)
        restored = SkillRuntime(self.root, state=runtime.snapshot())
        self.assertEqual(self.read(restored)['error_code'], 'skill_resource_limit')
        self.assertEqual(skill_allowance(restored.snapshot()), 5)

    def test_budget_rejection_preserves_no_body_or_accepted_resource(self):
        runtime = self.runtime()
        self.assertEqual(self.read(runtime, accept=lambda value: False)['error_code'], 'skill_resource_budget_exceeded')
        state = runtime.snapshot()
        self.assertEqual(state['resources'], {})
        self.assertNotIn('询问实际故障', json.dumps(state, ensure_ascii=False))

    def test_loaded_reference_survives_deletion_and_new_run_does_not_inherit_it(self):
        runtime = self.runtime()
        original = self.read(runtime)
        self.path.unlink()
        restored = SkillRuntime(self.root, state=json.loads(json.dumps(runtime.snapshot())))
        self.assertEqual(restored.resources['returns:references/missing.md'], original)
        self.assertEqual(self.read(restored)['status'], 'already_loaded')
        self.assertEqual(self.runtime().resources, {})

    def test_unread_reference_change_rejected_even_after_resume(self):
        runtime = self.runtime()
        self.path.write_text('new instructions', encoding='utf8')
        restored = SkillRuntime(self.root, state=runtime.snapshot())
        self.assertEqual(self.read(restored)['error_code'], 'skill_resource_version_changed')
        self.assertEqual(self.read(self.runtime())['content'], 'new instructions')

    def test_large_missing_and_invalid_utf8_references_are_recoverable_errors(self):
        for raw in (b'a' * 24001, b'\xff', None):
            with self.subTest(raw_type=type(raw).__name__):
                if raw is None:
                    self.path.unlink(missing_ok=True)
                else:
                    self.path.write_bytes(raw)
                runtime = self.runtime()
                self.assertIn('returns', runtime.loaded)
                self.assertIn('error', self.read(runtime))

    def test_reference_symlink_cannot_read_another_skill(self):
        outside = self.root / 'other.md'
        outside.write_text('must not be read', encoding='utf8')
        self.path.unlink()
        try:
            self.path.symlink_to(outside)
        except OSError:
            self.skipTest('Host cannot create symlinks')
        result = self.read(self.runtime())
        self.assertIn('error', result)
        self.assertNotIn('must not be read', json.dumps(result))

    @unittest.skipUnless(os.name == 'nt', 'Windows junction boundary')
    def test_reference_junction_cannot_escape_skill_directory(self):
        original = self.skill / 'references-original'
        (self.skill / 'references').rename(original)
        # Both paths are inside this test's temporary root; no user files involved.
        junction = self.skill / 'references'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(original)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.addCleanup(lambda: os.rmdir(junction) if junction.is_junction() else None)
        self.assertIn('error', self.read(self.runtime()))

    def test_history_projection_removes_reference_body_without_mutating_original(self):
        history = [{'role': 'assistant', 'tool_calls': [{'id': 'r1', 'function': {'name': 'read_skill_resource'}}]},
                   {'role': 'tool', 'tool_call_id': 'r1', 'content': json.dumps({'name': 'returns', 'path': 'references/missing.md', 'content': 'old private body'})}]
        view = historical_skill_view(history)
        self.assertNotIn('old private body', view[-1]['content'])
        self.assertIn('old private body', history[-1]['content'])

    def test_legacy_snapshot_has_no_new_resource_allowance(self):
        legacy = {'version': 1, 'catalog': [], 'loaded': {}, 'calls': 2, 'events': []}
        runtime = SkillRuntime(self.root, state=legacy)
        self.assertEqual(runtime.snapshot(), legacy)
        self.assertEqual(skill_allowance({**legacy, 'resource_calls': 4}), 2)

    def test_corrupted_resource_checkpoint_is_rejected(self):
        runtime = self.runtime()
        self.read(runtime)
        state = deepcopy(runtime.snapshot())
        state['resources']['returns:references/missing.md']['version'] = 'wrong'
        with self.assertRaises(ValueError):
            SkillRuntime(self.root, state=state)
