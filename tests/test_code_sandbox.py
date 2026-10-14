import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from code_sandbox import SandboxPolicy, run_code


class CodeSandboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_python_computation_and_fresh_sessions(self):
        first = await run_code('x = sum(row["n"] for row in rows)\nprint("computed", x)\n{"total": x}', {'rows': [{'n': 2}, {'n': 3}]})
        self.assertTrue(first['ok'], first)
        self.assertEqual(first['value'], {'total': 5})
        self.assertIn('computed 5', first['stdout'])
        second = await run_code('x', {})
        self.assertFalse(second['ok'])

    async def test_host_file_network_process_and_env_are_not_available(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'private.txt'
            sentinel.write_text('host-secret-sentinel', encoding='utf-8')
            attempts = [f'open({str(sentinel)!r}).read()',
                        'import socket\nsocket.socket()',
                        'import subprocess\nsubprocess.run(["whoami"])',
                        'import os\nos.environ["AGENT_TEST_SECRET"]',
                        'import ctypes\nctypes.CDLL(None)']
            with patch.dict(os.environ, {'AGENT_TEST_SECRET': 'host-secret-sentinel', 'MONTY_BIN': 'invalid-do-not-run.exe'}):
                for code in attempts:
                    with self.subTest(code=code):
                        result = await run_code(code, {})
                        self.assertFalse(result['ok'], result)
                        self.assertNotIn('host-secret-sentinel', str(result))
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'host-secret-sentinel')

    async def test_syntax_runtime_timeout_memory_and_print_limits(self):
        for code in ('if :', '1 / 0', 'while True:\n    pass', '"x" * 100_000_000', 'print("x" * 5000)'):
            with self.subTest(code=code):
                result = await run_code(code, {}, policy=SandboxPolicy(seconds=.15))
                self.assertFalse(result['ok'], result)
                self.assertLessEqual(len(result['stdout']), 4096)
        self.assertTrue((await run_code('1 + 1', {}))['ok'])

    async def test_non_json_and_large_results_are_rejected(self):
        for code in ('{1, 2}', 'float("nan")', '"x" * 70000'):
            with self.subTest(code=code):
                self.assertFalse((await run_code(code, {}))['ok'])

    async def test_cancellation_releases_real_worker(self):
        import asyncio
        task = asyncio.create_task(run_code('while True:\n    pass', {}))
        await asyncio.sleep(.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=4)
        self.assertTrue((await run_code('2 + 2', {}))['ok'])
