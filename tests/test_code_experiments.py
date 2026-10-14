import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx2
from fastapi import FastAPI

from code_experiments import CodeExperiments, register_code_routes
from test_code_agent import CORRECT_CODE, ScriptGenerator


class CodeExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = FastAPI()
        self.app.state.client = None
        self.app.state.model = 'scripted'
        self.root = Path(self.tmp.name)
        self.service = CodeExperiments(self.app, self.root, generator_factory=lambda: ScriptGenerator([CORRECT_CODE] * 3))
        await self.service.open()
        self.app.state.code_experiments = self.service
        register_code_routes(self.app, Path(__file__).resolve().parents[1])
        self.client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.service.close()
        await self.client.aclose()
        self.tmp.cleanup()

    async def wait(self, run_id):
        for _ in range(150):
            detail = (await self.client.get(f'/api/code/runs/{run_id}')).json()
            if detail['status'] != 'running': return detail
            await asyncio.sleep(.02)
        self.fail('任务未在时间内完成')

    async def test_api_persists_and_publishes_only_verified_fixed_artifacts(self):
        response = await self.client.post('/api/code/runs', json={'dataset_id': 'orders-demo', 'group_by': 'product', 'threshold': .3})
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['run_id']
        result = await self.wait(run_id)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['result']['report']['total_orders'], 18)
        for name in ('report.csv', 'report.md'):
            download = await self.client.get(f'/api/code/runs/{run_id}/artifacts/{name}')
            self.assertEqual(download.status_code, 200)
            self.assertIn('attachment', download.headers['content-disposition'])
            self.assertEqual(download.headers['x-content-type-options'], 'nosniff')
        self.assertEqual((await self.client.get(f'/api/code/runs/{run_id}/artifacts/input.csv')).status_code, 404)
        self.assertEqual((await self.client.get('/api/code/runs/not-a-uuid')).status_code, 422)
        self.assertEqual((await self.client.post('/api/code/runs', json={'dataset_id': '../secret'})).status_code, 422)
        restored = CodeExperiments(self.app, self.root)
        await restored.open()
        self.assertEqual(restored.read(run_id)['status'], 'completed')
        await restored.close()

    async def test_busy_cancel_and_restart_never_publish_unverified_result(self):
        entered = asyncio.Event()
        async def pending(messages):
            entered.set()
            await asyncio.Event().wait()
        self.service.generator_factory = lambda: pending
        response = await self.client.post('/api/code/runs', json={})
        run_id = response.json()['run_id']
        await entered.wait()
        self.assertEqual((await self.client.post('/api/code/runs', json={})).status_code, 409)
        self.assertEqual((await self.client.get(f'/api/code/runs/{run_id}/artifacts/report.md')).status_code, 409)
        self.assertEqual((await self.client.post(f'/api/code/runs/{run_id}/cancel')).status_code, 200)
        self.assertEqual(self.service.read(run_id)['status'], 'cancelled')
        self.assertFalse(self.service.jobs)
        path = self.root / run_id / 'run.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        state['status'] = 'running'
        path.write_text(json.dumps(state), encoding='utf-8')
        restored = CodeExperiments(self.app, self.root)
        await restored.open()
        self.assertEqual(restored.read(run_id)['status'], 'interrupted')
        await restored.close()

    async def test_failed_computation_has_no_downloadable_report(self):
        self.service.generator_factory = lambda: ScriptGenerator(['1 / 0'] * 3)
        run_id = (await self.client.post('/api/code/runs', json={})).json()['run_id']
        result = await self.wait(run_id)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.root / run_id / 'report.md').exists())
        self.assertEqual((await self.client.get(f'/api/code/runs/{run_id}/artifacts/report.md')).status_code, 409)

    async def test_transient_windows_record_lock_recovers_without_losing_run(self):
        original = Path.replace
        failed = 0
        def replace(path, target):
            nonlocal failed
            if path.name == 'run.json.tmp' and failed < 2:
                failed += 1
                raise PermissionError('simulated Windows sharing violation')
            return original(path, target)
        with patch.object(Path, 'replace', replace):
            response = await self.client.post('/api/code/runs', json={})
            self.assertEqual(response.status_code, 202)
            result = await self.wait(response.json()['run_id'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(failed, 2)
        self.assertFalse(self.service.jobs)

    async def test_persistent_record_failure_is_visible_and_releases_task_slot(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def pending(messages):
            entered.set()
            await release.wait()
            return {'code': CORRECT_CODE, 'note': ''}
        self.service.generator_factory = lambda: pending
        run_id = (await self.client.post('/api/code/runs', json={})).json()['run_id']
        await entered.wait()
        task = self.service.jobs[run_id]
        with patch.object(Path, 'replace', side_effect=PermissionError('persistent lock')):
            release.set()
            results = await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(results, [None])
            result = self.service.read(run_id)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('写入', result['error'])
            self.assertFalse(self.service.jobs)
            self.assertEqual(self.service.list()[0]['status'], 'failed')
            self.assertEqual((await self.client.get(f'/api/code/runs/{run_id}/artifacts/report.md')).status_code, 409)

    async def test_cancel_during_terminal_save_retry_keeps_a_visible_terminal_state(self):
        locked = asyncio.Event()
        original = Path.replace
        async def fatal(messages):
            return {'error': '模拟模型故障', 'fatal': True, 'sent': False}
        self.service.generator_factory = lambda: fatal
        def replace(path, target):
            if path.name == 'run.json.tmp' and json.loads(path.read_text(encoding='utf-8'))['status'] != 'running':
                locked.set()
                raise PermissionError('locked terminal record')
            return original(path, target)
        with patch.object(Path, 'replace', replace):
            run_id = (await self.client.post('/api/code/runs', json={})).json()['run_id']
            await asyncio.wait_for(locked.wait(), timeout=2)
            response = await self.client.post(f'/api/code/runs/{run_id}/cancel')
            self.assertEqual(response.status_code, 200)
            self.assertNotEqual(response.json()['status'], 'running')
            self.assertFalse(self.service.jobs)
            self.assertIn(run_id, self.service.unsaved)
            self.assertEqual((await self.client.get(f'/api/code/runs/{run_id}/artifacts/report.md')).status_code, 409)
