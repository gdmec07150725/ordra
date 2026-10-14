"""独立分析实验的任务生命周期、记录和报告接口。"""
import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from code_agent import ModelCodeGenerator, run_analysis
from code_data import DATASETS, FIELDS, load_dataset, render_reports
from code_sandbox import sandbox_status


LOG = logging.getLogger(__name__)


class CodeQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    dataset_id: Literal['orders-demo', 'orders-edge'] = 'orders-demo'
    group_by: Literal['product', 'channel'] = 'product'
    threshold: float = Field(default=.3, ge=0, le=1, allow_inf_nan=False)
    exercise: Literal['normal', 'runtime_error', 'wrong_result'] = 'normal'


def now():
    return datetime.now(UTC).isoformat()


class CodeExperiments:
    def __init__(self, app, root, *, generator_factory=None):
        self.app, self.root = app, Path(root).resolve()
        self.generator_factory = generator_factory
        self.lock, self.jobs, self.closing = asyncio.Lock(), {}, False
        self.unsaved = {}

    async def open(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                UUID(directory.name)
                record = self.read(directory.name)
            except (ValueError, HTTPException):
                continue
            if record['status'] == 'running':
                record.update(status='interrupted', error='服务重启中断了实验，已有代码与反馈保留，请重新运行。', updated_at=now())
                await self.save(record)
        return self

    def directory(self, run_id):
        try:
            key = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(404, '实验不存在。') from None
        path = self.root / key
        if path.resolve().parent != self.root:
            raise HTTPException(404, '实验不存在。')
        return path

    def read(self, run_id):
        directory = self.directory(run_id)
        if directory.name in self.unsaved:
            return deepcopy(self.unsaved[directory.name])
        try:
            return json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            raise HTTPException(404, '实验不存在或记录不完整。') from None

    async def save(self, record):
        directory = self.directory(record['run_id'])
        temp = directory / 'run.json.tmp'
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        # Windows readers/indexers may briefly deny atomic replacement. Yield to
        # the event loop between bounded retries instead of blocking other requests.
        for attempt in range(5):
            try:
                temp.write_text(payload, encoding='utf-8')
                temp.replace(directory / 'run.json')
                self.unsaved.pop(record['run_id'], None)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                await asyncio.sleep(.05 * (attempt + 1))

    async def save_terminal(self, record):
        """Always leave a visible terminal state, including cancellation during retry."""
        try:
            await self.save(record)
        except (OSError, asyncio.CancelledError) as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            record.update(status=('interrupted' if self.closing else 'cancelled') if cancelled else 'failed',
                          updated_at=now(),
                          error='实验已停止，终态记录未能写入磁盘，未发布报告；当前状态仅在服务运行期间保留，请检查文件权限或磁盘后重试。')
            self.unsaved[record['run_id']] = deepcopy(record)
            LOG.error('Code experiment %s could not save terminal state (%s)', record['run_id'], type(error).__name__)

    def list(self):
        records = []
        for path in self.root.glob('*/run.json'):
            try:
                record = self.read(path.parent.name)
            except HTTPException:
                continue
            records.append({key: record[key] for key in ('run_id', 'created_at', 'status', 'config', 'dataset_title')})
        return sorted(records, key=lambda row: row['created_at'], reverse=True)[:100]

    async def create(self, config):
        if not isinstance(config, CodeQuestion):
            config = CodeQuestion.model_validate(config)
        async with self.lock:
            if self.closing or self.jobs:
                raise HTTPException(409, '已有分析正在运行，请等待完成或停止当前任务。')
            readiness = sandbox_status()
            if not readiness['ready']:
                raise HTTPException(503, readiness['message'])
            if self.generator_factory is None and self.app.state.client is None:
                raise HTTPException(503, '请先配置模型服务并重启。')
            dataset = load_dataset(config.dataset_id)
            run_id = str(uuid4())
            directory = self.directory(run_id)
            directory.mkdir()
            (directory / 'input.csv').write_text(dataset['csv'], encoding='utf-8')
            record = {'run_id': run_id, 'created_at': now(), 'updated_at': now(), 'status': 'running',
                      'dataset_title': dataset['title'], 'source_sha256': dataset['sha256'],
                      'config': config.model_dump(), 'events': [], 'result': None, 'error': '', 'elapsed_ms': 0}
            await self.save(record)
            self.jobs[run_id] = asyncio.create_task(self._run(record, dataset), name='code-agent-' + run_id)
            return {'run_id': run_id, 'status': 'running'}

    async def _run(self, record, dataset):
        run_id, config = record['run_id'], record['config']
        started = monotonic()
        async def emit(event):
            record['events'].append({**event, 'at': now()})
            record['updated_at'] = now()
            await self.save(record)
        try:
            generator = self.generator_factory() if self.generator_factory else ModelCodeGenerator(
                self.app.state.client, self.app.state.model, self.app.state.context_policy)
            result = await run_analysis(dataset, config['group_by'], config['threshold'], generator,
                                        emit=emit, exercise=config['exercise'])
            record.update(result=result, status=result['status'])
            if result['status'] == 'completed':
                reports = render_reports(result['report'], title=dataset['title'], group_by=config['group_by'], threshold=config['threshold'])
                for name, content in reports.items():
                    (self.directory(run_id) / name).write_text(content, encoding='utf-8')
            else:
                record['error'] = '本次分析未通过校验，已停止自动尝试。可查看执行记录后新建实验。'
        except asyncio.CancelledError:
            record.update(status='interrupted' if self.closing else 'cancelled',
                          error='执行已停止，已有代码和反馈保留；未发布报告。')
        except Exception as error:
            LOG.error('Code experiment %s failed (%s)', run_id, type(error).__name__)
            record.update(status='failed', error='实验执行或记录保存异常，未发布报告。请检查服务日志后重新运行。')
        finally:
            record.update(updated_at=now(), elapsed_ms=round((monotonic() - started) * 1000))
            try:
                await self.save_terminal(record)
            finally:
                self.jobs.pop(run_id, None)

    async def cancel(self, run_id):
        record = self.read(run_id)
        task = self.jobs.get(run_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.jobs.pop(run_id, None)
            record = self.read(run_id)
            # Covers cancellation before the coroutine's first instruction.
            if record['status'] == 'running':
                record.update(status='interrupted' if self.closing else 'cancelled', updated_at=now(), error='执行已停止，未发布报告。')
                await self.save_terminal(record)
        return record

    async def close(self):
        self.closing = True
        for run_id in list(self.jobs):
            await self.cancel(run_id)

    def artifact(self, run_id, name):
        if name not in ('report.md', 'report.csv'):
            raise HTTPException(404, '报告不存在。')
        record = self.read(run_id)
        if record['status'] != 'completed' or not record.get('result', {}).get('verification', {}).get('passed'):
            raise HTTPException(409, '只有完成并通过校验的报告才能下载。')
        path = self.directory(run_id) / name
        if not path.is_file() or path.resolve().parent != self.directory(run_id):
            raise HTTPException(404, '报告不存在。')
        return path


def register_code_routes(app, project_dir):
    lock = asyncio.Lock()
    async def service():
        async with lock:
            if getattr(app.state, 'code_experiments', None) is None:
                db_path = app.state.store.db_path
                root = project_dir / '.local/code-agent' if db_path.resolve() == (project_dir / 'data/chat.db').resolve() else db_path.parent / 'code-agent'
                app.state.code_experiments = await CodeExperiments(app, root).open()
            return app.state.code_experiments

    @app.get('/code', include_in_schema=False)
    async def code_page():
        return FileResponse(project_dir / 'web/code.html')

    @app.get('/api/code/config')
    async def config():
        status = sandbox_status()
        status.pop('binary')
        return {'sandbox': status, 'configured': app.state.client is not None,
                'model': app.state.model, 'fields': FIELDS, 'datasets': [
                    {'id': key, 'title': title, 'rows': len((dataset := load_dataset(key))['rows']), 'preview': dataset['rows'][:5]}
                    for key, title in DATASETS.items()]}

    @app.get('/api/code/runs')
    async def list_runs():
        return {'runs': (await service()).list()}

    @app.post('/api/code/runs', status_code=202)
    async def create_run(body: CodeQuestion):
        return await (await service()).create(body)

    @app.get('/api/code/runs/{run_id}')
    async def read_run(run_id: UUID):
        return (await service()).read(str(run_id))

    @app.post('/api/code/runs/{run_id}/cancel')
    async def cancel_run(run_id: UUID):
        return await (await service()).cancel(str(run_id))

    @app.get('/api/code/runs/{run_id}/artifacts/{name}')
    async def artifact(run_id: UUID, name: str):
        path = (await service()).artifact(str(run_id), name)
        return FileResponse(path, filename=name, media_type='text/csv' if name.endswith('.csv') else 'text/plain',
                            headers={'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store',
                                     'Content-Security-Policy': "default-src 'none'; sandbox"})
