"""独立的 LangGraph 实验入口；图检查点与实验草稿均不写入正式聊天库。"""
import asyncio
from contextlib import AsyncExitStack, closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import UUID, uuid4

import anyio
from fastapi import HTTPException
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from return_requests import ReturnRequestStore
from return_tools import validate_return_arguments
from session_store import SessionStore


def public_error(error):
    """提供方错误体可能回显候选或消息；公开接口只给固定分类。"""
    if isinstance(error, APITimeoutError):
        return '模型请求超时，可从最近保存的节点继续。'
    if isinstance(error, APIConnectionError):
        return '暂时无法连接模型服务，可从最近保存的节点继续。'
    if isinstance(error, APIStatusError):
        return f'模型服务返回 HTTP {error.status_code}，可从最近保存的节点继续。'
    if isinstance(error, sqlite3.Error):
        return '保存实验进度时发生数据库错误，请检查本机存储后重试。'
    return '执行出现异常，可从最近保存的节点继续。'


class GraphQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000, pattern=r'\S')

    @model_validator(mode='after')
    def encodable(self):
        self.question.encode('utf8')
        if '\x00' in self.question:
            raise ValueError('问题不能包含空字符。')
        return self


class GraphResume(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    approval_id: str | None = Field(default=None, min_length=1, max_length=100)
    decision: Literal['approve', 'edit', 'reject'] | None = None
    arguments: dict | None = None
    reason: str = Field(default='', max_length=500)

    @model_validator(mode='after')
    def coherent(self):
        if self.decision is None:
            if self.approval_id is not None or self.arguments is not None or self.reason:
                raise ValueError('普通继续不包含审批参数。')
        elif self.approval_id is None or ((self.decision == 'edit') != (self.arguments is not None)):
            raise ValueError('审批需要有效 ID，只有修改并批准时提供完整参数。')
        json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False).encode('utf8')
        return self


class GraphExperiments:
    def __init__(self, app, path):
        self.app, self.path = app, Path(path).resolve()
        self.store = SessionStore(self.path)
        self.returns = ReturnRequestStore(self.path)
        self.stack = AsyncExitStack()
        self.jobs = {}
        self.lock = asyncio.Lock()
        self.closing = False

    def _sql(self, sql, arguments=(), *, many=False):
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            cursor = db.execute(sql, arguments)
            return [dict(row) for row in cursor.fetchall()] if many else None

    async def open(self):
        await anyio.to_thread.run_sync(self.store.initialize)
        await anyio.to_thread.run_sync(self._sql, '''CREATE TABLE IF NOT EXISTS graph_runs (
            run_id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '', resume_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        await anyio.to_thread.run_sync(self._sql,
            "UPDATE graph_runs SET status='interrupted', error='服务上次退出时执行尚未完成，可从保存的节点继续。' WHERE status='running'")
        try:
            self.saver = await self.stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(self.path)))
            await self.saver.setup()
        except BaseException:
            await self.stack.aclose()
            raise
        return self

    async def close(self):
        self.closing = True
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.stack.aclose()

    def agent(self, run_id):
        # 依赖只在实验请求时导入；原聊天无需初始化图运行时。
        from langgraph_agent import GraphAgent
        async def knowledge(query):
            return await anyio.to_thread.run_sync(self.app.state.knowledge_base.search, query, abandon_on_cancel=True)
        async def returns(name, arguments):
            if name != 'prepare_return_request':
                return {'error': '实验入口仅支持生成预览。'}
            return await anyio.to_thread.run_sync(self.returns.prepare_return_request,
                run_id, arguments['order_id'], arguments['reason'], abandon_on_cancel=True)
        return GraphAgent(self.app.state.client, self.app.state.model, self.saver,
            search_knowledge=knowledge, return_handler=returns, policy=self.app.state.context_policy,
            tool_policy=self.app.state.tool_policy)

    async def rows(self):
        return await anyio.to_thread.run_sync(lambda: self._sql(
            'SELECT * FROM graph_runs ORDER BY created_at DESC, run_id DESC LIMIT 100', many=True))

    async def require(self, run_id):
        rows = await anyio.to_thread.run_sync(lambda: self._sql(
            'SELECT * FROM graph_runs WHERE run_id=?', (run_id,), many=True))
        if not rows:
            raise HTTPException(status_code=404, detail='实验不存在。')
        return rows[0]

    async def set_status(self, run_id, status, error=''):
        await anyio.to_thread.run_sync(self._sql,
            'UPDATE graph_runs SET status=?,error=?,updated_at=? WHERE run_id=?',
            (status, error, datetime.now(UTC).isoformat(), run_id))

    async def detail(self, run_id):
        async with self.lock:
            # 若读取期间后台任务刚好结束，本次仍报告忙碌，让下次轮询
            # 读取一致的终态；避免旧索引配上 busy=False 而失去恢复按钮。
            was_busy = run_id in self.jobs
            return await self._detail(run_id, was_busy)

    async def _detail(self, run_id, was_busy):
        row = await self.require(run_id)
        public = await self.agent(run_id).get(run_id) or {
            'status': 'running', 'answer': '', 'approvals': [], 'events': [], 'metrics': {},
            'sources': [], 'return_previews': [], 'next': []}
        busy = was_busy or run_id in self.jobs
        status = 'running' if busy else row['status'] if row['status'] in ('failed', 'interrupted') else public['status']
        # 成功写入图状态后、更新索引前退出，也可直接恢复已完成的公开结果。
        if not busy and public['status'] in ('completed', 'unverified'):
            status = public['status']
        return {**public, **{key: row[key] for key in ('run_id', 'question', 'created_at', 'updated_at', 'resume_count')},
            'status': status, 'busy': busy, 'error': row['error'] if status in ('failed', 'interrupted') else '',
            'can_resume': not busy and status in ('failed', 'interrupted') and row['resume_count'] < 3}

    def launch(self, run_id, *, question=None, decision=None):
        async def work():
            try:
                result = await self.agent(run_id).run(run_id, question=question, decision=decision)
                await self.set_status(run_id, result['status'])
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    await self.set_status(run_id, 'interrupted', '执行已中断，可从保存的节点继续。')
                raise
            except Exception as error:
                await self.set_status(run_id, 'failed', public_error(error))
            finally:
                self.jobs.pop(run_id, None)
        self.jobs[run_id] = asyncio.create_task(work(), name=f'langgraph:{run_id}')

    async def create(self, question):
        if self.app.state.client is None:
            raise HTTPException(status_code=503, detail='请先配置聊天模型。')
        async with self.lock:
            if self.closing:
                raise HTTPException(status_code=503, detail='服务正在关闭。')
            run_id, now = str(uuid4()), datetime.now(UTC).isoformat()
            await anyio.to_thread.run_sync(self.store.create_session, run_id, [])
            await anyio.to_thread.run_sync(self._sql,
                'INSERT INTO graph_runs(run_id,question,status,created_at,updated_at) VALUES (?,?,?,?,?)',
                (run_id, question, 'running', now, now))
            self.launch(run_id, question=question)
            return {'run_id': run_id, 'question': question, 'status': 'running'}

    async def resume(self, run_id, body):
        if self.app.state.client is None:
            raise HTTPException(status_code=503, detail='请先配置聊天模型。')
        async with self.lock:
            if self.closing or run_id in self.jobs:
                raise HTTPException(status_code=409, detail='实验正在运行或服务正在关闭。')
            row = await self.require(run_id)
            state = await self.agent(run_id).get(run_id)
            if state and state['status'] in ('completed', 'unverified'):
                raise HTTPException(status_code=409, detail='本次实验已结束，请新建实验。')
            approvals = (state or {}).get('approvals', [])
            decision = None
            if approvals:
                item = next((a for a in approvals if a['approval_id'] == body.approval_id and not a.get('decision')), None)
                if body.decision is None or item is None:
                    raise HTTPException(status_code=409, detail='需要处理当前有效的审批，旧决定不能重用。')
                if body.decision == 'edit':
                    try:
                        validate_return_arguments(item['name'], body.arguments)
                    except (ValueError, TypeError) as error:
                        raise HTTPException(status_code=422, detail=str(error)) from None
                decision = body.model_dump(exclude_none=True)
            else:
                if body.decision is not None:
                    raise HTTPException(status_code=409, detail='当前没有待处理的审批。')
                if row['status'] not in ('failed', 'interrupted') or row['resume_count'] >= 3:
                    raise HTTPException(status_code=409, detail='本次实验不能继续或已达到三次恢复上限。')
                await anyio.to_thread.run_sync(self._sql,
                    'UPDATE graph_runs SET resume_count=resume_count+1 WHERE run_id=?', (run_id,))
            await self.set_status(run_id, 'running')
            self.launch(run_id, question=row['question'] if state is None else None, decision=decision)
            return {'run_id': run_id, 'status': 'running'}


def register_graph_routes(app, project_dir):
    async def service():
        async with app.state.graph_experiment_lock:
            if app.state.graph_experiments is None:
                path = app.state.store.db_path.with_name('langgraph.db')
                if path == app.state.store.db_path:
                    path = path.with_name('langgraph-experiments.db')
                app.state.graph_experiments = await GraphExperiments(app, path).open()
            return app.state.graph_experiments

    @app.get('/graph', include_in_schema=False)
    async def graph_page():
        return FileResponse(project_dir / 'web' / 'graph.html')

    @app.get('/api/graph/runs')
    async def list_graph_runs():
        manager = await service()
        return {'runs': await manager.rows(), 'configured': app.state.client is not None, 'model': app.state.model}

    @app.post('/api/graph/runs', status_code=202)
    async def start_graph_run(body: GraphQuestion):
        return await (await service()).create(body.question)

    @app.get('/api/graph/runs/{run_id}')
    async def read_graph_run(run_id: UUID):
        return await (await service()).detail(str(run_id))

    @app.post('/api/graph/runs/{run_id}/resume', status_code=202)
    async def resume_graph_run(run_id: UUID, body: GraphResume):
        return await (await service()).resume(str(run_id), body)
