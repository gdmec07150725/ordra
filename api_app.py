"""第六至九课：HTTP/SSE、SQLite 持久化、上下文管理与历史检索。"""

from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from html import escape
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4
from typing import Literal

import anyio
import httpx2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_service import describe_error, new_conversation, stream_agent
from session_store import SessionStore
from context_memory import ContextMemory, ContextPolicy
from local_embeddings import LocalEmbeddings
from knowledge_base import KnowledgeBase, KnowledgeReadError, KnowledgeVersionChanged
from return_requests import ReturnRequestStore
from tool_runtime import ToolPolicy
from checkpoint_store import CheckpointStore, CheckpointConflict, public_checkpoint
from human_approval import APPROVAL_POLICIES, HumanDecision
from trace_store import TraceStore
from agent_recovery import MAX_STALLED_BATCHES
from result_verification import verify_result
from graph_experiments import register_graph_routes
from code_experiments import register_code_routes


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "chat.db"
STORAGE_ERROR = "会话数据库读写失败，本次操作未确认完成。请稍后重试，刷新后以已保存的历史为准。"


def knowledge_page(title: str, content: str, status_code: int = 200):
    # 资料内容只作为转义文本展示，不能执行 HTML、脚本或加载外部资源。
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + escape(title) + """</title><style>
body{margin:0;background:#fcfdfb;color:#233e35;font:16px/1.9 system-ui,sans-serif}
main{max-width:800px;margin:32px auto;padding:0 24px}a{color:#315642}
h1{font-size:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;
background:#f0f4e9;border-radius:12px;padding:24px}
</style><main><a href="/">返回对话</a><h1>""" + escape(title) + "</h1><pre>" + escape(content) + "</pre></main></html>"
    return HTMLResponse(page, status_code=status_code, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"})


class ChatRequest(BaseModel):
    """类似 TS 请求类型，但 Pydantic 还会在运行时检查传入的 JSON。"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)


class ReturnConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmed: bool


class RunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    run_id: str = Field(min_length=1, max_length=64)


class ReturnCancellation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReturnReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    session_id: str = Field(min_length=1)
    decision: Literal['approved', 'rejected']
    reason: str = Field(min_length=1, max_length=500)


@dataclass
class Session:
    # default_factory 每次创建一个新列表，不能直接把 [] 当作共享默认值。
    history: list = field(default_factory=new_conversation)
    memory: ContextMemory = field(default_factory=ContextMemory)
    created_at: str = ""
    busy: bool = False


async def run_until_disconnect(work, receive):
    """请求工作与断连监听同时运行，先结束的一方决定本次等待何时停止。

    与 SDK 的网络层一样使用 AnyIO 取消范围；取消持续生效，直到工作退出。
    这段资源管理可以在理解聊天路由之后再读，关闭连接时另用 shield 保护。
    """
    result, failure = None, None
    async with anyio.create_task_group() as group:
        async def run_work():
            nonlocal result, failure  # 更新外层函数的变量，不创建同名局部变量。
            try:
                result = await work
            except Exception as error:
                failure = error
            finally:
                group.cancel_scope.cancel()

        group.start_soon(run_work)
        while True:
            if (await receive())["type"] == "http.disconnect":
                group.cancel_scope.cancel()
                break
    if failure is not None:
        raise failure
    return result


class SessionStreamResponse(StreamingResponse):
    """把流的关闭和会话解锁绑定到整个 HTTP 响应的生命周期。

    这部分可在理解主流程后再读：断连时取消网络等待，即使还没有第一个片段，
    也要关闭生成器并释放 busy，避免刷新页面后会话一直处于处理中。
    """
    def __init__(self, content, session: Session, on_close=None):
        super().__init__(content, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        self.session = session
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        try:
            # 一边发送流，一边监听浏览器断开；任意一边结束，就取消另一边。
            await run_until_disconnect(self.stream_response(send), receive)
        finally:
            # shield 允许清理完成，即使上层任务正在被取消。
            try:
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()
            finally:
                try:
                    if self.on_close is not None:
                        with anyio.CancelScope(shield=True):
                            await self.on_close()
                finally:
                    self.session.busy = False


def create_app(db_path: Path | None = None, *, knowledge_dir: Path | None = None, approval_tools: frozenset[str] | None = None, result_verification: bool = True, delegation_enabled: bool = True, tool_mode: str | None = None, skills_dir: Path | None = None, skills_enabled: bool = True) -> FastAPI:
    """默认写入项目 data/chat.db；测试传入临时路径，避免触碰真实记录。"""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_dotenv(PROJECT_DIR / ".env", override=False, encoding="utf-8-sig")
        app.state.context_policy = ContextPolicy.from_env()
        app.state.tool_policy = ToolPolicy()
        app.state.result_verification = result_verification
        app.state.delegation_enabled = delegation_enabled
        app.state.skills_dir = (skills_dir if skills_dir is not None else PROJECT_DIR / 'skills') if skills_enabled else None
        app.state.tool_mode = tool_mode if tool_mode is not None else os.getenv('AGENT_TOOL_MODE', 'mcp').strip().lower()
        if app.state.tool_mode not in ('local', 'mcp'):
            raise ValueError('AGENT_TOOL_MODE 只能是 local 或 mcp。')
        app.state.mcp_client = None
        app.state.approval_tools = frozenset(APPROVAL_POLICIES) if approval_tools is None else approval_tools
        if app.state.approval_tools - APPROVAL_POLICIES.keys():
            raise ValueError('审批工具必须在服务器策略中注册。')
        app.state.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        app.state.model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash"
        search_mode = os.getenv("AGENT_HISTORY_SEARCH", "hybrid").strip().lower()
        if search_mode not in ("hybrid", "keywords"):
            raise ValueError("AGENT_HISTORY_SEARCH 只能是 hybrid 或 keywords。")
        embeddings = LocalEmbeddings(PROJECT_DIR / ".local" / "embeddings") if search_mode == "hybrid" else None
        app.state.store = SessionStore(db_path if db_path is not None else DEFAULT_DB_PATH, semantic_search=embeddings)
        app.state.checkpoints = CheckpointStore(app.state.store.db_path)
        app.state.traces = TraceStore(app.state.store.db_path)
        app.state.returns = ReturnRequestStore(app.state.store.db_path)
        app.state.knowledge_base = KnowledgeBase(knowledge_dir if knowledge_dir is not None else PROJECT_DIR / "knowledge", semantic_search=embeddings)
        await anyio.to_thread.run_sync(app.state.store.initialize)
        await anyio.to_thread.run_sync(app.state.traces.recover_running)
        if embeddings is not None:
            await anyio.to_thread.run_sync(embeddings.warm_up)
        # 内存只做按需加载的缓存与 busy 协调；已完成历史以磁盘数据库为准。
        app.state.sessions = {}
        app.state.session_load_lock = anyio.Lock()
        app.state.graph_experiments = None
        app.state.code_experiments = None
        app.state.graph_experiment_lock = anyio.Lock()
        app.state.client = AsyncOpenAI(
            api_key=app.state.api_key, base_url="https://api.deepseek.com",
            timeout=30.0, max_retries=0,
        ) if app.state.api_key else None
        try:
            if app.state.tool_mode == 'mcp':
                from mcp_tool_client import LocalMCPClient
                app.state.mcp_client = LocalMCPClient(app.state.knowledge_base.directory,
                    search_mode=search_mode, embeddings_dir=PROJECT_DIR / '.local' / 'embeddings')
                await app.state.mcp_client.start()
            # 主图与细粒度业务进度使用同一数据库、不同 namespace。
            # saver 初始化在请求事务之外完成，连接随应用生命周期关闭。
            async with AsyncSqliteSaver.from_conn_string(str(app.state.store.db_path)) as checkpointer:
                await checkpointer.setup()
                app.state.graph_checkpointer = checkpointer
                yield
        finally:
            if app.state.code_experiments is not None:
                await app.state.code_experiments.close()
            if app.state.mcp_client is not None:
                await app.state.mcp_client.close()
            if app.state.graph_experiments is not None:
                await app.state.graph_experiments.close()
            if app.state.client is not None:
                await app.state.client.close()

    app = FastAPI(title="Ordra 订单客服接口", lifespan=lifespan)
    # 只提供 web 文件夹；绝不能把包含 .env 的项目根目录挂成静态目录。
    app.mount("/static", StaticFiles(directory=PROJECT_DIR / "web"), name="static")
    register_graph_routes(app, PROJECT_DIR)
    register_code_routes(app, PROJECT_DIR)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request: Request, error: sqlite3.Error):
        # 普通 JSON 接口的数据库错误统一转换；SSE 开始后的错误单独编码为事件。
        return JSONResponse(status_code=503, content={"detail": STORAGE_ERROR})

    @app.exception_handler(CheckpointConflict)
    async def checkpoint_conflict(request: Request, error: CheckpointConflict):
        return JSONResponse(status_code=409, content={'detail': str(error)})

    async def get_session(session_id: str) -> Session:
        session = app.state.sessions.get(session_id)
        if session is None:
            # 冷加载与删除共用锁，避免较慢的旧读取在删除后重新放回缓存。
            async with app.state.session_load_lock:
                session = app.state.sessions.get(session_id)
                if session is None:
                    stored = await anyio.to_thread.run_sync(app.state.store.load_session, session_id)
                    if stored is None:
                        raise HTTPException(status_code=404, detail="会话不存在，请新建对话。")
                    session = Session(history=stored["history"], memory=stored["memory"], created_at=stored["created_at"])
                    app.state.sessions[session_id] = session
        return session

    async def acquire_session(body: ChatRequest) -> Session:
        session = await get_session(body.session_id)
        if app.state.client is None:
            raise HTTPException(status_code=503, detail="请先在服务端 .env 中配置 DEEPSEEK_API_KEY，然后重启服务。")
        if session.busy:
            raise HTTPException(status_code=409, detail="这段对话正在生成回复，请稍后再发送。")
        # 检查和赋值之间没有 await；本课单进程事件循环中不会插入另一个请求。
        session.busy = True
        return session

    async def persist_history(session_id: str, session: Session, history: list, memory: ContextMemory | None = None, return_previews: list | None = None, checkpoint_id: str | None = None) -> None:
        # 在开始保存前允许断连取消。事务开始后必须等待它结束，
        # 不能让后台写库线程继续运行，而接口先释放 busy、开始另一个问题。
        await anyio.lowlevel.checkpoint()
        with anyio.CancelScope(shield=True):
            try:
                await anyio.to_thread.run_sync(app.state.store.replace_history, session_id, history, memory, return_previews or [], checkpoint_id)
            except sqlite3.Error:
                if checkpoint_id:
                    await record_stop(session_id, checkpoint_id, 'failed', STORAGE_ERROR)
                raise
            session.history = history  # 只有数据库提交成功，才同步更新缓存。
            session.memory = memory or ContextMemory()

    async def prepare_run(body: ChatRequest, resume_id: str | None = None):
        session = await acquire_session(body)
        try:
            with anyio.CancelScope(shield=True):
                if resume_id is None:
                    run = await anyio.to_thread.run_sync(app.state.checkpoints.begin, body.session_id, body.message, session.history)
                else:
                    run = await anyio.to_thread.run_sync(app.state.checkpoints.claim, body.session_id, resume_id)
            return session, run
        except BaseException:
            session.busy = False
            raise

    async def record_stop(session_id, run_id, status, message):
        try:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(app.state.traces.stop_running, session_id, run_id, status, message)
        except sqlite3.Error:
            # 原数据库故障仍交给业务错误处理；不让诊断记录异常覆盖原始错误。
            pass

    async def agent_events(session_id, session, run):
        async def save_checkpoint(state):
            await anyio.lowlevel.checkpoint()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(app.state.checkpoints.save, session_id, run['run_id'], state)
        async def review(**kwargs):
            return await verify_result(app.state.client, app.state.model, **kwargs)
        events = stream_agent(app.state.client, app.state.model, session.history, run['question'],
            policy=app.state.context_policy, memory=session.memory, search_history=history_reader(session_id),
            search_knowledge=knowledge_reader, return_handler=return_handler(session_id), tool_policy=app.state.tool_policy,
            checkpoint=run['state'], save_checkpoint=save_checkpoint, approval_tools=app.state.approval_tools,
            result_verifier=review if app.state.result_verification else None, delegation_enabled=app.state.delegation_enabled,
            graph_checkpointer=app.state.graph_checkpointer, graph_run_id=run['run_id'], mcp_client=app.state.mcp_client,
            skills_dir=app.state.skills_dir)
        try:
            async for event in events:
                yield event
        except Exception as error:
            await record_stop(session_id, run['run_id'], 'failed', STORAGE_ERROR if isinstance(error, sqlite3.Error) else describe_error(error, app.state.api_key))
            raise
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await events.aclose()
            finally:
                await record_stop(session_id, run['run_id'], 'interrupted', '本次连接已结束，任务尚未提交完成。可检查保存的进度。')

    async def require_no_pending_run(session_id):
        if await anyio.to_thread.run_sync(app.state.checkpoints.load, session_id) is not None:
            raise CheckpointConflict('这段对话有未完成的执行，请先继续或放弃，再操作申请。')

    def history_reader(session_id: str):
        """闭包绑定当前会话编号；模型只看得到 query/order，不能指定别的会话。"""
        async def search(query: str, order: str) -> dict:
            # 只读工作可在超时/断连时停止等待；迟到结果丢弃，不再追加聊天历史。
            # Python 不能强制终止已运行的线程，因此执行器不自动重试超时。
            return await anyio.to_thread.run_sync(app.state.store.search_history, session_id, query, order, abandon_on_cancel=True)
        return search

    async def knowledge_reader(query: str):
        return await anyio.to_thread.run_sync(app.state.knowledge_base.search, query, abandon_on_cancel=True)

    def return_handler(session_id: str):
        async def handle(name: str, arguments: dict):
            if name == "prepare_return_request":
                return await anyio.to_thread.run_sync(app.state.returns.prepare_return_request, session_id, arguments['order_id'], arguments['reason'], abandon_on_cancel=True)
            return await anyio.to_thread.run_sync(app.state.returns.get_return_request, session_id, arguments['order_id'], abandon_on_cancel=True)
        return handle

    async def change_return(session_id: str, preview_id: str, *, confirm: bool):
        session = await get_session(session_id)
        if session.busy:
            raise HTTPException(status_code=409, detail="这段对话正在处理中，请稍后再操作申请。")
        session.busy = True
        try:
            await require_no_pending_run(session_id)
            # 用户点击后的短事务完成前保持会话锁；确认与回执同一次提交。
            with anyio.CancelScope(shield=True):
                operation = app.state.returns.create_return_request if confirm else app.state.returns.cancel_preview
                result = await anyio.to_thread.run_sync(operation, session_id, preview_id)
                saved = await anyio.to_thread.run_sync(app.state.store.load_session, session_id)
                session.history, session.memory = saved['history'], saved['memory']
            return {"request": result}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except sqlite3.Error:
            # 提交可能已经完成；不能让旧缓存随后覆盖已落盘的确认与回执。
            app.state.sessions.pop(session_id, None)
            raise
        finally:
            session.busy = False

    @app.post("/api/sessions/{session_id}/return-previews/{preview_id}/confirm")
    async def confirm_return(session_id: str, preview_id: str, body: ReturnConfirmation):
        if not body.confirmed:
            raise HTTPException(status_code=422, detail="只有明确确认才能提交申请。")
        return await change_return(session_id, preview_id, confirm=True)

    @app.post("/api/sessions/{session_id}/return-previews/{preview_id}/cancel")
    async def cancel_return(session_id: str, preview_id: str, body: ReturnCancellation):
        return await change_return(session_id, preview_id, confirm=False)

    @app.get('/returns')
    async def return_review_page():
        return FileResponse(PROJECT_DIR / 'web' / 'returns.html')

    @app.get('/api/returns')
    async def list_returns(status: Literal['pending', 'approved', 'rejected', 'all'] = 'pending'):
        # 本机人工审核队列可以列出各会话申请；模型的查询工具仍绑定当前会话。
        return await anyio.to_thread.run_sync(app.state.returns.list_requests, status)

    @app.post('/api/returns/{request_id}/review')
    async def review_return(request_id: str, body: ReturnReview):
        session = await get_session(body.session_id)
        if session.busy:
            raise HTTPException(status_code=409, detail='这段对话正在处理中，请回复结束后再审核。')
        session.busy = True
        try:
            await require_no_pending_run(body.session_id)
            with anyio.CancelScope(shield=True):
                result = await anyio.to_thread.run_sync(app.state.returns.review_request, body.session_id, request_id, body.decision, body.reason)
            return {'request': result}
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        finally:
            session.busy = False

    @app.get("/api/knowledge/{document_id}")
    async def knowledge_document(document_id: str, version: str | None = None):
        try:
            document = await anyio.to_thread.run_sync(app.state.knowledge_base.read_document, document_id, version)
        except KnowledgeVersionChanged as error:
            return knowledge_page("资料版本已变化", str(error), 409)
        except KnowledgeReadError as error:
            return knowledge_page("暂时无法读取资料", str(error), 503)
        if document is None:
            return knowledge_page("资料不存在", "资料不存在或已移除。", 404)
        return knowledge_page(document["title"], document["content"])

    @app.get("/")
    async def home():
        return FileResponse(PROJECT_DIR / "web" / "index.html")

    @app.get('/runs')
    async def run_page():
        return FileResponse(PROJECT_DIR / 'web' / 'runs.html')

    @app.get('/api/sessions/{session_id}/runs')
    async def list_runs(session_id: str, before: int | None = Query(default=None, ge=1)):
        await get_session(session_id)
        return await anyio.to_thread.run_sync(app.state.traces.list_runs, session_id, before)

    @app.get('/api/sessions/{session_id}/runs/{run_id}')
    async def read_run(session_id: str, run_id: str):
        await get_session(session_id)
        trace = await anyio.to_thread.run_sync(app.state.traces.load, session_id, run_id)
        if trace is None:
            raise HTTPException(status_code=404, detail='执行记录不存在或不属于这段对话。')
        return trace

    @app.get("/api/health")
    async def health():
        policy = app.state.context_policy
        embeddings = app.state.store.semantic_search
        semantic_status = embeddings.status if embeddings is not None else "disabled"
        return {"status": "ok", "agent_engine": "langgraph", "model": app.state.model, "configured": bool(app.state.api_key), "storage": "sqlite", "context_strategy": "token_budget_summary", "context_tokens": policy.max_tokens, "recent_tokens": policy.recent_tokens, "summary_tokens": policy.summary_tokens, "history_search": "hybrid" if semantic_status == "ready" else "keywords", "semantic_status": semantic_status, "knowledge_base": "local_files", "result_verification": app.state.result_verification, "delegation_enabled": app.state.delegation_enabled,
                "tool_mode": app.state.tool_mode, "tool_discovery": app.state.tool_mode == 'mcp',
                "dynamic_replanning": True, "max_stalled_batches": MAX_STALLED_BATCHES,
                "skills_enabled": app.state.skills_dir is not None, "max_skill_loads": 3,
                "skill_resources_enabled": app.state.skills_dir is not None, "max_skill_resource_reads": 4}

    @app.post("/api/sessions", status_code=201)
    async def create_session():
        session_id = str(uuid4())
        stored = await anyio.to_thread.run_sync(app.state.store.create_session, session_id, new_conversation())
        app.state.sessions[session_id] = Session(history=stored["history"], created_at=stored["created_at"])
        return {"session_id": session_id, "created_at": stored["created_at"]}

    @app.get("/api/sessions")
    async def list_sessions():
        # 列表来自磁盘，因此丢失浏览器编号或重启 Python 后也能找回旧会话。
        sessions = await anyio.to_thread.run_sync(app.state.store.list_sessions)
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}")
    async def read_session(session_id: str, include_checkpoint: bool = False):
        session = await get_session(session_id)
        try:
            previews = await anyio.to_thread.run_sync(app.state.returns.list_previews, session_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail="会话不存在，请新建对话。") from error
        recovery = {'checkpoint': public_checkpoint(await anyio.to_thread.run_sync(app.state.checkpoints.load, session_id))} if include_checkpoint else {}
        return {"session_id": session_id, "messages": session.history, "busy": session.busy, "created_at": session.created_at, "return_previews": previews, **recovery}

    @app.get('/api/sessions/{session_id}/checkpoint')
    async def read_checkpoint(session_id: str):
        session = await get_session(session_id)
        run = await anyio.to_thread.run_sync(app.state.checkpoints.load, session_id)
        return {'checkpoint': public_checkpoint(run), 'busy': session.busy}

    @app.post('/api/sessions/{session_id}/checkpoint/discard', status_code=204)
    async def discard_checkpoint(session_id: str, body: RunRequest):
        session = await get_session(session_id)
        if session.busy:
            raise CheckpointConflict('任务正在执行，请等待结束后再放弃。')
        session.busy = True
        try:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(app.state.checkpoints.discard, session_id, body.run_id)
        finally:
            session.busy = False
        return Response(status_code=204)

    @app.post('/api/sessions/{session_id}/checkpoint/decision')
    async def decide_tool(session_id: str, body: HumanDecision):
        session = await get_session(session_id)
        if session.busy:
            raise CheckpointConflict('任务正在执行，请等待后再处理审批。')
        session.busy = True
        try:
            with anyio.CancelScope(shield=True):
                run = await anyio.to_thread.run_sync(app.state.checkpoints.decide, session_id, body)
            return {'checkpoint': public_checkpoint(run)}
        except CheckpointConflict:
            raise
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            session.busy = False

    async def resume_body(session_id, body):
        await get_session(session_id)
        run = await anyio.to_thread.run_sync(app.state.checkpoints.load, session_id)
        if run is None or run['run_id'] != body.run_id:
            raise CheckpointConflict('执行记录已变化或已完成，请刷新页面。')
        return ChatRequest(session_id=session_id, message=run['question'])

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        session = await get_session(session_id)
        if session.busy:
            raise HTTPException(status_code=409, detail="正在处理这段对话，请完成后再删除。")
        session.busy = True
        try:
            async with app.state.session_load_lock:
                await anyio.lowlevel.checkpoint()
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(app.state.store.delete_session, session_id)
                    app.state.sessions.pop(session_id, None)  # 提交成功后才移除缓存。
        finally:
            session.busy = False
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        session = await get_session(session_id)
        if session.busy:
            raise HTTPException(status_code=409, detail="正在生成回复，请完成后再清空对话。")
        session.busy = True  # 清空也会等待数据库，必须与聊天使用同一忙碌标记。
        try:
            await persist_history(session_id, session, new_conversation())
        finally:
            session.busy = False
        return {"session_id": session_id, "messages": session.history, "busy": False}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        return await respond_json(body, request)

    @app.post('/api/sessions/{session_id}/resume')
    async def resume(session_id: str, body: RunRequest, request: Request):
        return await respond_json(await resume_body(session_id, body), request, body.run_id)

    async def respond_json(body, request, resume_id=None):
        session, run = await prepare_run(body, resume_id)
        events = agent_events(body.session_id, session, run)

        async def collect_answer():
            # 两个接口使用同一个 Agent：这里收齐事件，只返回最终 JSON。
            async for event in events:
                if event['event'] == 'paused':
                    return {**event, 'checkpoint': public_checkpoint(await anyio.to_thread.run_sync(app.state.checkpoints.load, body.session_id))}
                if event["event"] == "complete":
                    # 保存也在断连监听范围内；SDK 关闭连接后仍需检查取消。
                    await persist_history(body.session_id, session, event["history"], event["memory"], event["return_previews"], run['run_id'])
                    return event

        try:
            completed = await run_until_disconnect(collect_answer(), request.receive)
            if completed is None:
                raise HTTPException(status_code=499, detail="连接已断开，本轮未保存。")
            if completed['event'] == 'paused':
                return {'session_id': body.session_id, 'status': completed['status'], 'checkpoint': completed['checkpoint']}
            return {"session_id": body.session_id, "answer": completed["answer"], "usage": completed["usage"], "context": completed["context"], "retrieval": completed["retrieval"], "knowledge": completed["knowledge"], "sources": completed["sources"], "return_previews": completed['return_previews'], "execution": completed['execution'], 'plan': completed['plan'], 'human_decisions': completed['human_decisions'], 'verification': completed['verification'], 'delegation': completed['delegation'], 'recovery': completed['recovery']}
        except (APIError, httpx2.HTTPError, ValueError) as error:
            raise HTTPException(status_code=502, detail=describe_error(error, app.state.api_key)) from error
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await events.aclose()
            finally:
                try:
                    await record_stop(body.session_id, run['run_id'], 'interrupted', '本次连接已结束，任务尚未提交完成。')
                finally:
                    session.busy = False

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest):
        return await respond_stream(body)

    @app.post('/api/sessions/{session_id}/resume/stream')
    async def resume_stream(session_id: str, body: RunRequest):
        return await respond_stream(await resume_body(session_id, body), body.run_id)

    async def respond_stream(body, resume_id=None):
        session, run = await prepare_run(body, resume_id)

        async def event_body():
            try:
                async with aclosing(agent_events(body.session_id, session, run)) as events:
                    async for event in events:
                        name = event["event"]
                        if name == "complete":
                            # 第七课先落盘再发 done，保存失败就不能通知网页成功。
                            await persist_history(body.session_id, session, event["history"], event["memory"], event["return_previews"], run['run_id'])
                            name = "done"
                            payload = {"answer": event["answer"], "usage": event["usage"], "session_id": body.session_id, "context": event["context"], "retrieval": event["retrieval"], "knowledge": event["knowledge"], "sources": event["sources"], "return_previews": event['return_previews'], "execution": event['execution'], 'plan': event['plan'], 'human_decisions': event['human_decisions'], 'verification': event['verification'], 'delegation': event['delegation'], 'recovery': event['recovery']}
                        else:
                            payload = {key: value for key, value in event.items() if key != "event"}
                            if name == 'paused':
                                payload['checkpoint'] = public_checkpoint(await anyio.to_thread.run_sync(app.state.checkpoints.load, body.session_id))
                        # 双换行分隔事件。ensure_ascii=False 让中文在网络面板中可读。
                        yield f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except sqlite3.Error:
                payload = {"message": STORAGE_ERROR}
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except (APIError, httpx2.HTTPError, ValueError) as error:
                # 响应开始后无法再改 HTTP 状态码，改用 error 事件通知网页。
                payload = {"message": describe_error(error, app.state.api_key)}
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def on_close():
            await record_stop(body.session_id, run['run_id'], 'interrupted', '本次连接已结束，任务尚未提交完成。')

        return SessionStreamResponse(event_body(), session, on_close)

    return app


app = create_app()
