"""Shared local MCP client with one task owning every SDK enter/exit pair."""
import asyncio
from copy import deepcopy
import math
from pathlib import Path
import sys

import anyio
from jsonschema import Draft202012Validator, ValidationError
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT


ALLOWED_TOOLS = frozenset({"get_order", "search_knowledge"})
_CLOSE_TIMEOUT = 8.0  # SDK stdio shutdown has its own bounded graceful/kill steps.


class MCPUnavailableError(ConnectionError):
    """A transport failure that ToolRuntime may retry within its total deadline."""


class MCPToolsChangedError(ValueError):
    """The run's loaded definition is no longer served by this connection."""


def _settle(future, *, result=None, error=None):
    if not future.done():
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)


def _validate_result(name, result):
    """Reject malformed evidence while preserving the original business object."""
    invalid = ValueError("MCP 工具返回了无效的结构化结果。")
    if not isinstance(result, dict):
        raise invalid
    if name == "get_order":
        if isinstance(result.get("error"), str) and result["error"]:
            return
        if not all(isinstance(result.get(key), str) and result[key]
                   for key in ("order_id", "status", "logistics")):
            raise invalid
        return
    if (not isinstance(result.get("query"), str) or not isinstance(result.get("results"), list)
            or any(type(result.get(key)) is not int or result[key] < 0
                   for key in ("document_count", "matched_chunks"))
            or any(type(result.get(key)) is not bool for key in ("has_more", "budget_limited"))
            or result.get("search_mode") not in ("keywords", "hybrid")
            or not isinstance(result.get("semantic_status"), str)):
        raise invalid
    for hit in result["results"]:
        if (not isinstance(hit, dict)
                or not all(isinstance(hit.get(key), str) for key in
                           ("source", "document_id", "version", "title", "section", "text"))
                or type(hit.get("start")) is not int or type(hit.get("end")) is not int
                or hit["start"] < 0 or hit["end"] <= hit["start"]
                or hit["end"] - hit["start"] != len(hit["text"])
                or hit["source"] != f"kb:{hit['document_id']}:{hit['start']}"):
            raise invalid


class LocalMCPClient:
    """Concurrent callers queue operations; failures are never retried here.

    A new invocation can establish a new connection after a failed invocation.
    expected_schema binds a call to the definition actually loaded by that run.
    close() is terminal; create a new client for a new application lifespan.
    """

    def __init__(self, knowledge_dir, *, search_mode="keywords", embeddings_dir=None, timeout_seconds=6):
        if search_mode not in ("keywords", "hybrid"):
            raise ValueError("search_mode must be keywords or hybrid")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        root = Path(__file__).resolve().parent
        arguments = ["-u", str(root / "mcp_tool_server.py"), "--knowledge-dir", str(Path(knowledge_dir).resolve()),
                     "--search-mode", search_mode]
        if embeddings_dir is not None:
            arguments.extend(["--embeddings-dir", str(Path(embeddings_dir).resolve())])
        # The pinned SDK inherits only its OS allowlist and uses CREATE_NO_WINDOW
        # on Windows. No caller-supplied command, environment, or .env is accepted.
        self._parameters = StdioServerParameters(command=sys.executable, args=arguments, cwd=root,
                                                 env={"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        self.timeout_seconds = float(timeout_seconds)
        self._owner_task = None
        self._ready = None
        self._scope = None
        self._queue = asyncio.Queue()
        self._closed = False
        self._retiring = False
        self._definitions = {}

    async def _ensure_started(self):
        if self._closed:
            raise MCPUnavailableError("本地 MCP 客户端已关闭。")
        if self._owner_task is not None and self._retiring:
            await asyncio.shield(self._owner_task)
        if self._closed:
            raise MCPUnavailableError("本地 MCP 客户端已关闭。")
        if self._owner_task is None or self._owner_task.done():
            self._retiring = False
            self._ready = asyncio.get_running_loop().create_future()
            # A cancelled startup waiter must not leave an unobserved exception.
            self._ready.add_done_callback(lambda future: None if future.cancelled() else future.exception())
            self._owner_task = asyncio.create_task(self._own_connection(), name="local-mcp-owner")
        await asyncio.shield(self._ready)

    async def start(self):
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self._ensure_started()
        except TimeoutError as error:
            self._retiring = True
            if self._scope is not None:
                self._scope.cancel()
            raise MCPUnavailableError("本地 MCP 服务连接超时。") from error

    async def close(self):
        self._closed = True
        if self._scope is not None:
            self._scope.cancel()
        if self._owner_task is not None:
            # Cancel the owner's AnyIO scope, never Task.cancel(): SDK cleanup
            # must remain shielded and run on the same task that opened it.
            with anyio.CancelScope(shield=True):
                deadline = asyncio.get_running_loop().time() + _CLOSE_TIMEOUT
                cancellation = None
                while True:
                    try:
                        remaining = max(0, deadline - asyncio.get_running_loop().time())
                        await asyncio.wait_for(asyncio.shield(self._owner_task), remaining)
                        break
                    except asyncio.CancelledError as error:
                        if self._owner_task.done():
                            raise
                        # Native Task.cancel() bypasses AnyIO shielding. Finish
                        # the owner's cleanup, then preserve the caller's cancel.
                        cancellation = error
                    except TimeoutError as error:
                        raise MCPUnavailableError("本地 MCP 服务关闭超时。") from error
                if cancellation is not None:
                    raise cancellation

    async def list_tools(self):
        return await self._request("list", None, None, None)

    async def call_tool(self, name, arguments: dict, *, expected_schema=None):
        if name not in ALLOWED_TOOLS:
            raise ValueError("未知或未授权的 MCP 工具。")
        if not isinstance(arguments, dict):
            raise ValueError("MCP 工具参数必须是对象。")
        return await self._request("call", name, deepcopy(arguments), deepcopy(expected_schema))

    async def _request(self, operation, name, arguments, expected_schema):
        future = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self._ensure_started()
                future = asyncio.get_running_loop().create_future()
                self._queue.put_nowait((operation, name, arguments, expected_schema, future))
                return await future
        except TimeoutError as error:
            if future is None:
                self._retiring = True
                if self._scope is not None:
                    self._scope.cancel()
            raise MCPUnavailableError("本地 MCP 工具调用超时。") from error
        finally:
            if future is not None and not future.done():
                future.cancel()

    async def _read_definitions(self, sdk):
        definitions, cursor, seen_cursors = {}, None, set()
        while True:
            page = await sdk.list_tools(cursor=cursor)
            for tool in page.tools:
                if tool.name not in ALLOWED_TOOLS:
                    continue
                if tool.name in definitions or tool.input_schema.get("type") != "object":
                    raise ValueError("MCP 返回了无效的工具定义。")
                Draft202012Validator.check_schema(tool.input_schema)
                definitions[tool.name] = {"type": "function", "function": {
                    "name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}}
            cursor = page.next_cursor
            if cursor is None:
                self._definitions = definitions
                return deepcopy(list(definitions.values()))
            if cursor in seen_cursors or len(seen_cursors) >= 10:
                raise ValueError("MCP 工具目录分页无效。")
            seen_cursors.add(cursor)

    async def _execute(self, sdk, operation, name, arguments, expected_schema):
        if operation == "list":
            return await self._read_definitions(sdk)
        actual = self._definitions.get(name)
        if actual is None or (expected_schema is not None and actual != expected_schema):
            raise MCPToolsChangedError("MCP 工具定义已变化或已移除，请重新发现工具。")
        try:
            Draft202012Validator(actual["function"]["parameters"]).validate(arguments)
        except ValidationError as error:
            raise ValueError("MCP 工具参数不符合定义。") from error
        result = await sdk.call_tool(name, arguments)
        if result.is_error:
            raise ValueError(f"MCP 工具 {name} 执行失败。")
        _validate_result(name, result.structured_content)
        return deepcopy(result.structured_content)

    async def _own_connection(self):
        failure = MCPUnavailableError("本地 MCP 连接已关闭。")
        active = None
        try:
            with anyio.CancelScope() as scope:
                self._scope = scope
                if self._closed:
                    return
                # Disable SDK caching: the actual tools/list response identifies
                # this connection; no schema survives reconnect without refresh.
                async with Client(self._parameters, cache=None, read_timeout_seconds=self.timeout_seconds) as sdk:
                    with anyio.fail_after(self.timeout_seconds):
                        await self._read_definitions(sdk)
                    _settle(self._ready)
                    while True:
                        operation, name, arguments, expected, active = await self._queue.get()
                        if active.cancelled():
                            active = None
                            continue
                        with anyio.CancelScope() as request_scope:
                            def cancelled(future, request_scope=request_scope):
                                if future.cancelled():
                                    request_scope.cancel()
                            active.add_done_callback(cancelled)
                            try:
                                with anyio.fail_after(self.timeout_seconds):
                                    result = await self._execute(sdk, operation, name, arguments, expected)
                                _settle(active, result=result)
                            except ValueError as error:
                                _settle(active, error=error)
                            except MCPError as error:
                                if error.code not in (CONNECTION_CLOSED, REQUEST_TIMEOUT):
                                    _settle(active, error=ValueError("MCP 协议请求失败。"))
                                else:
                                    raise
                            finally:
                                active.remove_done_callback(cancelled)
                        active = None
        except Exception as error:
            failure = (error if isinstance(error, ValueError)
                       else MCPUnavailableError("本地 MCP 服务连接失败或已断开。"))
        finally:
            self._retiring = True
            self._scope = None
            _settle(self._ready, error=failure)
            if active is not None:
                _settle(active, error=failure)
            while not self._queue.empty():
                _settle(self._queue.get_nowait()[-1], error=failure)
