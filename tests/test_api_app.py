"""接口集成测试：真实 FastAPI、真实 SDK，仅替换模型服务的 HTTP 流。

测试不会读取真实 .env，也不会访问外网或消耗模型额度。
"""

import asyncio
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
from contextlib import closing
import unittest
from unittest.mock import patch

import anyio
import httpx2
from openai import AsyncOpenAI

import api_app
from test_stream_agent import chunk, tool_delta, text_stream
from context_memory import ContextMemory, ContextPolicy, estimate_request
from test_context_memory import history_of


class AsyncEventBytes(httpx2.AsyncByteStream):
    def __init__(self, frames, entered, gate):
        self.frames, self.entered, self.gate = frames, entered, gate
        self.closed = False

    async def __aiter__(self):
        self.entered.set()
        await self.gate.wait()
        for frame in self.frames:
            if isinstance(frame, BaseException):
                raise frame
            data = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
            yield f"data: {data}\n\n".encode()

    async def aclose(self):
        # 模拟真实 httpcore2 PoolByteStream 关闭时使用的取消保护。
        # 这里包含一次 await，让测试覆盖异步连接清理，而非只改一个标志。
        with anyio.CancelScope(shield=True):
            await asyncio.sleep(0)
            self.closed = True


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def last_tool_result(request):
    """Read actual tool evidence even when optional actor feedback follows it."""
    return json.loads(next(message['content'] for message in reversed(request['messages'])
                           if message.get('role') == 'tool'))


class ApiAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.responses, self.requests, self.streams = [], [], []
        self.entered, self.gate = asyncio.Event(), asyncio.Event()
        self.gate.set()

        async def handle(request):
            self.requests.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertLessEqual(len(self.requests), len(self.responses), "出现了额外模型请求")
            response = self.responses[len(self.requests) - 1]
            if isinstance(response, dict):
                self.entered.set()
                await self.gate.wait()
                return httpx2.Response(200, json=response)
            stream = AsyncEventBytes(response, self.entered, self.gate)
            self.streams.append(stream)
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        def make_client(**kwargs):
            return AsyncOpenAI(
                **kwargs,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle), trust_env=False),
            )

        # Windows subprocesses need SystemRoot for asyncio/Winsock initialization.
        os_environment = {name: os.environ[name] for name in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP') if name in os.environ}
        # Keep legacy pressure cases at 8K even when the production default grows.
        self.environment = patch.dict(os.environ, {**os_environment, "DEEPSEEK_API_KEY": "offline-only-key", "AGENT_HISTORY_SEARCH": "keywords", "AGENT_CONTEXT_TOKENS": "8192"}, clear=True)
        self.dotenv = patch.object(api_app, "load_dotenv", return_value=False)
        self.factory = patch.object(api_app, "AsyncOpenAI", side_effect=make_client)
        self.environment.start()
        self.dotenv.start()
        self.factory.start()
        self.database_dir = TemporaryDirectory()
        self.db_path = Path(self.database_dir.name) / "chat.db"
        self.knowledge_dir = Path(self.database_dir.name) / "knowledge"
        # 既有业务用例独立验证直接执行；人工介入用例另行启用服务器策略。
        # Legacy budget scenarios retain their original prompt overhead; Skills
        # integration tests explicitly enable the new directory.
        self.app = api_app.create_app(db_path=self.db_path, knowledge_dir=self.knowledge_dir, approval_tools=frozenset(), result_verification=False, delegation_enabled=False, tool_mode=getattr(self, 'tool_mode', 'local'), skills_enabled=False)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=self.app), base_url="http://testserver")

    async def asyncTearDown(self):
        self.gate.set()
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.database_dir.cleanup()
        self.factory.stop()
        self.dotenv.stop()
        self.environment.stop()

    async def new_session(self):
        response = await self.client.post("/api/sessions")
        self.assertEqual(response.status_code, 201)
        return response.json()["session_id"]

    async def restart_app(self):
        """丢弃整个应用实例，保留磁盘数据库，模拟真正的服务重启。"""
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.app = api_app.create_app(db_path=self.db_path, knowledge_dir=self.knowledge_dir, approval_tools=frozenset(), result_verification=False, delegation_enabled=False, tool_mode=getattr(self, 'tool_mode', 'local'), skills_enabled=False)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=self.app), base_url="http://testserver")

    async def seed_retrievable_history(self, session):
        # 摘要故意遗漏收货要求，确保模型必须通过工具拿到旧原文。
        history = api_app.new_conversation() + [
            {"role": "user", "content": "O1001 的收货要求：请周五之后再送达。"},
            {"role": "assistant", "content": "已经了解你的要求。"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮忙？"},
        ]
        memory = ContextMemory("用户有一个订单，曾经聊过收货要求。", 3)
        await anyio.to_thread.run_sync(self.app.state.store.replace_history, session, history, memory)
        self.app.state.sessions.clear()
        return history, memory

    def history_tool_stream(self, arguments='{"query":"O1001 收货","order":"oldest"}', call_id="history-call"):
        return [chunk(calls=[tool_delta(0, call_id, "search_history", arguments)]), chunk(finish="tool_calls"), "[DONE]"]

    def seed_knowledge(self):
        self.knowledge_dir.mkdir(exist_ok=True)
        path = self.knowledge_dir / "policy.md"
        path.write_text("# 示例售后政策\n\n## 退款\n仓库验收后3个工作日原路退款。", encoding="utf-8")
        return path

    async def test_knowledge_tool_combines_order_and_policy_with_persisted_sources(self):
        self.seed_knowledge()
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                calls = [tool_delta(0, "policy", "search_knowledge", '{"query":"退款"}'),
                         tool_delta(1, "order", "get_order", '{"order_id":"O1001"}')]
                self.responses.extend([[chunk(calls=calls), chunk(finish="tool_calls"), "[DONE]"], text_stream("按示例售后政策，验收后3个工作日发起退款。")])
                result = await self.client.post(endpoint, json={"session_id": session, "message": "O1001 退款要多久？"})
                payload = parse_sse(result.text)[-1][1] if endpoint.endswith("stream") else result.json()
                self.assertEqual(result.status_code, 200)
                self.assertEqual(payload["knowledge"]["calls"], 1)
                source = payload["sources"][0]
                self.assertIn("3个工作日", source["text"])
                self.assertIn("search_knowledge", [tool["function"]["name"] for tool in self.requests[-1]["tools"]])
                self.assertLessEqual(estimate_request(self.requests[-1]["messages"], self.requests[-1]["tools"]), self.app.state.context_policy.input_limit)
                await self.restart_app()
                saved = (await self.client.get(f"/api/sessions/{session}")).json()["messages"]
                self.assertIn(source, json.loads(next(m["content"] for m in saved if m.get("tool_call_id") == "policy"))["results"])

    async def test_source_route_escapes_html_and_detects_changed_or_deleted_version(self):
        path = self.seed_knowledge()
        hit = self.app.state.knowledge_base.search("退款")["results"][0]
        url = f"/api/knowledge/{hit['document_id']}?version={hit['version']}"
        response = await self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("3个工作日", response.text)
        path.write_text("# 新版示例政策\n退款时间5天。", encoding="utf-8")
        self.assertEqual((await self.client.get(url)).status_code, 409)
        path.write_text("# <script>alert('title')</script>\n<script>alert('body')</script>", encoding="utf-8")
        latest = await self.client.get(url.split("?")[0])
        self.assertNotIn("<script>", latest.text)
        self.assertIn("&lt;script&gt;", latest.text)
        self.assertIn("default-src 'none'", latest.headers["content-security-policy"])
        path.unlink()
        self.assertEqual((await self.client.get(url)).status_code, 404)
        self.assertEqual((await self.client.get("/api/knowledge/.env")).status_code, 404)

    async def test_knowledge_absence_and_read_failure_are_distinct_tool_results(self):
        from knowledge_base import KnowledgeReadError
        session = await self.new_session()
        tool = [chunk(calls=[tool_delta(0, "policy", "search_knowledge", '{"query":"退款"}')]), chunk(finish="tool_calls"), "[DONE]"]
        self.responses = [tool, text_stream("资料没有相关规则。")]
        response = await self.client.post("/api/chat", json={"session_id": session, "message": "退款多久？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [])
        empty = last_tool_result(self.requests[-1])
        self.assertEqual(empty["document_count"], 0)
        self.responses.extend([tool, text_stream("资料读取失败，无法确认。")])
        with patch.object(self.app.state.knowledge_base, "search", side_effect=KnowledgeReadError("读取失败")):
            response = await self.client.post("/api/chat", json={"session_id": session, "message": "再查一次"})
        self.assertEqual(response.json()["sources"], [])
        self.assertIn("error", last_tool_result(self.requests[-1]))

    async def test_history_and_knowledge_share_search_cap_and_reject_extra_parameters(self):
        self.seed_knowledge()
        session = await self.new_session()
        calls = [tool_delta(0, "bad", "search_knowledge", '{"query":"退款","path":"../.env"}'),
                 tool_delta(1, "history", "search_history", '{"query":"退款"}')]
        calls += [tool_delta(index + 2, f"kb{index}", "search_knowledge", '{"query":"退款"}') for index in range(3)]
        self.responses = [[chunk(calls=calls), chunk(finish="tool_calls"), "[DONE]"], text_stream("按示例政策处理。")]
        response = await self.client.post("/api/chat", json={"session_id": session, "message": "退款规则"})
        data = response.json()
        self.assertEqual(data["retrieval"]["calls"], 1)
        self.assertEqual(data["knowledge"]["calls"], 2)
        self.assertEqual(len(data["sources"]), 1, "重复检索同一来源应去重")
        tool_messages = [m for m in self.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 5)
        self.assertIn("query", json.loads(tool_messages[0]["content"])["error"])
        self.assertIn("上限", json.loads(tool_messages[-1]["content"])["error"])

    async def test_prior_knowledge_sources_survive_later_context_exhaustion(self):
        self.seed_knowledge()
        session = await self.new_session()
        # 后续大结果挤掉政策时停止，而原始来源仍保存供查看。
        self.app.state.context_policy = ContextPolicy(max_tokens=5482, recent_tokens=128, summary_tokens=256)
        self.responses = [
            [chunk(calls=[tool_delta(0, "policy", "search_knowledge", '{"query":"退款"}')]), chunk(finish="tool_calls"), "[DONE]"],
            [chunk(calls=[tool_delta(0, "order", "get_order", '{"order_id":"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"],
            text_stream("示例政策规定验收后3个工作日退款。")]
        with patch("agent_service.execute_tool", return_value={"status": "已发货", "details": "x" * 2200}):
            response = await self.client.post("/api/chat", json={"session_id": session, "message": "O1001 退款需要多久？"})
        self.assertEqual(response.status_code, 200, response.text)
        first = json.loads(self.requests[1]["messages"][-1]["content"])["results"]
        self.assertTrue(first)
        self.assertEqual(len(self.requests), 2, "证据被挤空后不能继续请求模型")
        self.assertTrue(response.json()['recovery']['stopped'])
        self.assertIn('空间', response.json()['answer'])
        self.assertEqual(response.json()["sources"], first)
        saved = self.app.state.store.load_session(session)["history"]
        self.assertEqual(json.loads(next(m["content"] for m in saved if m.get("tool_call_id") == "policy"))["results"], first)

    async def test_cancelled_knowledge_read_does_not_save_or_request_model_again(self):
        self.seed_knowledge()
        session = await self.new_session()
        self.responses = [[chunk(calls=[tool_delta(0, "policy", "search_knowledge", '{"query":"退款"}')]), chunk(finish="tool_calls"), "[DONE]"]]
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        real_search = self.app.state.knowledge_base.search
        def slow_search(query):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise RuntimeError("测试未释放检索线程")
            return real_search(query)
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "退款多久？"}).encode(), "more_body": False})
        async def send(event):
            pass
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat/stream", "raw_path": b"/api/chat/stream", "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
        with patch.object(self.app.state.knowledge_base, "search", side_effect=slow_search):
            task = asyncio.create_task(self.app(scope, incoming.get, send))
            try:
                # Wait for the actual worker signal: graph cold start is not the
                # cancellation latency under test. Keep a bounded deadlock guard.
                await asyncio.wait_for(entered.wait(), 10)
                await incoming.put({"type": "http.disconnect"})
                self.assertEqual((await self.client.get("/api/health")).status_code, 200)
                self.assertTrue((await self.client.get(f"/api/sessions/{session}")).json()["busy"])
            finally:
                # Also disconnect if readiness or an earlier assertion failed;
                # otherwise shutdown can wait on receive and mask that failure.
                incoming.put_nowait({"type": "http.disconnect"})
                release.set()
                await asyncio.wait_for(task, 3)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.app.state.store.load_session(session)["history"]), 1)
        self.assertFalse((await self.client.get(f"/api/sessions/{session}")).json()["busy"])

    async def test_hybrid_search_wiring_recovers_paraphrase_in_both_routes(self):
        from test_semantic_search import FixedSimilarity

        class Encoder(FixedSimilarity):
            def warm_up(self):
                return True

        with patch.dict(os.environ, {"AGENT_HISTORY_SEARCH": "hybrid"}), \
             patch.object(api_app, "LocalEmbeddings", return_value=Encoder()):
            await self.restart_app()
        health = (await self.client.get("/api/health")).json()
        self.assertEqual(health["history_search"], "hybrid")
        self.assertEqual(health["semantic_status"], "ready")
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                history = api_app.new_conversation() + [
                    {"role": "user", "content": "请周五18点之后再送达。"},
                    {"role": "assistant", "content": "好的。"}]
                self.app.state.store.replace_history(session, history, ContextMemory("用户询问过订单。", 3))
                self.app.state.sessions.clear()
                self.responses.extend([self.history_tool_stream('{"query":"我之前希望什么时候收到包裹？"}'), text_stream("你希望周五18点后收到。")])
                response = await self.client.post(endpoint, json={"session_id": session, "message": "之前要求什么时候送来？"})
                payload = parse_sse(response.text)[-1][1] if endpoint.endswith("stream") else response.json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload["retrieval"]["returned_messages"], 1)
                result = json.loads(self.requests[-1]["messages"][-1]["content"])
                self.assertEqual(result["search_mode"], "hybrid")
                self.assertEqual(result["results"][0]["text"], history[1]["content"])
                self.assertLessEqual(estimate_request(self.requests[-1]["messages"], self.requests[-1]["tools"]), self.app.state.context_policy.input_limit)

    async def test_model_unavailable_keeps_api_and_keyword_search_usable(self):
        from local_embeddings import LocalEmbeddings
        with patch.dict(os.environ, {"AGENT_HISTORY_SEARCH": "hybrid"}), \
             patch.object(LocalEmbeddings, "_load_model", side_effect=OSError("missing weights")), \
             self.assertLogs("local_embeddings", level="WARNING"):
            await self.restart_app()
        health = (await self.client.get("/api/health")).json()
        self.assertEqual(health["history_search"], "keywords")
        self.assertEqual(health["semantic_status"], "unavailable")
        session = await self.new_session()
        await self.seed_retrievable_history(session)
        self.responses = [self.history_tool_stream(), text_stream("周五之后送达。")]
        with self.assertLogs("history_search", level="WARNING"):
            response = await self.client.post("/api/chat", json={"session_id": session, "message": "之前的要求？"})
        self.assertEqual(response.status_code, 200)
        result = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(result["search_mode"], "keywords")
        self.assertEqual(result["semantic_status"], "unavailable")
        self.assertIn("周五", result["results"][0]["text"])

    async def test_history_tool_recovers_omitted_original_for_both_routes_and_restart(self):
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                history, memory = await self.seed_retrievable_history(session)
                other = await self.new_session()
                await anyio.to_thread.run_sync(self.app.state.store.replace_history, other, api_app.new_conversation() + [{"role": "user", "content": "O1001 收货秘密：周日。"}, {"role": "assistant", "content": "好的"}])
                await self.restart_app()
                start = len(self.requests)
                self.responses.extend([self.history_tool_stream(), text_stream("你当时要求周五之后再送达。[message:1]")])
                response = await self.client.post(endpoint, json={"session_id": session, "message": "我最开始的收货要求是什么？"})
                self.assertEqual(response.status_code, 200)
                first = self.requests[start]
                self.assertIn("search_history", [tool["function"]["name"] for tool in first["tools"]])
                schema = next(tool["function"]["parameters"] for tool in first["tools"] if tool["function"]["name"] == "search_history")
                self.assertNotIn("session_id", schema["properties"])
                self.assertNotIn("周五", json.dumps(first["messages"], ensure_ascii=False))
                next_request = self.requests[start + 1]["messages"]
                tool_message = next_request[-1]
                self.assertEqual(tool_message["tool_call_id"], "history-call")
                result = json.loads(tool_message["content"])
                self.assertEqual(result["results"][0]["text"], history[1]["content"])
                self.assertNotIn("周日", tool_message["content"])
                payload = parse_sse(response.text)[-1][1] if endpoint.endswith("stream") else response.json()
                self.assertEqual(payload["retrieval"]["calls"], 1)
                self.assertEqual(payload["retrieval"]["returned_messages"], 1)
                if endpoint.endswith("stream"):
                    self.assertIn(("tool_result", {"name": "search_history", "result": result}), parse_sse(response.text))
                await self.restart_app()
                stored = self.app.state.store.load_session(session)
                self.assertEqual(stored["history"][:len(history)], history)
                self.assertEqual(stored["memory"], memory)
                self.assertEqual(stored["history"][-2]["tool_call_id"], "history-call")

    async def test_no_match_returns_empty_evidence_and_does_not_invent_a_tool_result(self):
        session = await self.new_session()
        await self.seed_retrievable_history(session)
        self.responses = [self.history_tool_stream('{"query":"不存在的备注"}'), text_stream("没有找到这项备注的原文依据。")]
        result = await self.client.post("/api/chat", json={"session_id": session, "message": "我以前提过什么备注？"})
        self.assertEqual(result.status_code, 200)
        data = last_tool_result(self.requests[-1])
        self.assertEqual(data["matched_messages"], 0)
        self.assertEqual(data["results"], [])

    async def test_search_arguments_cannot_select_another_session(self):
        session = await self.new_session()
        self.responses = [self.history_tool_stream('{"query":"收货","session_id":"someone-else"}'), text_stream("检索参数无效。")]
        result = await self.client.post("/api/chat", json={"session_id": session, "message": "查询历史"})
        self.assertEqual(result.status_code, 200)
        data = last_tool_result(self.requests[-1])
        self.assertIn("error", data)
        self.assertIn("query", data["error"])
        self.assertEqual(result.json()["retrieval"]["calls"], 0)

    async def test_failed_search_read_or_final_answer_keeps_old_state(self):
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                history, memory = await self.seed_retrievable_history(session)
                self.responses.append(self.history_tool_stream())
                with patch.object(self.app.state.store, "search_history", side_effect=sqlite3.OperationalError("disk read failed")):
                    response = await self.client.post(endpoint, json={"session_id": session, "message": "旧要求？"})
                if endpoint.endswith("stream"):
                    self.assertEqual(parse_sse(response.text)[-1][0], "error")
                else:
                    self.assertEqual(response.status_code, 503)
                self.responses.extend([self.history_tool_stream(), [chunk(content="未完成"), chunk(finish="length"), "[DONE]"]])
                await self.client.post(endpoint, json={"session_id": session, "message": "重试"})
                await self.restart_app()
                saved = self.app.state.store.load_session(session)
                self.assertEqual(saved["history"], history)
                self.assertEqual(saved["memory"], memory)
                self.assertFalse((await self.client.get(f"/api/sessions/{session}")).json()["busy"])

    async def test_mixed_tools_keep_request_result_pairs_and_search_has_per_turn_limit(self):
        session = await self.new_session()
        await self.seed_retrievable_history(session)
        calls = [tool_delta(0, "order", "get_order", '{"order_id":"O1001"}')]
        calls += [tool_delta(index + 1, f"search-{index}", "search_history", '{"query":"收货"}') for index in range(4)]
        self.responses = [[chunk(calls=calls), chunk(finish="tool_calls"), "[DONE]"], text_stream("已核对订单和旧要求。")]
        result = await self.client.post("/api/chat", json={"session_id": session, "message": "核对原话和订单"})
        self.assertEqual(result.status_code, 200)
        messages = self.requests[-1]["messages"]
        call_message = next(message for message in reversed(messages) if message.get('tool_calls'))
        tool_messages = [message for message in messages if message.get('role') == 'tool']
        self.assertEqual([call["id"] for call in call_message["tool_calls"]], ["order", "search-0", "search-1", "search-2", "search-3"])
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["order", "search-0", "search-1", "search-2", "search-3"])
        self.assertIn("上限", last_tool_result(self.requests[-1])["error"])
        self.assertEqual(result.json()["retrieval"]["calls"], 3)
        self.assertLessEqual(estimate_request(messages, self.requests[-1]["tools"]), self.app.state.context_policy.input_limit)

    async def test_final_batch_budget_does_not_depend_on_history_tool_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.requests.clear()
                self.responses.clear()
                session = await self.new_session()
                history = api_app.new_conversation() + [{"role": "user", "content": "delivery " + "Friday after " * 100}, {"role": "assistant", "content": "ok"}]
                memory = ContextMemory("Orders delivery " * 13, 3)
                self.app.state.store.replace_history(session, history, memory)
                self.app.state.sessions.clear()
                # 第三个工具及其规则增加固定输入；仍把预算压到必须回收片段。
                # 补齐计划工具 schema 后固定输入略增；仍必须实际触发下面的片段裁剪断言。
                self.app.state.context_policy = ContextPolicy(max_tokens=5000, recent_tokens=128, summary_tokens=256)
                tool_specs = [("search_history", '{"query":"delivery"}'), ("get_order", '{"order_id":"O1001"}'), ("get_order", '{"order_id":"O1002"}')]
                if reverse:
                    tool_specs.reverse()
                calls = [tool_delta(index, str(index), name, arguments) for index, (name, arguments) in enumerate(tool_specs)]
                self.responses.extend([[chunk(calls=calls), chunk(finish="tool_calls"), "[DONE]"], text_stream("done")])
                response = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "Verify delivery and orders O1001 O1002"})
                events = parse_sse(response.text)
                self.assertEqual(events[-1][0], "done", events[-1])
                result = next(data["result"] for name, data in events if name == "tool_result" and data["name"] == "search_history")
                self.assertTrue(result["budget_limited"])
                self.assertGreater(result["matched_messages"], 0)
                self.assertEqual(events[-1][1]["retrieval"]["returned_messages"], len(result["results"]))
                sent = self.requests[-1]
                self.assertLessEqual(estimate_request(sent["messages"], sent["tools"]), self.app.state.context_policy.input_limit)

    async def test_disconnect_during_history_scan_releases_wait_and_does_not_commit(self):
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                history, memory = await self.seed_retrievable_history(session)
                self.responses.append(self.history_tool_stream())
                entered, release, finished = threading.Event(), threading.Event(), threading.Event()
                actual_search = self.app.state.store.search_history

                def slow_search(*args):
                    entered.set()
                    try:
                        if not release.wait(5):
                            raise RuntimeError("测试未释放检索线程")
                        return actual_search(*args)
                    finally:
                        finished.set()

                incoming, sent = asyncio.Queue(), []
                await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "旧收货要求"}).encode(), "more_body": False})
                async def send(event):
                    sent.append(event)
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": endpoint, "raw_path": endpoint.encode(), "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
                with patch.object(self.app.state.store, "search_history", side_effect=slow_search):
                    task = asyncio.create_task(self.app(scope, incoming.get, send))
                    try:
                        self.assertTrue(await anyio.to_thread.run_sync(entered.wait, 3))
                        await incoming.put({"type": "http.disconnect"})
                        # 检索在线程中等待时，健康接口仍然正常响应。
                        self.assertEqual((await self.client.get("/api/health")).status_code, 200)
                        await asyncio.wait_for(task, timeout=3)
                        self.assertFalse(finished.is_set(), "断连无需等待只读线程结束")
                        self.assertFalse((await self.client.get(f"/api/sessions/{session}")).json()["busy"])
                        self.assertEqual(len(self.requests), len(self.responses), "断连后不应再请求模型")
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        self.assertTrue(await anyio.to_thread.run_sync(finished.wait, 3))
                self.assertFalse((await self.client.get(f"/api/sessions/{session}")).json()["busy"])
                self.assertEqual(self.app.state.store.load_session(session)["history"], history)
                self.assertEqual(self.app.state.store.load_session(session)["memory"], memory)

    async def test_restart_restores_tool_history_and_supplies_it_to_the_next_question(self):
        self.responses = [
            [chunk(calls=[tool_delta(0, "call-saved", "get_order", '{"order_id":"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"],
            text_stream("O1001 已发货。"), text_stream("这笔订单买的是无线键盘。"),
        ]
        session = await self.new_session()
        response = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "查 O1001"})
        self.assertEqual(parse_sse(response.text)[-1][0], "done")
        await self.restart_app()
        restored = await self.client.get(f"/api/sessions/{session}")
        self.assertEqual(restored.status_code, 200)
        state = restored.json()
        self.assertEqual([m["role"] for m in state["messages"]], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(state["messages"][2]["tool_calls"][0]["id"], "call-saved")
        self.assertEqual(state["messages"][3]["tool_call_id"], "call-saved")
        self.assertIn("created_at", state)
        self.assertFalse(state["busy"])
        result = await self.client.post("/api/chat", json={"session_id": session, "message": "这笔订单买的是什么？"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual([m["role"] for m in self.requests[2]["messages"]], ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(json.loads(self.requests[2]["messages"][3]["content"])["order_id"], "O1001")

    def summary_response(self, content="用户提供订单 O1001，需要继续查询。", finish="stop"):
        return {"id": "summary", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}

    async def seed_long_history(self, session):
        history = api_app.new_conversation() + history_of(8)[1:]
        await anyio.to_thread.run_sync(self.app.state.store.replace_history, session, history)
        self.app.state.sessions.clear()
        self.app.state.context_policy = ContextPolicy(max_tokens=5306, recent_tokens=500, summary_tokens=512)
        return history

    async def test_budget_summarizes_old_turns_and_persists_full_history_for_both_routes(self):
        for endpoint in ("/api/chat", "/api/chat/stream"):
            with self.subTest(endpoint=endpoint):
                session = await self.new_session()
                history = await self.seed_long_history(session)
                start = len(self.requests)
                self.responses.extend([
                    self.summary_response(),
                    [chunk(calls=[tool_delta(0, "new-call", "get_order", '{"order_id":"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"],
                    [chunk(content="O1001 买的是无线键盘。"), chunk(finish="stop", usage={"prompt_tokens": 123, "completion_tokens": 12, "total_tokens": 135}), "[DONE]"],
                ])
                result = await self.client.post(endpoint, json={"session_id": session, "message": "它买的是什么？"})
                self.assertEqual(result.status_code, 200)
                if endpoint.endswith("stream"):
                    events = parse_sse(result.text)
                    self.assertEqual(events[0][0], "summarizing")
                    self.assertEqual(events[-1][0], "done", events)
                    payload = events[-1][1]
                    self.assertEqual([data for name, data in events if name == "context"][-1], payload["context"])
                else:
                    payload = result.json()
                self.assertEqual(payload["context"]["summary_calls"], 1)
                self.assertEqual(payload["context"]["kept_turns"], 1)
                self.assertEqual(payload["context"]["summarized_turns"], 7)
                self.assertEqual(payload["context"]["summary_usage"]["total_tokens"], 120)
                self.assertEqual(payload["usage"]["total_tokens"], 135)
                initial = self.requests[start + 1]["messages"]
                self.assertEqual(initial[-3:-1], history[-2:])
                self.assertNotIn("问题 0", json.dumps(initial, ensure_ascii=False))
                self.assertIn("O1001", initial[1]["content"])
                next_request = self.requests[start + 2]["messages"]
                self.assertEqual(next_request[:-2], initial)
                self.assertEqual(next_request[-2]["tool_calls"][0]["id"], "new-call")
                self.assertEqual(next_request[-1]["tool_call_id"], "new-call")
                for request in self.requests[start:]:
                    self.assertLessEqual(estimate_request(request["messages"], request.get("tools")) + request["max_tokens"] + 256, 5306)
                await self.restart_app()
                restored = (await self.client.get(f"/api/sessions/{session}")).json()["messages"]
                self.assertEqual(restored[:len(history)], history)
                self.assertEqual([m["role"] for m in restored[len(history):]], ["user", "assistant", "tool", "assistant"])
                self.assertEqual(self.app.state.sessions[session].memory.covered_messages, 15)
                self.responses.append(text_stream("继续回复"))
                followup = await self.client.post("/api/chat", json={"session_id": session, "message": "继续"})
                self.assertEqual(followup.status_code, 200)
                self.assertEqual(followup.json()["context"]["summary_calls"], 0)
                self.assertIn("O1001", self.requests[-1]["messages"][1]["content"])
                await self.client.post(f"/api/sessions/{session}/reset")
                self.assertEqual(self.app.state.store.load_session(session)["memory"], ContextMemory())

    async def test_budget_configuration_is_loaded_and_legacy_turn_count_is_ignored(self):
        with patch.dict(os.environ, {"AGENT_CONTEXT_TOKENS": "6000", "AGENT_RECENT_TOKENS": "1000", "AGENT_CONTEXT_TURNS": "1"}):
            await self.restart_app()
            health = (await self.client.get("/api/health")).json()
            self.assertEqual(health["context_tokens"], 6000)
            self.assertEqual(health["recent_tokens"], 1000)
            self.assertNotIn("context_turns", health)

    async def test_invalid_context_configuration_fails_before_starting_service(self):
        for name in ("AGENT_CONTEXT_TOKENS", "AGENT_RECENT_TOKENS", "AGENT_SUMMARY_TOKENS"):
            for value in ("0", "-1", "two", "2.5", "", "100"):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    invalid_app = api_app.create_app(self.db_path)
                    with self.assertRaisesRegex(ValueError, "AGENT_"):
                        async with invalid_app.router.lifespan_context(invalid_app):
                            self.fail("无效预算不应启动服务")

    async def test_summary_or_answer_failure_keeps_both_saved_states_unchanged(self):
        for endpoint in ("/api/chat", "/api/chat/stream"):
            for fail_summary in (True, False):
                with self.subTest(endpoint=endpoint, fail_summary=fail_summary):
                    session = await self.new_session()
                    history = await self.seed_long_history(session)
                    self.responses.append(self.summary_response(finish="length" if fail_summary else "stop"))
                    if not fail_summary:
                        self.responses.append([chunk(content="不完整"), chunk(finish="length"), "[DONE]"])
                    result = await self.client.post(endpoint, json={"session_id": session, "message": "新问题"})
                    if endpoint.endswith("stream"):
                        self.assertEqual(parse_sse(result.text)[-1][0], "error")
                    else:
                        self.assertEqual(result.status_code, 502)
                    await self.restart_app()
                    saved = self.app.state.store.load_session(session)
                    self.assertEqual(saved["history"], history)
                    self.assertEqual(saved["memory"], ContextMemory())
                    self.assertFalse((await self.client.get(f"/api/sessions/{session}")).json()["busy"])

    async def test_tool_output_budget_is_rechecked_before_next_model_call(self):
        session = await self.new_session()
        self.app.state.context_policy = ContextPolicy(max_tokens=4659, recent_tokens=500, summary_tokens=256)
        self.responses = [[chunk(calls=[tool_delta(0, "huge", "get_order", '{"order_id":"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"]]
        # 放大只读工具依赖返回，验证真实 Agent 循环不能发出超预算的第二次请求。
        with patch("agent_service.execute_tool", return_value={"result": "大" * 5000}):
            result = await self.client.post("/api/chat", json={"session_id": session, "message": "查询 O1001"})
        self.assertEqual(result.status_code, 502)
        self.assertIn("预算", result.json()["detail"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.app.state.store.load_session(session)["history"]), 1)

    async def test_reset_persists_and_does_not_clear_another_session(self):
        self.responses = [text_stream("第一段回复"), text_stream("第二段回复")]
        first, second = await self.new_session(), await self.new_session()
        await self.client.post("/api/chat", json={"session_id": first, "message": "第一段问题"})
        await self.client.post("/api/chat", json={"session_id": second, "message": "第二段问题"})
        await self.client.post(f"/api/sessions/{first}/reset")
        await self.restart_app()
        cleared = await self.client.get(f"/api/sessions/{first}")
        kept = await self.client.get(f"/api/sessions/{second}")
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(kept.status_code, 200)
        self.assertEqual([m["role"] for m in cleared.json()["messages"]], ["system"])
        self.assertEqual(kept.json()["messages"][-1]["content"], "第二段回复")

    async def test_session_list_survives_restart_and_new_session_keeps_old_history(self):
        self.responses = [text_stream("原会话回复")]
        first = await self.new_session()
        await self.client.post("/api/chat", json={"session_id": first, "message": "查询 O1001 的物流"})
        original = (await self.client.get(f"/api/sessions/{first}")).json()
        second = await self.new_session()
        self.assertNotEqual(first, second)
        await self.restart_app()
        response = await self.client.get("/api/sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.json()["sessions"]
        self.assertEqual([item["session_id"] for item in sessions], [second, first])
        self.assertEqual(sessions[0]["title"], "新对话")
        self.assertEqual(sessions[1]["title"], "查询 O1001 的物流")
        self.assertEqual(sessions[1]["created_at"], original["created_at"])
        self.assertNotIn("messages", sessions[1], "列表只需要摘要，选中后再读取完整上下文")
        restored = (await self.client.get(f"/api/sessions/{first}")).json()
        self.assertEqual(restored["messages"], original["messages"])
        empty = (await self.client.get(f"/api/sessions/{second}")).json()
        self.assertEqual([message["role"] for message in empty["messages"]], ["system"])

    async def test_session_list_is_empty_before_creation_and_reports_storage_failure(self):
        response = await self.client.get("/api/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"sessions": []})
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("ALTER TABLE sessions RENAME TO unavailable_sessions")
        failed = await self.client.get("/api/sessions")
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("offline-only-key", failed.text)

    async def test_delete_removes_history_and_cache_after_restart_but_keeps_other_sessions(self):
        self.responses = [text_stream("保留下来的回复")]
        removed, kept = await self.new_session(), await self.new_session()
        await self.client.post("/api/chat", json={"session_id": kept, "message": "保留的问题"})
        original = (await self.client.get(f"/api/sessions/{kept}")).json()
        deleted = await self.client.delete(f"/api/sessions/{removed}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(deleted.content, b"")
        self.assertEqual((await self.client.get(f"/api/sessions/{removed}")).status_code, 404)
        self.assertEqual((await self.client.delete(f"/api/sessions/{removed}")).status_code, 404)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (removed,)).fetchone()[0], 0)
        await self.restart_app()
        self.assertEqual((await self.client.get(f"/api/sessions/{removed}")).status_code, 404)
        self.assertEqual((await self.client.get(f"/api/sessions/{kept}")).json(), original)
        self.assertEqual([s["session_id"] for s in (await self.client.get("/api/sessions")).json()["sessions"]], [kept])

    async def test_failed_delete_preserves_history_and_allows_retry(self):
        session = await self.new_session()
        original = (await self.client.get(f"/api/sessions/{session}")).json()
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON messages BEGIN SELECT RAISE(ABORT, 'delete failed'); END")
        result = await self.client.delete(f"/api/sessions/{session}")
        self.assertEqual(result.status_code, 503)
        self.assertEqual((await self.client.get(f"/api/sessions/{session}")).json(), original)
        await self.restart_app()
        self.assertEqual((await self.client.get(f"/api/sessions/{session}")).json(), original)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("DROP TRIGGER fail_delete")
        self.assertEqual((await self.client.delete(f"/api/sessions/{session}")).status_code, 204)
        self.assertEqual((await self.client.get("/api/sessions")).json(), {"sessions": []})

    async def test_failed_turn_does_not_replace_saved_history_after_restart(self):
        self.responses = [text_stream("已保存的回复"), [chunk(content="半段文字"), chunk(finish="length"), "[DONE]"]]
        session = await self.new_session()
        await self.client.post("/api/chat", json={"session_id": session, "message": "第一条问题"})
        failed = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "失败的问题"})
        self.assertEqual(parse_sse(failed.text)[-1][0], "error")
        await self.restart_app()
        restored = await self.client.get(f"/api/sessions/{session}")
        self.assertEqual(restored.status_code, 200)
        self.assertEqual([m["role"] for m in restored.json()["messages"]], ["system", "user", "assistant"])
        self.assertEqual(restored.json()["messages"][-1]["content"], "已保存的回复")

    async def test_database_failure_rolls_back_chat_and_reset_and_unlocks_session(self):
        self.responses = [text_stream("原来的回复"), text_stream("保存失败的回复"), text_stream("流式保存失败")]
        session = await self.new_session()
        await self.client.post("/api/chat", json={"session_id": session, "message": "原来的问题"})
        self.assertTrue(self.db_path.exists(), "会话必须实际写入磁盘数据库")
        # 用 SQLite 自己的触发器模拟插入失败，保留真实 DELETE/INSERT/ROLLBACK。
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_message_insert BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'test write failure'); END")
        failed_json = await self.client.post("/api/chat", json={"session_id": session, "message": "新问题"})
        self.assertEqual(failed_json.status_code, 503)
        pending = (await self.client.get(f'/api/sessions/{session}/checkpoint')).json()['checkpoint']
        failed_stream = await self.client.post(f'/api/sessions/{session}/resume/stream', json={'run_id': pending['run_id']})
        names = [name for name, _ in parse_sse(failed_stream.text)]
        self.assertEqual(names[-1], "error")
        self.assertNotIn("done", names)
        reset = await self.client.post(f"/api/sessions/{session}/reset")
        self.assertEqual(reset.status_code, 503)
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertFalse(state["busy"])
        self.assertEqual(state["messages"][-1]["content"], "原来的回复")
        await self.restart_app()
        restored = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(len(restored["messages"]), 3)
        self.assertEqual(restored["messages"][-1]["content"], "原来的回复")

    async def test_cold_session_loading_still_rejects_overlapping_questions(self):
        self.responses = [text_stream("回复")]
        session = await self.new_session()
        await self.restart_app()
        restored = await self.client.get(f"/api/sessions/{session}")
        self.assertEqual(restored.status_code, 200)
        self.app.state.sessions.clear()  # 再次从同一个磁盘会话冷加载。
        self.gate.clear()
        tasks = [asyncio.create_task(self.client.post("/api/chat", json={"session_id": session, "message": f"问题{index}"})) for index in range(2)]
        try:
            await asyncio.wait_for(self.entered.wait(), timeout=10)
            first_done, _ = await asyncio.wait(tasks, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            self.assertEqual(len(first_done), 1)
            self.assertEqual(next(iter(first_done)).result().status_code, 409)
        finally:
            self.gate.set()
            results = await asyncio.gather(*tasks)
        self.assertEqual(sorted(response.status_code for response in results), [200, 409])

    async def test_database_lock_wait_does_not_block_health_or_release_busy_early(self):
        self.responses = [text_stream("等待数据库后保存的回复")]
        session = await self.new_session()
        blocker = sqlite3.connect(self.db_path)
        blocker.execute("BEGIN IMMEDIATE")  # 另一个连接暂时占用写锁。
        entered = threading.Event()
        # 新执行的第一笔写入是检查点；同样应在线程等待锁、保持busy且让健康检查响应。
        begin_checkpoint = self.app.state.checkpoints.begin

        def observe_write(*args):
            entered.set()
            return begin_checkpoint(*args)  # 仍然执行真实 SQLite 写入。

        with patch.object(self.app.state.checkpoints, 'begin', side_effect=observe_write):
            pending = asyncio.create_task(self.client.post("/api/chat", json={"session_id": session, "message": "问题"}))
            try:
                self.assertTrue(await anyio.to_thread.run_sync(entered.wait, 3))
                health = await asyncio.wait_for(self.client.get("/api/health"), timeout=1)
                self.assertEqual(health.status_code, 200)
                reset = await self.client.post(f"/api/sessions/{session}/reset")
                self.assertEqual(reset.status_code, 409)
                self.assertFalse(pending.done(), "锁释放前应等待保存，不能提前结束请求")
            finally:
                blocker.rollback()
                blocker.close()
                response = await pending
        self.assertEqual(response.status_code, 200)
        await self.restart_app()
        restored = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(restored["messages"][-1]["content"], "等待数据库后保存的回复")

    async def test_json_endpoint_saves_the_answer_for_a_followup(self):
        self.responses = [text_stream("请提供订单号。"), text_stream("已收到订单号。")]
        session = await self.new_session()
        first = await self.client.post("/api/chat", json={"session_id": session, "message": "帮我查订单"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["answer"], "请提供订单号。")
        second = await self.client.post("/api/chat", json={"session_id": session, "message": "O1001"})
        self.assertEqual(second.status_code, 200)
        self.assertEqual([m["role"] for m in self.requests[1]["messages"]], ["system", "user", "assistant", "user"])

    async def test_stream_endpoint_pairs_fragmented_tools_and_commits_only_done(self):
        self.responses = [
            [chunk(calls=[tool_delta(0, "call-a", "get_order", '{"order_id":')]), chunk(calls=[tool_delta(0, arguments='"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"],
            text_stream("O1001 已发货。"),
        ]
        session = await self.new_session()
        response = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "查 O1001"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = parse_sse(response.text)
        self.assertIn("tool_result", [name for name, _ in events])
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1]["answer"], "O1001 已发货。")
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertFalse(state["busy"])
        self.assertEqual([m["role"] for m in state["messages"]], ["system", "user", "assistant", "tool", "assistant"])
        self.assertTrue(all(stream.closed for stream in self.streams))

    async def test_sessions_have_separate_histories_and_reset_removes_old_messages(self):
        self.responses = [text_stream("第一段回复"), text_stream("第二段回复")]
        first, second = await self.new_session(), await self.new_session()
        await self.client.post("/api/chat", json={"session_id": first, "message": "第一段问题"})
        await self.client.post("/api/chat", json={"session_id": second, "message": "第二段问题"})
        self.assertEqual(len(self.requests[1]["messages"]), 2)
        result = await self.client.post(f"/api/sessions/{first}/reset")
        self.assertEqual(result.status_code, 200)
        state = (await self.client.get(f"/api/sessions/{first}")).json()
        self.assertEqual([m["role"] for m in state["messages"]], ["system"])

    async def test_busy_session_rejects_overlap_without_blocking_health_requests(self):
        self.responses = [text_stream("已完成。")]
        session = await self.new_session()
        self.gate.clear()
        pending = asyncio.create_task(self.client.post("/api/chat/stream", json={"session_id": session, "message": "查订单"}))
        try:
            # 第一次调用 SDK 有惰性导入开销；等待测试闸门与接口响应耗时分开。
            await asyncio.wait_for(self.entered.wait(), timeout=10)
            overlap = await self.client.post("/api/chat", json={"session_id": session, "message": "另一个问题"})
            reset = await self.client.post(f"/api/sessions/{session}/reset")
            health = await asyncio.wait_for(self.client.get("/api/health"), timeout=1)
            deleted = await self.client.delete(f"/api/sessions/{session}")
            self.assertEqual(overlap.status_code, 409)
            self.assertEqual(reset.status_code, 409)
            self.assertEqual(deleted.status_code, 409)
            self.assertEqual(health.status_code, 200)
            self.assertNotIn("offline-only-key", health.text)
        finally:
            self.gate.set()
            await pending

    async def test_stream_error_keeps_history_and_releases_busy_flag(self):
        self.responses = [[chunk(content="部分文字"), {"error": {"message": "failed offline-only-key", "type": "server_error"}}]]
        session = await self.new_session()
        response = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "问题"})
        events = parse_sse(response.text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("failed [已隐藏 API Key]", events[-1][1]["message"])
        self.assertNotIn("offline-only-key", response.text)
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertFalse(state["busy"])
        self.assertEqual(len(state["messages"]), 1)

    async def test_invalid_requests_and_unknown_sessions_do_not_reach_the_model(self):
        session = await self.new_session()
        for message in ["  ", "x" * 2001, 123]:
            with self.subTest(message_type=type(message).__name__):
                response = await self.client.post("/api/chat", json={"session_id": session, "message": message})
                self.assertEqual(response.status_code, 422)
        response = await self.client.post("/api/chat", json={"session_id": session, "message": "问题", "messages": []})
        self.assertEqual(response.status_code, 422)
        missing = await self.client.post("/api/chat", json={"session_id": "unknown", "message": "问题"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(self.requests, [])

    async def test_root_serves_the_page_and_project_secrets_are_not_static_files(self):
        page = await self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertEqual((await self.client.get("/.env")).status_code, 404)
        self.assertEqual((await self.client.get("/static/%2e%2e/.env")).status_code, 404)
        self.assertEqual((await self.client.get("/data/chat.db")).status_code, 404)

    async def test_disconnect_cancels_a_waiting_model_stream_and_unlocks_the_session(self):
        # ASGITransport 会缓存完整响应；此测试直接发送 ASGI 断连消息，
        # 覆盖真实服务器使用的两种协议版本，以及模型迟迟不出字的场景。
        for endpoint, spec_version in (("/api/chat/stream", "2.3"), ("/api/chat/stream", "2.4"), ("/api/chat", "2.4")):
            with self.subTest(endpoint=endpoint, spec_version=spec_version):
                self.responses.append(text_stream("不应该保存的回复"))
                self.entered.clear()
                self.gate.clear()
                session = await self.new_session()
                incoming = asyncio.Queue()
                await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "测试断连"}).encode(), "more_body": False})
                sent = []

                async def send(event):
                    sent.append(event)

                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": endpoint, "raw_path": endpoint.encode(), "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
                task = asyncio.create_task(self.app(scope, incoming.get, send))
                try:
                    await asyncio.wait_for(self.entered.wait(), timeout=10)
                    await incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(task, timeout=2)
                finally:
                    self.gate.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                state = (await self.client.get(f"/api/sessions/{session}")).json()
                self.assertFalse(state["busy"])
                self.assertEqual(len(state["messages"]), 1)
                self.assertTrue(self.streams[-1].closed)
                self.assertTrue(any(event["type"] == "http.response.start" for event in sent))

    async def test_disconnect_during_summary_discards_draft_and_unlocks_session(self):
        # ASGITransport 会缓存完整响应；此测试直接发送 ASGI 断连消息，
        # 覆盖真实服务器使用的两种协议版本，以及模型迟迟不出字的场景。
        for endpoint, spec_version in (("/api/chat/stream", "2.3"), ("/api/chat/stream", "2.4"), ("/api/chat", "2.4")):
            with self.subTest(endpoint=endpoint, spec_version=spec_version):
                self.responses.append(self.summary_response())
                self.entered.clear()
                self.gate.clear()
                session = await self.new_session()
                original = await self.seed_long_history(session)
                incoming = asyncio.Queue()
                await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "测试断连"}).encode(), "more_body": False})
                sent = []

                async def send(event):
                    sent.append(event)

                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": endpoint, "raw_path": endpoint.encode(), "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
                task = asyncio.create_task(self.app(scope, incoming.get, send))
                try:
                    await asyncio.wait_for(self.entered.wait(), timeout=10)
                    await incoming.put({"type": "http.disconnect"})
                    await asyncio.wait_for(task, timeout=2)
                finally:
                    self.gate.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                state = (await self.client.get(f"/api/sessions/{session}")).json()
                self.assertFalse(state["busy"])
                self.assertEqual(state["messages"], original)
                self.assertEqual(self.app.state.store.load_session(session)["memory"], ContextMemory())
                self.assertTrue(any(event["type"] == "http.response.start" for event in sent))

    async def test_json_disconnect_during_stream_cleanup_does_not_save(self):
        # 真实 SDK 会保护关闭连接的过程。断连落在这里时，关闭仍会完成，
        # 但随后必须在原取消范围内检查取消，不能再把本轮写入数据库。
        self.responses = [text_stream("不应该保存的回复")]
        session = await self.new_session()
        close_entered, close_release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_close = AsyncEventBytes.aclose

        async def delayed_close(stream):
            with anyio.CancelScope(shield=True):
                close_entered.set()
                await close_release.wait()
                await original_close(stream)

        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "测试关闭时断连"}).encode(), "more_body": False})
        sent = []

        async def receive():
            event = await incoming.get()
            if event["type"] == "http.disconnect":
                disconnected.set()
            return event

        async def send(event):
            sent.append(event)

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat", "raw_path": b"/api/chat", "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
        with patch.object(AsyncEventBytes, "aclose", delayed_close):
            task = asyncio.create_task(self.app(scope, receive, send))
            try:
                await asyncio.wait_for(close_entered.wait(), timeout=10)
                await incoming.put({"type": "http.disconnect"})
                await asyncio.wait_for(disconnected.wait(), timeout=2)
                close_release.set()
                await asyncio.wait_for(task, timeout=2)
            finally:
                close_release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertFalse(state["busy"])
        self.assertEqual(len(state["messages"]), 1)
        self.assertTrue(self.streams[-1].closed)
        self.assertIn(499, [event["status"] for event in sent if event["type"] == "http.response.start"])
        await self.restart_app()
        restored = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(restored["messages"], state["messages"])

    async def test_failed_done_send_keeps_the_already_committed_turn_recoverable(self):
        self.responses = [text_stream("完整回复")]
        session = await self.new_session()
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": json.dumps({"session_id": session, "message": "问题"}).encode(), "more_body": False})

        async def send(event):
            if b"event: done" in event.get("body", b""):
                raise OSError("模拟客户端在完成事件发出前断开")

        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat/stream", "raw_path": b"/api/chat/stream", "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 12345), "server": ("testserver", 80)}
        # 第七课必须先持久化再确认完成：最终发送失败不能撤销已经提交的事务。
        try:
            await self.app(scope, incoming.get, send)
        except (OSError, ExceptionGroup):
            pass
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertFalse(state["busy"])
        self.assertEqual(len(state["messages"]), 3)
        self.assertTrue(self.streams[-1].closed)
        await self.restart_app()
        restored = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(restored["messages"][-1]["content"], "完整回复")

    async def test_truncated_tool_stream_is_not_executed_or_saved(self):
        self.responses = [[chunk(calls=[tool_delta(0, "call-a", "get_order", '{"order_id":')]), chunk(finish="length"), "[DONE]"]]
        session = await self.new_session()
        with patch("agent_service.execute_tool") as execute:
            response = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "查 O1001"})
            execute.assert_not_called()
        self.assertEqual(parse_sse(response.text)[-1][0], "error")
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(len(state["messages"]), 1)
        self.assertFalse(state["busy"])

    async def test_json_model_error_returns_502_and_missing_configuration_returns_503(self):
        self.responses = [[{"error": {"message": "provider failed", "type": "server_error"}}]]
        session = await self.new_session()
        response = await self.client.post("/api/chat", json={"session_id": session, "message": "问题"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("provider failed", response.json()["detail"])
        with patch.object(self.app.state, "client", None):
            missing = await self.client.post("/api/chat/stream", json={"session_id": session, "message": "问题"})
            self.assertEqual(missing.status_code, 503)
        state = (await self.client.get(f"/api/sessions/{session}")).json()
        self.assertEqual(len(state["messages"]), 1)
        self.assertFalse(state["busy"])


if __name__ == "__main__":
    unittest.main()
