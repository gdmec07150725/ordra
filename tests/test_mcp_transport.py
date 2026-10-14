"""Real local MCP subprocesses; all knowledge and caches belong to fixtures."""
import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mcp import Client, StdioServerParameters
import mcp.client.stdio as stdio


PROJECT = Path(__file__).resolve().parents[1]


class MCPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(importlib.util.find_spec("mcp_tool_client"),
                             "The real MCP client transport must be implemented")
        from mcp_tool_client import LocalMCPClient
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.knowledge = self.root / "knowledge"
        self.knowledge.mkdir()
        self.original = "# 测试退货政策\n\n## 退货条件\n签收后7天内，商品完好可申请退货。\n"
        (self.knowledge / "returns.md").write_text(self.original, encoding="utf-8")
        self.processes = []
        self.spawn_parameters = []
        self.server_override = None
        real_spawn = stdio._create_platform_compatible_process

        async def record_spawn(**kwargs):
            if self.server_override is not None:
                kwargs["args"] = ["-u", str(self.server_override)]
            self.spawn_parameters.append(kwargs)
            process = await real_spawn(**kwargs)
            self.processes.append(process)
            return process

        self.patch_spawn = patch.object(stdio, "_create_platform_compatible_process", record_spawn)
        self.patch_spawn.start()
        self.addCleanup(self.patch_spawn.stop)
        self.client = LocalMCPClient(self.knowledge, timeout_seconds=6)
        self.addAsyncCleanup(self.client.close)

    def fixture_server(self, body):
        """Use the real SDK with a deliberately broken test peer, never production hooks."""
        self.server_override = self.root / "fixture_server.py"
        self.server_override.write_text(
            "from mcp.server import MCPServer\n"
            "import asyncio\nfrom pathlib import Path\n"
            "server = MCPServer('transport-fixture')\n" + body.replace("-> dict:", "-> dict[str, str]:")
            + "\nserver.run()\n", encoding="utf-8")

    async def wait_for_file(self, path):
        async with asyncio.timeout(6):
            while not path.exists():
                await asyncio.sleep(0.01)

    async def test_malformed_business_object_is_rejected(self):
        self.fixture_server("@server.tool()\ndef get_order(order_id: str) -> dict:\n    return {}\n"
                            "@server.tool()\ndef search_knowledge(query: str) -> dict[str, object]:\n"
                            "    return {'query': query, 'results': [{}]}\n")
        for name, arguments in [("get_order", {"order_id": "O1003"}), ("search_knowledge", {"query": "退货"})]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "结果"):
                await self.client.call_tool(name, arguments)

    async def test_cancelled_call_does_not_cancel_other_caller_or_leak_process(self):
        marker = self.root / "started"
        self.fixture_server(
            "@server.tool()\nasync def get_order(order_id: str) -> dict:\n"
            f"    Path({str(marker)!r}).write_text('started')\n"
            "    if order_id == 'slow':\n        await asyncio.sleep(30)\n"
            "    return {'order_id': order_id, 'status': '已签收', 'logistics': '客户已签收'}\n")
        await self.client.start()
        waiting = asyncio.create_task(self.client.call_tool("get_order", {"order_id": "slow"}))
        await self.wait_for_file(marker)
        other = asyncio.create_task(self.client.call_tool("get_order", {"order_id": "O1003"}))
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual((await other)["status"], "已签收")
        await self.client.close()
        self.assertTrue(all(process.returncode is not None for process in self.processes))

    async def test_cancelled_close_still_waits_for_process_cleanup(self):
        marker = self.root / "shutting-down"
        self.fixture_server("@server.tool()\ndef get_order(order_id: str) -> dict:\n"
                            "    return {'order_id': order_id, 'status': '已签收', 'logistics': '客户已签收'}\n")
        script = self.server_override.read_text(encoding="utf-8")
        script = script.replace("server = MCPServer('transport-fixture')",
                                "from contextlib import asynccontextmanager\n"
                                "@asynccontextmanager\nasync def lifespan(server):\n"
                                "    yield\n"
                                f"    Path({str(marker)!r}).write_text('closing')\n"
                                "    await asyncio.sleep(0.5)\n"
                                "server = MCPServer('transport-fixture', lifespan=lifespan)")
        self.server_override.write_text(script, encoding="utf-8")
        await self.client.start()
        closing = asyncio.create_task(self.client.close())
        await self.wait_for_file(marker)
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertIsNotNone(self.processes[0].returncode, "Cancellation must finish child cleanup")

    async def test_timeout_cancels_request_and_next_call_still_works(self):
        from mcp_tool_client import MCPUnavailableError
        marker = self.root / "started"
        self.fixture_server(
            "@server.tool()\nasync def get_order(order_id: str) -> dict:\n"
            f"    Path({str(marker)!r}).write_text('started')\n"
            "    if order_id == 'slow':\n        await asyncio.sleep(30)\n"
            "    return {'order_id': order_id, 'status': '已签收', 'logistics': '客户已签收'}\n")
        await self.client.start()
        self.client.timeout_seconds = 0.2
        with self.assertRaises(MCPUnavailableError):
            await self.client.call_tool("get_order", {"order_id": "slow"})
        self.assertTrue(marker.exists())
        self.client.timeout_seconds = 6
        result = await self.client.call_tool("get_order", {"order_id": "O1003"})
        self.assertEqual(result["status"], "已签收")

    async def test_reconnect_refreshes_definition_before_calling_with_old_schema(self):
        from mcp_tool_client import MCPToolsChangedError, MCPUnavailableError
        self.fixture_server(
            "@server.tool(description='original')\ndef get_order(order_id: str) -> dict:\n"
            "    return {'order_id': order_id, 'status': '已签收', 'logistics': '客户已签收'}\n")
        old = (await self.client.list_tools())[0]
        self.processes[0].kill()
        await self.processes[0].wait()
        with self.assertRaises(MCPUnavailableError):
            await self.client.call_tool("get_order", {"order_id": "O1003"}, expected_schema=old)
        self.server_override.write_text(self.server_override.read_text(encoding="utf-8")
                                        .replace("description='original'", "description='changed'"), encoding="utf-8")
        with self.assertRaises(MCPToolsChangedError):
            await self.client.call_tool("get_order", {"order_id": "O1003"}, expected_schema=old)
        fresh = (await self.client.list_tools())[0]
        self.assertEqual(fresh["function"]["description"], "changed")
        self.assertEqual((await self.client.call_tool("get_order", {"order_id": "O1003"},
                                                     expected_schema=fresh))["status"], "已签收")

    async def test_real_list_tools_and_business_roundtrip_preserve_sources(self):
        await self.client.start()
        definitions = await self.client.list_tools()
        by_name = {tool["function"]["name"]: tool for tool in definitions}
        self.assertEqual(set(by_name), {"get_order", "search_knowledge"})
        self.assertEqual(by_name["get_order"]["function"]["parameters"]["required"], ["order_id"])
        self.assertFalse(by_name["search_knowledge"]["function"]["parameters"]["additionalProperties"])
        result = await self.client.call_tool("get_order", {"order_id": "O1003"})
        self.assertEqual(result, {"order_id": "O1003", "status": "已签收", "logistics": "客户已签收"})
        found = await self.client.call_tool("search_knowledge", {"query": "退货 条件"})
        self.assertEqual(found["document_count"], 1)
        self.assertEqual(found["search_mode"], "keywords")
        hit = found["results"][0]
        self.assertEqual(hit["title"], "测试退货政策")
        self.assertEqual(hit["text"], self.original[hit["start"]:hit["end"]])
        self.assertRegex(hit["source"], r"^kb:[0-9a-f]{12}:\d+$")
        self.assertRegex(hit["version"], r"^[0-9a-f]{12}$")
        self.assertEqual(len(self.processes), 1)

    async def test_raw_server_rejects_unknown_and_invalid_arguments(self):
        parameters = StdioServerParameters(command=sys.executable, args=[
            str(PROJECT / "mcp_tool_server.py"), "--knowledge-dir", str(self.knowledge)], cwd=PROJECT)
        async with Client(parameters, cache=None) as raw:
            for name, arguments in [
                ("delete_order", {"order_id": "O1003"}),
                ("get_order", {}),
                ("get_order", {"order_id": 1003}),
                ("get_order", {"order_id": "O1003", "extra": True}),
                ("search_knowledge", {"query": "x" * 101}),
            ]:
                with self.subTest(name=name, arguments=arguments):
                    result = await raw.call_tool(name, arguments)
                    self.assertTrue(result.is_error)

    async def test_client_rejects_unknown_and_invalid_arguments(self):
        for name, arguments in [
            ("delete_order", {"order_id": "O1003"}),
            ("get_order", {"order_id": "O1003", "extra": True}),
            ("get_order", {"order_id": None}),
            ("search_knowledge", {"query": ""}),
        ]:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                await self.client.call_tool(name, arguments)

    async def test_connection_failure_is_retryable_and_next_call_reconnects(self):
        from mcp_tool_client import MCPUnavailableError
        await self.client.start()
        self.processes[0].kill()
        await self.processes[0].wait()
        with self.assertRaises(MCPUnavailableError):
            await self.client.call_tool("get_order", {"order_id": "O1003"})
        self.assertEqual(len(self.processes), 1, "A failing call must not retry itself")
        result = await self.client.call_tool("get_order", {"order_id": "O1003"})
        self.assertEqual(result["status"], "已签收")
        self.assertEqual(len(self.processes), 2)

    async def test_per_run_schema_guard_rejects_stale_definition(self):
        from mcp_tool_client import MCPToolsChangedError
        definitions = await self.client.list_tools()
        original = next(tool for tool in definitions if tool["function"]["name"] == "get_order")
        stale = deepcopy(original)
        stale["function"]["parameters"]["properties"]["order_id"]["type"] = "integer"
        with self.assertRaises(MCPToolsChangedError):
            await self.client.call_tool("get_order", {"order_id": "O1003"}, expected_schema=stale)
        result = await self.client.call_tool("get_order", {"order_id": "O1003"}, expected_schema=original)
        self.assertEqual(result["status"], "已签收")

    async def test_concurrent_callers_share_connection_and_close_reaps_process(self):
        results = await asyncio.gather(*[
            self.client.call_tool("get_order", {"order_id": name})
            for name in ["O1001", "O1002", "O1003"]])
        self.assertEqual([row["status"] for row in results], ["已发货", "待发货", "已签收"])
        self.assertEqual(len(self.processes), 1)
        await asyncio.wait_for(asyncio.create_task(self.client.close()), 8)
        self.assertIsNotNone(self.processes[0].returncode)
        await self.client.close()
        from mcp_tool_client import MCPUnavailableError
        with self.assertRaises(MCPUnavailableError):
            await self.client.call_tool("get_order", {"order_id": "O1003"})

    async def test_subprocess_environment_omits_credentials(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-secret", "OPENAI_API_KEY": "fixture-secret",
                                       "PYTHONPATH": "fixture-injected-code"}):
            await self.client.start()
        environment = self.spawn_parameters[0]["env"]
        self.assertNotIn("DEEPSEEK_API_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(Path(self.spawn_parameters[0]["command"]), Path(sys.executable))
        self.assertIn(str(PROJECT / "mcp_tool_server.py"), self.spawn_parameters[0]["args"])


if __name__ == "__main__":
    unittest.main()
