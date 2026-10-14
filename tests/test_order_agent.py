"""第三课的离线验证：使用真实 SDK 和订单函数，只模拟 HTTP 响应。

这些测试不读取 .env，不连接 DeepSeek。先读主示例，再按需阅读测试。
"""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "04_order_agent.py"
spec = importlib.util.spec_from_file_location("order_agent_lesson", SCRIPT_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


def tool_call(call_id="call-1", name="get_order", arguments='{"order_id":"O1001"}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def completion(content=None, calls=None, finish_reason=None):
    """按官方响应结构构造离线模型输出，交给 SDK 解析。"""
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "offline-order-chat",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
            "message": message,
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class OrderAgentTests(unittest.TestCase):
    def run_lesson(self, responses, status=200):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertLessEqual(len(requests), len(responses), "出现了意外的额外请求")
            return httpx2.Response(status, json=responses[len(requests) - 1])

        def make_client(**kwargs):
            return OpenAI(
                **kwargs,
                http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
            )

        captured = io.StringIO()
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-only-key"}, clear=True),
            patch.object(sys, "argv", [str(SCRIPT_PATH), "我的 O1001 订单到哪里了？"]),
            patch.object(lesson, "load_dotenv", return_value=False),
            patch.object(lesson, "OpenAI", side_effect=make_client),
            contextlib.redirect_stdout(captured),
        ):
            code = lesson.main()
        return code, captured.getvalue(), requests

    def test_order_result_reaches_second_request_with_matching_call_id(self):
        code, output, requests = self.run_lesson([
            completion(calls=[tool_call()]),
            completion(content="O1001 已发货，包裹已到达武汉转运中心。"),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "get_order")
        self.assertFalse(requests[0]["stream"])
        self.assertEqual(requests[0]["thinking"]["type"], "disabled")
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user", "assistant", "tool"])
        assistant, result = requests[1]["messages"][-2:]
        self.assertEqual(assistant["tool_calls"][0]["id"], result["tool_call_id"])
        data = json.loads(result["content"])
        self.assertEqual(data["order_id"], "O1001")
        self.assertEqual(data["status"], "已发货")
        self.assertIn("武汉转运中心", data["logistics"])
        self.assertIn("O1001 已发货", output)

    def test_every_tool_call_gets_its_own_result_before_next_request(self):
        code, _, requests = self.run_lesson([
            completion(calls=[tool_call(), tool_call("call-2", arguments='{"order_id":"O1002"}')]),
            completion(content="两笔订单已查询。"),
        ])
        self.assertEqual(code, 0)
        results = requests[1]["messages"][-2:]
        self.assertEqual([item["tool_call_id"] for item in results], ["call-1", "call-2"])
        self.assertEqual([json.loads(item["content"])["status"] for item in results], ["已发货", "待发货"])

    def test_plain_answer_finishes_without_calling_an_order_tool(self):
        code, output, requests = self.run_lesson([completion(content="请提供订单号。")])
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn("请提供订单号", output)

    def test_unknown_tool_returns_an_error_instead_of_dispatching_it(self):
        code, _, requests = self.run_lesson([
            completion(calls=[tool_call(name="delete_order")]),
            completion(content="当前只能查询订单。"),
        ])
        self.assertEqual(code, 0)
        result = json.loads(requests[1]["messages"][-1]["content"])
        self.assertIn("error", result)
        self.assertIn("delete_order", result["error"])
        self.assertNotIn("status", result)

    def test_invalid_arguments_return_tool_errors_for_the_model_to_handle(self):
        # 每种输入会揭示一个不同的参数校验遗漏。
        for arguments in ['{', '[]', '{}', '{"order_id":1001}', '{"order_id":" "}', '{"order_id":"O1001","extra":true}']:
            with self.subTest(arguments=arguments):
                code, _, requests = self.run_lesson([
                    completion(calls=[tool_call(arguments=arguments)]),
                    completion(content="请核对订单号。"),
                ])
                self.assertEqual(code, 0)
                result = json.loads(requests[1]["messages"][-1]["content"])
                self.assertIn("error", result)
                self.assertNotIn("status", result)

    def test_missing_order_is_sent_back_as_a_business_error(self):
        code, _, requests = self.run_lesson([
            completion(calls=[tool_call(arguments='{"order_id":"O9999"}')]),
            completion(content="没有查到 O9999，请核对订单号。"),
        ])
        self.assertEqual(code, 0)
        result = json.loads(requests[1]["messages"][-1]["content"])
        self.assertIn("O9999", result["error"])
        self.assertNotIn("status", result)

    def test_repeated_calls_stop_at_the_round_limit(self):
        code, output, requests = self.run_lesson([completion(calls=[tool_call()])] * 4)
        self.assertNotEqual(code, 0)
        self.assertEqual(len(requests), 4)
        self.assertIn("上限", output)
        # 最后一轮不再执行无法发回模型的工具。
        self.assertEqual(output.count("工具结果："), 3)

    def test_truncated_tool_call_is_not_executed(self):
        code, output, requests = self.run_lesson([
            completion(calls=[tool_call()], finish_reason="length"),
        ])
        self.assertNotEqual(code, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn("length", output)
        self.assertNotIn("工具结果：", output)

    def test_provider_error_details_are_displayed_with_the_key_masked(self):
        code, output, requests = self.run_lesson([
            {"error": {"message": "rejected offline-only-key", "type": "authentication_error"}},
        ], status=401)
        self.assertEqual(code, 1)
        self.assertIn("401", output)
        self.assertIn("rejected [已隐藏 API Key]", output)
        self.assertNotIn("offline-only-key", output)
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
