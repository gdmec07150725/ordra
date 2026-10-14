"""第四课的会话行为测试：真实 SDK、模拟网络，不读取本机 .env。

重点验证历史是否传递、清空是否生效、失败后是否带入不完整消息。
"""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI

# 复用上一课测试中的接口响应样例构造函数。
from test_order_agent import completion, tool_call


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "05_chat_agent.py"
spec = importlib.util.spec_from_file_location("chat_agent_lesson", SCRIPT_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


class ChatAgentTests(unittest.TestCase):
    def run_session(self, inputs, responses):
        """模拟用户依次输入；只替换 input、配置来源和 HTTP 传输。"""
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertLessEqual(len(requests), len(responses), "出现了意外的额外请求")
            item = responses[len(requests) - 1]
            if isinstance(item, BaseException):
                raise item
            status, body = item if isinstance(item, tuple) else (200, item)
            return httpx2.Response(status, json=body)

        def make_client(**kwargs):
            return OpenAI(
                **kwargs,
                http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
            )

        captured = io.StringIO()
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-only-key"}, clear=True),
            patch.object(lesson, "load_dotenv", return_value=False),
            patch.object(lesson, "OpenAI", side_effect=make_client),
            patch("builtins.input", side_effect=inputs),
            contextlib.redirect_stdout(captured),
        ):
            code = lesson.main()
        return code, captured.getvalue(), requests

    def test_followup_includes_previous_questions_answers_and_tool_results(self):
        code, output, requests = self.run_session(
            ["帮我查订单", "订单号是 O1001", "我刚才提到哪个订单？", "/exit"],
            [
                completion(content="请提供订单号。"),
                completion(calls=[tool_call()]),
                completion(content="O1001 已发货。"),
                completion(content="你刚才提到的是 O1001。"),
            ],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 4)
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(requests[1]["messages"][1]["content"], "帮我查订单")
        self.assertEqual(requests[1]["messages"][2]["content"], "请提供订单号。")
        history = requests[3]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(history[4]["tool_calls"][0]["id"], history[5]["tool_call_id"])
        self.assertEqual(json.loads(history[5]["content"])["status"], "已发货")
        self.assertEqual(history[6]["content"], "O1001 已发货。")
        self.assertIn("你刚才提到的是 O1001", output)

    def test_reset_removes_previous_conversation_before_the_next_request(self):
        code, output, requests = self.run_session(
            ["记住订单号 O1001", "/reset", "我之前说了什么？", "/exit"],
            [completion(content="已收到订单号。"), completion(content="本轮还没有提供订单号。")],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 2)
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user"])
        self.assertEqual(requests[1]["messages"][1]["content"], "我之前说了什么？")
        self.assertIn("已清空", output)

    def test_history_command_shows_saved_messages_without_a_request(self):
        code, output, requests = self.run_session(
            ["你好", "/history", "/exit"],
            [completion(content="你好，请问需要查询哪笔订单？")],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn('"role": "user"', output)
        self.assertIn('"content": "你好"', output)
        self.assertIn('"role": "assistant"', output)

    def test_failed_tool_turn_does_not_pollute_the_next_question(self):
        code, output, requests = self.run_session(
            ["帮我查订单", "O1001", "请再告诉我需要什么信息", "/exit"],
            [
                completion(content="请提供订单号。"),
                completion(calls=[tool_call()]),
                (500, {"error": {"message": "server rejected offline-only-key"}}),
                completion(content="需要订单号。"),
            ],
        )
        self.assertEqual(code, 0)
        next_history = requests[3]["messages"]
        self.assertEqual([m["role"] for m in next_history], ["system", "user", "assistant", "user"])
        self.assertEqual(next_history[1]["content"], "帮我查订单")
        self.assertEqual(next_history[-1]["content"], "请再告诉我需要什么信息")
        self.assertIn("server rejected [已隐藏 API Key]", output)
        self.assertNotIn("offline-only-key", output)

    def test_truncated_reply_is_not_saved_as_a_completed_answer(self):
        code, output, requests = self.run_session(
            ["问题一", "问题二", "/exit"],
            [completion(content="一段不完整回答", finish_reason="length"), completion(content="第二个问题的回答")],
        )
        self.assertEqual(code, 0)
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user"])
        self.assertEqual(requests[1]["messages"][-1]["content"], "问题二")
        self.assertIn("length", output)

    def test_round_limit_stops_one_turn_and_allows_the_next_question(self):
        code, output, requests = self.run_session(
            ["一直查订单", "你好", "/exit"],
            [completion(calls=[tool_call()])] * 4 + [completion(content="你好。")],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 5)
        self.assertEqual([m["role"] for m in requests[4]["messages"]], ["system", "user"])
        self.assertIn("上限", output)

    def test_interrupted_request_keeps_the_session_available(self):
        code, output, requests = self.run_session(
            ["问题一", "问题二", "/exit"],
            [KeyboardInterrupt(), completion(content="第二个问题的回答")],
        )
        self.assertEqual(code, 0)
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user"])
        self.assertEqual(requests[1]["messages"][-1]["content"], "问题二")
        self.assertIn("已中断", output)

    def test_blank_input_and_eof_do_not_call_the_model(self):
        code, _, requests = self.run_session(["   ", EOFError()], [])
        self.assertEqual(code, 0)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
