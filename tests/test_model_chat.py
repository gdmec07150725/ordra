"""验证真实 SDK 的请求组装和错误处理；HTTP 响应在本机模拟。

这些检查不会调用 DeepSeek，也不会读取真实 .env 或消耗模型额度。
第一遍学习可以跳过本文件，先阅读根目录的 03_model_chat.py。
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


# 文件名以数字开头，测试中通过路径加载它，避免改动学习文件的命名。
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "03_model_chat.py"
spec = importlib.util.spec_from_file_location("model_chat_lesson", SCRIPT_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


def response_body(content="你好，这是离线模拟响应。", finish_reason="stop"):
    """构造符合对话接口结构的响应，供真实 SDK 解析。"""
    return {
        "id": "offline-chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


class ModelChatTests(unittest.TestCase):
    """只检查对外行为：请求发往哪里、显示什么、失败是否被识别。"""

    def run_lesson(self, args=None, key="offline-only-key", status=200, body=None, timeout=False, text_body=None):
        requests = []

        def handle(request):
            requests.append(request)
            if timeout:
                raise httpx2.ReadTimeout("offline timeout", request=request)
            if text_body is not None:
                return httpx2.Response(status, text=text_body)
            payload = response_body() if body is None else body
            return httpx2.Response(status, json=payload)

        def make_client(**kwargs):
            # 保留真实 SDK，只替换最外层的 HTTP 传输，所有请求都留在本机。
            return OpenAI(
                **kwargs,
                http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
            )

        captured = io.StringIO()
        variables = {"DEEPSEEK_API_KEY": key, "DEEPSEEK_MODEL": "deepseek-flash"}
        command = [str(SCRIPT_PATH)] + (["请介绍自己"] if args is None else args)
        with (
            patch.dict(os.environ, variables, clear=True),
            patch.object(sys, "argv", command),
            patch.object(lesson, "load_dotenv", return_value=False),
            patch.object(lesson, "OpenAI", side_effect=make_client) as constructor,
            contextlib.redirect_stdout(captured),
        ):
            exit_code = lesson.main()
        return exit_code, captured.getvalue(), requests, constructor.call_count

    def test_real_sdk_targets_deepseek_and_displays_the_answer(self):
        code, output, requests, _ = self.run_lesson()
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer offline-only-key")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "请介绍自己"})
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["thinking"]["type"], "disabled")
        self.assertIn("你好，这是离线模拟响应。", output)
        self.assertNotIn("offline-only-key", output)

    def test_missing_key_stops_before_creating_a_client(self):
        code, output, requests, calls = self.run_lesson(key="")
        self.assertEqual(code, 1)
        self.assertIn("DEEPSEEK_API_KEY", output)
        self.assertEqual(calls, 0)
        self.assertEqual(requests, [])

    def test_empty_question_does_not_send_a_request(self):
        code, output, requests, calls = self.run_lesson(args=["   "])
        self.assertEqual(code, 1)
        self.assertIn("问题不能为空", output)
        self.assertEqual(calls, 0)
        self.assertEqual(requests, [])

    def test_auth_failure_displays_provider_details_and_masks_the_key(self):
        code, output, requests, _ = self.run_lesson(
            status=401,
            body={"error": {"message": "rejected offline-only-key", "type": "authentication_error"}},
        )
        self.assertEqual(code, 1)
        self.assertIn("401", output)
        self.assertIn("rejected [已隐藏 API Key]", output)
        self.assertIn("authentication_error", output)
        self.assertNotIn("offline-only-key", output)
        self.assertEqual(len(requests), 1)

    def test_plain_text_error_displays_the_original_response(self):
        code, output, _, _ = self.run_lesson(status=503, text_body="Model service temporarily unavailable")
        self.assertEqual(code, 1)
        self.assertIn("503", output)
        self.assertIn("Model service temporarily unavailable", output)

    def test_empty_error_response_falls_back_to_the_sdk_exception(self):
        code, output, _, _ = self.run_lesson(status=500, text_body="")
        self.assertEqual(code, 1)
        self.assertIn("Error code: 500", output)

    def test_timeout_is_reported_without_automatic_retries(self):
        code, output, requests, _ = self.run_lesson(timeout=True)
        self.assertEqual(code, 1)
        self.assertIn("超时", output)
        self.assertEqual(len(requests), 1)

    def test_empty_answer_is_not_reported_as_success(self):
        code, output, _, _ = self.run_lesson(body=response_body(content=None))
        self.assertNotEqual(code, 0)
        self.assertIn("回答", output)

    def test_truncated_answer_is_marked_incomplete(self):
        code, output, _, _ = self.run_lesson(body=response_body(finish_reason="length"))
        self.assertNotEqual(code, 0)
        self.assertIn("length", output)
        self.assertIn("你好，这是离线模拟响应。", output)


if __name__ == "__main__":
    unittest.main()
