"""使用真实 SDK 解析模拟 SSE，验证分片显示、工具拼接与中断恢复。

模拟的是 HTTP 字节流，应用仍执行实际的流式读取和订单函数。
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


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "06_stream_agent.py"
spec = importlib.util.spec_from_file_location("stream_agent_lesson", SCRIPT_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


def chunk(content=None, calls=None, finish=None, usage=None, empty_choices=False):
    delta = {}
    if content is not None:
        delta["content"] = content
    if calls is not None:
        delta["tool_calls"] = calls
    return {
        "id": "stream-test-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [] if empty_choices else [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": usage,
    }


def tool_delta(index=0, call_id=None, name=None, arguments=None):
    part = {"index": index, "function": {}}
    if call_id is not None:
        part.update(id=call_id, type="function")
    if name is not None:
        part["function"]["name"] = name
    if arguments is not None:
        part["function"]["arguments"] = arguments
    return part


def text_stream(text="收到下一条问题。"):
    return [chunk(content=text), chunk(finish="stop"), "[DONE]"]


class EventBytes(httpx2.SyncByteStream):
    """按事件逐次提供字节，允许测试在后续片段到达前观察终端输出。"""
    def __init__(self, frames, before_frame):
        self.frames = frames
        self.before_frame = before_frame
        self.closed = False

    def __iter__(self):
        for index, frame in enumerate(self.frames):
            self.before_frame(index)
            if isinstance(frame, BaseException):
                raise frame
            data = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
            yield f"data: {data}\n\n".encode("utf-8")

    def close(self):
        self.closed = True


class StreamAgentTests(unittest.TestCase):
    def run_session(self, responses, inputs=None, observe=None):
        requests, streams = [], []
        captured = io.StringIO()

        def handle(request):
            request_number = len(requests)
            requests.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertLess(request_number, len(responses), "出现了额外请求")
            self.assertTrue(requests[-1]["stream"])
            self.assertTrue(requests[-1]["stream_options"]["include_usage"])

            def before_frame(index):
                if observe:
                    observe(request_number, index, captured.getvalue())

            stream = EventBytes(responses[request_number], before_frame)
            streams.append(stream)
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        def make_client(**kwargs):
            return OpenAI(
                **kwargs,
                http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
            )

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-only-key"}, clear=True),
            patch.object(lesson, "load_dotenv", return_value=False),
            patch.object(lesson, "OpenAI", side_effect=make_client),
            patch("builtins.input", side_effect=inputs or ["问题一", "问题二", "/exit"]),
            contextlib.redirect_stdout(captured),
        ):
            code = lesson.main()
        self.assertTrue(all(stream.closed for stream in streams), "响应结束或中断后必须关闭连接")
        return code, captured.getvalue(), requests

    def test_text_is_displayed_before_later_chunks_and_saved_as_one_message(self):
        def observe(request_number, index, output):
            if request_number == 0 and index == 1:
                self.assertIn("你好，", output, "下一片到达前必须显示上一片文字")

        code, output, requests = self.run_session([
            [chunk(content="你好，"), chunk(content="请提供订单号。"), chunk(finish="stop", usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}), "[DONE]"],
            text_stream(),
        ], observe=observe)
        self.assertEqual(code, 0)
        self.assertEqual(output.count("你好，请提供订单号。"), 1)
        self.assertEqual(requests[1]["messages"][2], {"role": "assistant", "content": "你好，请提供订单号。"})
        self.assertIn("合计 28", output)

    def test_interleaved_tool_fragments_are_joined_by_index_before_execution(self):
        frames = [
            chunk(calls=[tool_delta(0, "call-a", "get_", '{"order_id":')]),
            chunk(calls=[tool_delta(1, "call-b", "get_order", '{"order_id":"O')]),
            chunk(calls=[tool_delta(0, name="order", arguments='"O1001"}')]),
            chunk(calls=[tool_delta(1, arguments='1002"}')]),
            chunk(finish="tool_calls"),
            "[DONE]",
        ]

        def observe(request_number, index, output):
            if request_number == 0:
                self.assertNotIn("工具结果：", output, "整个响应收完前不能执行工具")

        code, _, requests = self.run_session([frames, text_stream("两笔订单已查询。")], inputs=["查两笔订单", "/exit"], observe=observe)
        self.assertEqual(code, 0)
        assistant, first_result, second_result = requests[1]["messages"][-3:]
        self.assertEqual([call["function"]["name"] for call in assistant["tool_calls"]], ["get_order", "get_order"])
        self.assertEqual([first_result["tool_call_id"], second_result["tool_call_id"]], ["call-a", "call-b"])
        self.assertEqual(json.loads(first_result["content"])["status"], "已发货")
        self.assertEqual(json.loads(second_result["content"])["status"], "待发货")

    def test_truncated_tool_arguments_are_not_executed_or_saved(self):
        code, output, requests = self.run_session([
            [chunk(calls=[tool_delta(0, "call-a", "get_order", '{"order_id":"O1001"}')]), chunk(finish="length"), "[DONE]"],
            text_stream(),
        ])
        self.assertEqual(code, 0)
        self.assertIn("length", output)
        self.assertNotIn("工具结果：", output)
        self.assertEqual([m["role"] for m in requests[1]["messages"]], ["system", "user"])

    def test_eof_without_finish_reason_does_not_commit_partial_text(self):
        code, output, requests = self.run_session([[chunk(content="尚未完成的片段")], text_stream()])
        self.assertEqual(code, 0)
        self.assertIn("尚未完成的片段", output)
        self.assertIn("结束状态", output)
        self.assertEqual(len(requests[1]["messages"]), 2)

    def test_sse_error_is_shown_and_failed_turn_is_not_saved(self):
        code, output, requests = self.run_session([
            [chunk(content="部分文字"), {"error": {"message": "stream rejected offline-only-key", "type": "server_error"}}],
            text_stream(),
        ])
        self.assertEqual(code, 0)
        self.assertIn("stream rejected [已隐藏 API Key]", output)
        self.assertIn("server_error", output)
        self.assertNotIn("offline-only-key", output)
        self.assertEqual(len(requests[1]["messages"]), 2)

    def test_transport_error_closes_stream_and_allows_another_question(self):
        code, output, requests = self.run_session([
            [chunk(content="部分文字"), httpx2.ReadError("offline connection closed")],
            text_stream(),
        ])
        self.assertEqual(code, 0)
        self.assertIn("连接", output)
        self.assertEqual(len(requests[1]["messages"]), 2)

    def test_keyboard_interrupt_during_stream_leaves_history_unchanged(self):
        code, output, requests = self.run_session([[chunk(content="部分文字"), KeyboardInterrupt()], text_stream()])
        self.assertEqual(code, 0)
        self.assertIn("已中断", output)
        self.assertEqual(len(requests[1]["messages"]), 2)

    def test_usage_only_chunk_does_not_hide_the_completed_answer(self):
        code, output, _ = self.run_session([
            [chunk(content="你好"), chunk(finish="stop"), chunk(empty_choices=True, usage={"prompt_tokens": 18, "completion_tokens": 2, "total_tokens": 20}), "[DONE]"],
        ], inputs=["你好", "/exit"])
        self.assertEqual(code, 0)
        self.assertIn("合计 20", output)

    def test_tool_call_without_id_is_rejected_before_execution(self):
        code, output, requests = self.run_session([
            [chunk(calls=[tool_delta(name="get_order", arguments='{"order_id":"O1001"}')]), chunk(finish="tool_calls"), "[DONE]"],
            text_stream(),
        ])
        self.assertEqual(code, 0)
        self.assertNotIn("工具结果：", output)
        self.assertIn("调用 ID", output)
        self.assertEqual(len(requests[1]["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
