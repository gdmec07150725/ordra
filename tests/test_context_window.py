"""第八课：按用户回合选择上下文，原始历史不能被改写。"""

from copy import deepcopy
import unittest

from context_window import build_context


class ContextWindowTests(unittest.TestCase):
    def test_keeps_system_and_latest_turn_without_modifying_nested_history(self):
        history = [
            {"role": "system", "content": "客服规则"},
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回复"},
            {"role": "user", "content": "查询 O1001"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "function": {"name": "get_order", "arguments": '{"order_id":"O1001"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "订单结果"},
            {"role": "assistant", "content": "最终回复"},
        ]
        original = deepcopy(history)
        messages, info = build_context(history, max_turns=1)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["content"], "查询 O1001")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], messages[3]["tool_call_id"])
        self.assertEqual(info, {"max_turns": 1, "total_turns": 2, "kept_turns": 1, "omitted_turns": 1, "stored_messages": 7, "request_messages": 6})
        messages.append({"role": "user", "content": "下一轮"})
        self.assertEqual(history, original)

    def test_short_or_empty_history_is_kept_in_a_separate_list(self):
        for history in ([], [{"role": "system", "content": "规则"}], [{"role": "system", "content": "规则"}, {"role": "user", "content": "问题"}, {"role": "assistant", "content": "回复"}]):
            with self.subTest(history_length=len(history)):
                messages, info = build_context(history, max_turns=2)
                self.assertEqual(messages, history)
                self.assertIsNot(messages, history)
                self.assertEqual(info["omitted_turns"], 0)
                self.assertEqual(info["request_messages"], len(history) + 1)

    def test_default_window_keeps_six_completed_turns(self):
        history = [{"role": "system", "content": "规则"}]
        for index in range(8):
            history.extend([{"role": "user", "content": f"问题 {index}"}, {"role": "assistant", "content": "回复"}])
        messages, info = build_context(history)
        self.assertEqual(len(messages), 13)
        self.assertEqual(messages[1]["content"], "问题 2")
        self.assertEqual(info["kept_turns"], 6)
        self.assertEqual(info["omitted_turns"], 2)

    def test_rejects_non_positive_or_non_integer_window(self):
        for invalid in (0, -1, True, 2.5, "2", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_context([], max_turns=invalid)


if __name__ == "__main__":
    unittest.main()
