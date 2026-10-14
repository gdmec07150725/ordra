"""用真实临时 SQLite 验证历史检索；不读取真实聊天记录。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from session_store import SessionStore
from context_memory import ContextMemory, estimate_request
import json


class HistorySearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SessionStore(Path(self.directory.name) / "chat.db")
        self.store.initialize()

    def seed(self, session="a", text="O1001 请周五之后再送达。"):
        history = [{"role": "system", "content": "隐藏规则"}, {"role": "user", "content": text}, {"role": "assistant", "content": "好的，已了解。"}]
        self.store.create_session(session, history)
        return history

    def test_search_finds_original_even_when_summary_omits_it(self):
        self.assertTrue(hasattr(self.store, "search_history"), "需要新增按会话检索原文的方法")
        history = self.seed()
        self.store.replace_history("a", history, ContextMemory("用户查询过订单。", 3))
        result = self.store.search_history("a", "o1001 送达")
        self.assertEqual(result["results"][0]["text"], history[1]["content"])
        self.assertEqual(result["results"][0]["source"], "message:1")
        self.assertEqual(result["results"][0]["role"], "user")
        self.assertEqual(self.store.load_session("a")["history"], history)

    def test_other_sessions_system_and_tool_data_are_not_searchable(self):
        history = self.seed()
        history += [
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "工具阶段片段", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "工具秘密"},
            {"role": "assistant", "content": "完成"},
        ]
        self.store.replace_history("a", history)
        self.seed("b", "其他会话秘密 周五")
        for query in ("其他会话秘密", "隐藏规则", "工具秘密", "工具阶段片段"):
            with self.subTest(query=query):
                self.assertEqual(self.store.search_history("a", query)["results"], [])
        self.assertEqual(len(self.store.search_history("a", "周五")["results"]), 1)

    def test_no_match_and_deleted_or_reset_history_stay_empty(self):
        self.seed()
        self.assertEqual(self.store.search_history("a", "不存在的词")["matched_messages"], 0)
        self.store.replace_history("a", [{"role": "system", "content": "客服"}])
        self.assertEqual(self.store.search_history("a", "O1001")["results"], [])
        self.store.delete_session("a")
        self.assertEqual(self.store.search_history("a", "O1001")["results"], [])

    def test_ranking_and_order_preserve_user_assistant_sources(self):
        history = self.seed()
        history += [{"role": "user", "content": "更正：O1001 改成周六送达。"}, {"role": "assistant", "content": "O1001 送达要求已收到。"}]
        self.store.replace_history("a", history)
        self.assertEqual(self.store.search_history("a", "O1001 送达")["results"][0]["position"], 3)
        self.assertEqual(self.store.search_history("a", "O1001", "oldest")["results"][0]["position"], 1)
        latest = self.store.search_history("a", "O1001", "newest")["results"][0]
        self.assertEqual(latest["position"], 4)
        self.assertEqual(latest["role"], "assistant")

    def test_long_hit_is_exact_excerpt_with_offsets_and_bounded_result(self):
        original = "开头内容" * 500 + "送达要求：请周五之后到。" + "后文内容" * 500
        self.seed(text=original)
        result = self.store.search_history("a", "送达")
        hit = result["results"][0]
        self.assertIn("请周五之后到", hit["text"])
        self.assertTrue(hit["truncated"])
        self.assertEqual(hit["text"], original[hit["start"]:hit["end"]])
        self.assertLessEqual(len(hit["text"]), 240)
        message = {"role": "tool", "tool_call_id": "history-result", "content": json.dumps(result, ensure_ascii=False)}
        self.assertLessEqual(estimate_request([message]), 900)

    def test_result_limit_reports_more_matches_without_faking_no_results(self):
        history = self.seed()
        history += [{"role": "user", "content": "订单要求" * 100}, {"role": "assistant", "content": "订单要求" * 100}] * 6
        self.store.replace_history("a", history)
        result = self.store.search_history("a", "订单")
        self.assertGreater(result["matched_messages"], len(result["results"]))
        self.assertTrue(result["has_more"])
        self.assertLessEqual(len(result["results"]), 3)

    def test_bad_arguments_fail_and_query_is_never_sql(self):
        self.seed()
        for query, order in (("", "relevance"), (" ", "relevance"), ("词" * 101, "relevance"), (None, "relevance"), ("?!", "relevance"), ("订单", "random")):
            with self.subTest(query=query, order=order), self.assertRaises(ValueError):
                self.store.search_history("a", query, order)
        self.store.search_history("a", "' OR 1=1 --")
        self.assertIsNotNone(self.store.load_session("a"))
