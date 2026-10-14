"""混合检索边界测试；用固定相似度替代外部模型，SQLite 与检索逻辑使用真实实现。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from history_search import query_terms, result_tokens
from session_store import SessionStore


class FixedSimilarity:
    status = "ready"

    def scores(self, query, texts):
        # 不复刻排序算法，只控制模型边界的相关/无关得分。
        return [0.92 if "周五18点" in text else 0.15 for text in texts]


class SemanticSearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SessionStore(Path(self.directory.name) / "chat.db")
        self.store.initialize()
        self.store.semantic_search = FixedSimilarity()

    def seed(self, session="a", text="请周五18点之后再送达。"):
        history = [{"role": "system", "content": "系统规则"}, {"role": "user", "content": text},
                   {"role": "assistant", "content": "好的，已经了解。"}]
        self.store.create_session(session, history)
        return history

    def test_paraphrase_recovers_source_without_keyword_overlap(self):
        history = self.seed()
        result = self.store.search_history("a", "我之前希望什么时候收到包裹？")
        self.assertEqual([hit["source"] for hit in result["results"]], ["message:1"])
        self.assertEqual(result["results"][0]["text"], history[1]["content"])
        self.assertEqual(result["search_mode"], "hybrid")
        self.assertLessEqual(result_tokens(result), 900)

    def test_relevance_threshold_does_not_return_unrelated_history(self):
        self.seed(text="我的键盘是黑色的。")
        result = self.store.search_history("a", "发票抬头是什么？")
        self.assertEqual(result["results"], [])

    def test_exact_order_ids_do_not_match_prefix_or_different_order(self):
        history = self.seed(text="O10010 请周五18点之后再送达。")
        history += [{"role": "user", "content": "O1002 请周五18点之后再送达。"},
                    {"role": "user", "content": "O1001 请周五18点之后再送达。"}]
        self.store.replace_history("a", history)
        result = self.store.search_history("a", "O1001 到货时间")
        self.assertEqual([hit["position"] for hit in result["results"]], [4])

    def test_multi_order_message_does_not_return_another_orders_best_chunk(self):
        class Similarity:
            status = "ready"
            def scores(self, query, texts):
                return [0.99 if "周六早上" in text else 0.7 if "周五18点" in text else 0.1 for text in texts]
        self.store.semantic_search = Similarity()
        # 测试带其他订单号的片段，以及订单号在前一个片段的情况。
        for padding in ("", "普通说明。" * 70):
            with self.subTest(padding=bool(padding)):
                original = "O1001 请周五18点之后再送达。" + "背景文字。" * 80 + "O1002 " + padding + "包裹到货时间为周六早上。"
                if self.store.load_session("a"):
                    self.store.delete_session("a")
                self.seed(text=original)
                hit = self.store.search_history("a", "O1001 到货时间")["results"][0]
                self.assertIn("周五18点", hit["text"])
                self.assertNotIn("周六早上", hit["text"])

    def test_semantic_excerpt_finds_tail_and_deduplicates_overlapping_chunks(self):
        original = "背景文字。" * 100 + "请周五18点之后再送达。" + "补充说明。" * 100
        self.seed(text=original)
        result = self.store.search_history("a", "我希望何时收件？")
        self.assertEqual(len(result["results"]), 1)
        hit = result["results"][0]
        self.assertIn("周五18点", hit["text"])
        self.assertGreater(hit["start"], 0)
        self.assertEqual(hit["text"], original[hit["start"]:hit["end"]])
        self.assertLessEqual(len(hit["text"]), 240)

    def test_semantic_excerpt_beats_distant_order_id_keyword_anchor(self):
        self.seed(text="O1001 " + "背景文字。" * 100 + "请周五18点之后再送达。")
        result = self.store.search_history("a", "O1001 何时收件")
        self.assertIn("周五18点", result["results"][0]["text"])

    def test_semantic_acknowledgments_do_not_crowd_out_substantive_messages(self):
        class Similarity:
            status = "ready"
            def scores(self, query, texts):
                return [0.99 if text in ("收到。", "好的。", "明白。") else 0.7 for text in texts]
        self.store.semantic_search = Similarity()
        history = self.seed()
        history[2]["content"] = "收到。"
        history += [{"role": "assistant", "content": "好的。"}, {"role": "assistant", "content": "明白。"}]
        self.store.replace_history("a", history)
        result = self.store.search_history("a", "何时收件")
        self.assertEqual([hit["source"] for hit in result["results"]], ["message:1"])

    def test_only_live_current_session_messages_can_be_retrieved(self):
        history = self.seed(text="普通问候")
        history += [{"role": "system", "content": "周五18点系统规则"},
                    {"role": "tool", "content": "周五18点工具结果"},
                    {"role": "assistant", "content": "周五18点工具阶段", "tool_calls": [{"id": "a"}]}]
        self.store.replace_history("a", history)
        self.seed("b")
        self.assertEqual(self.store.search_history("a", "何时收件")["results"], [])
        self.assertTrue(self.store.search_history("b", "何时收件")["results"])
        self.store.replace_history("b", history)
        self.assertEqual(self.store.search_history("b", "何时收件")["results"], [])
        self.store.delete_session("b")
        self.assertEqual(self.store.search_history("b", "何时收件")["results"], [])

    def test_failing_embeddings_fall_back_to_keywords_and_report_degradation(self):
        class Unavailable:
            status = "unavailable"

            def scores(self, query, texts):
                raise RuntimeError("model unavailable")

        self.seed()
        self.store.semantic_search = Unavailable()
        with self.assertLogs("history_search", level="WARNING"):
            result = self.store.search_history("a", "周五18点")
        self.assertEqual(result["results"][0]["source"], "message:1")
        self.assertEqual(result["search_mode"], "keywords")
        self.assertEqual(result["semantic_status"], "unavailable")

    def test_semantic_results_respect_explicit_chronological_order(self):
        history = self.seed()
        history += [{"role": "user", "content": "更正：周五18点改为周六。"}]
        self.store.replace_history("a", history)
        oldest = self.store.search_history("a", "何时收件", "oldest")
        newest = self.store.search_history("a", "何时收件", "newest")
        self.assertEqual(oldest["results"][0]["position"], 1)
        self.assertEqual(newest["results"][0]["position"], 3)

    def test_natural_language_query_is_not_limited_to_six_words(self):
        self.assertTrue(query_terms("When did I want to receive my parcel?"))


class LocalEmbeddingsTests(unittest.TestCase):
    def test_scan_larger_than_cache_reuses_existing_vectors(self):
        from local_embeddings import LocalEmbeddings
        encoded = []
        class Model:
            def query_embed(self, query):
                return iter([[1.0, 0.0]])
            def passage_embed(self, texts, **kwargs):
                encoded.extend(texts)
                return iter([[1.0, 0.0] for _ in texts])
        with patch.object(LocalEmbeddings, "_load_model", return_value=Model()):
            encoder = LocalEmbeddings(Path("unused-model-cache"))
            texts = [f"历史片段 {index}" for index in range(2080)]
            self.assertEqual(len(encoder.scores("历史", texts)), 2080)
            encoded.clear()
            self.assertEqual(len(encoder.scores("历史", texts)), 2080)
            self.assertLessEqual(len(encoded), 32, "一次扫描不应把尚未使用的缓存逐出")

    def test_missing_model_never_attempts_network_in_chat_runtime(self):
        from local_embeddings import LocalEmbeddings
        with TemporaryDirectory() as directory, \
             patch("fastembed.common.model_management.logger"), \
             patch("httpx.Client.send", side_effect=OSError("network forbidden")) as hub_network, \
             patch("requests.get", side_effect=OSError("network forbidden")) as fallback_network:
            with self.assertLogs("local_embeddings", level="WARNING"):
                self.assertFalse(LocalEmbeddings(Path(directory)).warm_up())
            self.assertEqual(hub_network.call_count, 0, "聊天服务不应触发模型下载")
            self.assertEqual(fallback_network.call_count, 0, "聊天服务不应尝试其他下载源")

    def test_cosine_similarity_cache_and_changed_text(self):
        from local_embeddings import LocalEmbeddings
        encoded = []

        class Model:
            def query_embed(self, query):
                return iter([[3.0, 0.0]])

            def passage_embed(self, texts, **kwargs):
                encoded.extend(texts)
                return iter([[2.0, 0.0] if "周五" in text else [0.0, 4.0] for text in texts])

        with patch.object(LocalEmbeddings, "_load_model", return_value=Model()):
            encoder = LocalEmbeddings(Path("unused-model-cache"))
            self.assertEqual(encoder.scores("何时收件", ["周五", "发票"]), [1.0, 0.0])
            self.assertEqual(encoder.scores("何时收件", ["周五", "发票"]), [1.0, 0.0])
            self.assertEqual(encoded, ["周五", "发票"], "重复检索应复用文本向量")
            self.assertEqual(encoder.scores("何时收件", ["周六"]), [0.0])

    def test_model_failure_has_cooldown_and_later_recovers(self):
        from local_embeddings import LocalEmbeddings
        clock = [10.0]
        with patch("local_embeddings.monotonic", side_effect=lambda: clock[0]), \
             patch.object(LocalEmbeddings, "_load_model", side_effect=OSError("missing weights")) as loader:
            encoder = LocalEmbeddings(Path("unused-model-cache"))
            with self.assertLogs("local_embeddings", level="WARNING"):
                self.assertFalse(encoder.warm_up())
            self.assertFalse(encoder.warm_up())
            self.assertEqual(loader.call_count, 1, "失败后不应每次检索重新加载")
            loader.side_effect = None
            loader.return_value = object()
            clock[0] = 80.0
            self.assertTrue(encoder.warm_up())
            self.assertEqual(encoder.status, "ready")
