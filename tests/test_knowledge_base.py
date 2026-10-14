"""知识库检索：真实临时文件，验证来源、更新与错误边界。"""
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from history_search import result_tokens


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        from knowledge_base import KnowledgeBase
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "knowledge"
        self.root.mkdir()
        self.path = self.root / "returns.md"
        self.original = "# 示例售后政策\n\n## 退货条件\n签收后7天内，商品完好且配件齐全，可以申请退货。\n\n## 退款时间\n仓库验收后3个工作日原路退款。\n"
        self.path.write_text(self.original, encoding="utf-8")
        self.kb = KnowledgeBase(self.root)

    def test_keyword_result_has_verifiable_original_and_section(self):
        result = self.kb.search("退货 条件")
        hit = result["results"][0]
        self.assertEqual(hit["title"], "示例售后政策")
        self.assertEqual(hit["section"], "退货条件")
        self.assertIn("7天", hit["text"])
        self.assertEqual(hit["text"], self.original[hit["start"]:hit["end"]])
        doc = self.kb.read_document(hit["document_id"], hit["version"])
        self.assertEqual(doc["content"], self.original)
        self.assertLessEqual(result_tokens(result), 900)

    def test_semantic_paraphrase_works_without_shared_words(self):
        class Embeddings:
            status = "ready"
            def scores(self, query, texts):
                return [0.8 if "商品完好" in text else 0.1 for text in texts]
        self.kb.semantic_search = Embeddings()
        result = self.kb.search("买完不想要了怎么办？")
        self.assertEqual(result["search_mode"], "hybrid")
        self.assertIn("7天", result["results"][0]["text"])

    def test_update_delete_and_old_source_version(self):
        from knowledge_base import KnowledgeVersionChanged
        old = self.kb.search("退款 时间")["results"][0]
        self.path.write_text(self.original.replace("3个工作日", "5个工作日"), encoding="utf-8")
        fresh = self.kb.search("退款 时间")["results"][0]
        self.assertIn("5个工作日", fresh["text"])
        self.assertNotEqual(old["version"], fresh["version"])
        with self.assertRaises(KnowledgeVersionChanged):
            self.kb.read_document(old["document_id"], old["version"])
        self.path.unlink()
        self.assertEqual(self.kb.search("退款")["results"], [])
        self.assertIsNone(self.kb.read_document(old["document_id"], old["version"]))

    def test_empty_and_unrelated_query_do_not_invent_evidence(self):
        self.assertEqual(self.kb.search("火星运费")["results"], [])
        self.path.unlink()
        self.assertEqual(self.kb.search("退货")["document_count"], 0)

    def test_only_designated_text_files_are_read(self):
        (self.root.parent / "secret.txt").write_text("机密内容", encoding="utf-8")
        (self.root / ".env").write_text("机密内容", encoding="utf-8")
        (self.root / "settings.json").write_text('"机密内容"', encoding="utf-8")
        self.assertEqual(self.kb.search("机密内容")["results"], [])
        self.assertIsNone(self.kb.read_document("../secret.txt"))

    def test_long_sections_are_bounded_exact_excerpts(self):
        self.path.write_text("# 示例政策\n\n## 说明\n" + "普通内容。" * 150 + "退货暗号：青山。" + "背景。" * 100, encoding="utf-8")
        original = self.path.read_text(encoding="utf-8")
        result = self.kb.search("青山")
        self.assertTrue(result["results"])
        for hit in result["results"]:
            self.assertIn("青山", hit["text"])
            self.assertEqual(hit["text"], original[hit["start"]:hit["end"]])
        self.assertLessEqual(result_tokens(result), 900)

    def test_embedding_failure_preserves_keywords(self):
        class Broken:
            def scores(self, query, texts):
                raise OSError("offline")
        self.kb.semantic_search = Broken()
        with self.assertLogs("knowledge_base", level="WARNING"):
            result = self.kb.search("退款")
        self.assertTrue(result["results"])
        self.assertEqual(result["semantic_status"], "unavailable")

    def test_unreadable_or_oversized_file_is_not_a_false_no_match(self):
        from knowledge_base import KnowledgeReadError
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")), self.assertRaises(KnowledgeReadError):
            self.kb.search("退货")
        self.path.write_bytes(b"x" * (128 * 1024 + 1))
        with self.assertRaises(KnowledgeReadError):
            self.kb.search("退货")

    def test_directory_scan_failures_are_not_empty_or_partial_results(self):
        from knowledge_base import KnowledgeReadError
        with patch("os.scandir", side_effect=PermissionError("denied")), self.assertRaises(KnowledgeReadError):
            self.kb.search("退货")
        hidden = self.root / "restricted"
        hidden.mkdir()
        real_scandir = os.scandir
        def scan(path):
            if Path(path) == hidden:
                raise PermissionError("nested denied")
            return real_scandir(path)
        with patch("os.scandir", side_effect=scan), self.assertRaises(KnowledgeReadError):
            self.kb.search("退货")
