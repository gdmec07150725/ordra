"""人工审核存储：真实临时 SQLite，原申请和历史消息始终保留。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from orders import ORDERS
from return_requests import ReturnRequestStore, initialize_return_tables, save_return_previews


class ReturnReviewStoreTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "chat.db"
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE messages (session_id TEXT NOT NULL, position INTEGER NOT NULL CHECK(position >= 0), message_json TEXT NOT NULL, PRIMARY KEY(session_id, position), FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE)")
            initialize_return_tables(connection)
            for sid in ("a", "b"):
                connection.execute("INSERT INTO sessions VALUES (?, ?)", (sid, "2026-09-13T00:00:00+00:00"))
                connection.execute("INSERT INTO messages VALUES (?, 0, ?)", (sid, json.dumps({"role": "system", "content": "中文客服"}, ensure_ascii=False)))
        self.store = ReturnRequestStore(self.path)

    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def prepare(self, order_id="O1003", sid="a"):
        preview = self.store.prepare_return_request(sid, order_id, "商品外壳破损")["preview"]
        with closing(self.connect()) as connection, connection:
            save_return_previews(connection, sid, [preview])
        return preview

    def submit(self, order_id="O1003", sid="a"):
        preview = self.prepare(order_id, sid)
        return self.store.create_return_request(sid, preview["preview_id"])

    def review(self, request, decision="approved", reason="已人工核对，符合本地模拟审核条件。", sid="a"):
        return self.store.review_request(sid, request["request_id"], decision, reason)

    def queue(self, status="pending"):
        return self.store.list_requests(status)

    def original_records(self):
        with closing(self.connect()) as connection:
            return {
                "sessions": connection.execute("SELECT * FROM sessions ORDER BY session_id").fetchall(),
                "previews": connection.execute("SELECT * FROM return_previews ORDER BY rowid").fetchall(),
                "messages": connection.execute("SELECT * FROM messages ORDER BY session_id, position").fetchall(),
            }

    def test_approve_and_reject_preserve_original_applications_messages_and_orders(self):
        approved = self.submit("O1001")
        rejected = self.submit("O1002")
        original = self.original_records()
        original_orders = json.dumps(ORDERS, ensure_ascii=False, sort_keys=True)
        for request, decision, status in ((approved, "approved", "审核通过"), (rejected, "rejected", "审核拒绝")):
            with self.subTest(decision=decision):
                result = self.review(request, decision, "  已核实；'说明'\n第二行  ")
                self.assertEqual(result["status"], status)
                self.assertEqual(result["state"], "submitted")
                self.assertEqual(result["review"]["decision"], decision)
                self.assertEqual(result["review"]["reason"], "已核实；'说明'\n第二行")
                self.assertTrue(result["review"]["reviewed_at"])
                for key in ("request_id", "preview_id", "reason", "order", "created_at", "confirmed_at", "simulated"):
                    self.assertEqual(result[key], request[key])
        self.assertEqual(self.original_records(), original)
        self.assertEqual(json.dumps(ORDERS, ensure_ascii=False, sort_keys=True), original_orders)

    def test_all_read_paths_and_confirmation_retry_return_latest_review(self):
        pending = self.prepare()
        self.assertIn("review", pending)
        self.assertIsNone(pending["review"])
        submitted = self.store.create_return_request("a", pending["preview_id"])
        self.assertIn("review", submitted)
        self.assertIsNone(submitted["review"])
        reviewed = self.review(submitted)
        self.assertEqual(self.store.get_return_request("a", "o1003"), {"request": reviewed})
        self.assertEqual(self.store.list_previews("a"), [reviewed])
        self.assertEqual(self.store.prepare_return_request("a", "O1003", "新的退货原因"), {"request": reviewed, "already_submitted": True})
        self.assertEqual(self.store.create_return_request("a", pending["preview_id"]), reviewed)

    def test_queue_filters_counts_and_order_use_only_submitted_applications(self):
        first_pending = self.submit("O1001")
        second_pending = self.submit("O1002")
        first_approved = self.submit("O1003")
        rejected = self.submit("O1001", "b")
        last_approved = self.submit("O1002", "b")
        unsubmitted = self.prepare("O1003", "b")
        self.store.cancel_preview("b", unsubmitted["preview_id"])
        self.review(first_approved)
        self.review(rejected, "rejected", "人工核对后拒绝。", sid="b")
        self.review(last_approved, sid="b")
        expected_counts = {"pending": 2, "approved": 2, "rejected": 1, "all": 5}
        expected_ids = {
            "pending": [first_pending["request_id"], second_pending["request_id"]],
            "approved": [last_approved["request_id"], first_approved["request_id"]],
            "rejected": [rejected["request_id"]],
            "all": [last_approved["request_id"], rejected["request_id"], first_approved["request_id"], second_pending["request_id"], first_pending["request_id"]],
        }
        for status, ids in expected_ids.items():
            with self.subTest(status=status):
                result = self.queue(status)
                self.assertEqual(result["counts"], expected_counts)
                self.assertEqual([item["request_id"] for item in result["requests"]], ids)
                self.assertTrue(all(item["session_id"] in ("a", "b") for item in result["requests"]))
                self.assertTrue(all(item["state"] == "submitted" for item in result["requests"]))
        self.assertEqual(self.queue(), self.queue("pending"))

    def test_empty_queue_has_explicit_zero_counts(self):
        self.prepare()
        self.assertEqual(self.queue(), {"requests": [], "counts": {"pending": 0, "approved": 0, "rejected": 0, "all": 0}})

    def test_normalized_identical_retry_preserves_timestamp_and_conflicts_are_rejected(self):
        request = self.submit()
        first = self.review(request, " APPROVED ", "  已人工核对  ")
        self.assertEqual(self.review(request, "approved", "已人工核对"), first)
        for decision, reason in (("rejected", "已人工核对"), ("approved", "替换审核原因")):
            with self.subTest(decision=decision, reason=reason):
                with self.assertRaises(ValueError):
                    self.review(request, decision, reason)
        self.assertEqual(self.store.get_return_request("a", "O1003")["request"], first)
        with closing(self.connect()) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM return_reviews").fetchone()[0], 1)

    def test_invalid_decisions_reasons_and_filters_never_write_a_review(self):
        request = self.submit()
        original = self.original_records()
        for decision in ("", "pending", "all", "refund", None, 1, [], {}):
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    self.review(request, decision)
        for reason in ("", " \n\t ", "长" * 501, None, 1, True, [], {}, "\x00原因", "坏字符\ud800"):
            with self.subTest(reason=repr(reason)):
                with self.assertRaises(ValueError):
                    self.review(request, reason=reason)
        for status in ("", "submitted", "unknown", None, 1, [], {}):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    self.queue(status)
        self.assertEqual(self.queue()["counts"], {"pending": 1, "approved": 0, "rejected": 0, "all": 1})
        self.assertEqual(self.original_records(), original)
        self.assertEqual(len(self.review(request, reason="长" * 500)["review"]["reason"]), 500)

    def test_missing_cross_session_and_unsubmitted_ids_are_rejected(self):
        request = self.submit()
        for sid, request_id in (("b", request["request_id"]), ("missing", request["request_id"]), ("a", "missing")):
            with self.subTest(sid=sid, request_id=request_id):
                with self.assertRaises(LookupError):
                    self.review({"request_id": request_id}, sid=sid)
        pending = self.prepare("O1002")
        with self.assertRaises(LookupError):
            self.review({"request_id": pending["preview_id"]})
        self.assertEqual(self.queue()["counts"], {"pending": 1, "approved": 0, "rejected": 0, "all": 1})

    def test_concurrent_identical_reviews_return_one_persisted_decision(self):
        request = self.submit()
        barrier = Barrier(6)

        def submit_review(_):
            barrier.wait(timeout=5)
            return ReturnRequestStore(self.path).review_request("a", request["request_id"], "approved", "已人工核对")

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit_review, range(6)))
        self.assertTrue(all(result == results[0] for result in results))
        with closing(self.connect()) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM return_reviews").fetchone()[0], 1)

    def test_concurrent_conflicting_reviews_choose_one_winner(self):
        request = self.submit()
        barrier = Barrier(2)

        def submit_review(decision):
            barrier.wait(timeout=5)
            try:
                return ReturnRequestStore(self.path).review_request("a", request["request_id"], decision, "已核对")
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit_review, ("approved", "rejected")))
        winners = [result for result in results if result is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.store.get_return_request("a", "O1003")["request"], winners[0])
        self.assertEqual(self.queue()["counts"]["pending"], 0)

    def test_review_insert_failure_rolls_back_partial_write(self):
        request = self.submit()
        original = self.original_records()
        with closing(self.connect()) as connection, connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertIn("return_reviews", tables)
            # AFTER + FAIL 会保留语句已写的数据，必须依靠整个事务回滚清除它。
            connection.execute("CREATE TRIGGER fail_review AFTER INSERT ON return_reviews BEGIN SELECT RAISE(FAIL, 'simulated review failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.review(request)
        self.assertEqual(self.queue()["counts"]["pending"], 1)
        self.assertEqual(self.original_records(), original)
        with closing(self.connect()) as connection, connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM return_reviews").fetchone()[0], 0)
            connection.execute("DROP TRIGGER fail_review")
        self.assertEqual(self.review(request)["status"], "审核通过")

    def test_additive_migration_preserves_pre_review_schema_data(self):
        request = self.submit()
        original = self.original_records()
        with closing(self.connect()) as connection, connection:
            connection.execute("DROP TABLE IF EXISTS return_reviews")
            original_schema = connection.execute("SELECT sql FROM sqlite_master WHERE name = 'return_previews'").fetchone()[0]
            initialize_return_tables(connection)
            self.assertIn("return_reviews", {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")})
            self.assertEqual(connection.execute("SELECT sql FROM sqlite_master WHERE name = 'return_previews'").fetchone()[0], original_schema)
        self.assertEqual(self.original_records(), original)
        self.assertEqual(self.queue()["requests"][0]["request_id"], request["request_id"])
        self.assertIsNone(self.queue()["requests"][0]["review"])

    def test_restart_and_reinitialization_preserve_review_and_retry_result(self):
        request = self.submit()
        reviewed = self.review(request, "rejected", "资料不足，审核拒绝。")
        original = self.original_records()
        with closing(self.connect()) as connection, connection:
            initialize_return_tables(connection)
        restarted = ReturnRequestStore(self.path)
        self.assertEqual(restarted.get_return_request("a", "O1003")["request"], reviewed)
        self.assertEqual(restarted.review_request("a", request["request_id"], "rejected", "资料不足，审核拒绝。"), reviewed)
        self.assertEqual(self.original_records(), original)

    def test_session_deletion_cascades_to_review_without_touching_other_session(self):
        first = self.submit()
        other = self.submit(sid="b")
        self.review(first)
        other_review = self.review(other, "rejected", "人工拒绝。", sid="b")
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE session_id = 'a'")
            self.assertEqual(connection.execute("SELECT request_id FROM return_reviews").fetchall(), [(other["request_id"],)])
        self.assertEqual(self.store.get_return_request("b", "O1003")["request"], other_review)
        self.assertEqual(self.queue("all")["counts"], {"pending": 0, "approved": 0, "rejected": 1, "all": 1})


if __name__ == "__main__":
    unittest.main()
