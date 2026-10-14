"""退货申请存储测试：仅使用临时 SQLite 和本地模拟订单。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from return_requests import ReturnRequestStore, initialize_return_tables, save_return_previews


class ReturnRequestStoreTests(unittest.TestCase):
    def setUp(self):
        self.ReturnRequestStore = ReturnRequestStore
        self.initialize_return_tables = initialize_return_tables
        self.save_return_previews = save_return_previews
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "chat.db"
        self.original_history = [{"role": "system", "content": "中文客服"}]
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE messages (session_id TEXT NOT NULL, position INTEGER NOT NULL CHECK(position >= 0), message_json TEXT NOT NULL, PRIMARY KEY(session_id, position), FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE)")
            self.initialize_return_tables(connection)
            for sid in ("a", "b"):
                connection.execute("INSERT INTO sessions VALUES (?, ?)", (sid, "2026-09-13T00:00:00+00:00"))
                connection.execute("INSERT INTO messages VALUES (?, 0, ?)", (sid, json.dumps(self.original_history[0], ensure_ascii=False)))
        self.store = ReturnRequestStore(self.path)

    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def prepare(self, order_id="O1003", reason="收到的商品无法正常使用", sid="a"):
        return self.store.prepare_return_request(sid, order_id, reason)["preview"]

    def persist(self, preview=None, sid="a"):
        with closing(self.connect()) as connection, connection:
            self.save_return_previews(connection, sid, [] if preview is None else [preview])

    def history(self, sid="a"):
        with closing(self.connect()) as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT message_json FROM messages WHERE session_id = ? ORDER BY position", (sid,))]

    def test_preparation_normalizes_and_has_no_database_side_effects(self):
        before = self.path.read_bytes()
        result = self.prepare(" o1003 ", "  外壳破损\n无法使用  ")
        self.assertEqual(result["order_id"], "O1003")
        self.assertEqual(result["order"], {"order_id": "O1003", "status": "已签收", "logistics": "客户已签收"})
        self.assertEqual(result["reason"], "外壳破损\n无法使用")
        self.assertEqual(result["state"], "pending")
        self.assertIsNone(result["request_id"])
        self.assertIsNone(result["confirmed_at"])
        self.assertTrue(result["simulated"])
        self.assertTrue(result["status"])
        self.assertTrue(result["created_at"])
        self.assertNotEqual(self.prepare()["preview_id"], result["preview_id"])
        self.assertEqual(self.store.list_previews("a"), [])
        self.assertIn("error", self.store.get_return_request("a", "O1003"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.history(), self.original_history)

    def test_confirmation_is_idempotent_and_persists_exact_reason_in_receipt(self):
        preview = self.prepare(reason="商品破损；'引号'\n第二行 \\\"")
        self.persist(preview)
        self.assertIn("error", self.store.get_return_request("a", "O1003"))
        first = self.store.create_return_request("a", preview["preview_id"])
        second = self.store.create_return_request("a", preview["preview_id"])
        self.assertEqual(first, second)
        self.assertEqual(first["state"], "submitted")
        self.assertTrue(first["request_id"].startswith("RET-"))
        self.assertEqual(first["reason"], preview["reason"])
        self.assertTrue(first["confirmed_at"])
        self.assertEqual(first["status"], "待人工审核")
        history = self.history()
        self.assertEqual(len(history), 3)
        self.assertEqual([message["role"] for message in history], ["system", "user", "assistant"])
        self.assertIn(preview["reason"], history[-1]["content"])
        self.assertIn("O1003", history[-1]["content"])
        self.assertIn(first["request_id"], history[-1]["content"])
        self.assertIn("本地模拟", history[-1]["content"])
        self.assertEqual(self.store.get_return_request("a", " o1003 "), {"request": first})

    def test_concurrent_confirmations_create_exactly_one_request_and_receipt(self):
        preview = self.prepare()
        self.persist(preview)
        barrier = Barrier(6)

        def confirm(_):
            barrier.wait(timeout=5)
            return self.ReturnRequestStore(self.path).create_return_request("a", preview["preview_id"])

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(confirm, range(6)))
        self.assertEqual(len({result["request_id"] for result in results}), 1)
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(self.history()), 3)
        self.assertEqual(len(self.store.list_previews("a")), 1)

    def test_confirmation_failure_rolls_back_request_and_partial_messages(self):
        preview = self.prepare()
        self.persist(preview)
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON messages WHEN NEW.position = 2 BEGIN SELECT RAISE(ABORT, 'simulated receipt failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.create_return_request("a", preview["preview_id"])
        self.assertEqual(self.history(), self.original_history)
        self.assertEqual(self.store.list_previews("a"), [preview])
        self.assertIn("error", self.store.get_return_request("a", "O1003"))
        with closing(self.connect()) as connection, connection:
            connection.execute("DROP TRIGGER fail_receipt")
        self.assertEqual(self.store.create_return_request("a", preview["preview_id"])["state"], "submitted")

    def test_cross_session_and_missing_session_are_rejected(self):
        preview = self.prepare()
        self.persist(preview)
        for sid in ("b", "missing"):
            with self.subTest(sid=sid):
                with self.assertRaises(LookupError):
                    self.store.create_return_request(sid, preview["preview_id"])
                with self.assertRaises(LookupError):
                    self.store.cancel_preview(sid, preview["preview_id"])
        for action in (
            lambda: self.store.prepare_return_request("missing", "O1003", "坏了"),
            lambda: self.store.list_previews("missing"),
            lambda: self.store.get_return_request("missing", "O1003"),
            lambda: self.store.create_return_request("a", "missing"),
            lambda: self.persist(preview, sid="missing"),
        ):
            with self.assertRaises(LookupError):
                action()
        self.assertEqual(self.store.list_previews("a"), [preview])
        self.assertEqual(self.store.list_previews("b"), [])

    def test_invalid_order_and_reason_are_rejected_without_writing(self):
        for order_id, reason in (
            ("missing", "坏了"), (" ", "坏了"), (123, "坏了"),
            ("O1003", ""), ("O1003", " \n\t "), ("O1003", "长" * 501),
            ("O1003", 7), ("O1003", None), ("O1003", ["坏了"]), ("O1003", "\x00坏了"),
        ):
            with self.subTest(order_id=order_id, reason=reason):
                with self.assertRaises(ValueError):
                    self.store.prepare_return_request("a", order_id, reason)
        self.assertEqual(len(self.prepare(reason="长" * 500)["reason"]), 500)
        self.assertEqual(self.store.list_previews("a"), [])

    def test_new_successful_turn_expires_old_pending_even_without_a_new_preview(self):
        first = self.prepare()
        self.persist(first)
        second = self.prepare(reason="补充新原因")
        self.persist(second)
        self.assertEqual([item["state"] for item in self.store.list_previews("a")], ["expired", "pending"])
        with self.assertRaises(ValueError):
            self.store.create_return_request("a", first["preview_id"])
        self.persist()
        with self.assertRaises(ValueError):
            self.store.create_return_request("a", second["preview_id"])
        self.assertEqual([item["state"] for item in self.store.list_previews("a")], ["expired", "expired"])

    def test_cancel_is_idempotent_and_submitted_or_expired_cannot_be_cancelled(self):
        first = self.prepare()
        self.persist(first)
        cancelled = self.store.cancel_preview("a", first["preview_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(self.store.cancel_preview("a", first["preview_id"]), cancelled)
        with self.assertRaises(ValueError):
            self.store.create_return_request("a", first["preview_id"])
        expired = self.prepare()
        self.persist(expired)
        submitted = self.prepare()
        self.persist(submitted)
        self.store.create_return_request("a", submitted["preview_id"])
        for preview in (expired, submitted):
            with self.assertRaises(ValueError):
                self.store.cancel_preview("a", preview["preview_id"])

    def test_existing_submission_is_reused_within_session_but_not_other_sessions(self):
        preview = self.prepare()
        self.persist(preview)
        submitted = self.store.create_return_request("a", preview["preview_id"])
        result = self.store.prepare_return_request("a", "o1003", "新的原因")
        self.assertEqual(result, {"request": submitted, "already_submitted": True})
        other = self.prepare(sid="b")
        self.persist(other, sid="b")
        self.assertNotEqual(self.store.create_return_request("b", other["preview_id"])["request_id"], submitted["request_id"])
        self.assertEqual(len(self.store.list_previews("a")), 1)

    def test_preview_save_failure_rolls_back_caller_messages_and_invalidation(self):
        first = self.prepare()
        self.persist(first)
        second = self.prepare(reason="修改原因")
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TRIGGER fail_preview BEFORE INSERT ON return_previews BEGIN SELECT RAISE(ABORT, 'simulated preview failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            with closing(self.connect()) as connection, connection:
                connection.execute("INSERT INTO messages VALUES ('a', 1, ?)", (json.dumps({"role": "user", "content": "新消息"}),))
                self.save_return_previews(connection, "a", [second])
        self.assertEqual(self.history(), self.original_history)
        self.assertEqual(self.store.list_previews("a"), [first])

    def test_caller_rollback_undoes_a_successful_preview_save(self):
        preview = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "caller failed"):
            with closing(self.connect()) as connection, connection:
                connection.execute("INSERT INTO messages VALUES ('a', 1, ?)", (json.dumps({"role": "user", "content": "新消息"}),))
                self.save_return_previews(connection, "a", [preview])
                raise RuntimeError("caller failed")
        self.assertEqual(self.store.list_previews("a"), [])
        self.assertEqual(self.history(), self.original_history)

    def test_malformed_unicode_reason_cannot_produce_an_unsavable_preview(self):
        with self.assertRaises(ValueError):
            self.prepare(reason="损坏\ud800")
        self.assertEqual(self.store.list_previews("a"), [])

    def test_non_json_order_snapshot_is_a_validation_error_and_preserves_old_preview(self):
        first = self.prepare()
        self.persist(first)
        for value in (object(), float("nan"), "坏数据\ud800"):
            with self.subTest(value=repr(value)):
                invalid = self.prepare()
                invalid["order"]["invalid"] = value
                with self.assertRaises(ValueError):
                    self.persist(invalid)
                self.assertEqual(self.store.list_previews("a"), [first])

    def test_late_transient_preview_cannot_replace_an_already_submitted_order(self):
        first = self.prepare()
        late = self.prepare(reason="稍后返回的旧回合")
        self.persist(first)
        submitted = self.store.create_return_request("a", first["preview_id"])
        with self.assertRaises(ValueError):
            self.persist(late)
        self.assertEqual(self.store.list_previews("a"), [submitted])
        self.assertEqual(len(self.history()), 3)

    def test_one_preview_per_turn_and_database_uniqueness_are_enforced(self):
        first = self.prepare()
        second = self.prepare()
        with self.assertRaises(ValueError):
            with closing(self.connect()) as connection, connection:
                self.save_return_previews(connection, "a", [first, second])
        self.persist(first)
        submitted = self.store.create_return_request("a", first["preview_id"])
        with self.assertRaises(sqlite3.IntegrityError):
            with closing(self.connect()) as connection, connection:
                connection.execute("INSERT INTO return_previews SELECT 'duplicate', session_id, order_id, reason, order_json, state, 'RET-duplicate', created_at, confirmed_at FROM return_previews WHERE preview_id = ?", (first["preview_id"],))
        self.assertEqual(self.store.list_previews("a"), [submitted])

    def test_restart_and_schema_reinitialization_preserve_requests(self):
        preview = self.prepare()
        self.persist(preview)
        submitted = self.store.create_return_request("a", preview["preview_id"])
        saved_history = self.history()
        with closing(self.connect()) as connection, connection:
            self.initialize_return_tables(connection)
        restarted = self.ReturnRequestStore(self.path)
        self.assertEqual(restarted.list_previews("a"), [submitted])
        self.assertEqual(restarted.get_return_request("a", "O1003"), {"request": submitted})
        self.assertEqual(restarted.create_return_request("a", preview["preview_id"]), submitted)
        self.assertEqual(self.history(), saved_history)

    def test_deleting_session_cascades_previews_and_receipts(self):
        preview = self.prepare()
        self.persist(preview)
        self.store.create_return_request("a", preview["preview_id"])
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE session_id = 'a'")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM return_previews WHERE session_id = 'a'").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages WHERE session_id = 'a'").fetchone()[0], 0)
        with self.assertRaises(LookupError):
            self.store.create_return_request("a", preview["preview_id"])


if __name__ == "__main__":
    unittest.main()
