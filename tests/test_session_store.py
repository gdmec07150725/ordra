"""真实临时 SQLite 文件：验证完整消息保存和事务回滚，不模拟数据库。"""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from session_store import SessionStore
from context_memory import ContextMemory


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data" / "chat.db"
        self.store = SessionStore(self.path)
        self.store.initialize()

    def test_new_store_instance_restores_nested_messages_and_creation_time(self):
        # 引号与 SQL 片段作为普通数据存取，不能改变 SQL 语句的含义。
        session = "test'; DROP TABLE sessions; --"
        history = [
            {"role": "system", "content": "中文客服"},
            {"role": "user", "content": "查一下 'O1001'"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id":"O1001"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"status":"已发货"}'},
            {"role": "assistant", "content": "已发货。"},
        ]
        created = self.store.create_session(session, history)
        self.assertIsNotNone(created)
        another_store = SessionStore(self.path)
        another_store.initialize()  # 再次启动不能覆盖已有表和数据。
        restored = another_store.load_session(session)
        self.assertEqual(restored["history"], history)
        self.assertEqual(restored["created_at"], created["created_at"])
        self.assertTrue(restored["created_at"])
        another_store.replace_history(session, [{"role": "system", "content": "新的系统消息"}])
        reset = self.store.load_session(session)
        self.assertEqual(reset["created_at"], created["created_at"])
        self.assertEqual(reset["history"], [{"role": "system", "content": "新的系统消息"}])
        self.assertIsNone(another_store.load_session("missing"))

    def test_failed_insert_rolls_back_the_deletion_and_partial_insert(self):
        original = [{"role": "system", "content": "客服"}, {"role": "user", "content": "原来的问题"}, {"role": "assistant", "content": "原来的回复"}]
        self.assertIsNotNone(self.store.create_session("session-1", original))
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_second_message BEFORE INSERT ON messages WHEN NEW.position = 1 BEGIN SELECT RAISE(ABORT, 'simulated disk write failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.replace_history("session-1", [{"role": "system", "content": "新的系统消息"}, {"role": "user", "content": "新的问题"}])
        # 若 DELETE 提前提交、或逐条插入提交，这里会缺失原来的三条消息。
        self.assertEqual(SessionStore(self.path).load_session("session-1")["history"], original)

    def test_summary_persists_atomically_and_reset_delete_clear_it(self):
        history = [{"role": "system", "content": "客服"}, {"role": "user", "content": "O1001"}, {"role": "assistant", "content": "好的"}]
        self.store.create_session("a", history)
        memory = ContextMemory("用户提供 O1001", 3)
        self.store.replace_history("a", history, memory)
        self.assertEqual(SessionStore(self.path).load_session("a")["memory"], memory)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_summary BEFORE INSERT ON context_memory BEGIN SELECT RAISE(ABORT, 'summary write failed'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.replace_history("a", history + [{"role": "user", "content": "新问题"}, {"role": "assistant", "content": "新回复"}], ContextMemory("新摘要", 5))
        saved = self.store.load_session("a")
        self.assertEqual(saved["history"], history)
        self.assertEqual(saved["memory"], memory)
        self.store.replace_history("a", history[:1])
        self.assertEqual(self.store.load_session("a")["memory"], ContextMemory())
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DROP TRIGGER fail_summary")
        self.store.replace_history("a", history, memory)
        self.store.delete_session("a")
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM context_memory").fetchone()[0], 0)

    def test_initialize_migrates_legacy_database_without_changing_messages(self):
        history = [{"role": "system", "content": "已有规则"}]
        self.store.create_session("legacy", history)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DROP TABLE context_memory")
        self.store.initialize()
        restored = self.store.load_session("legacy")
        self.assertEqual(restored["history"], history)
        self.assertEqual(restored["memory"], ContextMemory())


if __name__ == "__main__":
    unittest.main()
