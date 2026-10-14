"""检查点使用真实临时SQLite，验证幂等边界及事务回滚。"""
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from session_store import SessionStore


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'chat.db'
        self.sessions = SessionStore(self.path)
        self.sessions.initialize()
        self.history = [{'role': 'system', 'content': '订单助手'}]
        self.sessions.create_session('a', self.history)

    def store(self):
        from checkpoint_store import CheckpointStore
        return CheckpointStore(self.path)

    def test_restart_restores_checkpoint_without_adding_partial_chat(self):
        store = self.store()
        run = store.begin('a', '查询订单', self.history)
        store.save('a', run['run_id'], {'version': 1, 'rounds': 2, 'active': [{'role': 'user', 'content': '查询订单'}]})
        restored = self.store().load('a')
        self.assertEqual(restored['run_id'], run['run_id'])
        self.assertEqual(restored['state']['rounds'], 2)
        self.assertEqual(self.sessions.load_session('a')['history'], self.history)
        self.assertEqual(self.sessions.list_sessions()[0]['title'], '查询订单')

    def test_wrong_run_and_duplicate_begin_cannot_replace_pending_work(self):
        store = self.store()
        run = store.begin('a', '原问题', self.history)
        for operation in [lambda: store.begin('a', '新问题', self.history),
                          lambda: store.save('a', 'other', {}),
                          lambda: store.claim('a', 'other'),
                          lambda: store.discard('a', 'other'),
                          lambda: self.sessions.replace_history('a', [], checkpoint_id='other')]:
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(store.load('a')['run_id'], run['run_id'])
        self.assertEqual(self.sessions.load_session('a')['history'], self.history)

    def test_failed_final_commit_keeps_checkpoint_and_history_then_commits_once(self):
        store = self.store()
        run = store.begin('a', '查询订单', self.history)
        final = self.history + [{'role': 'user', 'content': '查询订单'}, {'role': 'assistant', 'content': '已签收'}]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TRIGGER fail_final BEFORE INSERT ON messages WHEN NEW.position=1 BEGIN SELECT RAISE(ABORT,'disk failure'); END")
        with self.assertRaises(sqlite3.Error):
            self.sessions.replace_history('a', final, checkpoint_id=run['run_id'])
        self.assertIsNotNone(store.load('a'))
        self.assertEqual(self.sessions.load_session('a')['history'], self.history)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('DROP TRIGGER fail_final')
        self.sessions.replace_history('a', final, checkpoint_id=run['run_id'])
        self.assertIsNone(store.load('a'))
        with self.assertRaises(ValueError):
            self.sessions.replace_history('a', final, checkpoint_id=run['run_id'])
        self.assertEqual(self.sessions.load_session('a')['history'], final)

    def test_history_change_rejects_stale_checkpoint(self):
        store = self.store()
        run = store.begin('a', '查询订单', self.history)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE messages SET message_json = ? WHERE session_id='a'", ('{"role":"system","content":"changed"}',))
        for operation in [lambda: store.save('a', run['run_id'], {}), lambda: store.claim('a', run['run_id'])]:
            with self.assertRaises(ValueError):
                operation()
        store.discard('a', run['run_id'])
        self.assertIsNone(store.load('a'))

    def test_resume_limit_and_reset_delete_cleanup(self):
        store = self.store()
        run = store.begin('a', '查询订单', self.history)
        for count in range(1, 4):
            self.assertEqual(store.claim('a', run['run_id'])['resume_count'], count)
        with self.assertRaises(ValueError):
            store.claim('a', run['run_id'])
        self.sessions.replace_history('a', self.history)
        self.assertIsNone(store.load('a'))
        store.begin('a', '另一个任务', self.history)
        self.sessions.delete_session('a')
        self.assertIsNone(store.load('a'))

    def test_exhausted_model_budget_is_not_resumable_and_does_not_spend_claims(self):
        from checkpoint_store import public_checkpoint
        store = self.store()
        run = store.begin('a', '查询订单', self.history)
        store.save('a', run['run_id'], {'rounds': 10, 'plan': {'snapshot': {'steps': [{}]}}})
        saved = store.load('a')
        self.assertFalse(public_checkpoint(saved)['can_resume'])
        self.assertIn('模型请求', public_checkpoint(saved)['resume_reason'])
        with self.assertRaisesRegex(ValueError, '模型请求'):
            store.claim('a', run['run_id'])
        self.assertEqual(store.load('a')['resume_count'], 0)
        store.save('a', run['run_id'], {'rounds': 10, 'answer': '已完成的候选回复'})
        self.assertTrue(public_checkpoint(store.load('a'))['can_resume'])
