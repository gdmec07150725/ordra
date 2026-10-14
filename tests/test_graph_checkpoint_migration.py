"""真实SQLite：旧检查点读取、LangGraph存储迁移和同事务清理。"""
from contextlib import closing
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
import threading

from checkpoint_store import CheckpointStore
from session_store import SessionStore
from human_approval import HumanDecision, make_approval


class GraphCheckpointMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'chat.db'
        self.sessions = SessionStore(self.path)
        self.sessions.initialize()
        self.store = CheckpointStore(self.path)
        self.history = [{'role': 'system', 'content': '客服'}]
        self.sessions.create_session('a', self.history)
        self.run = self.store.begin('a', '查询订单', self.history)
        self.state = {'version': 1, 'rounds': 1, 'active': [{'role': 'user', 'content': '查询订单'}]}

    def legacy(self, state=None):
        encoded = json.dumps(state or self.state, ensure_ascii=False)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('UPDATE agent_checkpoints SET state_json=? WHERE run_id=?', (encoded, self.run['run_id']))
        return encoded

    def raw_state(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute('SELECT state_json FROM agent_checkpoints WHERE run_id=?', (self.run['run_id'],)).fetchone()[0]

    def test_read_preserves_legacy_then_save_migrates_to_langgraph(self):
        encoded = self.legacy()
        self.assertEqual(self.store.load('a')['state'], self.state)
        self.assertEqual(self.raw_state(), encoded)
        updated = {**self.state, 'rounds': 2}
        self.store.save('a', self.run['run_id'], updated)
        self.assertIsNone(self.raw_state())
        self.assertEqual(CheckpointStore(self.path).load('a')['state'], updated)
        from graph_checkpoint_storage import load_progress
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(load_progress(db, self.run['run_id']), updated)
            self.assertGreater(db.execute('SELECT COUNT(*) FROM checkpoints WHERE checkpoint_ns=?', ('agent-progress',)).fetchone()[0], 0)

    def test_metadata_failure_rolls_back_graph_save_and_legacy_migration(self):
        encoded = self.legacy()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TRIGGER fail_migration BEFORE UPDATE OF state_json ON agent_checkpoints BEGIN SELECT RAISE(ABORT,'disk error'); END")
        with self.assertRaises(sqlite3.Error):
            self.store.save('a', self.run['run_id'], {**self.state, 'rounds': 2})
        self.assertEqual(self.raw_state(), encoded)
        self.assertEqual(self.store.load('a')['state'], self.state)
        from graph_checkpoint_storage import load_progress
        with closing(sqlite3.connect(self.path)) as db:
            self.assertIsNone(load_progress(db, self.run['run_id']))

    def test_claim_migrates_legacy_without_resetting_budget(self):
        self.legacy()
        claimed = self.store.claim('a', self.run['run_id'])
        self.assertEqual(claimed['state'], self.state)
        self.assertEqual(claimed['resume_count'], 1)
        self.assertIsNone(self.raw_state())

    def test_edit_migrates_old_approval_and_preserves_decision(self):
        call = {'id': 'preview', 'type': 'function', 'function': {'name': 'prepare_return_request', 'arguments': '{"order_id":"O1003","reason":"无声"}'}}
        approval = make_approval(call)
        state = {**self.state, 'pending_batch': {'message': {'role': 'assistant', 'tool_calls': [call]}, 'approvals': [approval], 'reserved': [True]}}
        self.legacy(state)
        decided = self.store.decide('a', HumanDecision(run_id=self.run['run_id'], approval_id=approval['approval_id'], decision='edit', arguments={'order_id': 'O1003', 'reason': '右侧无声'}))
        self.assertTrue(decided['state']['decision_ready'])
        self.assertIsNone(self.raw_state())
        restored = CheckpointStore(self.path).load('a')
        self.assertEqual(restored['state']['pending_batch']['approvals'][0]['arguments']['reason'], '右侧无声')
        self.assertEqual(self.store.claim('a', self.run['run_id'])['resume_count'], 0)

    def seed_graph(self):
        self.store.save('a', self.run['run_id'], self.state)
        from graph_checkpoint_storage import TransactionSqliteSaver
        from langgraph.checkpoint.base import empty_checkpoint
        with closing(sqlite3.connect(self.path)) as db, db:
            saver = TransactionSqliteSaver(db)
            cp = empty_checkpoint(); cp['channel_values'] = {'stage': 'tools'}
            config = saver.put({'configurable': {'thread_id': self.run['run_id'], 'checkpoint_ns': ''}}, cp, {'source': 'input', 'step': 0}, {})
            saver.put_writes(config, [('__interrupt__', 'pending')], 'task')

    def assert_graph_gone(self):
        with closing(sqlite3.connect(self.path)) as db:
            for table in ('checkpoints', 'writes'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table} WHERE thread_id=?', (self.run['run_id'],)).fetchone()[0], 0)

    def test_discard_and_delete_clear_both_graph_namespaces(self):
        self.seed_graph()
        self.store.discard('a', self.run['run_id'])
        self.assert_graph_gone()
        self.run = self.store.begin('a', '查询订单', self.history)
        self.seed_graph()
        self.sessions.delete_session('a')
        self.assert_graph_gone()

    def test_final_commit_failure_keeps_graph_then_success_cleans_it(self):
        self.seed_graph()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TRIGGER fail_final BEFORE INSERT ON messages WHEN NEW.position=1 BEGIN SELECT RAISE(ABORT,'disk error'); END")
        final = self.history + [{'role': 'user', 'content': '查询订单'}, {'role': 'assistant', 'content': '请提供订单号'}]
        with self.assertRaises(sqlite3.Error):
            self.sessions.replace_history('a', final, checkpoint_id=self.run['run_id'])
        self.assertEqual(self.store.load('a')['state'], self.state)
        self.assertEqual(self.sessions.load_session('a')['history'], self.history)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('DROP TRIGGER fail_final')
        self.sessions.replace_history('a', final, checkpoint_id=self.run['run_id'])
        self.assert_graph_gone()

    def test_reset_cleans_graph_and_other_session_is_untouched(self):
        self.seed_graph()
        self.sessions.create_session('b', self.history)
        another = self.store.begin('b', '另一条问题', self.history)
        self.store.save('b', another['run_id'], {'rounds': 1})
        self.sessions.replace_history('a', self.history)
        self.assert_graph_gone()
        self.assertEqual(self.store.load('b')['state'], {'rounds': 1})

    def test_session_writes_wait_for_competing_short_writer(self):
        for mode in ('DELETE', 'WAL'):
            with closing(sqlite3.connect(self.path, autocommit=True)) as db:
                db.execute(f'PRAGMA journal_mode={mode}')
            for operation in ('complete', 'reset', 'delete'):
                with self.subTest(mode=mode, operation=operation):
                    sid = mode + operation
                    self.sessions.create_session(sid, self.history)
                    run = self.store.begin(sid, '查询订单', self.history)
                    locked, release = threading.Event(), threading.Event()

                    def hold_writer():
                        with closing(sqlite3.connect(self.path)) as db, db:
                            db.execute('UPDATE sessions SET created_at=created_at WHERE session_id=?', (sid,))
                            locked.set()
                            self.assertTrue(release.wait(2))

                    with ThreadPoolExecutor(max_workers=1) as pool:
                        held = pool.submit(hold_writer)
                        self.assertTrue(locked.wait(2))
                        timer = threading.Timer(.15, release.set)
                        timer.start()
                        try:
                            if operation == 'delete':
                                self.assertTrue(self.sessions.delete_session(sid))
                            else:
                                self.sessions.replace_history(sid, self.history,
                                    checkpoint_id=run['run_id'] if operation == 'complete' else None)
                        finally:
                            release.set()
                            timer.cancel()
                            timer.join()
                            held.result(timeout=2)
