"""真实 SQLite 验证去重、保留上限、重启和级联删除。"""
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from session_store import SessionStore
from checkpoint_store import CheckpointStore
from trace_store import TraceStore, MAX_EVENTS, add_event, safe_data, start_trace


class TraceStoreTests(unittest.TestCase):
    def test_subagent_counts_and_tool_evidence_without_local_messages_or_summary(self):
        run = self.checkpoints.begin('s', '问题', self.history)
        outcome = {'call_id': 'specialist_1:r1:read', 'name': 'get_order', 'attempts': 1, 'retries': [], 'elapsed_ms': 2,
                   'status': 'ok', 'result': {'order_id': 'O1003', 'status': '已签收'}}
        state = {'delegation': {'requests': 2, 'usage': {'reported_rounds': 2, 'total_tokens': 70}, 'tasks': [
            {'id': 'specialist_1', 'agent': 'orders', 'task': '查询订单', 'status': 'completed', 'rounds': 2,
             'messages': [{'role': 'system', 'content': '私有子任务提示'}], 'tool_results': [outcome], 'result': {'summary': '未公开子结论'}}]},
            'runtime': {'trace': [{k:v for k,v in outcome.items() if k != 'result'}]}}
        self.checkpoints.save('s', run['run_id'], state)
        self.checkpoints.save('s', run['run_id'], state)
        trace = self.traces.load('s', run['run_id'])
        self.assertEqual(trace['metrics']['subagent_requests'], 2)
        self.assertEqual(trace['metrics']['total_tokens'], 70)
        self.assertEqual(sum(e['kind'] == 'specialist_done' for e in trace['events']), 1)
        tool = next(e['data'] for e in trace['events'] if e['kind'] == 'tool_result')
        self.assertEqual(tool['result']['order_id'], 'O1003')
        self.assertNotIn('私有子任务提示', json.dumps(trace, ensure_ascii=False))
        self.assertNotIn('未公开子结论', json.dumps(trace, ensure_ascii=False))

    def setUp(self):
        directory = TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / 'chat.db'
        self.store = SessionStore(self.db); self.store.initialize()
        self.history = [{'role': 'system', 'content': '助手'}]
        self.store.create_session('s', self.history)
        self.traces = TraceStore(self.db); self.checkpoints = CheckpointStore(self.db)

    def test_checkpoint_save_deduplicates_events_and_restart_preserves_counts(self):
        run = self.checkpoints.begin('s', '问题', self.history)
        state = {'rounds': 1, 'last_model_result': {'round': 1, 'finish_reason': 'stop'}, 'totals': {'reported_rounds': 1, 'total_tokens': 55}}
        self.checkpoints.save('s', run['run_id'], state)
        self.checkpoints.save('s', run['run_id'], state)
        before = self.traces.load('s', run['run_id'])
        self.assertEqual([e['kind'] for e in before['events']], ['started', 'model_request', 'model_response'])
        self.traces.recover_running(); self.traces.recover_running()
        after = self.traces.load('s', run['run_id'])
        self.assertEqual(after['status'], 'interrupted')
        self.assertEqual(after['metrics'], before['metrics'])
        self.assertEqual(sum(e['kind'] == 'interrupted' for e in after['events']), 1)

    def test_event_limit_still_allows_final_status_and_metrics(self):
        run = self.checkpoints.begin('s', '问题', self.history)
        with closing(self.traces._connect()) as db, db:
            for index in range(MAX_EVENTS + 5):
                add_event(db, run['run_id'], 'diagnostic', {'index': index})
        self.checkpoints.save('s', run['run_id'], {'rounds': 4})
        self.checkpoints.discard('s', run['run_id'])
        trace = self.traces.load('s', run['run_id'])
        self.assertEqual(len(trace['events']), MAX_EVENTS + 1)
        self.assertEqual(trace['events'][-1]['kind'], 'truncated')
        self.assertEqual(trace['status'], 'discarded')
        self.assertEqual(trace['metrics']['model_requests'], 4)

    def test_cursor_has_no_duplicates_and_deletion_cascades(self):
        with closing(self.traces._connect()) as db, db:
            for index in range(25):
                start_trace(db, 's', str(index), '问题')
        first = self.traces.list_runs('s')
        second = self.traces.list_runs('s', first['next_cursor'])
        ids = [r['run_id'] for r in first['runs'] + second['runs']]
        self.assertEqual(len(ids), 25); self.assertEqual(len(set(ids)), 25)
        self.assertIsNone(second['next_cursor'])
        self.store.delete_session('s')
        with closing(self.traces._connect()) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM agent_run_events').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM agent_runs').fetchone()[0], 0)

    def test_sanitization_bounds_and_nonfinite_values(self):
        value = safe_data({'api_key': 'secret-value', 'nested': {'authorization': 'private'}, 'text': 'Bearer abcdef', 'long': 'x' * 10000, 'number': float('inf')})
        encoded = json.dumps(value, allow_nan=False)
        for secret in ('secret-value', 'private', 'abcdef'):
            self.assertNotIn(secret, encoded)
        self.assertLess(len(value['long']), 1050)
        self.assertIn('无效数值', value['number'])

    def test_review_events_count_usage_without_saving_candidate_or_prompt(self):
        run = self.checkpoints.begin('s', '问题', self.history)
        state = {'rounds': 1, 'verification': {'checks': 1, 'corrections': 1, 'status': 'correcting',
            'candidate': '不可展示的错误草稿', 'reports': [{'number': 1, 'verdict': 'revise', 'issues': ['不可展示的错误草稿，依据不足'], 'evidence_ids': [], 'elapsed_ms': 2}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'reported_rounds': 1}}}
        self.checkpoints.save('s', run['run_id'], state)
        self.checkpoints.save('s', run['run_id'], state)
        trace = self.traces.load('s', run['run_id'])
        self.assertEqual(trace['metrics']['verification_requests'], 1)
        self.assertEqual(trace['metrics']['total_tokens'], 120)
        self.assertEqual(trace['metrics']['corrections'], 1)
        self.assertEqual(sum(e['kind'] == 'verification_result' for e in trace['events']), 1)
        self.assertNotIn('不可展示的错误草稿', json.dumps(trace, ensure_ascii=False))
