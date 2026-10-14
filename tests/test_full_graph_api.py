"""正式聊天的真实 SQLite 图状态、重启审批及旧检查点兼容。"""
from contextlib import closing
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_api_app as fixture
import test_human_approval_api as human
from test_stream_agent import text_stream


class FullGraphApiTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = human.HumanApprovalTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown
    new_session = fixture.ApiAppTests.new_session
    restart = human.HumanApprovalTests.restart
    pause = human.HumanApprovalTests.pause
    decide = human.HumanApprovalTests.decide

    def graph_rows(self, run_id):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return [connection.execute(f'SELECT COUNT(*) FROM {table} WHERE thread_id=?',
                                       (run_id,)).fetchone()[0] for table in ('checkpoints', 'writes')]

    async def test_health_identifies_formal_graph_engine(self):
        health = (await self.client.get('/api/health')).json()
        self.assertEqual(health.get('agent_engine'), 'langgraph')

    async def test_pause_is_persisted_interrupt_and_restart_executes_once_then_cleans_graph(self):
        sid, cp = await self.pause(stream=True)
        saved = await self.app.state.graph_checkpointer.aget_tuple({'configurable': {'thread_id': cp['run_id']}})
        self.assertIsNotNone(saved)
        self.assertTrue(any(channel == '__interrupt__' for _, channel, _ in saved.pending_writes))
        self.assertTrue(all(self.graph_rows(cp['run_id'])))
        await self.restart()
        self.assertEqual((await self.decide(sid, cp)).status_code, 200)
        self.responses.append(text_stream('预览已生成，等待确认提交。'))
        with patch.object(self.app.state.returns, 'prepare_return_request', wraps=self.app.state.returns.prepare_return_request) as execute:
            result = await self.client.post(f'/api/sessions/{sid}/resume/stream', json={'run_id': cp['run_id']})
            events = fixture.parse_sse(result.text)
            self.assertEqual(events[-1][0], 'done', result.text)
            self.assertEqual(len(events[-1][1]['return_previews']), 1)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(self.graph_rows(cp['run_id']), [0, 0])
        self.assertIsNone(self.app.state.checkpoints.load(sid))
        await self.restart()
        self.assertEqual(self.graph_rows(cp['run_id']), [0, 0])

    async def test_legacy_approval_without_graph_state_survives_restart_and_migrates(self):
        sid, cp = await self.pause()
        state = self.app.state.checkpoints.load(sid)['state']
        # 重建原版本的落盘格式：业务表内的 version1 JSON，没有任何图检查点。
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('UPDATE agent_checkpoints SET state_json=? WHERE run_id=?',
                               (json.dumps(state, ensure_ascii=False), cp['run_id']))
            for table in ('checkpoints', 'writes'):
                connection.execute(f'DELETE FROM {table} WHERE thread_id=?', (cp['run_id'],))
        await self.restart()
        restored = (await self.client.get(f'/api/sessions/{sid}/checkpoint')).json()['checkpoint']
        self.assertEqual(restored['approvals'], cp['approvals'])
        self.assertEqual(self.graph_rows(cp['run_id']), [0, 0], '读取旧记录不触发迁移')
        edited = {'order_id': 'O1003', 'reason': '耳机左侧没有声音'}
        self.assertEqual((await self.decide(sid, cp, 'edit', arguments=edited)).status_code, 200)
        self.responses.append(text_stream('已按核对后的原因生成预览。'))
        result = await self.client.post(f'/api/sessions/{sid}/resume', json={'run_id': cp['run_id']})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['return_previews'][0]['reason'], edited['reason'])
        self.assertEqual(self.graph_rows(cp['run_id']), [0, 0])
        self.assertEqual(len(self.requests), 2)
