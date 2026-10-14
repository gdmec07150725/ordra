"""执行的业务索引与准入规则；进度内容由 LangGraph saver 持久化。"""
from contextlib import closing
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
from agent_plan import MAX_PLANNED_ROUNDS
from skill_runtime import skill_allowance
from tool_discovery import discovery_allowance
from human_approval import pending_approvals, public_approvals, record_decision
from trace_store import start_trace, sync_trace, finish_trace, add_event
from result_verification import public_plan
from graph_checkpoint_storage import initialize_graph_checkpoints, load_progress, save_progress, delete_graph_state


class CheckpointConflict(ValueError):
    pass


def history_digest(history):
    return hashlib.sha256(json.dumps(history, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def initialize_checkpoints(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS agent_checkpoints (
        session_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE,
        question TEXT NOT NULL,
        base_hash TEXT NOT NULL,
        state_json TEXT,
        updated_at TEXT NOT NULL,
        resume_count INTEGER NOT NULL DEFAULT 0 CHECK(resume_count >= 0),
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )''')
    initialize_graph_checkpoints(connection)


def require_checkpoint(connection, session_id, run_id):
    # 先获取写锁；检查与随后更新、最终历史提交处于同一事务。
    changed = connection.execute('UPDATE agent_checkpoints SET run_id=run_id WHERE session_id=? AND run_id=?', (session_id, run_id))
    if not changed.rowcount:
        raise CheckpointConflict('执行记录已变化或已完成，请刷新页面。')
    row = connection.execute('SELECT * FROM agent_checkpoints WHERE session_id=?', (session_id,)).fetchone()
    history = [json.loads(item[0]) for item in connection.execute('SELECT message_json FROM messages WHERE session_id=? ORDER BY position', (session_id,))]
    if history_digest(history) != row['base_hash']:
        raise CheckpointConflict('会话内容已变化，不能恢复旧执行。请放弃该执行后重新提问。')
    return row


def checkpoint_state(connection, row):
    # 纯读兼容旧任务；只有合法写入时才迁移并清空旧字段。
    return json.loads(row['state_json']) if row['state_json'] else load_progress(connection, row['run_id'])


def _decode(connection, row):
    return {**dict(row), 'state': checkpoint_state(connection, row)} if row else None


def resume_reason(run):
    state = run['state'] or {}
    if state.get('answer'):
        return ''
    if state.get('recovery', {}).get('stopped'):
        return ''  # 只提交固定的停止说明，不再请求模型或工具。
    verification = state.get('verification', {})
    if verification.get('candidate') is not None:
        # 候选可单独重新校验；次数耗尽时只需本地保存未核实答复。
        if verification.get('checks', 0) >= 3:
            return ''
        if run['resume_count'] >= 3:
            return '本次任务已恢复三次，请放弃本次执行后重新提问。'
        return ''
    if (state.get('plan', {}).get('snapshot') or {}).get('status') == 'blocked':
        return ''  # 阻塞提问只需本地提交，不会再请求模型。
    if any(a['decision'] is None for a in pending_approvals(state)):
        return '正在等待你核对工具参数并作出决定。'
    limit = ((MAX_PLANNED_ROUNDS if state.get('plan', {}).get('snapshot') else 4) +
             min(verification.get('corrections', 0), 2) + discovery_allowance(state.get('tool_selection')) + skill_allowance(state.get('skills')))
    if state.get('rounds', 0) >= limit:
        if verification.get('status') == 'correcting':
            return ''  # 只保存未核实说明，不追加模型请求。
        return '本次任务已达到模型请求上限，请放弃本次执行后补充信息或调整问题。'
    if run['resume_count'] >= 3 and not state.get('decision_ready'):
        return '本次任务已恢复三次，请放弃本次执行后重新提问。'
    return ''


def public_checkpoint(run):
    if run is None:
        return None
    state = run['state'] or {}
    return {'run_id': run['run_id'], 'question': run['question'], 'updated_at': run['updated_at'],
            'resume_count': run['resume_count'], 'can_resume': not bool(resume_reason(run)), 'resume_reason': resume_reason(run),
            'plan': public_plan(state.get('plan', {}).get('snapshot'), state.get('verification', {}).get('enabled', False)), 'rounds': state.get('rounds', 0),
            'approvals': public_approvals(state)}


class CheckpointStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=2.0, autocommit=True)
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.autocommit = False
            connection.row_factory = sqlite3.Row
            return connection
        except BaseException:
            connection.close()
            raise

    def begin(self, session_id, question, history):
        run_id = str(uuid4())
        with closing(self._connect()) as connection, connection:
            inserted = connection.execute('''INSERT OR IGNORE INTO agent_checkpoints
                (session_id,run_id,question,base_hash,updated_at) VALUES (?,?,?,?,?)''',
                (session_id, run_id, question, history_digest(history), datetime.now(UTC).isoformat()))
            if not inserted.rowcount:
                raise CheckpointConflict('这段对话有未完成的执行，请先继续或放弃，也可以新建对话。')
            row = require_checkpoint(connection, session_id, run_id)
            start_trace(connection, session_id, run_id, question)
            return _decode(connection, row)

    def load(self, session_id):
        with closing(self._connect()) as connection, connection:
            return _decode(connection, connection.execute('SELECT * FROM agent_checkpoints WHERE session_id=?', (session_id,)).fetchone())

    def save(self, session_id, run_id, state):
        with closing(self._connect()) as connection, connection:
            row = require_checkpoint(connection, session_id, run_id)
            sync_trace(connection, session_id, run_id, checkpoint_state(connection, row), state)
            save_progress(connection, run_id, state)
            connection.execute('UPDATE agent_checkpoints SET state_json=NULL,updated_at=? WHERE session_id=? AND run_id=?',
                               (datetime.now(UTC).isoformat(), session_id, run_id))

    def claim(self, session_id, run_id):
        with closing(self._connect()) as connection, connection:
            row = require_checkpoint(connection, session_id, run_id)
            reason = resume_reason(_decode(connection, row))
            if reason:
                raise CheckpointConflict(reason)
            state = checkpoint_state(connection, row) or {}
            increment = 0 if state.pop('decision_ready', False) else 1
            start_trace(connection, session_id, run_id, row['question'], legacy=True, state=state)
            connection.execute("UPDATE agent_runs SET status='running',updated_at=? WHERE run_id=?", (datetime.now(UTC).isoformat(), run_id))
            add_event(connection, run_id, 'resumed', {'resume_count': row['resume_count'] + increment, 'after_human_decision': not bool(increment)})
            if state:
                save_progress(connection, run_id, state)
            connection.execute('UPDATE agent_checkpoints SET resume_count=resume_count+?,state_json=NULL,updated_at=? WHERE session_id=?',
                               (increment, datetime.now(UTC).isoformat(), session_id))
            return _decode(connection, connection.execute('SELECT * FROM agent_checkpoints WHERE session_id=?', (session_id,)).fetchone())

    def decide(self, session_id, body):
        with closing(self._connect()) as connection, connection:
            row = require_checkpoint(connection, session_id, body.run_id)
            state = checkpoint_state(connection, row) or {}
            try:
                record_decision(state, body)
            except LookupError as error:
                raise CheckpointConflict(str(error)) from error
            start_trace(connection, session_id, body.run_id, row['question'], legacy=True, state=state)
            approval = next(a for a in pending_approvals(state) if a['approval_id'] == body.approval_id)
            add_event(connection, body.run_id, 'human_decision', approval, 'decision:' + body.approval_id)
            connection.execute('UPDATE agent_runs SET status=?,updated_at=? WHERE run_id=?',
                               ('ready' if state.get('decision_ready') else 'awaiting_approval', datetime.now(UTC).isoformat(), body.run_id))
            save_progress(connection, body.run_id, state)
            connection.execute('UPDATE agent_checkpoints SET state_json=NULL,updated_at=? WHERE session_id=?',
                               (datetime.now(UTC).isoformat(), session_id))
            return _decode(connection, connection.execute('SELECT * FROM agent_checkpoints WHERE session_id=?', (session_id,)).fetchone())

    def discard(self, session_id, run_id):
        with closing(self._connect()) as connection, connection:
            deleted = connection.execute('DELETE FROM agent_checkpoints WHERE session_id=? AND run_id=?', (session_id, run_id))
            if not deleted.rowcount:
                raise CheckpointConflict('执行记录已变化或已完成，请刷新页面。')
            delete_graph_state(connection, [run_id])
            finish_trace(connection, session_id, run_id, 'discarded')
