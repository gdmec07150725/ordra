"""将业务进度交给 LangGraph saver，保留调用方的 SQLite 事务边界。"""
from contextlib import contextmanager
import json

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver


PROGRESS_NAMESPACE = 'agent-progress'


def initialize_graph_checkpoints(connection):
    # 与已锁定的 langgraph-checkpoint-sqlite 3.1.1 schema 一致。
    # 不调用 executescript/PRAGMA journal_mode，避免隐式提交业务初始化事务。
    connection.execute('''CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT,
        type TEXT, checkpoint BLOB, metadata BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))''')
    connection.execute('''CREATE TABLE IF NOT EXISTS writes (
        thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, idx INTEGER NOT NULL,
        channel TEXT NOT NULL, type TEXT, value BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))''')


class TransactionSqliteSaver(SqliteSaver):
    """库负责格式和读写，外层 SessionStore/CheckpointStore 负责提交与回滚。"""
    def __init__(self, connection):
        super().__init__(connection)
        self.is_setup = True  # 由 SessionStore.initialize 统一建立 schema。

    @contextmanager
    def cursor(self, transaction=True):
        with self.lock:
            cursor = self.conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()


def load_progress(connection, run_id):
    saved = TransactionSqliteSaver(connection).get_tuple({
        'configurable': {'thread_id': run_id, 'checkpoint_ns': PROGRESS_NAMESPACE}})
    return saved.checkpoint['channel_values'].get('execution') if saved else None


def save_progress(connection, run_id, state):
    # 保持旧接口的纯 JSON 数据约束，不接受可执行对象或非有限数字。
    json.dumps(state, ensure_ascii=False, allow_nan=False).encode('utf8')
    saver = TransactionSqliteSaver(connection)
    config = {'configurable': {'thread_id': run_id, 'checkpoint_ns': PROGRESS_NAMESPACE}}
    previous = saver.get_tuple(config)
    checkpoint = empty_checkpoint()
    step = previous.metadata.get('step', 0) + 1 if previous else 0
    version = saver.get_next_version(previous.checkpoint['channel_versions'].get('execution') if previous else None, None)
    checkpoint['channel_values'] = {'execution': state}
    checkpoint['channel_versions'] = {'execution': version}
    checkpoint['updated_channels'] = ['execution']
    saver.put(previous.config if previous else config, checkpoint,
              {'source': 'update', 'step': step, 'parents': {}}, {'execution': version})


def delete_graph_state(connection, run_ids):
    saver = TransactionSqliteSaver(connection)
    for run_id in set(run_ids):
        saver.delete_thread(run_id)
