"""第七课：用标准库 sqlite3 保存会话与完整消息。

本模块只处理数据库，不依赖 FastAPI 或模型 SDK。
每个方法自己创建并关闭连接，调用方可以把整个方法放到工作线程执行。
"""

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3

from context_memory import ContextMemory
from history_search import query_terms, search_messages
from return_requests import initialize_return_tables, save_return_previews
from checkpoint_store import initialize_checkpoints, require_checkpoint, checkpoint_state
from graph_checkpoint_storage import delete_graph_state
from trace_store import initialize_traces, finish_trace


class SessionStore:
    def __init__(self, db_path: Path, *, semantic_search=None):
        self.db_path = Path(db_path).resolve()
        self.semantic_search = semantic_search

    def _connect(self) -> sqlite3.Connection:
        """连接必须在实际使用它的线程中创建，保持默认的同线程检查。"""
        # 先在事务外启用外键，再明确采用需要 commit/rollback 的事务模式。
        connection = sqlite3.connect(self.db_path, timeout=2.0, autocommit=True)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.autocommit = False
            connection.row_factory = sqlite3.Row  # 允许用 row["created_at"] 读取列。
            return connection
        except BaseException:
            connection.close()
            raise

    def initialize(self) -> None:
        """首次运行创建目录与表；IF NOT EXISTS 让重启不会覆盖已有数据。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # connection 上下文负责提交/回滚，closing 负责关闭连接，二者分工不同。
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    message_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, position),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS context_memory (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    covered_messages INTEGER NOT NULL CHECK (covered_messages > 0),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            initialize_return_tables(connection)
            initialize_checkpoints(connection)
            initialize_traces(connection)

    def create_session(self, session_id: str, history: list) -> dict:
        """会话记录与初始消息在同一个事务中保存。"""
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        rows = [(session_id, index, json.dumps(message, ensure_ascii=False)) for index, message in enumerate(history)]
        with closing(self._connect()) as connection, connection:
            # ? 是参数占位符：传入的数据不会被当成 SQL 命令的一部分。
            connection.execute("INSERT INTO sessions(session_id, created_at) VALUES (?, ?)", (session_id, created_at))
            connection.executemany("INSERT INTO messages(session_id, position, message_json) VALUES (?, ?, ?)", rows)
        return {"session_id": session_id, "created_at": created_at, "history": history}

    def load_session(self, session_id: str) -> dict | None:
        """按会话编号恢复原始消息字典，包含工具调用等嵌套字段。"""
        with closing(self._connect()) as connection, connection:
            session = connection.execute("SELECT session_id, created_at FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if session is None:
                return None
            rows = connection.execute("SELECT message_json FROM messages WHERE session_id = ? ORDER BY position", (session_id,)).fetchall()
            history = [json.loads(row["message_json"]) for row in rows]
            memory_row = connection.execute("SELECT summary_text, covered_messages FROM context_memory WHERE session_id = ?", (session_id,)).fetchone()
            memory = ContextMemory(memory_row["summary_text"], memory_row["covered_messages"]) if memory_row else ContextMemory()
        return {"session_id": session["session_id"], "created_at": session["created_at"], "history": history, "memory": memory}

    def list_sessions(self) -> list[dict]:
        """列出会话摘要；旧数据库直接可用，不需要修改表或重新创建记录。"""
        with closing(self._connect()) as connection, connection:
            # 本项目第 0 条始终是 system，第 1 条是首次成功保存的用户问题。
            # LEFT JOIN 保留还没有问题的空会话；只读一条消息，不加载整段上下文。
            rows = connection.execute("""
                SELECT s.session_id, s.created_at, m.message_json, c.question AS pending_question
                FROM sessions AS s
                LEFT JOIN messages AS m ON m.session_id = s.session_id AND m.position = 1
                LEFT JOIN agent_checkpoints AS c ON c.session_id = s.session_id
                ORDER BY s.created_at DESC, s.rowid DESC
            """).fetchall()
        summaries = []
        for row in rows:
            first_question = json.loads(row["message_json"]) if row["message_json"] else {}
            title = " ".join((first_question.get("content") or row['pending_question'] or "").split()) or "新对话"
            summaries.append({"session_id": row["session_id"], "created_at": row["created_at"], "title": title[:40] + ("…" if len(title) > 40 else "")})
        return summaries

    def delete_session(self, session_id: str) -> bool:
        """删除会话；外键的 ON DELETE CASCADE 在同一事务中删除它的消息。"""
        with closing(self._connect()) as connection, connection:
            # 先取得写锁，再读取即将级联删除的 run ID；竞争时使用 busy timeout。
            connection.execute('UPDATE sessions SET created_at=created_at WHERE session_id=?', (session_id,))
            runs = [row[0] for row in connection.execute('SELECT run_id FROM agent_runs WHERE session_id=? UNION SELECT run_id FROM agent_checkpoints WHERE session_id=?', (session_id, session_id))]
            delete_graph_state(connection, runs)
            deleted = connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return deleted.rowcount > 0

    def search_history(self, session_id: str, query: str, order: str = "relevance") -> dict:
        """只读当前会话原文；先释放 SQLite 连接，再计算向量与排序。"""
        query_terms(query, order)  # 参数有误时不打开数据库。
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT position, message_json FROM messages WHERE session_id = ? ORDER BY position", (session_id,))
            messages = [(row["position"], json.loads(row["message_json"])) for row in rows]
        return search_messages(messages, query, order, semantic_search=self.semantic_search)

    def replace_history(self, session_id: str, history: list, memory: ContextMemory | None = None, return_previews: list | None = None, checkpoint_id: str | None = None) -> None:
        """完整历史和摘要一起提交；不传摘要表示清空摘要（用于 reset）。"""
        # 先把全部消息转成 JSON，再进入短事务；不在数据库事务中请求模型。
        rows = [(session_id, index, json.dumps(message, ensure_ascii=False)) for index, message in enumerate(history)]
        with closing(self._connect()) as connection, connection:
            if checkpoint_id is not None:
                checkpoint = require_checkpoint(connection, session_id, checkpoint_id)
            # 先开始写操作取得写锁，避免多个连接先读后写时争抢锁升级。
            # 后面的检查和插入失败时，这次删除也会由同一事务回滚。
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            runs = [row[0] for row in connection.execute('SELECT run_id FROM agent_runs WHERE session_id=? UNION SELECT run_id FROM agent_checkpoints WHERE session_id=?', (session_id, session_id))]
            if connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
                raise sqlite3.IntegrityError("待保存的会话不存在。")
            connection.executemany("INSERT INTO messages(session_id, position, message_json) VALUES (?, ?, ?)", rows)
            connection.execute("DELETE FROM context_memory WHERE session_id = ?", (session_id,))
            if memory is not None and memory.text:
                connection.execute("INSERT INTO context_memory(session_id, summary_text, covered_messages) VALUES (?, ?, ?)", (session_id, memory.text, memory.covered_messages))
            if return_previews is not None:
                save_return_previews(connection, session_id, return_previews)
            if checkpoint_id is not None:
                state = checkpoint_state(connection, checkpoint) or {}
                status = 'waiting_input' if (state.get('plan', {}).get('snapshot') or {}).get('status') == 'blocked' else 'completed'
                if state.get('verification', {}).get('status') == 'unverified':
                    status = 'unverified'
                if state.get('recovery', {}).get('stopped'):
                    status = 'stalled'
                finish_trace(connection, session_id, checkpoint_id, status, {'answer_chars': len(state.get('answer') or '')})
            else:
                connection.execute('DELETE FROM agent_runs WHERE session_id=?', (session_id,))
            # 完整回合与检查点删除一起提交；失败时两者一起回滚。reset也清理旧执行。
            delete_graph_state(connection, [checkpoint_id] if checkpoint_id else runs)
            connection.execute('DELETE FROM agent_checkpoints WHERE session_id=?', (session_id,))
