"""本地模拟退货：模型只准备预览，明确确认后才在短事务中提交。"""

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import orders


_STATUSES = {"pending": "待确认", "submitted": "待人工审核", "cancelled": "已取消", "expired": "已失效"}
_REVIEW_STATUSES = {"approved": "审核通过", "rejected": "审核拒绝"}
_PREVIEW_SELECT = """
    SELECT p.*, r.decision AS review_decision, r.reason AS review_reason, r.reviewed_at
    FROM return_previews AS p
    LEFT JOIN return_reviews AS r ON r.request_id = p.request_id
"""


def initialize_return_tables(connection: sqlite3.Connection) -> None:
    """由 SessionStore 初始化；不提交调用者正在执行的事务。"""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS return_previews (
            preview_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
            order_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'submitted', 'cancelled', 'expired')),
            request_id TEXT UNIQUE,
            created_at TEXT NOT NULL,
            confirmed_at TEXT,
            CHECK ((state = 'submitted' AND request_id IS NOT NULL AND confirmed_at IS NOT NULL)
                OR (state != 'submitted' AND request_id IS NULL AND confirmed_at IS NULL)),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS one_submitted_return_per_order
        ON return_previews(session_id, order_id) WHERE state = 'submitted'
    """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS return_previews_by_session
        ON return_previews(session_id, created_at)
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS return_reviews (
            request_id TEXT PRIMARY KEY NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('approved', 'rejected')),
            reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
            reviewed_at TEXT NOT NULL,
            FOREIGN KEY (request_id) REFERENCES return_previews(request_id) ON DELETE CASCADE
        )
    """)


def _require_session(connection: sqlite3.Connection, session_id: str) -> None:
    if connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
        raise LookupError("会话不存在。")


def _normalize_order_id(order_id: str) -> str:
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("订单号必须是非空字符串。")
    return order_id.strip().upper()


def _normalize_reason(reason: str, *, label: str = "退货原因") -> str:
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 500:
        raise ValueError(f"{label}必须是 1–500 字的非空字符串。")
    if "\x00" in reason:
        raise ValueError(f"{label}不能包含空字符。")
    try:
        reason.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label}包含无效的 Unicode 字符。") from error
    return reason.strip()


def _json_encode(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        encoded.encode("utf-8")
        return encoded
    except (TypeError, ValueError) as error:
        raise ValueError("退货申请包含无法保存的 JSON 数据。") from error


def _serialize(row: sqlite3.Row) -> dict:
    review = None
    if row["review_decision"] is not None:
        review = {"decision": row["review_decision"], "reason": row["review_reason"], "reviewed_at": row["reviewed_at"]}
    return {
        "preview_id": row["preview_id"],
        "order_id": row["order_id"],
        "reason": row["reason"],
        "order": json.loads(row["order_json"]),
        "state": row["state"],
        "request_id": row["request_id"],
        "created_at": row["created_at"],
        "confirmed_at": row["confirmed_at"],
        "status": _REVIEW_STATUSES[review["decision"]] if review else _STATUSES[row["state"]],
        "simulated": True,
        "review": review,
    }


def save_return_previews(connection: sqlite3.Connection, session_id: str, previews: list[dict]) -> None:
    """成功聊天回合同事务保存；包括没有新预览时，也使旧 pending 失效。"""
    if not isinstance(previews, list) or len(previews) > 1:
        raise ValueError("每个回合最多保存一份退货申请预览。")
    rows = []
    for preview in previews:
        if not isinstance(preview, dict) or preview.get("state") != "pending":
            raise ValueError("只能保存待确认的退货申请预览。")
        order_id = _normalize_order_id(preview.get("order_id"))
        reason = _normalize_reason(preview.get("reason"))
        order = preview.get("order")
        if (
            not isinstance(preview.get("preview_id"), str) or not preview["preview_id"]
            or not isinstance(preview.get("created_at"), str) or not preview["created_at"]
            or preview.get("request_id") is not None or preview.get("confirmed_at") is not None
            or not isinstance(order, dict) or order.get("order_id") != order_id
        ):
            raise ValueError("退货申请预览无效。")
        order_json = _json_encode(order)
        rows.append((preview["preview_id"], session_id, order_id, reason, order_json, preview["created_at"]))

    # 先写后读，适配调用者的 DEFERRED 事务并取得写锁；异常交由调用者整体回滚。
    connection.execute("UPDATE return_previews SET state = 'expired' WHERE session_id = ? AND state = 'pending'", (session_id,))
    _require_session(connection, session_id)
    for row in rows:
        if connection.execute("SELECT 1 FROM return_previews WHERE session_id = ? AND order_id = ? AND state = 'submitted'", (session_id, row[2])).fetchone():
            raise ValueError("该订单已经提交过退货申请，请查询已有申请。")
    connection.executemany("""
        INSERT INTO return_previews
            (preview_id, session_id, order_id, reason, order_json, state, request_id, created_at, confirmed_at)
        VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)
    """, rows)


class ReturnRequestStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()

    def _connect(self, *, writable: bool = False) -> sqlite3.Connection:
        # mode=ro 使准备/查询不可能创建或写入数据库；建表归 SessionStore 管理。
        mode = "rw" if writable else "ro"
        connection = sqlite3.connect(f"{self.db_path.as_uri()}?mode={mode}", uri=True, timeout=2.0, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.row_factory = sqlite3.Row
            return connection
        except BaseException:
            connection.close()
            raise

    def prepare_return_request(self, session_id: str, order_id: str, reason: str) -> dict:
        """只读取订单和会话，返回可由成功回合保存的临时预览。"""
        order_id = _normalize_order_id(order_id)
        reason = _normalize_reason(reason)
        order = orders.get_order(order_id)
        if "error" in order:
            raise ValueError(order["error"])
        # 复制整个 JSON 快照，嵌套字段也不会与订单资料共享可变对象。
        order = json.loads(_json_encode(order))
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN")
            _require_session(connection, session_id)
            submitted = connection.execute(_PREVIEW_SELECT + " WHERE p.session_id = ? AND p.order_id = ? AND p.state = 'submitted'", (session_id, order_id)).fetchone()
            if submitted is not None:
                return {"request": _serialize(submitted), "already_submitted": True}
        return {"preview": {
            "preview_id": "PRE-" + uuid4().hex,
            "order_id": order_id,
            "reason": reason,
            "order": order,
            "state": "pending",
            "request_id": None,
            "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
            "confirmed_at": None,
            "status": _STATUSES["pending"],
            "simulated": True,
            "review": None,
        }}

    def list_previews(self, session_id: str) -> list[dict]:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN")
            _require_session(connection, session_id)
            return [_serialize(row) for row in connection.execute(_PREVIEW_SELECT + " WHERE p.session_id = ? ORDER BY p.created_at, p.rowid", (session_id,))]

    def get_return_request(self, session_id: str, order_id: str) -> dict:
        order_id = _normalize_order_id(order_id)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN")
            _require_session(connection, session_id)
            row = connection.execute(_PREVIEW_SELECT + " WHERE p.session_id = ? AND p.order_id = ? AND p.state = 'submitted'", (session_id, order_id)).fetchone()
            if row is None:
                return {"error": f"当前会话中订单 {order_id} 尚未提交退货申请。"}
            return {"request": _serialize(row)}

    @staticmethod
    def _find_preview(connection: sqlite3.Connection, session_id: str, preview_id: str) -> sqlite3.Row:
        _require_session(connection, session_id)
        row = connection.execute(_PREVIEW_SELECT + " WHERE p.session_id = ? AND p.preview_id = ?", (session_id, preview_id)).fetchone()
        if row is None:
            raise LookupError("退货申请预览不存在。")
        return row

    def create_return_request(self, session_id: str, preview_id: str) -> dict:
        """仅供用户明确确认的服务器入口；提交及两条回执消息不可分割。"""
        with closing(self._connect(writable=True)) as connection, connection:
            # 在任何读取前取得写锁，让重复请求顺序看到已经提交的结果。
            connection.execute("BEGIN IMMEDIATE")
            preview = self._find_preview(connection, session_id, preview_id)
            if preview["state"] == "submitted":
                return _serialize(preview)
            if preview["state"] != "pending":
                raise ValueError("该退货申请预览已取消或失效，请重新生成预览。")
            existing = connection.execute("SELECT 1 FROM return_previews WHERE session_id = ? AND order_id = ? AND state = 'submitted'", (session_id, preview["order_id"])).fetchone()
            if existing is not None:
                raise ValueError("该订单已经提交过退货申请，请查询已有申请。")
            request_id = "RET-" + uuid4().hex.upper()
            confirmed_at = datetime.now(UTC).isoformat(timespec="microseconds")
            connection.execute("UPDATE return_previews SET state = 'submitted', request_id = ?, confirmed_at = ? WHERE preview_id = ? AND session_id = ?", (request_id, confirmed_at, preview_id, session_id))
            position = connection.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            details = f"订单号：{preview['order_id']}\n退货原因：{preview['reason']}"
            messages = [
                {"role": "user", "content": f"确认提交这份退货申请（本地模拟）。\n{details}"},
                {"role": "assistant", "content": f"退货申请已提交（本地模拟）。\n申请编号：{request_id}\n{details}\n状态：待人工审核。"},
            ]
            connection.executemany("INSERT INTO messages(session_id, position, message_json) VALUES (?, ?, ?)", [(session_id, position + index, _json_encode(message)) for index, message in enumerate(messages)])
            return _serialize(self._find_preview(connection, session_id, preview_id))

    def cancel_preview(self, session_id: str, preview_id: str) -> dict:
        """仅取消 pending 预览；重复取消返回相同状态，不产生聊天消息。"""
        with closing(self._connect(writable=True)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            preview = self._find_preview(connection, session_id, preview_id)
            if preview["state"] == "cancelled":
                return _serialize(preview)
            if preview["state"] != "pending":
                raise ValueError("只能取消待确认的退货申请预览。")
            connection.execute("UPDATE return_previews SET state = 'cancelled' WHERE session_id = ? AND preview_id = ?", (session_id, preview_id))
            return _serialize(self._find_preview(connection, session_id, preview_id))

    def list_requests(self, review_status: str = "pending") -> dict:
        """人工审核队列；计数和列表从同一份已提交申请快照生成。"""
        if not isinstance(review_status, str) or review_status not in ("pending", "approved", "rejected", "all"):
            raise ValueError("审核筛选必须是 pending、approved、rejected 或 all。")
        ordering = {
            "pending": "p.confirmed_at ASC, p.rowid ASC",
            "approved": "r.reviewed_at DESC, p.rowid DESC",
            "rejected": "r.reviewed_at DESC, p.rowid DESC",
            "all": "p.confirmed_at DESC, p.rowid DESC",
        }[review_status]
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN")
            rows = connection.execute(_PREVIEW_SELECT + " WHERE p.state = 'submitted' ORDER BY " + ordering).fetchall()
        counts = {"pending": 0, "approved": 0, "rejected": 0, "all": len(rows)}
        requests = []
        for row in rows:
            status = row["review_decision"] or "pending"
            counts[status] += 1
            if review_status == "all" or status == review_status:
                requests.append({**_serialize(row), "session_id": row["session_id"]})
        return {"requests": requests, "counts": counts}

    def review_request(self, session_id: str, request_id: str, decision: str, reason: str) -> dict:
        """人工入口记录一次审核；相同重试幂等，不修改原申请或聊天消息。"""
        if not isinstance(decision, str) or decision.strip().lower() not in _REVIEW_STATUSES:
            raise ValueError("审核决定必须是 approved 或 rejected。")
        decision = decision.strip().lower()
        reason = _normalize_reason(reason, label="审核原因")
        with closing(self._connect(writable=True)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            _require_session(connection, session_id)
            request = connection.execute(_PREVIEW_SELECT + " WHERE p.session_id = ? AND p.request_id = ?", (session_id, request_id)).fetchone()
            if request is None:
                raise LookupError("退货申请不存在。")
            if request["state"] != "submitted":
                raise ValueError("只能审核已提交的退货申请。")
            if request["review_decision"] is not None:
                if request["review_decision"] == decision and request["review_reason"] == reason:
                    return _serialize(request)
                raise ValueError("该退货申请已有审核结果，不能重复修改。")
            reviewed_at = datetime.now(UTC).isoformat(timespec="microseconds")
            connection.execute("INSERT INTO return_reviews(request_id, decision, reason, reviewed_at) VALUES (?, ?, ?, ?)", (request_id, decision, reason, reviewed_at))
            return _serialize(self._find_preview(connection, session_id, request["preview_id"]))
