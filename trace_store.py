"""本机可观察执行轨迹；与检查点/最终历史共用事务，不记录隐藏思考。"""
from contextlib import closing
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
from result_verification import public_review, public_plan


MAX_EVENTS = 400


def now():
    return datetime.now(UTC).isoformat()


def safe_data(value, depth=0):
    if depth > 7:
        return '[内容已省略]'
    if isinstance(value, dict):
        return {str(key): '[已隐藏]' if re.search(r'api.?key|authorization|password|secret|access.?token', str(key), re.I)
                else safe_data(item, depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [safe_data(item, depth + 1) for item in value[:30]]
    if isinstance(value, str):
        value = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [已隐藏]', value)
        return value if len(value) <= 1000 else value[:1000] + '…[已截断]'
    if isinstance(value, float) and not math.isfinite(value):
        return '[无效数值：' + str(value) + ']'
    return value


def proposed_arguments(raw):
    """异常参数也能留下诊断，不改变业务侧的解析和校验结果。"""
    try:
        return safe_data(json.loads(raw))
    except (ValueError, TypeError, RecursionError):
        return safe_data({'invalid_json': raw})


def initialize_traces(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS agent_runs (
        run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, question TEXT NOT NULL,
        status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE)''')
    connection.execute('CREATE INDEX IF NOT EXISTS runs_by_session ON agent_runs(session_id)')
    connection.execute('''CREATE TABLE IF NOT EXISTS agent_run_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
        event_key TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL, data_json TEXT NOT NULL,
        UNIQUE(run_id,event_key), FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE)''')


def add_event(connection, run_id, kind, data, key=None):
    key = key or str(uuid4())
    count = connection.execute('SELECT count(*) FROM agent_run_events WHERE run_id=?', (run_id,)).fetchone()[0]
    if count >= MAX_EVENTS:
        kind, key, data = 'truncated', 'truncated', {'message': '事件已达保存上限，后续详情省略，最终状态和统计继续更新。'}
    connection.execute('''INSERT OR IGNORE INTO agent_run_events(run_id,event_key,kind,created_at,data_json)
                          VALUES(?,?,?,?,?)''', (run_id, key, kind, now(), json.dumps(safe_data(data), ensure_ascii=False, allow_nan=False)))


def start_trace(connection, session_id, run_id, question, *, legacy=False, state=None):
    stamp = now()
    inserted = connection.execute('''INSERT OR IGNORE INTO agent_runs(run_id,session_id,question,status,created_at,updated_at)
                                    VALUES(?,?,?,'running',?,?)''', (run_id, session_id, safe_data(question), stamp, stamp))
    if inserted.rowcount:
        if state:
            connection.execute('UPDATE agent_runs SET metrics_json=? WHERE run_id=?', (json.dumps(trace_metrics(state)), run_id))
        add_event(connection, run_id, 'started', {'legacy_checkpoint': legacy,
            'message': '旧检查点，从本次恢复开始记录。' if legacy else '开始执行本轮请求。'}, 'started')


def finish_trace(connection, session_id, run_id, status, details=None):
    row = connection.execute('SELECT status FROM agent_runs WHERE session_id=? AND run_id=?', (session_id, run_id)).fetchone()
    if row is None:
        return
    connection.execute('UPDATE agent_runs SET status=?,updated_at=? WHERE session_id=? AND run_id=?',
                       (status, now(), session_id, run_id))
    add_event(connection, run_id, status, details or {}, status if status in ('completed', 'waiting_input', 'discarded', 'unverified', 'stalled') else None)


def trace_metrics(state):
    runtime, context = state.get('runtime', {}), state.get('context', {})
    totals, summary = state.get('totals', {}), context.get('summary_usage', {})
    verification = state.get('verification', {})
    review_usage = verification.get('usage', {})
    delegation = state.get('delegation', {})
    child_usage = delegation.get('usage', {})
    business = [t for t in runtime.get('trace', []) if t['name'] not in ('update_plan', 'delegate_tasks', 'discover_tools', 'load_skill', 'read_skill_resource')]
    return {'model': state.get('model', ''), 'model_requests': state.get('rounds', 0), 'summary_requests': context.get('summary_calls', 0),
            'verification_requests': verification.get('checks', 0), 'corrections': verification.get('corrections', 0),
            'verification_status': verification.get('status', 'disabled'),
            'subagent_requests': delegation.get('requests', 0), 'delegated_tasks': len(delegation.get('tasks', [])),
            'subagent_failures': sum(task['status'] == 'failed' for task in delegation.get('tasks', [])),
            'reported_requests': totals.get('reported_rounds', 0) + summary.get('reported_rounds', 0) + review_usage.get('reported_rounds', 0) + child_usage.get('reported_rounds', 0),
            **{key: totals.get(key, 0) + summary.get(key, 0) + review_usage.get(key, 0) + child_usage.get(key, 0) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'tool_calls': len(business), 'tool_attempts': sum(t['attempts'] for t in business),
            'tool_retries': sum(len(t['retries']) for t in business),
            'max_parallel': runtime.get('peak_parallel', 0), 'reserved_business_calls': runtime.get('calls', 0),
            'plan_updates': (state.get('plan', {}).get('snapshot') or {}).get('revision', 0),
            'tool_mode': state.get('tool_selection', {}).get('mode', 'local'),
            'tool_discoveries': state.get('tool_selection', {}).get('calls', 0),
            'skill_loads': state.get('skills', {}).get('calls', 0),
            'skills_loaded': len(state.get('skills', {}).get('loaded', {})),
            'skill_resource_reads': state.get('skills', {}).get('resource_calls', 0),
            'skill_resources_loaded': len(state.get('skills', {}).get('resources', {})),
            'replans': state.get('recovery', {}).get('replans', 0),
            'stalled_batches': state.get('recovery', {}).get('stalled_batches', 0),
            'recovery_stopped': state.get('recovery', {}).get('stopped', False)}


def tool_result_summary(result):
    allowed = ('error', 'error_code', 'status', 'order_id', 'retryable', 'budget_limited', 'search_mode', 'name', 'path', 'version', 'estimated_tokens')
    summary = {key: result[key] for key in allowed if key in result}
    if 'delegations' in result:
        summary['delegations'] = [{key: item[key] for key in ('task_id', 'agent', 'status')} for item in result['delegations']]
    if 'results' in result:
        summary['returned_items'] = len(result['results'])
        summary['sources'] = [item['source'] for item in result['results'] if 'source' in item]
    for key in ('preview', 'request'):
        if isinstance(result.get(key), dict):
            summary[key] = {name: result[key].get(name) for name in ('preview_id', 'request_id', 'order_id', 'reason', 'state')}
    return summary


def sync_trace(connection, session_id, run_id, previous, state):
    """只投影已保存的事实。稳定事件键使重复保存、恢复不重复计数。"""
    previous = previous or {}
    round_number = state.get('rounds', 0)
    context = state.get('context', {})
    for loaded in state.get('skills', {}).get('events', []):
        if loaded['sequence'] > previous.get('skills', {}).get('calls', 0):
            add_event(connection, run_id, 'skill_load', loaded, f"skill:{loaded['sequence']}")
    previous_resources = {event['sequence'] for event in previous.get('skills', {}).get('resource_events', [])}
    for resource in state.get('skills', {}).get('resource_events', []):
        if resource['sequence'] not in previous_resources:
            add_event(connection, run_id, 'skill_resource', resource, f"skill-resource:{resource['sequence']}")
    for event in state.get('recovery', {}).get('events', []):
        if event['sequence'] > previous.get('recovery', {}).get('sequence', 0):
            add_event(connection, run_id, 'recovery', event, f"recovery:{event['sequence']}")
    for discovered in state.get('tool_selection', {}).get('events', []):
        if discovered['sequence'] > previous.get('tool_selection', {}).get('calls', 0):
            add_event(connection, run_id, 'tool_discovery', discovered, f"discovery:{discovered['sequence']}")
    for observation in context.get('budget_observations', []):
        if observation['sequence'] > previous.get('context', {}).get('budget_sequence', 0):
            add_event(connection, run_id, 'context_budget', observation, f"budget:{observation['sequence']}")
    if round_number > previous.get('rounds', 0):
        add_event(connection, run_id, 'model_request', {'round': round_number, 'model': state.get('model'), 'context': context.get('info', {})}, f'model:{round_number}')
    response = state.get('last_model_result')
    if response:
        if state.get('verification', {}).get('enabled'):
            response = {**response, 'tool_calls': [{**call, 'arguments': public_plan(call.get('arguments'))} if call.get('name') == 'update_plan' else call for call in response.get('tool_calls', [])]}
        add_event(connection, run_id, 'model_response', response, f"response:{response['round']}")
    verification = state.get('verification', {})
    old_tasks = {task['id']: task for task in previous.get('delegation', {}).get('tasks', [])}
    for task in state.get('delegation', {}).get('tasks', []):
        public = {key: task[key] for key in ('id', 'agent', 'status', 'rounds')}
        add_event(connection, run_id, 'delegated_task', {**public, 'task': task['task']}, f"delegated:{task['id']}")
        for number in range(old_tasks.get(task['id'], {}).get('rounds', 0) + 1, task['rounds'] + 1):
            add_event(connection, run_id, 'specialist_request', {'task_id': task['id'], 'agent': task['agent'], 'number': number}, f"specialist-request:{task['id']}:{number}")
        if task['status'] in ('completed', 'needs_input', 'failed'):
            add_event(connection, run_id, 'specialist_done', public, f"specialist-done:{task['id']}")
    previous_verification = previous.get('verification', {})
    for number in range(previous_verification.get('checks', 0) + 1, verification.get('checks', 0) + 1):
        add_event(connection, run_id, 'verification_request', {'number': number}, f'verification-request:{number}')
    for report in verification.get('reports', []):
        add_event(connection, run_id, 'verification_result', public_review(report), f"verification-result:{report['number']}")
    for number in range(previous_verification.get('corrections', 0) + 1, verification.get('corrections', 0) + 1):
        add_event(connection, run_id, 'result_correction', {'number': number}, f'result-correction:{number}')
    for count in range(previous.get('context', {}).get('summary_calls', 0) + 1, context.get('summary_calls', 0) + 1):
        add_event(connection, run_id, 'summary_request', {'number': count}, f'summary:{count}')
    summary_usage = context.get('summary_usage', {})
    if summary_usage.get('reported_rounds', 0) > previous.get('context', {}).get('summary_usage', {}).get('reported_rounds', 0):
        add_event(connection, run_id, 'summary_result', {'usage': summary_usage, 'covered_messages': context.get('memory', {}).get('covered_messages', 0)},
                  f"summary-result:{summary_usage['reported_rounds']}")
    plan = state.get('plan', {}).get('snapshot')
    if plan and 'revision' in plan:
        add_event(connection, run_id, 'plan', public_plan(plan, verification.get('enabled', False)), f"plan:{plan['revision']}")
    active = state.get('active', [])
    calls = {call['id']: call for message in active for call in message.get('tool_calls', [])}
    results = {m['tool_call_id']: json.loads(m['content']) for m in active if m.get('role') == 'tool'}
    for task in state.get('delegation', {}).get('tasks', []):
        calls.update({call['id']: call for message in task.get('messages', []) for call in message.get('tool_calls', [])})
        results.update({outcome['call_id']: outcome['result'] for outcome in task.get('tool_results', [])})
    old_tools = len(previous.get('runtime', {}).get('trace', []))
    for index, tool in enumerate(state.get('runtime', {}).get('trace', [])):
        if index < old_tools:
            continue
        result = results.get(tool['call_id'], {})
        call = calls.get(tool['call_id'], {})
        arguments = proposed_arguments(call.get('function', {}).get('arguments', '{}'))
        arguments = result.get('_human_decision', {}).get('arguments', arguments)
        if tool['name'] == 'update_plan':
            arguments = public_plan(arguments, verification.get('enabled', False))
        add_event(connection, run_id, 'tool_result', {**tool, 'arguments': arguments, 'result': tool_result_summary(result)}, f'tool:{index}')
    approvals = (state.get('pending_batch') or {}).get('approvals', [])
    for approval in approvals:
        add_event(connection, run_id, 'approval_requested', {key: approval[key] for key in ('approval_id', 'call_id', 'name', 'original_arguments')},
                  'approval:' + approval['approval_id'])
    status = 'awaiting_approval' if any(a['decision'] is None for a in approvals) else 'running'
    connection.execute('UPDATE agent_runs SET metrics_json=?,status=?,updated_at=? WHERE session_id=? AND run_id=?',
                       (json.dumps(trace_metrics(state)), status, now(), session_id, run_id))


class TraceStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=2, autocommit=True)
        connection.execute('PRAGMA foreign_keys=ON')
        connection.autocommit = False
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _run(row):
        data = dict(row)
        data['metrics'] = json.loads(data.pop('metrics_json'))
        return data

    def list_runs(self, session_id, before=None):
        with closing(self._connect()) as connection, connection:
            rows = connection.execute('''SELECT rowid AS cursor,* FROM agent_runs WHERE session_id=?
                AND (? IS NULL OR rowid<?) ORDER BY rowid DESC LIMIT 21''', (session_id, before, before)).fetchall()
            return {'runs': [self._run(row) for row in rows[:20]], 'next_cursor': rows[19]['cursor'] if len(rows) > 20 else None}

    def load(self, session_id, run_id):
        with closing(self._connect()) as connection, connection:
            row = connection.execute('SELECT * FROM agent_runs WHERE session_id=? AND run_id=?', (session_id, run_id)).fetchone()
            if row is None:
                return None
            events = connection.execute('SELECT event_id,kind,created_at,data_json FROM agent_run_events WHERE run_id=? ORDER BY event_id', (run_id,)).fetchall()
            return {**self._run(row), 'events': [{'id': e['event_id'], 'kind': e['kind'], 'created_at': e['created_at'], 'data': json.loads(e['data_json'])} for e in events]}

    def stop_running(self, session_id, run_id, status, message):
        with closing(self._connect()) as connection, connection:
            # 占写锁后检查，避免迟到的清理覆盖已完成/已批准状态。
            changed = connection.execute("UPDATE agent_runs SET status=status WHERE session_id=? AND run_id=? AND status='running'", (session_id, run_id))
            if changed.rowcount:
                finish_trace(connection, session_id, run_id, status, {'message': message})

    def recover_running(self):
        with closing(self._connect()) as connection, connection:
            for row in connection.execute("SELECT session_id,run_id FROM agent_runs WHERE status='running'").fetchall():
                finish_trace(connection, row['session_id'], row['run_id'], 'interrupted', {'message': '服务已重启，可检查保存的进度后继续。'})
