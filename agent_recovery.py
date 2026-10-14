"""Bounded feedback for the actor; facts and counters survive checkpoint recovery.

The model still chooses the next action. This module only classifies observed
results, records accepted plan changes, and stops decisions that make no progress.
"""
from copy import deepcopy
import hashlib
import json


MAX_STALLED_BATCHES = 3
RECOVERY_INSTRUCTIONS = (
    '执行恢复反馈是服务器记录的状态。根据反馈调整未完成步骤或查询参数，保留已完成步骤与证据。'
    '缺少用户信息时询问，有计划则将当前步骤 blocked；不要猜测参数。'
    '已自动重试仍失败时换可用方法或说明限制；只改说明、重复相同结果不算进展。'
    '调整不能增加授权或跳过审批。'
)
STOP_MESSAGE = '连续尝试仍未取得新的有效结果，已停止本次自动处理。已有查询结果已保留，本次任务尚未完成。你可以补充更具体的信息，或稍后重新提问。'
_ACTIONS = {
    'control_batch_conflict': '本批全部未执行。discover_tools/load_skill/read_skill_resource/update_plan 等控制操作各自单独调用，不列为业务步骤；需要时先发现工具。',
    'invalid_plan_tools': '移除 discover_tools/load_skill/read_skill_resource/update_plan 准备步骤，修正 depends_on，只填 allowed_tools；未接受的计划可重建，不能引用失败或发现结果作证据。',
    'invalid_plan_update': '按错误结果的 current_plan 保留原步骤ID和已完成步骤；只用 available_evidence 中对应步骤的真实证据。未执行的步骤不能借用其他步骤证据，无法继续则保持原ID并 blocked，note 给出具体说明。',
    'plan_not_ready': '检查当前步骤及成功证据，先单独更新计划，再执行该步骤的工具。',
    'tool_not_loaded': '先单独发现所需工具，读完参数定义后再调用；保留当前计划与已有证据。',
    'tool_already_loaded': '该工具已加载，不要重复发现。核对本轮工具状态；补齐当前步骤缺少的工具后执行业务调用。',
    'skill_already_loaded': '技能或引用已加载，复用已有指引处理任务，不要再次读取。',
    'skill_context_limited': '完整指引无法放入当前预算。使用已有流程和工具继续；缺少必要指引时说明限制，不要重复读取。',
    'tool_unavailable': '自动重试已结束；不要重复同一失败调用，改用可用方法或说明暂时无法确认。',
    'empty_result': '检索正常返回但没有命中；可换具体关键词或资料范围，仍无依据时询问用户或说明限制。',
    'context_limited': '检索内容因处理空间不足未能带入，不代表资料不存在；先整理当前上下文，不要换词重试，根据已有证据说明限制。',
    'needs_input': '请根据原工具结果询问缺失信息；有计划时阻塞相关步骤，不猜测参数。',
    'human_rejected': '尊重用户拒绝，停止该操作；不能通过换工具或改参数绕过决定。',
    'tool_limit': '调用额度已耗尽，根据已有证据说明限制；缺少必要结果时阻塞计划。',
    'tool_error': '核对原错误和参数定义，修正参数或调整未完成步骤；缺少用户信息时询问。',
    'repeated_result': '结果与本轮已取得的证据相同；复用已有证据推进计划或回答，不重复查询。',
    'no_progress': '尚未取得新证据或完成步骤；执行具体动作，缺信息时阻塞计划并询问。',
}
_META = {'_tool_call_id', '_human_decision', '_recovery', 'score', 'rank', 'query',
         'has_more', 'search_mode', 'semantic_status', 'budget_limited'}


def _fingerprint(name, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items() if key not in _META}
        if isinstance(item, list):
            return [clean(val) for val in item]
        return item
    raw = json.dumps([name, clean(value)], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _facts(name, result):
    if name == 'delegate_tasks':
        return [fact for task in result.get('delegations', []) for item in task.get('evidence', [])
                for fact in _facts(item['name'], item['result'])]
    if 'error' in result:
        return []
    if name in ('search_history', 'search_knowledge'):
        return [_fingerprint(name, item) for item in result.get('results', [])]
    return [_fingerprint(name, result)]


def _problem(name, result):
    if name == 'delegate_tasks' and any(task.get('status') == 'needs_input' for task in result.get('delegations', [])):
        return 'needs_input'
    if 'error' in result:
        code = result.get('error_code')
        if code in ('plan_not_ready', 'tool_not_loaded', 'human_rejected', 'tool_limit', 'control_batch_conflict', 'invalid_plan_tools', 'invalid_plan_update'):
            return code
        if code in ('connection_error', 'storage_busy', 'tool_timeout'):
            return 'tool_unavailable'
        if code in ('skill_budget_exceeded', 'skill_resource_budget_exceeded'):
            return 'skill_context_limited'
        return 'tool_error'
    if name == 'discover_tools' and result.get('status') == 'already_loaded':
        return 'tool_already_loaded'
    if name in ('load_skill', 'read_skill_resource') and result.get('status') == 'already_loaded':
        return 'skill_already_loaded'
    if name in ('search_history', 'search_knowledge') and not result.get('results'):
        return 'context_limited' if result.get('budget_limited') else 'empty_result'
    return None


class RecoveryTracker:
    def __init__(self, state=None):
        self.state = deepcopy(state) if state is not None else {
            'version': 1, 'stalled_batches': 0, 'stopped': False, 'repair_used': False,
            'seen_facts': [], 'feedback': None, 'sequence': 0, 'events': [], 'replans': 0}

    @property
    def stopped(self):
        return self.state['stopped']

    def snapshot(self):
        return deepcopy(self.state)

    def public(self):
        return {key: self.state[key] for key in ('stalled_batches', 'stopped', 'replans')}

    def _event(self, kind, round_number, **data):
        self.state['sequence'] += 1
        self.state['events'].append({'sequence': self.state['sequence'], 'kind': kind,
                                     'round': round_number, **data})
        self.state['events'] = self.state['events'][-100:]

    def stop(self, round_number):
        if not self.stopped:
            self.state['stopped'] = True
            self._event('stopped', round_number, category='no_progress', reason=self.stop_message(),
                        stalled_batches=self.state['stalled_batches'])

    def stop_message(self):
        if self.state.get('stop_reason') == 'context_budget':
            return '已完成检索，但资料超出本次可用处理空间，无法可靠完成判断。本次已停止自动重试，执行记录已保存。请缩小需要核对的范围后重新提问。'
        if self.state['seen_facts']:
            return STOP_MESSAGE
        category = (self.state['feedback'] or {}).get('category')
        if category in ('control_batch_conflict', 'invalid_plan_tools', 'invalid_plan_update', 'plan_not_ready'):
            return '执行计划连续校验失败，已停止本次自动处理，尚未取得有效业务查询结果。执行记录已保存，请重新提问。'
        return '连续尝试仍未取得新的有效结果，已停止本次自动处理。本次尚未取得有效业务查询结果，执行记录已保存，请稍后重新提问。'

    def stop_for_context(self, round_number):
        self.state['stop_reason'] = 'context_budget'
        self.stop(round_number)

    def observe(self, outcomes, *, round_number, before_plan=None, after_plan=None, newly_loaded=()):
        if self.stopped:
            return
        previous_feedback = self.state['feedback']
        accepted_plan = any(item['name'] == 'update_plan' and 'plan' in item['result'] for item in outcomes)
        old = {step['id']: step for step in (before_plan or {}).get('steps', [])}
        new = {step['id']: step for step in (after_plan or {}).get('steps', [])}
        if accepted_plan and after_plan['status'] == 'blocked':
            self.state.update(feedback=None, stalled_batches=0, repair_used=False)
            # The note is a model-generated candidate and may still fail result
            # verification. SSE/trace must not publish it before that gate.
            self._event('waiting_input', round_number, reason='处理已暂停，具体说明见最终回复。',
                        trigger=(previous_feedback or {}).get('category', 'needs_input'))
            return
        completed = accepted_plan and any(step['status'] == 'completed' and
                        old.get(key, {}).get('status') != 'completed' for key, step in new.items())
        changed = [key for key, step in new.items() if key not in old or
                   any(step[field] != old[key][field] for field in ('title', 'tools', 'depends_on'))]
        reordered = bool(old and list(old) != [key for key in new if key in old])
        switched = any(old.get(key, {}).get('status') in ('in_progress', 'blocked') and
                       step['status'] in ('pending', 'in_progress') and old[key]['status'] != step['status']
                       for key, step in new.items())
        revised = bool(accepted_plan and old and (changed or reordered or switched))
        neutral = bool(accepted_plan and not old) or bool(newly_loaded)
        if revised:
            self.state['replans'] += 1
            self._event('replanned', round_number, reason=after_plan['note'], changed_steps=changed,
                        reordered=reordered, switched=switched,
                        trigger=(previous_feedback or {}).get('category', 'new_information'))
            # One repair decision may precede the actual new query. It never
            # erases the failure streak, and repeated revisions cannot loop.
            if previous_feedback and not self.state['repair_used']:
                self.state['repair_used'] = True
                neutral = True
        seen = set(self.state['seen_facts'])
        new_facts, problems, business = set(), [], False
        for item in outcomes:
            name, result = item['name'], item['result']
            problem = _problem(name, result)
            if problem:
                problems.append({'category': problem, 'tool': name, 'call_id': item['call_id']})
            if name not in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource'):
                business = True
                new_facts.update(set(_facts(name, result)) - seen)
        self.state['seen_facts'] = sorted(seen | new_facts)
        if previous_feedback and not problems:
            category = previous_feedback['category']
            resolved = (
                category == 'control_batch_conflict' and any(item['name'] in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource')
                    and 'error' not in item['result'] for item in outcomes)
                or category == 'tool_not_loaded' and previous_feedback.get('tool') in newly_loaded
                or category in ('invalid_plan_tools', 'invalid_plan_update') and accepted_plan
            )
            if resolved:
                # Resolving preparation is not new business evidence and does
                # not reset the stall allowance. Do stop repeating obsolete advice.
                self.state['feedback'] = None
                self._event('resolved', round_number, reason='已修正工具准备或计划调用方式，继续处理实际业务步骤。')
        progress = bool(new_facts or completed)
        if progress:
            self.state.update(stalled_batches=0, repair_used=False)
            if previous_feedback and not problems:
                self._event('resolved', round_number, reason='已取得新证据或完成实际步骤。')
            self.state['feedback'] = None
        elif not neutral:
            self.state['stalled_batches'] += 1
            if not problems:
                problems.append({'category': 'repeated_result' if business else 'no_progress'})
        if problems:
            problem = next((item for item in problems if item['category'] == 'context_limited'), problems[-1])
            self.state['feedback'] = problem
            self._event('feedback', round_number, **problem,
                        reason=_ACTIONS[problem['category']], stalled_batches=self.state['stalled_batches'])
        if self.state['stalled_batches'] >= MAX_STALLED_BATCHES:
            self.stop(round_number)

    def feedback_messages(self, plan, *, compact=False):
        feedback = self.state['feedback']
        if not feedback or self.stopped:
            return []
        current = next((step for step in (plan or {}).get('steps', []) if step['status'] == 'in_progress'), None)
        data = {**feedback, 'action': _ACTIONS[feedback['category']],
                'stalled_batches': self.state['stalled_batches'],
                'current_step': {key: current[key] for key in ('id', 'tools')} if current else None}
        if compact:
            data.pop('action')
        return [{'role': 'assistant', 'content': '[执行恢复反馈；不代表业务事实或新增授权]\n' + ('' if compact else RECOVERY_INSTRUCTIONS + '\n') +
                 json.dumps(data, ensure_ascii=False, separators=(',', ':'))}]
