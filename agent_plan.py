"""可验证的任务清单：模型维护步骤，服务器校验依赖、证据和完成状态。"""
from copy import deepcopy
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MAX_PLANNED_ROUNDS = 10
PLAN_CONTROL_TOOLS = frozenset({'update_plan', 'discover_tools', 'load_skill', 'read_skill_resource'})
PLAN_INSTRUCTIONS = (
    '仅有依赖、需跟踪的任务才单独 update_plan；订单与政策等独立查询直接处理。'
    'steps.tools 仅填业务枚举；发现工具、加载技能不作步骤或证据。'
    '只计划待做的工具动作；汇总、判断、询问、答复不列步骤，不为这些动作重复查询或追认历史完成。'
    '提交完整计划：一步 in_progress，其余 pending，已完成保留 completed；业务调用仅限当前步骤，独立查询可同一步。'
    '仅 completed 的 evidence 填真实成功结果的 _tool_call_id，须覆盖该步全部工具；pending/in_progress/blocked 必须为 []。'
    '修订需顶层 note，不删步骤、不改已完成项；全部 completed 才完成。'
    '缺信息且无计划时直接追问；已有计划无法继续则 blocked，可同时完成前一步，note 说明待补信息，再直接答复，不再调用。预览不等于提交。'
)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,24}$')
    title: str = Field(min_length=1, max_length=60)
    tools: list[str] = Field(min_length=1, max_length=3)
    status: Literal['pending', 'in_progress', 'completed', 'blocked']
    depends_on: list[str] = Field(default_factory=list, max_length=3)
    evidence: list[str] = Field(default_factory=list, max_length=6)


class PlanUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    steps: list[PlanStep] = Field(min_length=1, max_length=4)
    note: str = Field(default='', max_length=160)


PLAN_TOOL = {'type': 'function', 'function': {
    'name': 'update_plan', 'description': '创建或更新当前回合的执行计划；仅记录进度，不执行业务操作。',
    'parameters': {'type': 'object', 'properties': {
        'steps': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {
            'type': 'object', 'properties': {
                'id': {'type': 'string', 'pattern': r'^[a-zA-Z0-9_-]{1,24}$', 'description': '稳定的步骤ID，如 order。'},
                'title': {'type': 'string', 'minLength': 1, 'maxLength': 60},
                'tools': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {'type': 'string'}, 'description': '业务能力名（含委派）；不填 discover_tools/load_skill/update_plan。'},
                'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed', 'blocked']},
                'depends_on': {'type': 'array', 'maxItems': 3, 'items': {'type': 'string'}, 'description': '依赖的前面步骤ID。'},
                'evidence': {'type': 'array', 'maxItems': 6, 'items': {'type': 'string'}, 'description': '仅 completed 填真实成功的 _tool_call_id；pending/in_progress/blocked 必须为 []。'},
            }, 'required': ['id', 'title', 'tools', 'status', 'depends_on', 'evidence'], 'additionalProperties': False}},
        'note': {'type': 'string', 'maxLength': 160, 'description': '修订或阻塞的简短说明。'},
    }, 'required': ['steps', 'note'], 'additionalProperties': False},
}}


def plan_tool(available_tools):
    """Publish the same per-run capability boundary that validates plans."""
    tool = deepcopy(PLAN_TOOL)
    tools = tool['function']['parameters']['properties']['steps']['items']['properties']['tools']
    tools['items']['enum'] = sorted(set(available_tools) - PLAN_CONTROL_TOOLS)
    return tool


class AgentPlan:
    def __init__(self, available_tools):
        self.available_tools = set(available_tools) - PLAN_CONTROL_TOOLS
        self.steps = []
        self.note = ''
        self.revision = 0
        self.observations = {}

    @property
    def current(self):
        return next((step for step in self.steps if step['status'] == 'in_progress'), None)

    @property
    def status(self):
        if not self.steps:
            return 'none'
        if all(step['status'] == 'completed' for step in self.steps):
            return 'completed'
        return 'in_progress' if self.current else 'blocked'

    def snapshot(self):
        return {'revision': self.revision, 'status': self.status, 'steps': deepcopy(self.steps), 'note': self.note} if self.steps else None

    def can_call(self, name):
        return not self.steps or bool(self.current and name in self.current['tools'])

    def observe(self, outcome):
        step = self.current
        if step and outcome['name'] in step['tools'] and outcome['attempts']:
            self.observations.setdefault(step['id'], {})[outcome['call_id']] = {
                'name': outcome['name'], 'ok': 'error' not in outcome['result']}

    def update(self, arguments):
        try:
            return self._update(arguments)
        except ValidationError as error:
            first = error.errors(include_input=False)[0]
            field = '.'.join(str(part) for part in first['loc'])
            hint = ' 每步 tools 至少一个业务工具；不要单列询问步骤，缺信息时将相关工具步骤设 blocked 并在 note 提问。' if field.endswith('.tools') else ''
            return self._validation_error(f"计划格式无效：{field}: {first['msg']}{hint}")
        except (ValueError, TypeError) as error:
            return self._validation_error(str(error))

    def _validation_error(self, message, *, repair_hint=None):
        evidence = {step['id']: {call_id: item['name'] for call_id, item in self.observations.get(step['id'], {}).items()
                                 if item['ok']} for step in self.steps}
        result = {'error': message, 'error_code': 'invalid_plan_update', 'current_plan': self.snapshot(),
                  'available_evidence': {step: items for step, items in evidence.items() if items}}
        if repair_hint:
            result['repair_hint'] = repair_hint
        return result

    def _update(self, arguments):
        update = PlanUpdate.model_validate_json(arguments)
        candidate = [step.model_dump() for step in update.steps]
        encoded = json.dumps(candidate, ensure_ascii=False) + update.note
        encoded.encode('utf-8')
        if '\\u0000' in encoded or '\x00' in update.note:
            raise ValueError('计划不能包含空字符。')
        terminal_block = self.status == 'in_progress' and any(step['status'] == 'blocked' for step in candidate)
        if (self.revision >= 6 and not terminal_block) or self.status == 'completed':
            raise ValueError('计划已完成或达到更新上限，请根据已保存结果回复。')
        ids = [step['id'] for step in candidate]
        if len(set(ids)) != len(ids):
            raise ValueError('步骤ID不能重复。')
        old = {step['id']: step for step in self.steps}
        new = {step['id']: step for step in candidate}
        if set(old) - set(new):
            missing = ', '.join(sorted(set(old) - set(new)))
            raise ValueError(f'不能删除或重命名原有步骤 {missing}；保留 current_plan 中的 id，无法继续时将对应步骤 blocked 并说明原因。')
        changed = set()
        for index, step in enumerate(candidate):
            invalid = sorted(set(step['tools']) - self.available_tools)
            if invalid:
                return {'error': f"步骤 {step['id']} 的 tools 不允许 {', '.join(invalid)}。"
                        'discover_tools/load_skill/update_plan 只可单独调用，不列业务步骤或证据。'
                        '移除准备步骤并修正 depends_on；未接受的计划无需保留，无实际成功证据时 evidence 留空。',
                        'error_code': 'invalid_plan_tools', 'invalid_tools': invalid,
                        'allowed_tools': sorted(self.available_tools)}
            if len(set(step['tools'])) != len(step['tools']):
                raise ValueError('同一步的业务工具名不能重复。')
            if set(step['depends_on']) - set(ids[:index]) or len(set(step['depends_on'])) != len(step['depends_on']):
                raise ValueError('依赖只能指向前面的不同步骤。')
            if step['status'] in ('in_progress', 'completed') and any(new[dependency]['status'] != 'completed' for dependency in step['depends_on']):
                raise ValueError('依赖步骤尚未完成。')
            previous = old.get(step['id'])
            modified = previous and any(previous[key] != step[key] for key in ('title', 'tools', 'depends_on'))
            if previous and previous['status'] == 'completed':
                if previous != step:
                    raise ValueError('不能改写已经完成的步骤和证据。')
                continue
            if modified or (self.steps and previous is None):
                changed.add(step['id'])
            if step['status'] == 'completed':
                observed = self.observations.get(step['id'], {})
                if not previous:
                    raise ValueError('新计划只列尚需执行的工具步骤，不能填 completed 或引用旧凭据；已查结果直接使用，不补建已完成步骤。')
                if not previous or previous['status'] != 'in_progress' or modified:
                    raise ValueError('只能完成已实际执行的步骤；修订后需要重新执行。')
                if not step['evidence'] or any(not observed.get(call_id, {}).get('ok') for call_id in step['evidence']):
                    raise ValueError('完成步骤必须引用该步骤真实成功的工具调用ID，失败结果不能作为完成证据。')
                if set(step['tools']) - {observed[call_id]['name'] for call_id in step['evidence']}:
                    raise ValueError('完成证据没有覆盖该步骤声明的全部工具。')
            elif step['evidence']:
                return self._validation_error(
                    '尚未完成的步骤应将 evidence 留空。',
                    repair_hint=f"步骤 {step['id']} 为 {step['status']}，将 steps[{index}].evidence 改为 []；"
                    '阻塞原因写在顶层 note，保留已完成步骤的真实证据，再单独提交完整 update_plan。')
            if step['status'] == 'blocked':
                next_ready = previous and previous['status'] == 'pending' and all(
                    earlier['status'] == 'completed' for earlier in candidate[:index])
                if not previous or (previous['status'] not in ('in_progress', 'blocked') and not next_ready):
                    raise ValueError('只能阻塞当前步骤或前面步骤均已完成的下一步。')
        reordered = [step['id'] for step in self.steps] != [step_id for step_id in ids if step_id in old]
        suspended = any((previous['status'] == 'in_progress' and new[step_id]['status'] == 'pending')
                        or (previous['status'] == 'blocked' and new[step_id]['status'] != 'blocked')
                        for step_id, previous in old.items())
        if (changed or reordered or suspended) and not update.note:
            raise ValueError('修订未完成步骤时必须填写 note。')
        active = sum(step['status'] == 'in_progress' for step in candidate)
        blocked = any(step['status'] == 'blocked' for step in candidate)
        if active > 1 or (active == 0 and not blocked and any(step['status'] != 'completed' for step in candidate)):
            raise ValueError('计划应有且仅有一个执行中步骤，或已全部完成、明确阻塞。')
        if blocked and (active or not update.note):
            raise ValueError('阻塞计划必须暂停执行，并填写需要补充或确认的具体事项。')
        if candidate == self.steps and update.note == self.note:
            raise ValueError('计划没有变化，请先执行当前步骤或说明阻塞原因。')
        # 全部校验完成后才更新临时状态，错误不能留下半份计划。
        for step_id in changed:
            self.observations.pop(step_id, None)
        self.steps, self.note = candidate, update.note
        self.revision += 1
        return {'plan': self.snapshot()}
