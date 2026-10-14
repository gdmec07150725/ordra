"""服务器控制的工具审批策略与可持久化决定，不执行任何业务。"""
from copy import deepcopy
from typing import Literal
from uuid import uuid4
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator
from return_tools import validate_return_arguments


APPROVAL_POLICIES = {
    'prepare_return_request': {
        'title': '生成退货预览',
        'description': '核对或修改下面的信息，批准后生成本地预览。申请提交仍需在预览卡中确认。',
        'fields': [{'name': 'order_id', 'label': '订单号', 'max_length': 50},
                   {'name': 'reason', 'label': '退货原因', 'max_length': 500}],
        'validate': lambda args: validate_return_arguments('prepare_return_request', args),
    },
}
APPROVAL_INSTRUCTIONS = 'prepare_return_request 执行前网页会请用户批准、修改或拒绝。拒绝后本轮停止该操作，不换参数重试；批准仅生成预览。'


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    run_id: str = Field(min_length=1, max_length=64)
    approval_id: str = Field(min_length=1, max_length=64)
    decision: Literal['approve', 'edit', 'reject']
    arguments: dict | None = None
    reason: str = Field(default='', max_length=500)

    @model_validator(mode='after')
    def check_arguments(self):
        if (self.decision == 'edit') != (self.arguments is not None):
            raise ValueError('仅修改并批准时需要提交完整参数。')
        return self


def make_approval(call):
    name = call['function']['name']
    policy = APPROVAL_POLICIES[name]
    try:
        arguments = policy['validate'](json.loads(call['function']['arguments']))
    except (ValueError, TypeError):
        return None  # 非法模型参数由现有执行器返回校验错误，不请用户批准非法请求。
    return {'approval_id': str(uuid4()), 'call_id': call['id'], 'name': name,
            'original_arguments': deepcopy(arguments), 'arguments': deepcopy(arguments),
            'decision': None, 'reason': ''}


def pending_approvals(state):
    return (state.get('pending_batch') or {}).get('approvals', [])


def public_approvals(state):
    return [{**deepcopy(item), **{key: value for key, value in APPROVAL_POLICIES[item['name']].items() if key != 'validate'}}
            for item in pending_approvals(state)]


def record_decision(state, body):
    """在存储事务内调用；仅从服务器批次中找到的审批可被决定。"""
    item = next((a for a in pending_approvals(state) if a['approval_id'] == body.approval_id), None)
    if item is None or item['decision'] is not None:
        raise LookupError('审批已处理或已失效，请刷新查看最新状态。')
    arguments = item['arguments']
    if body.decision == 'edit':
        arguments = APPROVAL_POLICIES[item['name']]['validate'](body.arguments)
    # 也验证编码，避免不可存储字符或非有限数穿过数据库边界。
    json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode('utf-8')
    item.update(decision=body.decision, arguments=deepcopy(arguments), reason=body.reason)
    if all(a['decision'] is not None for a in pending_approvals(state)):
        state['decision_ready'] = True
