"""仅序列化可恢复执行状态；不保存SDK连接、半截模型输出或数据库句柄。"""
from copy import deepcopy
from dataclasses import asdict, replace

from context_memory import ContextMemory, ContextPolicy
from tool_runtime import ToolPolicy


def pack_checkpoint(plan, runtime, context, **state):
    return deepcopy({**state, 'version': 1,
        'plan': {'snapshot': plan.snapshot(), 'observations': plan.observations, 'available_tools': sorted(plan.available_tools)},
        'runtime': {'calls': runtime.calls, 'trace': runtime.trace, 'peak_parallel': runtime.peak_parallel,
                    'failures': [[*key, value] for key, value in runtime.failures.items()], 'policy': asdict(runtime.policy)},
        'context': {'memory': asdict(context.memory), 'policy': asdict(context.policy), 'info': context.info,
                    'summary_calls': context.summary_calls, 'summary_usage': context.summary_usage,
                    'pending_summary': context.pending_summary,
                    'budget_observations': context.budget_observations, 'budget_sequence': context.budget_sequence}})


def restore_checkpoint(checkpoint, question, plan, runtime, context, *, allow_budget_increase=False):
    state = deepcopy(checkpoint)
    if state.get('version') != 1 or state.get('active', [{}])[0] != {'role': 'user', 'content': question}:
        raise ValueError('执行检查点版本或问题不匹配，请放弃后重新提问。')
    if state['plan']['available_tools'] != sorted(plan.available_tools):
        raise ValueError('可用工具已变化，请放弃旧执行后重新提问。')
    saved_plan = state['plan']['snapshot']
    if saved_plan:
        plan.steps, plan.note, plan.revision = saved_plan['steps'], saved_plan['note'], saved_plan['revision']
    plan.observations = state['plan']['observations']
    execution = state['runtime']
    runtime.calls, runtime.trace, runtime.peak_parallel = execution['calls'], execution['trace'], execution['peak_parallel']
    runtime.failures = {(name, arguments): result for name, arguments, result in execution['failures']}
    runtime.policy = ToolPolicy(**execution['policy'])
    saved_context = state['context']
    # A user raising the application budget must also unblock saved runs.
    # All execution/approval counters and summary settings stay pinned.
    configured_max = context.policy.max_tokens
    context.memory, context.policy = ContextMemory(**saved_context['memory']), ContextPolicy(**saved_context['policy'])
    if allow_budget_increase and configured_max > context.policy.max_tokens:
        context.policy = replace(context.policy, max_tokens=configured_max)
    context.info = saved_context['info']
    context.summary_calls, context.summary_usage = saved_context['summary_calls'], saved_context['summary_usage']
    context.pending_summary = saved_context.get('pending_summary')  # 兼容升级前的版本 1 检查点。
    context.budget_observations = saved_context.get('budget_observations', [])
    context.budget_sequence = saved_context.get('budget_sequence', 0)
    state['knowledge_snapshots'] = {int(index): result for index, result in state['knowledge_snapshots'].items()}
    return state
