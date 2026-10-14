"""Formal chat orchestration: distinct persisted LangGraph phases.

Business snapshots remain version 1. The graph owns phase routing and interrupts;
request-local closures own SDK handles and business helpers, which never enter a saver.
"""
import asyncio
from uuid import uuid4
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
import agent_service as service
import json
from copy import deepcopy
from collections.abc import Awaitable, Callable
from time import monotonic
import anyio
from openai import AsyncOpenAI
from context_memory import ContextMemory, ContextPolicy, RollingContext, empty_usage, add_usage, estimate_request
from context_observation import observe_request, request_observer
from history_search import HISTORY_INSTRUCTIONS, HISTORY_TOOL
from knowledge_base import KNOWLEDGE_INSTRUCTIONS, KNOWLEDGE_TOOL
from return_tools import RETURN_INSTRUCTIONS, RETURN_TOOLS
from tool_runtime import ToolJob, ToolPolicy, ToolRuntime
from agent_plan import AgentPlan, plan_tool, PLAN_INSTRUCTIONS, MAX_PLANNED_ROUNDS
from human_approval import APPROVAL_INSTRUCTIONS, make_approval
from result_verification import ReviewBudgetError, public_review, public_plan
from subagent_runtime import (DELEGATE_TOOL, DELEGATION_INSTRUCTIONS, new_delegation_state,
                              register_tasks, run_specialists)
from trace_store import proposed_arguments
from agent_service import (MAX_MODEL_ROUNDS, MAX_HISTORY_SEARCHES, MAX_RESULT_CHECKS,
    MAX_RESULT_CORRECTIONS, UNVERIFIED_ANSWER, RESULT_CHECK_INSTRUCTIONS,
    TOOL_EXECUTION_INSTRUCTIONS, TOOLS, HistoryReader, KnowledgeReader, ReturnHandler,
    history_arguments, knowledge_arguments, execute_history_search, execute_knowledge_search,
    plan_context, model_context, fit_history_results)
from execution_state import pack_checkpoint, restore_checkpoint
from tool_discovery import ToolDiscovery, DISCOVERY_TOOL
from agent_recovery import RecoveryTracker
from skill_runtime import SkillRuntime, SKILL_TOOL, public_skill, historical_skill_view
from skill_resources import RESOURCE_TOOL


class ChatState(TypedDict):
    snapshot: dict
    route: str


async def stream_graph(client: AsyncOpenAI, model: str, history: list, question: str, *, policy: ContextPolicy | None = None, memory: ContextMemory | None = None, search_history: HistoryReader | None = None, search_knowledge: KnowledgeReader | None = None, return_handler: ReturnHandler | None = None, tool_policy: ToolPolicy | None = None, checkpoint: dict | None = None, save_checkpoint: Callable[[dict], Awaitable[None]] | None = None, approval_tools: frozenset[str] = frozenset(), result_verifier: Callable[..., Awaitable[dict]] | None = None, delegation_enabled: bool = False, graph_checkpointer=None, graph_run_id=None, mcp_client=None, skills_dir=None):
    saver = graph_checkpointer if graph_checkpointer is not None else InMemorySaver()
    config = {'configurable': {'thread_id': graph_run_id or uuid4().hex, 'checkpoint_ns': ''},
              'recursion_limit': 200}
    root = await saver.aget_tuple(config)
    if checkpoint is None and root is not None:
        checkpoint = root.checkpoint['channel_values'].get('snapshot')
    root_interrupted = bool(root and any(channel == '__interrupt__' for _, channel, _ in root.pending_writes))
    next_message = deepcopy(checkpoint.get('graph_message')) if checkpoint else None
    if checkpoint is not None and 'delegate_tasks' not in checkpoint.get('plan', {}).get('available_tools', []):
        delegation_enabled = False
    # Legacy runs retain their original execution path and approval contract.
    if checkpoint is not None and 'tool_selection' not in checkpoint:
        mcp_client = None
    if checkpoint and 'tool_selection' in checkpoint and mcp_client is None:
        raise ValueError('此任务使用 MCP 工具，请在启用 MCP 的服务中继续。')
    if checkpoint is not None and 'skills' not in checkpoint:
        skills_dir = None
    if checkpoint and 'skills' in checkpoint and skills_dir is None:
        raise ValueError('此任务使用技能，请在启用 Skills 的服务中继续。')
    skills = await anyio.to_thread.run_sync(lambda: SkillRuntime(skills_dir,
        state=checkpoint.get('skills') if checkpoint else None)) if skills_dir is not None else None
    remote_tools = {}
    if mcp_client is not None:
        remote_tools = {tool['function']['name']: tool for tool in await mcp_client.list_tools()
                        if tool['function']['name'] in ('get_order', 'search_knowledge')}
        required_remote = {'get_order'} | ({'search_knowledge'} if search_knowledge is not None else set())
        if required_remote - remote_tools.keys():
            raise ValueError('MCP 服务缺少已注册工具，无法开始执行。')
    tools_by_agent = {'orders': TOOLS}
    if search_knowledge is not None:
        tools_by_agent['knowledge'] = [KNOWLEDGE_TOOL]
    tools = TOOLS + ([HISTORY_TOOL] if search_history is not None else []) + ([KNOWLEDGE_TOOL] if search_knowledge is not None else []) + (RETURN_TOOLS if return_handler is not None else []) + ([DELEGATE_TOOL] if delegation_enabled else [])
    plan = AgentPlan(tool['function']['name'] for tool in tools)
    tools = tools + [plan_tool(plan.available_tools)] + ([SKILL_TOOL] if skills is not None else [])
    if skills is not None and skills.resources_enabled:
        tools.append(RESOURCE_TOOL)
    if remote_tools:
        tools = [deepcopy(remote_tools.get(tool['function']['name'], tool)) for tool in tools]
        tools_by_agent = {agent: [deepcopy(remote_tools[t['function']['name']]) for t in items]
                          for agent, items in tools_by_agent.items()}
    selection = ToolDiscovery(tools, remote_names=remote_tools, state=checkpoint.get('tool_selection') if checkpoint else None) if mcp_client is not None else None
    if selection is not None:
        tools = selection.request_tools()
    request_history = historical_skill_view(history)
    request_history[0] = {**history[0], "content": history[0]["content"] + TOOL_EXECUTION_INSTRUCTIONS + PLAN_INSTRUCTIONS}
    if selection is not None:
        request_history[0]['content'] += selection.instructions()
    if skills is not None:
        request_history[0]['content'] += skills.instructions()
    if result_verifier is not None:
        request_history[0]['content'] += RESULT_CHECK_INSTRUCTIONS
    if delegation_enabled:
        request_history[0]['content'] += DELEGATION_INSTRUCTIONS
    if approval_tools:
        if save_checkpoint is None:
            raise ValueError('工具审批需要持久化检查点。')
        request_history[0]['content'] += APPROVAL_INSTRUCTIONS
    if search_history is not None or search_knowledge is not None or return_handler is not None:
        # 请求时补充新工具规则，旧会话立即生效；不改写数据库里的原始 system。
        instructions = (HISTORY_INSTRUCTIONS if search_history is not None else "") + (KNOWLEDGE_INSTRUCTIONS if search_knowledge is not None else "") + (RETURN_INSTRUCTIONS if return_handler is not None else "")
        request_history[0] = {**request_history[0], "content": request_history[0]["content"] + instructions}
    context = RollingContext(request_history, memory or ContextMemory(), policy or ContextPolicy(), tools)
    # 当前回合单独积累。上下文可反复压缩，保存时仍只追加这些原始消息。
    active = [{"role": "user", "content": question}]
    totals = empty_usage()
    retrieval = {"calls": 0, "returned_messages": 0}
    retrieval_results = []
    retrieval_snapshots = {}
    knowledge = {"calls": 0, "returned_chunks": 0}
    knowledge_results = []
    knowledge_snapshots = {}
    return_previews = []
    runtime = ToolRuntime(tool_policy)
    recovery = RecoveryTracker(checkpoint.get('recovery') if checkpoint else None)
    search_issued = 0
    completion_corrections = 0
    staged_plan = None
    staged_selection, staged_discovery_result = None, None
    staged_skills, staged_skill_result = None, None
    resource_attempts = {}
    rounds, answer = 0, None
    pending_batch, rejected_tools = None, set()
    last_model_result = None
    delegation, pending_delegation = new_delegation_state(), None
    save_lock = anyio.Lock()
    verification = {'enabled': result_verifier is not None, 'checks': 0, 'corrections': 0,
                    'status': 'pending' if result_verifier is not None else 'disabled', 'candidate': None,
                    'feedback': [], 'reports': [], 'usage': empty_usage()}
    correction = '上次答复提前结束，但计划尚未完成。请继续执行并更新计划，或明确标记 blocked 后说明原因。'
    if checkpoint is not None:
        restored = restore_checkpoint(checkpoint, question, plan, runtime, context, allow_budget_increase=policy is not None)
        active, totals = restored['active'], restored['totals']
        retrieval, retrieval_results = restored['retrieval'], restored['retrieval_results']
        retrieval_snapshots = {int(index): result for index, result in restored.get('retrieval_snapshots', {}).items()}
        knowledge, knowledge_results = restored['knowledge'], restored['knowledge_results']
        knowledge_snapshots, return_previews = restored['knowledge_snapshots'], restored['return_previews']
        search_issued, completion_corrections = restored['search_issued'], restored['completion_corrections']
        rounds, answer = restored['rounds'], restored['answer']
        pending_batch = restored.get('pending_batch')
        last_model_result = restored.get('last_model_result')
        delegation = restored.get('delegation', delegation)
        pending_delegation = restored.get('pending_delegation')
        rejected_tools = set(restored.get('rejected_tools', []))
        verification = restored.get('verification', verification)
        if result_verifier is not None and not verification['enabled']:
            verification.update(enabled=True, status='pending')
        if verification['enabled'] and result_verifier is None:
            raise ValueError('本次任务需要结果校验，请在启用校验的服务中继续。')
        if completion_corrections:
            context.history[0] = {**context.history[0], 'content': context.history[0]['content'] + correction}

    def snapshot(stable_active=None):
        return pack_checkpoint(plan, runtime, context, active=active if stable_active is None else stable_active, totals=totals,
            retrieval_snapshots=retrieval_snapshots,
            retrieval=retrieval, retrieval_results=retrieval_results, knowledge=knowledge,
            knowledge_results=knowledge_results, knowledge_snapshots=knowledge_snapshots,
            return_previews=return_previews, search_issued=search_issued,
            completion_corrections=completion_corrections, rounds=rounds, answer=answer,
            pending_batch=pending_batch, rejected_tools=sorted(rejected_tools), model=model, last_model_result=last_model_result,
            verification=verification, delegation=delegation, pending_delegation=pending_delegation,
            graph_message=next_message, recovery=recovery.snapshot(),
            **({'skills': skills.snapshot()} if skills is not None else {}),
            **({'tool_selection': selection.snapshot()} if selection is not None else {}))

    async def save_progress(stable_active=None):
        if save_checkpoint is not None:
            async with save_lock:
                await save_checkpoint(snapshot(stable_active))

    async def record_budget(data):
        context.budget_sequence += 1
        context.budget_observations.append({**data, 'sequence': context.budget_sequence})
        # Each entry is flushed immediately; snapshots need only a bounded tail.
        context.budget_observations = context.budget_observations[-100:]
        await save_progress()

    async def emit(event):
        # Acknowledgement is released only when the consumer requests its next event.
        # Closing at round/tool_result cannot race ahead into another model request.
        acknowledged = asyncio.Event()
        get_stream_writer()((event, acknowledged))
        await acknowledged.wait()
        await anyio.lowlevel.checkpoint()

    async def forward(events):
        try:
            async for event in events:
                await emit(event)
        finally:
            await events.aclose()

    def update(route):
        return {'snapshot': snapshot(), 'route': route}

    def evidence_active():
        raw_results = {**retrieval_snapshots, **knowledge_snapshots}
        return [{**message, 'content': json.dumps(raw_results[index], ensure_ascii=False)}
                if index in raw_results else message for index, message in enumerate(active)]

    def complete_event():
        saved_active = evidence_active()
        sources = {(hit['source'], hit['version']): hit for result in knowledge_snapshots.values() for hit in result['results']}
        child_knowledge = [outcome['result'] for task in delegation['tasks'] for outcome in task.get('tool_results', [])
                           if outcome['name'] == 'search_knowledge' and 'error' not in outcome['result']]
        sources.update({(hit['source'], hit['version']): hit for result in child_knowledge for hit in result.get('results', [])})
        all_knowledge = {**knowledge, 'calls': knowledge['calls'] + len(child_knowledge),
                         'returned_chunks': knowledge['returned_chunks'] + sum(len(r.get('results', [])) for r in child_knowledge)}
        decisions = [result['_human_decision'] for message in saved_active if message['role'] == 'tool'
                     for result in [json.loads(message['content'])] if '_human_decision' in result]
        return {'event': 'complete', 'answer': answer, 'history': history + saved_active, 'memory': context.memory, 'human_decisions': decisions,
                'usage': totals.copy(), 'context': context.info, 'retrieval': retrieval.copy(), 'knowledge': all_knowledge, 'recovery': recovery.public(),
                'sources': list(sources.values()), 'return_previews': return_previews, 'execution': runtime.summary(), 'plan': public_plan(plan.snapshot(), verification['enabled']),
                'verification': {key: deepcopy(verification[key]) for key in ('enabled', 'checks', 'corrections', 'status', 'usage')},
                'delegation': {'requests': delegation['requests'], 'usage': deepcopy(delegation['usage']),
                               'tasks': [{key: deepcopy(task[key]) for key in ('id', 'agent', 'status', 'rounds')} for task in delegation['tasks']]}}

    async def finish_unverified():
        nonlocal answer
        verification.update(status='unverified', candidate=None)
        # 这些预览尚未写入业务表；不能把未核实结果发布为可确认的申请卡片。
        return_previews.clear()
        answer = UNVERIFIED_ANSWER
        active.append({'role': 'assistant', 'content': answer})
        await save_progress()
        yield {'event': 'verification', 'verdict': 'unverified', 'checks': verification['checks'], 'corrections': verification['corrections']}
        yield {'event': 'delta', 'text': answer}

    async def settle_candidate():
        nonlocal answer
        candidate = verification['candidate']
        if verification['enabled']:
            if verification['checks'] >= MAX_RESULT_CHECKS:
                async for event in finish_unverified():
                    yield event
                return
            verification['checks'] += 1
            verification['status'] = 'checking'
            await save_progress()  # 请求前预留额度；中断后仍保存同一份待核对答复。
            yield {'event': 'verifying', 'number': verification['checks']}
            try:
                result = await result_verifier(question=question, candidate=candidate,
                    messages=context.compose(plan_context(evidence_active())), previews=deepcopy(return_previews), policy=context.policy)
            except ReviewBudgetError:
                async for event in finish_unverified():
                    yield event
                return
            if result['verdict'] not in ('pass', 'revise'):
                raise ValueError('结果校验返回了无效状态。')
            usage = result.get('usage')
            if usage is not None:
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    verification['usage'][key] += usage[key]
                verification['usage']['reported_rounds'] += 1
            report = {'number': verification['checks'], **result}
            verification['reports'].append(report)
            if result['verdict'] == 'revise':
                verification['feedback'] = result['issues']
                correction_limit = 1 if plan.status == 'blocked' else MAX_RESULT_CORRECTIONS
                if verification['corrections'] >= correction_limit:
                    async for event in finish_unverified():
                        yield event
                    return
                verification['corrections'] += 1
                verification.update(status='correcting', candidate=None)
                await save_progress()
                yield {'event': 'verification', **public_review(report)}
                yield {'event': 'correcting', 'number': verification['corrections']}
                return
            verification.update(status='passed', candidate=None, feedback=[])
        else:
            verification['candidate'] = None
        answer = candidate
        active.append({'role': 'assistant', 'content': answer})
        await save_progress()
        if verification['enabled']:
            yield {'event': 'verification', **public_review(report)}
            yield {'event': 'delta', 'text': answer}

    def make_job(call, mixed_plan_batch=False, *, reserve_search=True, decision=None, preflight=None, delegated=False):
        nonlocal search_issued
        name, arguments = call['function']['name'], call['function']['arguments']
        if decision is not None:
            arguments = json.dumps(decision['arguments'], ensure_ascii=False)
        immediate = None
        if mixed_plan_batch:
            immediate = {'error': '规划、委派、工具发现和技能加载必须各自单独一批调用；本批未执行。先单独完成所需控制调用；discover_tools/load_skill/read_skill_resource/update_plan 不列为业务计划步骤。', 'error_code': 'control_batch_conflict'}
        elif selection is not None and not delegated and not selection.permits(name):
            immediate = {'error': '该工具尚未加载。请先单独调用 discover_tools，读取参数定义后再调用。', 'error_code': 'tool_not_loaded'}
        elif not delegated and name not in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource') and not plan.can_call(name) and not (
                verification['status'] == 'correcting' and plan.status == 'completed' and
                name in ('get_order', 'search_history', 'search_knowledge', 'get_return_request', 'delegate_tasks')):
            immediate = {'error': '该工具不属于当前执行步骤。请先完成当前步骤或修订计划。', 'error_code': 'plan_not_ready'}
        elif name in rejected_tools and (decision is None or decision['decision'] == 'reject'):
            immediate = {'error': '用户已拒绝本轮该操作，请停止调用并说明结果。', 'error_code': 'human_rejected'}
        if immediate is None and not reserve_search and preflight is not None:
            immediate = deepcopy(preflight.get(call['id']))
        resource_sequence = None
        if immediate is None and name == 'read_skill_resource' and skills is not None and skills.resources_enabled:
            resource_sequence = skills.reserve_resource_read()
            if resource_sequence is None:
                immediate = skills.resource_limit_result()
            else:
                resource_attempts[call['id']] = (resource_sequence, arguments)
        # 先按模型的原始顺序验证和预留检索名额，避免并行任务竞争预算。
        if reserve_search and immediate is None and name in ('search_history', 'search_knowledge'):
            try:
                (history_arguments if name == 'search_history' else knowledge_arguments)(arguments)
            except (ValueError, TypeError) as error:
                immediate = {'error': f'检索参数无效：{error}'}
            else:
                if search_issued >= MAX_HISTORY_SEARCHES:
                    immediate = {'error': '本轮检索已达到 3 次上限，请根据已有依据回答，缺少依据时如实说明。'}
                else:
                    search_issued += 1
        if preflight is not None and immediate is not None:
            preflight[call['id']] = deepcopy(immediate)

        async def invoke():
            nonlocal staged_plan, staged_selection, staged_discovery_result, staged_skills, staged_skill_result
            if immediate is not None:
                return immediate.copy()
            if name == 'discover_tools' and selection is not None:
                staged_selection = deepcopy(selection)
                staged_discovery_result = staged_selection.discover(arguments)
                return staged_discovery_result
            if name in ('load_skill', 'read_skill_resource') and skills is not None:
                candidate = deepcopy(skills)
                def fits(result):
                    prospective = active + [{'role': 'tool', 'tool_call_id': call['id'],
                                              'content': json.dumps(result, ensure_ascii=False)}]
                    # Reserve every schema this run can subsequently discover;
                    # accepting a body must not poison the next discovery step.
                    # The request fitter can use compact recovery feedback;
                    # don't reject a complete skill for optional verbose advice.
                    feedback = execution_feedback(recovery, compact=True)
                    essential = context.history[:context.prefix_end] + model_context(prospective) + feedback
                    future_tools = list(selection.tools.values()) + [DISCOVERY_TOOL] if selection is not None else context.tools
                    reserve = (context.policy.summary_tokens if context.starts else 0) + 1024
                    return estimate_request(essential, future_tools) + reserve <= context.policy.input_limit
                def read():
                    return (candidate.load(arguments, accept=fits) if name == 'load_skill' else
                            candidate.read_resource(arguments, accept=fits, reserved_sequence=resource_sequence))
                result = await anyio.to_thread.run_sync(read, abandon_on_cancel=True)
                staged_skills, staged_skill_result = candidate, result
                return result
            if name == 'update_plan':
                # 先在副本上验证，执行器确认未超时后再接受新计划。
                staged_plan = deepcopy(plan)
                return staged_plan.update(arguments)
            if name == 'search_history':
                return await execute_history_search(arguments, search_history)
            if name == 'search_knowledge':
                if mcp_client is not None:
                    async def remote_search(query):
                        return await service.execute_mcp_tool(mcp_client, name, {'query': query}, remote_tools[name])
                    return await execute_knowledge_search(arguments, remote_search)
                return await execute_knowledge_search(arguments, search_knowledge)
            if name == 'get_order' and mcp_client is not None:
                try:
                    payload = json.loads(arguments)
                    if not isinstance(payload, dict):
                        raise ValueError('参数必须是 JSON 对象。')
                except (ValueError, TypeError) as error:
                    return {'error': f'订单参数无效：{error}'}
                return await service.execute_mcp_tool(mcp_client, name, payload, remote_tools[name])
            if name in ('prepare_return_request', 'get_return_request'):
                if name == 'prepare_return_request' and return_previews:
                    return {'error': '本轮已有一份退货预览，请先核对；修改原因需要发起下一轮对话。'}
                return await service.execute_return_tool(name, arguments, return_handler)
            # 当前只有只读的 get_order；未知工具仍由白名单检查拒绝。
            return await anyio.to_thread.run_sync(service.execute_tool, name, arguments, abandon_on_cancel=True)

        safe = name in ('get_order', 'search_history', 'search_knowledge', 'get_return_request')
        cache_failures = name not in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource') and not (immediate and immediate.get('error_code') in ('plan_not_ready', 'control_batch_conflict', 'human_rejected', 'tool_not_loaded'))
        return ToolJob(call['id'], name, arguments, invoke, parallel_safe=safe, retry_safe=safe,
                       cache_failures=cache_failures, budgeted=name not in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource'))

    def mark_tool_source(outcome):
        outcome['source'] = 'mcp' if outcome['name'] in remote_tools else 'local'

    def remember_preview(outcome):
        nonlocal plan, staged_plan, selection, staged_selection, staged_discovery_result, skills, staged_skills, staged_skill_result
        mark_tool_source(outcome)
        if outcome['name'] in ('load_skill', 'read_skill_resource'):
            if staged_skills is not None and outcome['result'] is staged_skill_result:
                skills = staged_skills
            elif outcome['name'] == 'read_skill_resource' and outcome['call_id'] in resource_attempts:
                sequence, arguments = resource_attempts[outcome['call_id']]
                skills.record_resource_event(sequence, arguments, outcome['result'])
            resource_attempts.pop(outcome['call_id'], None)
            staged_skills, staged_skill_result = None, None
            return
        if outcome['name'] == 'discover_tools':
            # Accept only the result that survived ToolRuntime's deadline/cancel checks.
            if staged_selection is not None and outcome['result'] is staged_discovery_result:
                selection = staged_selection
                # Retrieval fitting runs before the next context node. Use the
                # newly accepted schemas now, while original evidence stays saved.
                context.tools = selection.request_tools()
            staged_selection, staged_discovery_result = None, None
            return
        if outcome['name'] == 'update_plan':
            if 'plan' in outcome['result'] and staged_plan is not None:
                plan = staged_plan
            staged_plan = None
            return
        plan.observe(outcome)
        result = outcome['result']
        if outcome['name'] == 'prepare_return_request' and 'error' not in result and 'preview' in result:
            return_previews.append(result['preview'])

    def prepare_child_tools(task, calls):
        allowed = {tool['function']['name'] for tool in tools_by_agent[task['agent']]}
        if any(call['function']['name'] not in allowed for call in calls):
            raise ValueError('子 Agent 请求了超出职责的工具。')
        preflight = {}
        jobs = [make_job(call, preflight=preflight, delegated=True) for call in calls]
        return {'reserved': runtime.reserve_batch(jobs), 'preflight': preflight}

    async def execute_child_tools(task, pending):
        calls = pending['message']['tool_calls']
        allowed = {tool['function']['name'] for tool in tools_by_agent[task['agent']]}
        if any(call['function']['name'] not in allowed for call in calls):
            raise ValueError('保存的子 Agent 工具超出职责。')
        jobs = [make_job(call, reserve_search=False, preflight=pending.get('preflight', {}), delegated=True) for call in calls]
        return await runtime.run_batch(jobs, reserved=pending['reserved'], on_result=mark_tool_source)

    async def execute_delegation(message):
        nonlocal pending_delegation
        call = message['tool_calls'][0]
        started = monotonic()
        error, results = None, []
        if pending_delegation is None:
            permitted = plan.can_call('delegate_tasks') or (verification['status'] == 'correcting' and plan.status == 'completed')
            if not delegation_enabled or not permitted:
                error = '当前步骤不能委派子任务，请先核对计划。'
            else:
                try:
                    task_ids = register_tasks(delegation, call['function']['arguments'], set(tools_by_agent))
                except (ValueError, TypeError) as failure:
                    error = str(failure)
                else:
                    pending_delegation = {'message': deepcopy(message), 'task_ids': task_ids}
                    if selection is not None:
                        for task in delegation['tasks']:
                            if task['id'] in task_ids:
                                selection.bind_specialist(tool['function']['name'] for tool in tools_by_agent[task['agent']])
                    await save_progress()
        if error is None:
            async for event in run_specialists(client, model, pending_delegation['task_ids'], question=question,
                    policy=context.policy, state=delegation, tools_by_agent=tools_by_agent,
                    prepare_tools=prepare_child_tools, execute_tools=execute_child_tools, save_checkpoint=save_progress):
                if event['event'] == 'delegation_complete':
                    results = event['results']
                else:
                    yield event
            if not results or any(result['status'] != 'completed' for result in results):
                error = '部分子任务未完成，请根据各项状态与实际证据说明限制或询问缺失信息。'
        result = {'delegations': results, 'status': 'partial' if error and results else 'failed' if error else 'completed'}
        if error:
            result['error'] = error
        elif plan.steps:
            result['_tool_call_id'] = call['id']
        outcome = {'call_id': call['id'], 'name': 'delegate_tasks', 'result': result, 'status': 'error' if error else 'ok',
                   'attempts': 1, 'retries': [], 'elapsed_ms': round((monotonic() - started) * 1000)}
        plan.observe(outcome)
        recovery_start = recovery.state['sequence']
        recovery.observe([outcome], round_number=rounds)
        stamp_recovery_stop(result)
        runtime.trace.append({key: value for key, value in outcome.items() if key != 'result'})
        active.extend([deepcopy(message), {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)}])
        pending_delegation = None
        await save_progress()
        # 子任务结论只交给主管核对；网页阶段事件不公开子 Agent 的答复草稿。
        visible = {**result, 'delegations': [{key: item[key] for key in ('task_id', 'agent', 'status')} for item in results]}
        yield {'event': 'tool_result', 'name': 'delegate_tasks', 'result': visible}
        yield {'event': 'tool_execution', **runtime.summary()}
        for event in recovery.state['events']:
            if event['sequence'] > recovery_start:
                yield {'event': 'recovery', **event}

    def stamp_recovery_stop(result):
        if not recovery.stopped:
            return
        # Server-owned execution metadata is attached to the actual final tool
        # result. No fabricated model call or success evidence is introduced.
        if plan.current:
            plan.current['status'] = 'blocked'
            plan.note = recovery.stop_message()
            plan.revision += 1
        result['_recovery'] = {**recovery.public(), 'plan': plan.snapshot()}

    def retrieval_exhausted():
        return any(result.get('budget_limited') and not result.get('results')
                   for _, result in retrieval_results + knowledge_results)

    async def finish_stalled():
        nonlocal answer
        # This fixed execution-status notice contains no model-generated facts
        # and needs no additional actor/verifier call. Unverified previews stay
        # in raw evidence only and are never published as actionable cards.
        return_previews.clear()
        verification.update(status='not_applicable', candidate=None, feedback=[])
        answer = recovery.stop_message()
        active.append({'role': 'assistant', 'content': answer})
        await save_progress()
        yield {'event': 'delta', 'text': answer}

    async def execute_batch(message, *, saved_batch=None):
        nonlocal pending_batch, recovery
        tool_calls = message['tool_calls']
        approvals = saved_batch['approvals'] if saved_batch else []
        decisions = {item['call_id']: item for item in approvals}
        rejected_tools.update(item['name'] for item in approvals if item['decision'] == 'reject')
        active.append(deepcopy(message))
        batch_results = []
        preflight = deepcopy(saved_batch.get('preflight', {})) if saved_batch else {}
        mixed = len(tool_calls) > 1 and any(call['function']['name'] in ('update_plan', 'delegate_tasks', 'discover_tools', 'load_skill', 'read_skill_resource') for call in tool_calls)
        jobs = [make_job(call, mixed, reserve_search=saved_batch is None, decision=decisions.get(call['id']), preflight=preflight) for call in tool_calls]
        reserved = saved_batch['reserved'] if saved_batch else runtime.reserve_batch(jobs)
        if saved_batch is None:
            for call, job, allowed in zip(tool_calls, jobs, reserved):
                name = job.name
                if (allowed and not mixed and not preflight.get(call['id']) and name in approval_tools and plan.can_call(name)
                        and name not in rejected_tools and not (name == 'prepare_return_request' and return_previews)
                        and runtime._key(job) not in runtime.failures):
                    approval = make_approval(call)
                    if approval:
                        approvals.append(approval)
            if approvals:
                pending_batch = {'message': deepcopy(message), 'reserved': reserved, 'approvals': approvals, 'preflight': preflight}
                await save_progress(active[:-1])
                active.pop()
                return
        await save_progress(active[:-1])
        for job in jobs:
            yield {'event': 'tool_start', 'name': job.name, 'batch_size': len(jobs)}
        before_plan = plan.snapshot()
        before_loaded = set(selection.loaded) if selection else set()
        before_skills = set(skills.loaded) if skills else set()
        before_resources = set(skills.resources) if skills else set()
        recovery_start = recovery.state['sequence']
        outcomes = await runtime.run_batch(jobs, on_result=remember_preview, reserved=reserved)
        for outcome in outcomes:
            name, result = outcome['name'], outcome['result']
            if outcome['call_id'] in decisions:
                result['_human_decision'] = deepcopy(decisions[outcome['call_id']])
            if plan.steps and name not in ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource') and 'error' not in result:
                # 模型服务可能不展示协议层 ID，正文中提供可复制的可信凭据。
                result['_tool_call_id'] = outcome['call_id']
            if name == 'search_history' and 'error' not in result:
                retrieval['calls'] += 1
                retrieval_results.append((len(active), result))
                retrieval_snapshots[len(active)] = deepcopy(result)
            elif name == 'search_knowledge' and 'error' not in result:
                knowledge['calls'] += 1
                knowledge_results.append((len(active), result))
                # Capture original evidence BEFORE any request-budget fitting.
                knowledge_snapshots[len(active)] = deepcopy(result)
            active.append({"role": "tool", "tool_call_id": outcome['call_id'], "content": json.dumps(result, ensure_ascii=False)})
            batch_results.append((name, result))
        # 先收齐同批订单/检索结果再裁剪，避免工具排列顺序决定能否继续。
        retrieved = sorted(retrieval_results + knowledge_results, key=lambda row: row[0])
        fit_history_results(context, active, retrieved)
        while True:
            candidate = RecoveryTracker(recovery.snapshot())
            candidate.observe(outcomes, round_number=rounds, before_plan=before_plan, after_plan=plan.snapshot(),
                              newly_loaded=(set(selection.loaded) - before_loaded if selection else set()) |
                                           ({'skill:' + name for name in set(skills.loaded) - before_skills} if skills else set()) |
                                           ({'resource:' + key for key in set(skills.resources) - before_resources} if skills else set()))
            count_before = sum(len(result['results']) for _, result in retrieved)
            fit_execution_feedback(candidate)
            count_after = sum(len(result['results']) for _, result in retrieved)
            if count_after == count_before:
                recovery = candidate
                break
            # Each iteration removes at least one retrieval item. Re-evaluate
            # only that smaller view; no speculative event/counter is committed.
        if retrieval_exhausted():
            # Compaction and fitting already ran. Rewording a query cannot fix
            # a result that cannot fit; do not spend another model/search call.
            recovery.stop_for_context(rounds)
        if outcomes and recovery.stopped:
            stamp_recovery_stop(outcomes[-1]['result'])
            active[-1]['content'] = json.dumps(outcomes[-1]['result'], ensure_ascii=False)
            if len(active) - 1 in knowledge_snapshots:
                knowledge_snapshots[len(active) - 1]['_recovery'] = deepcopy(outcomes[-1]['result']['_recovery'])
            if len(active) - 1 in retrieval_snapshots:
                retrieval_snapshots[len(active) - 1]['_recovery'] = deepcopy(outcomes[-1]['result']['_recovery'])
        retrieval["returned_messages"] = sum(len(result["results"]) for _, result in retrieval_results)
        knowledge["returned_chunks"] = sum(len(result["results"]) for result in knowledge_snapshots.values())
        # 已完成批次才清除待审批调用；断连可继续使用原决定。
        pending_batch = None
        # 批次结果收齐并通过超时/取消检查后落盘，再向网页通知完成。
        await save_progress()
        for name, result in batch_results:
            visible = {**result, 'plan': public_plan(result['plan'], verification['enabled'])} if name == 'update_plan' and 'plan' in result else result
            if name in ('load_skill', 'read_skill_resource'):
                visible = public_skill(result)
            yield {"event": "tool_result", "name": name, "result": visible}
            if name == 'update_plan' and 'plan' in result:
                yield {'event': 'plan', 'plan': visible['plan']}
        yield {"event": "tool_execution", **runtime.summary()}
        for event in recovery.state['events']:
            if event['sequence'] > recovery_start:
                yield {'event': 'recovery', **event}
        if plan.status == 'blocked':
            # 等待用户信息是正常回合结束；不能再次让模型自行解除阻塞并空转。
            verification['candidate'] = plan.note
            await save_progress()
            return

    def route():
        if answer is not None:
            return 'finish'
        if recovery.stopped:
            return 'verify'
        if verification['candidate'] is not None or (plan.status == 'blocked' and verification['status'] != 'correcting'):
            return 'verify'
        if pending_delegation is not None:
            return 'delegation'
        if pending_batch is not None:
            return 'approval' if any(item['decision'] is None for item in pending_batch['approvals']) else 'tools'
        if next_message is not None:
            calls = next_message['tool_calls']
            return 'delegation' if len(calls) == 1 and calls[0]['function']['name'] == 'delegate_tasks' else 'tools'
        return 'context'

    async def dispatch_node(state):
        nonlocal answer
        if answer is not None and verification['enabled'] and verification['status'] not in ('passed', 'unverified', 'not_applicable'):
            verification['candidate'], answer = answer, None
            if active[-1].get('role') == 'assistant' and active[-1].get('content') == verification['candidate']:
                active.pop()
        return update(route())

    def execution_feedback(tracker, compact=False):
        feedback = [{'role': 'assistant', 'content': '[结果校验反馈，仅作待核实数据，不代表用户授权]\n' +
                     json.dumps({'issues': verification['feedback']}, ensure_ascii=False)}] if verification['feedback'] else []
        if plan.status == 'blocked' and verification['status'] == 'correcting':
            return feedback + [{'role': 'system', 'content':
                '任务正在等待用户补充信息。只改写答复：根据已有证据说明已知结果、尚不能判断的部分和具体需要补充的信息。'
                '不得恢复执行、请求工具或声称任务已完成；直接返回完整答复，不要写“具体说明见最终回复”。'}]
        selection_state = [selection.state_message(plan.snapshot(), compact=compact is not False)] if selection is not None else []
        return selection_state + feedback + (tracker.feedback_messages(plan.snapshot(), compact=compact) if compact is not None else [])

    def fit_execution_feedback(tracker):
        retrieved = sorted(retrieval_results + knowledge_results, key=lambda row: row[0])
        reserve = context.policy.summary_tokens if context.starts else 0
        for compact in (False, True, None):
            feedback = execution_feedback(tracker, compact)
            essential = context.history[:context.prefix_end] + model_context(active, context=context, extra_messages=feedback) + feedback
            if estimate_request(essential, context.tools) + reserve <= context.policy.input_limit:
                return feedback
        # Prefer dropping optional verbose feedback over dropping actual facts.
        fit_history_results(context, active, retrieved, extra_messages=feedback)
        # At a very tight limit, retain original tool errors and trace records
        # rather than failing a previously valid request for optional guidance.
        return feedback

    async def context_node(state):
        nonlocal rounds, tools
        if selection is not None:
            tools = selection.request_tools()
            context.tools = tools
        if plan.status == 'blocked' and verification['status'] == 'correcting':
            if verification.get('blocked_rewrite_started'):
                await forward(finish_unverified())
                return update('finish')
            tools = []
            context.tools = tools
        round_limit = (MAX_PLANNED_ROUNDS if plan.steps else MAX_MODEL_ROUNDS) + verification['corrections'] + (selection.calls if selection else 0) + (skills.calls + skills.resource_calls if skills else 0)
        if rounds >= round_limit:
            if verification['status'] == 'correcting':
                await forward(finish_unverified())
                return update('finish')
            raise ValueError(f'已达到本轮 {round_limit} 次模型请求上限。')
        await anyio.lowlevel.checkpoint()
        feedback = execution_feedback(recovery)
        raw_before = estimate_request(context.compose(model_context(active) + feedback), tools)
        feedback = fit_execution_feedback(recovery)
        retrieval['returned_messages'] = sum(len(result['results']) for _, result in retrieval_results)
        if retrieval_exhausted():
            recovery.stop_for_context(rounds)
            for index in range(len(active) - 1, -1, -1):
                if active[index].get('role') == 'tool':
                    result = json.loads(active[index]['content'])
                    stamp_recovery_stop(result)
                    active[index]['content'] = json.dumps(result, ensure_ascii=False)
                    for snapshots in (retrieval_snapshots, knowledge_snapshots):
                        if index in snapshots:
                            snapshots[index]['_recovery'] = deepcopy(result['_recovery'])
                    break
            await save_progress()
            await emit({'event': 'recovery', **recovery.state['events'][-1]})
            return update('verify')
        try:
            await forward(context.prepare(client, model, model_context(active, context=context, extra_messages=feedback) + feedback, on_checkpoint=save_progress))
        except ValueError:
            if verification['status'] != 'correcting':
                raise
            await forward(finish_unverified())
            return update('finish')
        context.info['before_estimated_tokens'] = raw_before
        rounds += 1
        await save_progress()
        return update('model')

    async def model_node(state):
        nonlocal last_model_result, completion_corrections, next_message
        if plan.status == 'blocked' and verification['status'] == 'correcting':
            if verification.get('blocked_rewrite_started'):
                await forward(finish_unverified())
                return update('finish')
            # Charge the one rewrite before contacting the model, including on resume.
            verification['blocked_rewrite_started'] = True
            await save_progress()
        round_number = rounds
        round_limit = (MAX_PLANNED_ROUNDS if plan.steps else MAX_MODEL_ROUNDS) + verification['corrections'] + (selection.calls if selection else 0) + (skills.calls + skills.resource_calls if skills else 0)
        await emit({'event': 'round', 'round': rounds})
        text_parts, calls_by_index = [], {}
        finish_reason, usage = None, None
        model_started = monotonic()

        # await 等待网络时让出执行机会，其他会话和健康检查仍能得到响应。
        await observe_request(context.messages, tools, context.policy, role='actor',
            before_tokens=context.info.get('before_estimated_tokens'),
            history_end=context.prefix_end + bool(context.memory.text) +
                len(context.history) - max(context.prefix_end, context.memory.covered_messages))
        stream = await client.chat.completions.create(
            model=model, messages=context.messages, **({'tools': tools, 'tool_choice': 'auto'} if tools else {}),
            stream=True, stream_options={"include_usage": True}, max_tokens=context.policy.output_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        try:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    text_parts.append(delta.content)
                    if not verification['enabled']:
                        await emit({"event": "delta", "text": delta.content})

                for part in delta.tool_calls or []:
                    index = part.index
                    if not isinstance(index, int) or index < 0:
                        raise ValueError("工具调用缺少有效的序号。")
                    if index not in calls_by_index:
                        calls_by_index[index] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    call = calls_by_index[index]
                    if part.type and part.type != "function":
                        raise ValueError("收到不支持的工具调用类型。")
                    if part.id:
                        call["id"] += part.id
                    if part.function is not None:
                        if part.function.name:
                            call["function"]["name"] += part.function.name
                        if part.function.arguments:
                            call["function"]["arguments"] += part.function.arguments
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
        finally:
            # 正常结束、异常或断连取消都会走到这里。屏蔽取消直到清理结束，
            # 避免关闭连接所需的 await 也被外层取消打断。
            with anyio.CancelScope(shield=True):
                await stream.close()

        if usage is not None:
            add_usage(totals, usage)
        last_model_result = {'round': rounds, 'finish_reason': finish_reason, 'elapsed_ms': round((monotonic() - model_started) * 1000),
                             'text_chars': sum(len(text) for text in text_parts),
                             'tool_calls': [{'call_id': call['id'], 'name': call['function']['name'], 'arguments': proposed_arguments(call['function']['arguments'])} for call in calls_by_index.values()],
                             'usage': {key: getattr(usage, key, None) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')} if usage else None}
        await save_progress()
        if usage is not None:
            await emit({"event": "usage", "usage": totals.copy()})
        if finish_reason is None:
            raise ValueError("流已关闭，但没有收到结束状态。")
        if finish_reason not in ("stop", "tool_calls"):
            raise ValueError(f"模型输出未正常完成，结束原因：{finish_reason}")

        content = "".join(text_parts)
        tool_calls = [calls_by_index[index] for index in sorted(calls_by_index)]
        if tool_calls:
            if plan.status == 'blocked' and verification['status'] == 'correcting':
                # A blocked clarification may be rewritten, never resumed by the model.
                await forward(finish_unverified())
                return update('finish')
            if finish_reason != "tool_calls":
                raise ValueError("工具调用与结束原因不一致。")
            if round_number == round_limit:
                if verification['status'] == 'correcting':
                    await forward(finish_unverified())
                    return update('finish')
                raise ValueError(f"已达到本轮 {round_limit} 次模型请求上限。")
            call_ids = [call["id"] for call in tool_calls]
            if not all(call_ids) or len(set(call_ids)) != len(call_ids):
                raise ValueError("工具调用 ID 缺失或重复。")
            if any(not call["function"]["name"] for call in tool_calls):
                raise ValueError("工具名称不完整。")

            next_message = {'role': 'assistant', 'content': content or None, 'tool_calls': tool_calls}
            await save_progress()
            return update(route())

        if finish_reason != 'stop' or not content.strip():
            raise ValueError('接口没有返回可显示的最终回答。')
        if plan.status == 'in_progress':
            if completion_corrections or round_number == round_limit:
                raise ValueError('执行计划仍有未完成步骤，本轮未保存。请重试或补充所需信息。')
            completion_corrections += 1
            context.history[0] = {**context.history[0], 'content': context.history[0]['content'] + correction}
            await save_progress()
            await emit({'event': 'plan_pending', 'plan': plan.snapshot()})
            return update('context')
        verification['candidate'] = content
        await save_progress()
        return update('verify')

    async def tools_node(state):
        nonlocal next_message
        message = pending_batch['message'] if pending_batch is not None else next_message
        next_message = None
        await forward(execute_batch(message, saved_batch=pending_batch))
        return update(route())

    async def delegation_node(state):
        nonlocal next_message
        message = pending_delegation['message'] if pending_delegation is not None else next_message
        next_message = None
        await forward(execute_delegation(message))
        return update(route())

    async def approval_node(state):
        # The application validates and persists decisions before this invocation.
        # Replay the same interrupt on resume; consume Command without replacing
        # the interrupted task or the root checkpoint with the progress snapshot.
        interrupt({'status': 'awaiting_approval'})
        if pending_batch is not None and any(item['decision'] is None for item in pending_batch['approvals']):
            raise ValueError('人工审批尚未完成。')
        return update(route())

    async def verify_node(state):
        if recovery.stopped:
            await forward(finish_stalled())
            return update('finish')
        if verification['candidate'] is None and plan.status == 'blocked':
            verification['candidate'] = plan.note
            await save_progress()
        await forward(settle_candidate())
        return update(route())

    async def finish_node(state):
        return update('end')

    builder = StateGraph(ChatState)
    phases = {'dispatch': dispatch_node, 'context': context_node, 'model': model_node,
              'tools': tools_node, 'delegation': delegation_node, 'approval': approval_node,
              'verify': verify_node, 'finish': finish_node}
    for name, node in phases.items():
        builder.add_node(name, node)
    builder.add_edge(START, 'dispatch')
    for name in phases:
        builder.add_conditional_edges(name, lambda state: state['route'],
                                      {**{phase: phase for phase in phases}, 'end': END})
    graph = builder.compile(checkpointer=saver)
    if checkpoint is not None:
        yield {'event': 'resumed', 'plan': public_plan(plan.snapshot(), verification['enabled']), 'rounds': rounds,
               'return_ids': [preview['preview_id'] for preview in return_previews]}
    if root_interrupted and pending_batch is not None and any(item['decision'] is None for item in pending_batch['approvals']):
        yield {'event': 'paused', 'status': 'awaiting_approval'}
        return
    graph_input = Command(resume=True) if root_interrupted else update('dispatch')
    # At most one unacknowledged event plus the terminal marker can be queued.
    events = asyncio.Queue(maxsize=2)
    finished = object()
    paused = False

    async def run_graph():
        nonlocal paused
        # LangGraph uses asyncio cancellation: its executor cancels workers once,
        # then awaits their cleanup. AnyIO's repeated cancellation must not cut
        # that wait short. Isolate the pump; the consumer cancels it once and
        # waits for every graph/SDK task before releasing the request.
        with anyio.CancelScope(shield=True), request_observer(record_budget):
            stream = graph.astream(graph_input, config=config, stream_mode=['custom', 'updates'])
            try:
                async for mode, item in stream:
                    if mode == 'custom':
                        events.put_nowait(item)
                    elif '__interrupt__' in item:
                        paused = True
            finally:
                await stream.aclose()

    runner = asyncio.create_task(run_graph(), name='formal-chat-graph')
    runner.add_done_callback(lambda task: events.put_nowait(finished))
    try:
        while True:
            item = await events.get()
            if item is finished:
                await runner  # Propagate graph failures after cleanup.
                break
            event, acknowledged = item
            yield event
            acknowledged.set()
    finally:
        if not runner.done():
            runner.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(runner, return_exceptions=True)
    # The API commits history and deletes graph state upon complete. Root writes
    # must be fully drained before exposing either terminal event to the caller.
    if paused:
        yield {'event': 'paused', 'status': 'awaiting_approval'}
    elif answer is not None:
        yield complete_event()
