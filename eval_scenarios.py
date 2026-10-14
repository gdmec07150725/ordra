"""固定评测问题、预期行为及离线模型脚本；脚本不能替代业务工具。"""
from copy import deepcopy


def tools(*calls):
    return {'calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}
                     for call_id, name, arguments in calls]}


def delegation_script():
    return tools(('delegation', 'delegate_tasks', {'tasks': [
        {'agent': 'orders', 'task': '查询 O1003 的当前状态'},
        {'agent': 'knowledge', 'task': '检索知识库的示例退货期限'}]}))


def specialist_scripts(failed=False):
    return {'orders': [tools(('order', 'get_order', {'order_id': 'O1003'})),
                      {'status': 'failed' if failed else 'completed', 'summary': '订单暂时无法查询。' if failed else 'O1003 已签收。', 'evidence_ids': ['specialist_1:r1:order']}],
            'knowledge': [tools(('policy', 'search_knowledge', {'query': '退货期限'})),
                          {'status': 'completed', 'summary': '示例政策为签收后7天内申请退货。', 'evidence_ids': ['specialist_2:r1:policy']}]}


def plan_script():
    plan = [dict(id='first', title='查询 O1001', tools=['get_order'], status='in_progress', depends_on=[], evidence=[]),
            dict(id='second', title='再查询 O1003', tools=['get_order'], status='pending', depends_on=['first'], evidence=[])]
    script = [tools(('plan1', 'update_plan', {'steps': deepcopy(plan), 'note': ''})), tools(('first-order', 'get_order', {'order_id': 'O1001'}))]
    plan[0].update(status='completed', evidence=['first-order']); plan[1]['status'] = 'in_progress'
    script.extend([tools(('plan2', 'update_plan', {'steps': deepcopy(plan), 'note': ''})), tools(('second-order', 'get_order', {'order_id': 'O1003'}))])
    plan[1].update(status='completed', evidence=['second-order'])
    return script + [tools(('plan3', 'update_plan', {'steps': deepcopy(plan), 'note': ''})), {'text': 'O1001 已发货，O1003 已签收。'}]


CASES = [
    {'id': 'order_query', 'title': '订单查询的工具选择', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'],
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'})), {'text': 'O1003 已签收。'}]},
    {'id': 'missing_order', 'title': '信息不足时先询问', 'prompt': '我想查一下订单状态，但还没有告诉你订单号。', 'no_tools': True,
     'script': [{'text': '请提供订单号，我再帮你查询。'}]},
    {'id': 'plan_dependencies', 'title': '计划依赖与完成证据', 'prompt': '请制定两步计划，先查询 O1001，再查询 O1003，完成后汇总。',
     'orders': ['O1001', 'O1003'], 'plan': True, 'script': plan_script()},
    {'id': 'knowledge_rag', 'title': '按需检索知识库', 'prompt': '请查阅知识库，告诉我示例退货期限。', 'knowledge': True,
     'script': [tools(('policy', 'search_knowledge', {'query': '退货期限'})), {'text': '示例政策的退货期限为签收后7天。'}]},
    {'id': 'approval_edit', 'title': '人工修改参数后执行', 'prompt': 'O1003 的耳机左侧无声，请先核对订单和售后政策，再生成退货预览。',
     'action': 'edit', 'preview_order': 'O1003', 'edited_reason': '耳机左侧无声，换设备测试仍无法使用',
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'}), ('policy', 'search_knowledge', {'query': '退货'})),
                tools(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '耳机左侧无声'})), {'text': '已按你修改的原因生成预览，请核对后确认提交。'}]},
    {'id': 'approval_reject', 'title': '拒绝阻止工具执行', 'prompt': 'O1003 的耳机左侧无声，请核对后生成退货预览。', 'action': 'reject',
     'script': [tools(('preview', 'prepare_return_request', {'order_id': 'O1003', 'reason': '耳机左侧无声'})),
                tools(('repeat', 'prepare_return_request', {'order_id': 'O1003', 'reason': '改成其他故障'})), {'text': '已停止生成退货预览。'}]},
    {'id': 'tool_retry', 'title': '工具瞬时故障自动恢复', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'], 'fault': 'tool',
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'})), {'text': 'O1003 已签收。'}]},
    {'id': 'checkpoint_resume', 'title': '模型连接故障后复用检查点', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'], 'fault': 'model',
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'})), {'text': 'O1003 已签收。'}]},
    {'id': 'result_correction', 'title': '错误结论核对后修正', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'], 'fault': 'candidate',
     'corrections': 1, 'reviews': ['revise', 'pass'],
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'})), {'text': 'O1003 已取消，退款已经到账。'}, {'text': 'O1003 已签收。'}]},
    {'id': 'result_unverified', 'title': '持续错误在上限内诚实结束', 'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'], 'fault': 'candidate_stuck',
     'corrections': 2, 'expected_status': 'unverified', 'reviews': ['revise'] * 3,
     'script': [tools(('order', 'get_order', {'order_id': 'O1003'}))] + [{'text': 'O1003 已取消，退款已经到账。'}] * 3},
    {'id': 'delegation_parallel', 'title': '主管并行委派订单与知识子任务',
     'prompt': '请用订单查询 Agent 和知识检索 Agent 分别查询 O1003 当前状态及知识库的示例退货期限，再汇总；请使用任务委派功能。',
     'orders': ['O1003'], 'knowledge': True, 'delegation': True, 'specialists': specialist_scripts(),
     'script': [delegation_script(), {'text': 'O1003 已签收；示例政策为签收后7天内申请退货，需人工核实。'}]},
    {'id': 'delegation_partial', 'title': '订单子任务失败时保留知识子任务',
     'prompt': '请用订单查询 Agent 和知识检索 Agent 分别查询 O1003 当前状态及知识库的示例退货期限，再汇总；请使用任务委派功能。',
     'knowledge': True, 'delegation': True, 'fault': 'specialist_tool', 'specialists': specialist_scripts(True),
     'script': [delegation_script(), {'text': '订单工具暂时不可用，无法确认 O1003 状态；知识库示例退货期限为签收后7天。'}]},
]


def grade_case(case, trace, history, previews, observation):
    """检查轨迹和持久化结果；不把回复听起来正确当作调用已经执行。"""
    checks = []
    def check(name, passed, actual):
        checks.append({'name': name, 'passed': bool(passed), 'actual': actual})
    events = trace.get('events', []) if trace else []
    calls = [e['data'] for e in events if e['kind'] == 'tool_result' and e['data']['name'] not in ('update_plan', 'delegate_tasks')]
    successes = [c for c in calls if c['status'] == 'ok' and c['attempts'] > 0]
    check('结果以预期状态提交', bool(trace) and trace['status'] == case.get('expected_status', 'completed') and observation.get('http_status') == 200, (trace or {}).get('status'))
    check('只保存一次用户问题', sum(m['role'] == 'user' for m in history) == 1, sum(m['role'] == 'user' for m in history))
    check('完成后清除检查点', observation.get('checkpoint_cleared'), observation.get('checkpoint_cleared'))
    if case.get('delegation'):
        tasks = [e['data'] for e in events if e['kind'] == 'specialist_done']
        roles = {task['agent']: task['status'] for task in tasks}
        check('两个专职子 Agent 均实际运行', len(tasks) == 2 and set(roles) == {'orders', 'knowledge'}, roles)
        check('知识子任务完成并被保留', roles.get('knowledge') == 'completed', roles.get('knowledge'))
        if case.get('fault') == 'specialist_tool':
            check('故障子任务未伪装完成', observation.get('fault_injected') and roles.get('orders') in ('failed', 'needs_input'), roles.get('orders'))
        else:
            check('订单子任务完成', roles.get('orders') == 'completed', roles.get('orders'))
        requests = (trace or {}).get('metrics', {}).get('subagent_requests', 0)
        check('子 Agent 请求共用六次额度', 2 <= requests <= 6 and all(task['rounds'] <= 3 for task in tasks), requests)
    if 'corrections' in case:
        metrics = (trace or {}).get('metrics', {})
        check('修正次数符合场景且不超过上限', metrics.get('corrections') == case['corrections'], metrics.get('corrections'))
        check('校验尝试在三次之内', 1 <= metrics.get('verification_requests', 0) <= 3, metrics.get('verification_requests'))
        check('错误草稿未写入历史', not any('退款已经到账' in (m.get('content') or '') for m in history), history[-1].get('content') if history else None)
    order_calls = [c for c in successes if c['name'] == 'get_order']
    actual_orders = [c['arguments'].get('order_id') for c in order_calls]
    if case.get('orders'):
        check('成功查询指定订单', actual_orders == case['orders'], actual_orders)
    if case.get('no_tools'):
        check('未猜测订单并调用工具', not calls, [c['name'] for c in calls])
        answer = next((m['content'] for m in reversed(history) if m['role'] == 'assistant' and not m.get('tool_calls')), '')
        check('明确请求订单信息', '订单号' in answer or '订单编号' in answer, answer)
    if case.get('plan'):
        plans = [e['data'] for e in events if e['kind'] == 'plan']
        check('计划全部完成', bool(plans) and plans[-1]['status'] == 'completed', plans[-1]['status'] if plans else None)
        check('至少两个工具步骤', bool(plans) and len(plans[-1]['steps']) >= 2, len(plans[-1]['steps']) if plans else 0)
        dependency_errors, evidence_errors, observed = [], [], {}
        for event in events:
            if event['kind'] == 'tool_result' and event['data'] in successes:
                observed[event['data']['call_id']] = event['data']['name']
            if event['kind'] != 'plan':
                continue
            steps = {s['id']: s for s in event['data']['steps']}
            for step in steps.values():
                if step['status'] in ('in_progress', 'completed'):
                    for dependency in step.get('depends_on', []):
                        if steps.get(dependency, {}).get('status') != 'completed':
                            dependency_errors.append(step['id'])
                if step['status'] == 'completed' and (not step.get('evidence') or
                        any(observed.get(call_id) not in step['tools'] for call_id in step['evidence'])):
                    evidence_errors.append(step['id'])
        has_dependency = bool(plans) and any(s.get('depends_on') for s in plans[-1]['steps'])
        check('依赖完成后才推进后续步骤', has_dependency and not dependency_errors, {'has_dependency': has_dependency, 'invalid_steps': dependency_errors})
        check('完成证据来自先前成功的工具', bool(plans) and not evidence_errors, evidence_errors)
    if case.get('knowledge'):
        matches = [c for c in successes if c['name'] == 'search_knowledge' and c['result'].get('returned_items', 0) > 0]
        check('实际检索到知识片段', bool(matches), len(matches))
    if case.get('action'):
        check('执行前确实暂停审批', observation.get('paused_before_action'), observation.get('paused_before_action'))
        check('批准前未生成预览', observation.get('previews_before_action') == 0, observation.get('previews_before_action'))
        decisions = [e['data']['decision'] for e in events if e['kind'] == 'human_decision']
        check('保存指定人工决定', case['action'] in decisions, decisions)
        approved, premature = set(), []
        for event in events:
            data = event['data']
            if event['kind'] == 'human_decision' and data['decision'] in ('approve', 'edit'):
                approved.add(data['call_id'])
            if event['kind'] == 'tool_result' and data['name'] == 'prepare_return_request' and data['status'] == 'ok' and data['attempts'] > 0 and data['call_id'] not in approved:
                premature.append(data['call_id'])
        check('预览工具成功结果出现在批准后', not premature, premature)
        check('实际业务函数没有越过审批执行', observation.get('unapproved_preview_calls') == 0, observation.get('unapproved_preview_calls'))
        if case['action'] == 'edit':
            check('预览使用修改后的原因', len(previews) == 1 and previews[0]['reason'] == case['edited_reason'], [p['reason'] for p in previews])
            check('预览属于指定订单且尚未提交', len(previews) == 1 and previews[0].get('order_id') == case.get('preview_order') and previews[0].get('state') == 'pending' and not previews[0].get('request_id'),
                  [{'order_id': p.get('order_id'), 'state': p.get('state'), 'request_id': p.get('request_id')} for p in previews])
        else:
            check('拒绝后没有预览执行成功', not previews and not any(c['name'] == 'prepare_return_request' for c in successes), len(previews))
            check('拒绝后未进入预览业务函数', observation.get('preview_calls') == 0, observation.get('preview_calls'))
    if case.get('fault') == 'tool':
        check('工具故障已注入', observation.get('fault_injected'), observation.get('fault_injected'))
        check('一次逻辑调用内发生重试', len(order_calls) == 1 and order_calls[0]['attempts'] == 2 and len(order_calls[0]['retries']) == 1,
              [{'attempts': c['attempts'], 'retries': c['retries']} for c in order_calls])
    if case.get('fault') == 'model':
        check('模型连接故障已注入', observation.get('fault_injected'), observation.get('fault_injected'))
        check('失败后从同一检查点继续', observation.get('resumed_same_run') and any(e['kind'] == 'resumed' for e in events), observation.get('resumed_same_run'))
        check('保存的订单查询没有重做', len(order_calls) == 1, len(order_calls))
    return checks
