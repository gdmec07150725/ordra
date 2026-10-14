"""Versioned, controlled reproductions; no private conversation export."""
from eval_scenarios import delegation_script, specialist_scripts, tools


DATASET_VERSION = 'decision-cases-1'
POLICY_TEXT = '''# 评测专用售后政策

## 退货与退款政策
签收后7天内可申请模拟退货，需核对商品情况、具体退货原因和签收时间，并由人工审核。
退款需在退货审核通过后处理，具体到账时间待确认。预览不等于提交或退款。
'''
STRATEGIES = {
    'current': {'title': '当前策略', 'instruction': ''},
    'direct': {'title': '主管直接查询', 'instruction':
               '本次执行优先由主管直接使用已发现的业务工具完成查询，独立只读查询可并行。'
               '只有用户明确要求子 Agent 时才委派。保持原有事实核对、缺失信息询问和审批规则。'},
    'selective': {'title': '独立子任务按需委派', 'instruction':
                  '本次执行中，若同时需要独立获取订单事实和政策资料，优先一次委派 orders 与 knowledge 两个子任务，'
                  '主管汇总证据。单一查询或只需补充信息时直接处理。保持原有事实核对、缺失信息询问和审批规则。'},
}
CASES = [
    {'id': 'simple_order', 'title': '简单订单查询', 'prompt': '请查询 O1003 当前订单状态。',
     'orders': ['O1003'], 'origin': 'synthetic_control'},
    {'id': 'order_policy', 'title': '订单与政策综合查询',
     'prompt': '帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。',
     'orders': ['O1003'], 'knowledge': True, 'origin': 'reported_failure_controlled_reproduction'},
    {'id': 'history_discovery', 'title': '已有历史后重新发现工具',
     'prompt': '帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。',
     'orders': ['O1003'], 'knowledge': True, 'seed_history': True,
     'origin': 'reported_failure_controlled_reproduction'},
    {'id': 'missing_reason', 'title': '缺少退货原因时澄清',
     'prompt': '我要退 O1003，还没有提供具体退货原因，请先告诉我需要补充什么。',
     'clarification': 'reason', 'origin': 'synthetic_guardrail'},
    {'id': 'missing_order', 'title': '缺少订单号时澄清',
     'prompt': '帮我查订单状态，我还没提供订单号，请先问我需要的信息。',
     'clarification': 'order', 'origin': 'synthetic_guardrail'},
    {'id': 'tool_transient', 'title': '订单工具首次连接失败',
     'prompt': '请查询 O1003 当前订单状态。', 'orders': ['O1003'],
     'fault': 'tool', 'origin': 'synthetic_fault_injection'},
]


def scripted_case(case, strategy):
    """Script differences verify both paths, never constitute strategy evidence."""
    if case.get('clarification'):
        text = '请提供具体退货原因，必要时补充商品情况；目前未生成预览。' if case['clarification'] == 'reason' else '请提供订单号。'
        return {'script': [{'text': text}]}
    answer = 'O1003 已签收。'
    if case.get('knowledge'):
        answer += '示例退货政策为签收后7天内申请，需人工审核。退款在审核通过后处理，到账时间待确认。请补充具体退货原因、商品情况和签收时间。'
    if strategy == 'selective' and case.get('knowledge'):
        delegation, specialists = delegation_script(), specialist_scripts()
        delegation['calls'][0]['function']['arguments']['tasks'][1]['task'] = '检索退货与退款政策，说明适用条件和缺失信息'
        specialists['knowledge'][0] = tools(('policy', 'search_knowledge', {'query': '退货 退款 政策'}))
        specialists['knowledge'][1]['summary'] = answer.split('。', 1)[1]
        return {'script': [delegation, {'text': answer}], 'specialists': specialists}
    names = ['get_order'] + (['search_knowledge'] if case.get('knowledge') else [])
    calls = [('order', 'get_order', {'order_id': 'O1003'})]
    if case.get('knowledge'):
        calls.append(('policy', 'search_knowledge', {'query': '退货 退款 政策'}))
    return {'script': [tools(('discover', 'discover_tools', {'names': names})), tools(*calls), {'text': answer}]}
