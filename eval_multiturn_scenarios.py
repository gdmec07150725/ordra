"""Scripted response fixtures are labelled separately from real tool/outcome evidence."""
from eval_scenarios import tools, delegation_script, specialist_scripts


def order_turn(prompt, order_id, *, fault=None):
    status = '已发货' if order_id == 'O1001' else '已签收'
    return {'prompt': prompt, 'orders': [order_id], 'fault': fault,
            'script': [tools(('order', 'get_order', {'order_id': order_id})), {'text': f'{order_id} {status}。'}]}


MULTITURN_CASES = [
    {'id': 'mt_followup', 'title': '连续追问与订单指代', 'turns': [
        order_turn('请查询 O1003 当前订单状态。', 'O1003'),
        order_turn('这笔订单现在的物流状态呢？请再查询一下。', 'O1003'),
        {'prompt': '请重新查询这笔订单当前状态，再查阅知识库，告诉我可参考的示例退货期限。', 'orders': ['O1003'], 'knowledge': True,
         'script': [tools(('policy', 'search_knowledge', {'query': '退货期限'}), ('order', 'get_order', {'order_id': 'O1003'})),
                    {'text': 'O1003 已签收；示例政策为签收后7天内申请退货，需人工核实。'}]},
    ]},
    {'id': 'mt_correction', 'title': '明确更正后保持新订单指代', 'turns': [
        order_turn('请查询 O1001 当前订单状态。', 'O1001'),
        order_turn('刚才订单号说错了，更正为 O1003。请查询这笔订单。', 'O1003'),
        order_turn('那这笔更正后的订单现在是什么状态？请重新确认。', 'O1003'),
    ]},
    {'id': 'mt_long_history', 'title': '长历史摘要后连续委派与刷新（压力夹具）',
     'fixture': {'kind': 'seeded_long_history', 'seeded_turns': 16,
                 'description': '在临时数据库预置16轮合成背景对话以触发摘要；这些回合未经过模型，后续3轮通过真实本地API。'},
     'summary': True, 'summary_text': '用户当前查询订单为 O1003；背景回合仅为压力夹具。状态必须重新查询；退货依据示例政策。',
     'turns': [
         order_turn('请继续查询之前提到的那笔订单当前状态。', 'O1003'),
         {'prompt': '请使用任务委派功能，让订单查询 Agent 和知识检索 Agent 分别核实这笔订单状态及示例退货期限，再汇总。',
          'orders': ['O1003'], 'knowledge': True, 'delegation': True,
          'script': [delegation_script(), {'text': 'O1003 已签收；示例政策为签收后7天内申请退货，需人工核实。'}],
          'specialists': specialist_scripts()},
         order_turn('最后再查询这笔订单的最新状态。', 'O1003'),
     ]},
    {'id': 'mt_recovery', 'title': '多轮中模型故障恢复与工具不重放',
     'fixture': {'kind': 'model_connection_failure', 'seeded_turns': 0,
                 'description': '第二轮完成工具调用后的下一次主管模型请求注入一次连接故障，然后恢复同一run。'},
     'turns': [
         order_turn('请查询 O1003 当前订单状态。', 'O1003'),
         order_turn('请再查这笔订单现在的状态。', 'O1003', fault='model'),
         order_turn('继续确认刚才那笔订单当前状态。', 'O1003'),
     ]},
]


def seeded_history(case, prefix):
    history = list(prefix)
    for index in range(case.get('fixture', {}).get('seeded_turns', 0)):
        history.extend([
            {'role': 'user', 'content': f'压力夹具背景回合 {index + 1}：我当前查询的订单是 O1003。' + '这是合成背景说明，不包含新的请求或订单号。' * 20},
            {'role': 'assistant', 'content': '已记录当前订单为 O1003；订单状态需要重新使用工具核实。' + '背景信息已记录。' * 10},
        ])
    return history
