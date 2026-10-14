"""Escape every data field in the standalone multi-turn report."""
from html import escape
import json


def pretty(value):
    return '<pre>' + escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'


def checks_html(checks):
    return '<ul>' + ''.join('<li>' + ('通过' if c['passed'] else '失败') + ' · ' + escape(c['name']) + pretty(c['actual']) + '</li>' for c in checks) + '</ul>'


def usage_html(metrics):
    reported = metrics['reported_requests']
    counts = '主管 {model_requests} · 摘要 {summary_requests} · 子任务 {subagent_requests} · 校验 {verification_requests}'.format(**metrics)
    usage = '未报告' if metrics['total_tokens'] is None else str(metrics['total_tokens']) + ' tokens'
    return '<p>' + escape(counts) + '</p><p>' + escape(f"服务端用量覆盖：{reported}/{metrics['request_count']} 次请求 · {usage} · {metrics['usage_status']}") + '</p>'


def budgets_html(budgets):
    if not budgets:
        return '<p>旧记录没有请求预算观测。</p>'
    labels = {'system': '系统指令', 'tools': '工具定义', 'tool_schemas': '工具定义', 'history_summary': '历史摘要',
              'recent_messages': '近期消息', 'current_messages': '本轮消息', 'tool_results': '工具结果',
              'request_data': '请求数据整体', 'format': '格式开销'}
    roles = {'actor': '主管', 'specialist': '子任务', 'summary': '摘要', 'verification': '校验'}
    rows = []
    for budget in budgets:
        role = roles.get(budget.get('role'), budget.get('role', '未知'))
        if budget.get('agent'):
            role += ' / ' + {'orders': '订单', 'knowledge': '知识'}.get(budget['agent'], str(budget['agent']))
        parts = '<br>'.join(escape(labels.get(key, key)) + '：' + escape(str(value)) for key, value in budget.get('components', {}).items())
        cells = [budget.get('sequence', '?'), role, budget.get('status', '未知'),
                 str(budget.get('estimated_input_tokens', '?')) + ' / ' + str(budget.get('input_limit', '?')),
                 str(budget.get('before_estimated_tokens', '?')) + ' → ' + str(budget.get('estimated_input_tokens', '?')) + '（节省 ' + str(budget.get('saved_tokens', '?')) + '）']
        rows.append('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in cells) + '<td>' + parts + '</td></tr>')
    return '<table><thead><tr><th>序号</th><th>角色</th><th>状态</th><th>输入 / 上限</th><th>整理收益</th><th>估算组成</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'


def multiturn_rows(result):
    metrics = result['metrics']
    total = metrics['total_tokens'] if metrics['total_tokens'] is not None else '未报告'
    rows = [f"<tr><td>{escape(result['title'])}</td><td>{'通过' if result['passed'] else '失败'} · {escape(str(result['trajectory']['score']))}/100</td><td>{escape(str(metrics['request_count']))}</td><td>{escape(str(total))} ({escape(metrics['usage_status'])})</td><td>{escape(str(metrics['tool_attempts']))}</td><td>{escape(str(result['elapsed_ms']))}</td></tr>"]
    details = '<p>trajectory：规则约束满足比例（等权），不代表完整语义质量；离线决策、摘要、校验与 usage 为脚本。</p>'
    details += '<p>' + escape(result['fixture']['description']) + '</p>'
    details += '<p>费用：未估算。离线 usage 为模拟；真实运行没有配置价格，缺失 usage 不按零成本计算。</p>'
    details += '<h3>会话检查</h3>' + checks_html(result['checks'])
    details += '<h3>会话用量与请求数</h3>' + usage_html(metrics)
    for turn in result['turns']:
        details += '<details><summary>第 ' + str(turn['turn']) + ' 轮 · ' + escape(str(turn['trajectory']['score'])) + '/100 · ' + str(turn['elapsed_ms']) + ' ms</summary>'
        details += '<p>' + escape(turn['prompt']) + '</p><p>' + escape(turn['answer']) + '</p>'
        details += checks_html(turn['checks']) + '<h4>本轮用量</h4>' + usage_html(turn['metrics'])
        budgets = [event['data'] for event in (turn['trace'] or {}).get('events', []) if event['kind'] == 'context_budget']
        details += '<h4>context_budget：请求预算（估算值）</h4>'
        details += '<p>prepared 表示准备发送；blocked 表示预算拦截。估算不等于服务端 usage；request_data 是摘要/校验嵌套数据的整体。</p>'
        details += budgets_html(budgets)
        details += '<details><summary>完整工具与执行证据</summary>' + pretty(turn['trace']) + '</details></details>'
    rows.append("<tr><td colspan='6'><details><summary>查看连续对话与评分依据</summary>" + details + '</details></td></tr>')
    return rows
