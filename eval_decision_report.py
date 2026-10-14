"""Standalone, escaped experiment report; no automatic promotion or winner."""
from html import escape
import json


def pretty(value):
    return '<pre>' + escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'


def report_html(report):
    simulated = report['mode'] == 'offline'
    mode = '离线流程验收：决策、校验与 tokens 均为脚本，不能据此比较策略效果。' if simulated else '真实模型实验：规则通过率不代表完整语义质量，小样本不足以断言某策略更好。'
    rows = []
    for row in report['summary']:
        compliance = '自由选择' if row['strategy_compliance_rate'] is None else f"{row['strategy_compliance_rate']:.0%}"
        usage = '模拟' if simulated else str(row['mean_total_tokens']) if row['mean_total_tokens'] is not None else '覆盖不完整，未计算'
        cells = [report['strategies'][row['strategy']]['title'], f"{row['passed']}/{row['trials']} ({row['success_rate']:.0%})", compliance,
                 row['mean_requests'], usage, f"{row['complete_usage_trials']}/{row['trials']}", row['mean_elapsed_ms'],
                 str(row['invalid_calls']) + ('（轨迹计数，可能少计）' if row.get('invalid_call_coverage') == 'partial' else ''), row['repeated_calls'], row['control_calls']]
        rows.append('<tr>' + ''.join('<td>' + escape(str(c)) + '</td>' for c in cells) + '</tr>')
    details = []
    for item in report['results']:
        title = f"{item['case_id']} · {item['strategy']} · 第 {item['repeat']} 次 · {'通过' if item['passed'] else '失败'}"
        details.append('<details><summary>' + escape(title) + '</summary><p>' + escape(item.get('answer', '')) + '</p>' + pretty(item) + '</details>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent 执行策略实验</title><style>body{{max-width:1280px;margin:36px auto;padding:0 20px;font:15px/1.7 system-ui;background:#f7f9f5;color:#243e33}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce4db}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 monospace}}details{{margin:14px 0}}summary{{cursor:pointer}}.table{{overflow:auto}}</style>
<h1>Agent 执行策略实验</h1><p>{escape(mode)}</p><p>已记录 {report['completed_trials']} / {report['total']} 次；通过 {report['passed']} 次。{'实验完成' if report['complete'] else '实验仍在进行，以下是部分结果'}。</p>
<p>{escape('环境运行异常：' + report['lifecycle_error']) if report.get('lifecycle_error') else ''}</p>
<p>{escape(report['promotion'])}</p><p>{escape(report['method'])}</p>
{('<details><summary>本报告已统一重新评分，展开查看评分版本与变化</summary>' + pretty(report['analysis']) + '</details>') if report.get('analysis') else ''}
<div class="table"><table><thead><tr><th>策略</th><th>任务通过</th><th>策略遵循</th><th>平均全部模型请求</th><th>平均 tokens</th><th>完整用量覆盖</th><th>平均 ms</th><th>无效调用</th><th>重复调用</th><th>控制调用</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>重复调用按同一轮同名同参数统计，不含执行计划更新；这是效率提示，合理复查也可能计入。无效调用指参数、发现或控制协议拒绝；业务暂时故障单独留在执行轨迹中。耗时包含模型和工具调用，不包含环境启动；未执行项记为失败、耗时未知，并停止计算该组平均效率。</p>
<details><summary>每个案例的重复一致性</summary>{pretty([{k: r[k] for k in ('strategy', 'cases')} for r in report['summary']])}</details>
<details><summary>固定案例、策略、配置与版本指纹</summary>{pretty({k: report[k] for k in ('dataset', 'strategies', 'environment', 'schedule')})}</details>
<h2>逐次执行证据</h2>{''.join(details)}</html>'''
