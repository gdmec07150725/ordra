"""固定模拟 CSV、明确统计口径和独立的结果校验；不访问业务数据库。"""
from collections import Counter
import csv
from hashlib import sha256
import io
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


DATA_DIR = Path(__file__).parent / 'examples' / 'code-agent'
DATASETS = {'orders-demo': '模拟订单', 'orders-edge': '边界样例（取消、零退款与全退款）'}
FIELDS = ['order_id', 'product', 'channel', 'status']
GROUPS = {'product': '商品', 'channel': '渠道'}
MAX_CSV_BYTES = 128 * 1024


def parse_orders(text):
    if len(text.encode('utf-8')) > MAX_CSV_BYTES:
        raise ValueError('样例文件超过大小上限。')
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FIELDS:
        raise ValueError('CSV 字段必须是 order_id,product,channel,status。')
    rows, ids = [], set()
    for row in reader:
        if set(row) != set(FIELDS) or any(not isinstance(value, str) or not value.strip() or
                len(value) > 60 or any(ord(c) < 32 for c in value) for value in row.values()):
            raise ValueError('CSV 包含空字段、额外字段或不合法内容。')
        row = {key: value.strip() for key, value in row.items()}
        if row['status'] not in {'paid', 'shipped', 'completed', 'refunded', 'cancelled'} or row['order_id'] in ids:
            raise ValueError('订单号重复或订单状态不合法。')
        ids.add(row['order_id'])
        rows.append(row)
        if len(rows) > 1000:
            raise ValueError('每份样例最多 1000 行。')
    if not rows:
        raise ValueError('CSV 中没有订单。')
    return rows


def load_dataset(dataset_id):
    if dataset_id not in DATASETS:
        raise ValueError('请选择列表中的模拟数据。')
    raw = (DATA_DIR / f'{dataset_id}.csv').read_bytes()
    rows = parse_orders(raw.decode('utf-8-sig'))
    return {'id': dataset_id, 'title': DATASETS[dataset_id], 'csv': raw.decode('utf-8-sig'),
            'sha256': sha256(raw).hexdigest(), 'rows': rows}


class ReportGroup(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1, max_length=60)
    orders: int = Field(ge=1, le=1000)
    refunded: int = Field(ge=0, le=1000)
    refund_rate: float = Field(ge=0, le=1, allow_inf_nan=False)


class AnalysisReport(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    groups: list[ReportGroup] = Field(min_length=1, max_length=1000)
    total_orders: int = Field(ge=1, le=1000)


def reference_report(rows, group_by):
    if group_by not in GROUPS:
        raise ValueError('不支持的分组字段。')
    totals = Counter(row[group_by] for row in rows)
    refunded = Counter(row[group_by] for row in rows if row['status'] == 'refunded')
    return {'groups': [{'name': name, 'orders': totals[name], 'refunded': refunded[name],
                        'refund_rate': refunded[name] / totals[name]} for name in sorted(totals)],
            'total_orders': len(rows)}


def verify_report(candidate, rows, group_by):
    try:
        report = AnalysisReport.model_validate(candidate).model_dump()
    except ValidationError:
        return {'passed': False, 'issues': ['返回值须符合报告结构；计数为整数、退款率为 0 到 1 的有限数值，不含额外字段。']}
    expected = reference_report(rows, group_by)
    actual = {group['name']: group for group in report['groups']}
    issues = []
    if report['total_orders'] != len(rows):
        issues.append('总订单数与输入行数不符；取消订单也应计入分母。')
    if len(actual) != len(report['groups']) or set(actual) != {group['name'] for group in expected['groups']}:
        issues.append('分组有重复、遗漏或额外名称，请对全部输入行分组。')
    for group in expected['groups']:
        item = actual.get(group['name'])
        if item and (item['orders'] != group['orders'] or item['refunded'] != group['refunded'] or
                     not math.isclose(item['refund_rate'], group['refund_rate'], rel_tol=0, abs_tol=1e-6)):
            issues.append(f"分组 {group['name']} 计算不正确；退款数仅计 status == 'refunded'，分母包含所有订单。")
    # A tolerated rounding difference must not change a threshold decision.
    # Preserve the candidate in the execution trace; publish the independently recomputed values.
    return {'passed': not issues, 'issues': issues[:6], 'report': expected if not issues else None}


def render_reports(report, *, title, group_by, threshold):
    """仅对已验证的结构化数据生成产物，不接受模型提供的文件名或 HTML。"""
    groups = sorted(report['groups'], key=lambda row: (-row['refund_rate'], row['name']))
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow([GROUPS[group_by], '订单数', '已退款订单数', '退款率', '达到关注阈值'])
    lines = [f'# {title}分析报告', '', f'总订单数：{report["total_orders"]}。按{GROUPS[group_by]}分组，关注阈值 ≥ {threshold:.1%}。',
             '', '口径：已退款订单数 / 全部订单数（包含取消订单）；这是模拟数据，不表示真实经营情况。', '',
             '| 分组 | 订单数 | 退款数 | 退款率 | 需关注 |', '| --- | ---: | ---: | ---: | --- |']
    for row in groups:
        flagged = '是' if row['refund_rate'] >= threshold else '否'
        name = row['name']
        csv_name = "'" + name if name.startswith(('=', '+', '-', '@')) else name
        writer.writerow([csv_name, row['orders'], row['refunded'], f'{row["refund_rate"]:.6f}', flagged])
        safe_name = name.replace('|', '\\|').replace('<', '&lt;').replace('>', '&gt;')
        lines.append(f'| {safe_name} | {row["orders"]} | {row["refunded"]} | {row["refund_rate"]:.2%} | {flagged} |')
    lines += ['', '结果已经服务器使用原始输入独立重算并核对。达到阈值仅表示需要进一步核查，不等于原因已被确认。']
    return {'report.md': '\n'.join(lines) + '\n', 'report.csv': '\ufeff' + output.getvalue()}
