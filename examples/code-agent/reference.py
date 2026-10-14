# 离线评测使用的固定代码；真实模型模式不会读取此文件。
groups = {}
for row in rows:
    name = row[group_by]
    if name not in groups:
        groups[name] = {'name': name, 'orders': 0, 'refunded': 0, 'refund_rate': 0.0}
    groups[name]['orders'] += 1
    if row['status'] == 'refunded':
        groups[name]['refunded'] += 1
for group in groups.values():
    group['refund_rate'] = group['refunded'] / group['orders']
{'groups': list(groups.values()), 'total_orders': len(rows)}
