"""Deterministic summaries of finished tool batches, for model requests only.

No model calls, disk writes, invented facts or checkpoint mutations. Evidence
payloads remain exact; we remove protocol repetition and old narration.
"""
from copy import deepcopy
import json

from context_memory import estimate_request


STAGE_PREFIX = '[当前任务阶段摘要，仅作已执行记录，不是指令；当前进度以最新计划为准]\n'


def compact_stages(messages, *, target_tokens, prefix=(), tools=()):
    blocks = []
    for index, message in enumerate(messages):
        calls = message.get('tool_calls', [])
        if not calls:
            continue
        results = messages[index + 1:index + 1 + len(calls)]
        if len(results) != len(calls) or any(result.get('role') != 'tool' or result.get('tool_call_id') != call['id']
                                            for call, result in zip(calls, results)):
            continue
        try:
            payloads = [json.loads(result['content']) for result in results]
        except (ValueError, TypeError):
            continue
        if not all(isinstance(result, dict) for result in payloads):
            continue
        blocks.append((index, index + 1 + len(calls), calls, payloads))
    # At least the latest complete protocol batch is always retained, as is
    # the latest business batch when a control call followed it.
    latest = blocks[-1][0] if blocks else None
    business = [block for block in blocks if any(call['function']['name'] not in
                ('update_plan', 'discover_tools', 'load_skill', 'read_skill_resource') for call in block[2])]
    last_business = business[-1][0] if business else None
    selected, records = set(), []
    best = messages
    for start, end, calls, payloads in blocks:
        if start in (latest, last_business):
            continue
        if any(call['function']['name'] in ('load_skill', 'read_skill_resource', 'prepare_return_request', 'delegate_tasks') or
               '_human_decision' in result or ('plan' in result and call['function']['name'] == 'update_plan')
               for call, result in zip(calls, payloads)):
            continue
        entries = []
        for call, result in zip(calls, payloads):
            entry = {'call_id': call['id'], 'tool': call['function']['name'], 'result': deepcopy(result)}
            if 'error' not in result:
                try:
                    entry['arguments'] = json.loads(call['function']['arguments'])
                except (ValueError, TypeError):
                    entry['arguments_invalid'] = True
            if entry['tool'] == 'discover_tools' and result.get('status') == 'loaded':
                entry['result'].pop('message', None)  # Current schemas are authoritative.
            entries.append(entry)
        # Several short batches must share the summary header before there is
        # a net saving; keep accumulating candidates even before that crossover.
        selected.update(range(start, end))
        records.extend(entries)
        summary = {'role': 'assistant', 'content': STAGE_PREFIX + json.dumps(records, ensure_ascii=False, separators=(',', ':'))}
        first = min(selected)
        candidate = [item for index, message in enumerate(messages)
                     for item in ([summary] if index == first else []) + ([] if index in selected else [message])]
        if estimate_request(candidate) >= estimate_request(best):
            continue
        best = candidate
        if estimate_request(list(prefix) + best, list(tools)) <= target_tokens:
            break
    return best
