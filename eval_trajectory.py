"""Evidence-based conversation constraints, not a general semantic-quality judge."""
from collections import Counter
import json
import unicodedata

from agent_service import history_arguments, knowledge_arguments
from return_tools import validate_return_arguments


def criterion(key, name, passed, actual):
    return {'id': key, 'name': name, 'passed': bool(passed), 'actual': actual}


def score_checks(checks):
    passed = sum(check['passed'] for check in checks)
    return {'score': round(100 * passed / len(checks), 1) if checks else None,
            'passed': bool(checks) and passed == len(checks),
            'satisfied': passed, 'total': len(checks),
            'failed_criteria': [check.get('id', check['name']) for check in checks if not check['passed']],
            'checks': checks}


def tool_results(trace):
    return [event['data'] for event in (trace or {}).get('events', []) if event['kind'] == 'tool_result']


def normalized_arguments(arguments):
    def normalize(value):
        if isinstance(value, str):
            return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return json.dumps(normalize(arguments), ensure_ascii=False, sort_keys=True)


def order_argument(call):
    arguments = call.get('arguments')
    value = arguments.get('order_id') if isinstance(arguments, dict) else None
    return value.strip().upper() if isinstance(value, str) else None


def valid_arguments(call):
    """Reuse pure runtime parsers, so network/business errors are not parameter errors."""
    name, arguments = call.get('name'), call.get('arguments')
    try:
        if name == 'search_knowledge':
            knowledge_arguments(json.dumps(arguments, ensure_ascii=False))
        elif name == 'search_history':
            history_arguments(json.dumps(arguments, ensure_ascii=False))
        elif name == 'get_order':
            return isinstance(arguments, dict) and set(arguments) == {'order_id'} and bool(order_argument(call))
        elif name in ('prepare_return_request', 'get_return_request'):
            validate_return_arguments(name, arguments)
        elif name == 'update_plan':
            from agent_plan import PlanUpdate
            PlanUpdate.model_validate(arguments)
        elif name == 'delegate_tasks':
            from subagent_runtime import new_delegation_state, register_tasks
            register_tasks(new_delegation_state(), json.dumps(arguments, ensure_ascii=False), {'orders', 'knowledge'})
        else:
            return False
    except (ValueError, TypeError, KeyError):
        return False
    return True


def search_key(call):
    arguments = call.get('arguments', {})
    if call.get('name') == 'search_history' and isinstance(arguments, dict):
        arguments = {'order': 'relevance', **arguments}
    return call.get('name'), normalized_arguments(arguments)


def grade_trajectory(turn, trace):
    """Grade a single turn; tool order is unrestricted and refresh across turns is allowed."""
    checks = []
    calls = tool_results(trace)
    successful = [c for c in calls if c.get('status') == 'ok' and c.get('attempts', 0) > 0]
    checks.append(criterion('completed', '本轮执行已完成', (trace or {}).get('status') == 'completed', (trace or {}).get('status')))
    invalid = []
    for call in calls:
        result = call.get('result') or {}
        # Plan updates validate only local control state. Their plain error
        # result is the authoritative structural/dependency/evidence rejection;
        # transient failures have runtime error codes and are not this signal.
        rejected_plan = (call.get('name') == 'update_plan' and call.get('status') == 'error' and result.get('error')
                         and result.get('error_code') in (None, 'invalid_plan_tools'))
        if (not valid_arguments(call) or rejected_plan or call.get('status') in ('invalid_arguments', 'invalid') or
                result.get('error_code') in ('invalid_arguments', 'invalid_json', 'unknown_tool')):
            invalid.append(call.get('call_id'))
    expected_orders = set(turn.get('orders', []))
    if 'orders' in turn:
        invalid += [c.get('call_id') for c in calls if c.get('name') in ('get_order', 'get_return_request', 'prepare_return_request') and order_argument(c) not in expected_orders]
    checks.append(criterion('valid_tool_arguments', '工具参数符合本轮目标', not invalid, invalid))
    orders = [c for c in successful if c.get('name') == 'get_order']
    if expected_orders:
        evidence = {order_argument(c) for c in orders if order_argument(c) and
                    (c.get('result') or {}).get('order_id') == order_argument(c)}
        checks.append(criterion('order_evidence', '指定订单均有实际成功查询证据', expected_orders.issubset(evidence), sorted(evidence)))
    if turn.get('knowledge'):
        evidence = [c.get('call_id') for c in successful if c.get('name') == 'search_knowledge' and
                    (c.get('result') or {}).get('returned_items', 0) > 0]
        checks.append(criterion('knowledge_evidence', '知识检索实际返回片段', bool(evidence), evidence))
    # One runtime retry is a single logical call. Different calls with equivalent
    # searches in this turn are repeated work, including actor/specialist overlap.
    searches = Counter(search_key(c) for c in calls
                       if c.get('name') in ('search_knowledge', 'search_history'))
    repeated = [{'name': name, 'arguments': json.loads(arguments), 'calls': count}
                for (name, arguments), count in searches.items() if count > 1]
    checks.append(criterion('no_repeated_searches', '同轮没有重复的等价检索', not repeated, repeated))
    if turn.get('delegation'):
        tasks = [event['data'] for event in (trace or {}).get('events', []) if event['kind'] == 'specialist_done']
        roles = {task.get('agent'): task.get('status') for task in tasks}
        checks.append(criterion('delegation_evidence', '订单与知识子任务均实际完成',
                                roles.get('orders') == roles.get('knowledge') == 'completed', roles))
    if turn.get('fault') == 'model':
        counts = Counter((c.get('name'), normalized_arguments(c.get('arguments', {}))) for c in calls
                         if c.get('name') not in ('update_plan', 'delegate_tasks'))
        repeated_tools = [name for (name, _), count in counts.items() if count > 1]
        checks.append(criterion('no_replayed_tools', '恢复没有重新执行已保存的工具', not repeated_tools, repeated_tools))
    return score_checks(checks)


def summarize_metrics(traces, *, simulated=False):
    """Token sums include only reported usage; unknown pricing/usage stays unknown."""
    keys = ('model_requests', 'summary_requests', 'verification_requests', 'subagent_requests',
            'reported_requests', 'tool_calls', 'tool_attempts', 'tool_retries', 'corrections')
    result = {key: sum((trace or {}).get('metrics', {}).get(key, 0) or 0 for trace in traces) for key in keys}
    result['request_count'] = sum(result[key] for key in ('model_requests', 'summary_requests', 'verification_requests', 'subagent_requests'))
    reported = result['reported_requests']
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        result[key] = sum((trace or {}).get('metrics', {}).get(key, 0) or 0 for trace in traces) if reported else None
    result['usage_status'] = 'unreported' if not reported else 'complete' if reported == result['request_count'] else 'partial'
    result['usage_simulated'] = simulated
    result['estimated_cost'] = None
    result['cost_status'] = 'scripted_usage_not_billable' if simulated else 'price_not_configured'
    return result
