"""Request estimates only: never retain prompts, retrieved text or credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
import json


_observer = ContextVar('model_request_observer', default=None)


def request_budget(messages, tools, policy, *, role, output_tokens=None,
                   before_tokens=None, history_end=0, status=None, agent=None):
    # Local import avoids a cycle: RollingContext also reports summary requests.
    from context_memory import estimate_request, SUMMARY_PREFIX
    components = dict.fromkeys(('system', 'tools', 'history_summary', 'recent_messages',
                               'current_messages', 'tool_results', 'skills', 'request_data', 'format'), 0)
    skill_calls = {call['id'] for message in messages for call in message.get('tool_calls', [])
                   if call.get('function', {}).get('name') in ('load_skill', 'read_skill_resource')}
    components['format'] = estimate_request([])
    previous = components['format']
    encoded_bytes = len(b'{"messages":[]}')
    for index, message in enumerate(messages):
        if message.get('role') == 'system':
            key = 'system'
        elif message.get('role') == 'tool':
            key = 'skills' if message.get('tool_call_id') in skill_calls else 'tool_results'
        elif (message.get('role') == 'assistant' and
              (message.get('content') or '').startswith(SUMMARY_PREFIX)):
            key = 'history_summary'
        elif role in ('summary', 'verification'):
            key = 'request_data'  # Their JSON payload contains nested history/evidence.
        else:
            key = 'recent_messages' if index < history_end else 'current_messages'
        encoded_bytes += len(json.dumps(message, ensure_ascii=False, separators=(',', ':')).encode('utf8')) + bool(index)
        total = (encoded_bytes + 1) // 2 + 64
        components[key] += total - previous
        previous = total
    estimated = estimate_request(messages, tools)
    components['tools'] = estimated - previous
    output = policy.output_tokens if output_tokens is None else output_tokens
    limit = policy.max_tokens - output - policy.safety_tokens
    before = estimated if before_tokens is None else before_tokens
    return {'role': role, **({'agent': agent} if agent is not None else {}),
            'status': status or ('blocked' if estimated > limit else 'prepared'),
            'estimator': 'utf8_bytes_conservative', 'estimated_input_tokens': estimated,
            'input_limit': limit, 'max_tokens': policy.max_tokens, 'output_reserve': output,
            'safety_tokens': policy.safety_tokens, 'before_estimated_tokens': before,
            'saved_tokens': max(0, before - estimated), 'components': components,
            'request_messages': len(messages),
            'tool_names': [tool['function']['name'] for tool in tools or []]}


@contextmanager
def request_observer(callback):
    """Per graph invocation; ContextVar propagates to concurrent child tasks."""
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


async def observe_request(messages, tools, policy, **kwargs):
    callback = _observer.get()
    if callback is not None:
        await callback(request_budget(messages, tools, policy, **kwargs))
