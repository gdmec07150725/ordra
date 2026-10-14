"""Load tool definitions per execution; selection never grants new capabilities."""
from copy import deepcopy
from hashlib import sha256
import json


DISCOVERY_TOOL = {'type': 'function', 'function': {
    'name': 'discover_tools',
    'description': '按当前任务从工具目录选择需要的工具，加载其参数定义；必须单独调用，下一次请求再使用所选工具。',
    'parameters': {'type': 'object', 'properties': {
        'names': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 5},
    }, 'required': ['names'], 'additionalProperties': False},
}}
CORE_NAMES = frozenset({'update_plan', 'delegate_tasks', 'load_skill', 'read_skill_resource'})
MAX_DISCOVERIES = 3


def discovery_allowance(state):
    if not state or state.get('mode') != 'mcp':
        return 0
    calls = state.get('calls', 0)
    return min(MAX_DISCOVERIES, max(0, calls)) if type(calls) is int else 0


def definition_hash(tool):
    return sha256(json.dumps(tool, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ToolDiscovery:
    def __init__(self, tools, *, remote_names=(), state=None):
        self.tools = {tool['function']['name']: deepcopy(tool) for tool in tools}
        self.remote_names = frozenset(remote_names) & self.tools.keys()
        self.loaded, self.specialist, self.calls, self.events = set(), set(), 0, []
        if state is not None:
            if state.get('mode') != 'mcp' or state.get('version') != 1:
                raise ValueError('工具发现检查点不兼容。')
            if type(state.get('calls')) is not int or not 0 <= state['calls'] <= MAX_DISCOVERIES:
                raise ValueError('工具发现检查点次数无效。')
            self.calls = state['calls']
            self.events = deepcopy(state.get('events', []))
            for field, target in (('loaded', self.loaded), ('specialist', self.specialist)):
                for name, fingerprint in state.get(field, {}).items():
                    if name not in self.tools or fingerprint != definition_hash(self.tools[name]):
                        raise ValueError('已加载工具定义发生变化，不能继续使用旧参数；请放弃旧执行后重新提问。')
                    target.add(name)

    def bind_specialist(self, names):
        names = set(names)
        if not names <= self.remote_names:
            raise ValueError('子 Agent 工具必须属于已注册的 MCP 职责工具。')
        self.specialist.update(names)

    def request_tools(self):
        remaining = sorted(self.tools.keys() - CORE_NAMES - self.loaded)
        discovery = []
        if remaining and self.calls < MAX_DISCOVERIES:
            definition = deepcopy(DISCOVERY_TOOL)
            definition['function']['parameters']['properties']['names']['items']['enum'] = remaining
            discovery = [definition]
        return deepcopy([tool for name, tool in self.tools.items() if name in CORE_NAMES] +
                        discovery + [tool for name, tool in self.tools.items() if name in self.loaded])

    def state_message(self, plan=None, *, compact=False):
        current = next((step for step in (plan or {}).get('steps', []) if step['status'] == 'in_progress'), None)
        missing = sorted(set(current['tools']) - CORE_NAMES - self.loaded) if current else []
        available = sorted(tool['function']['name'] for tool in self.request_tools())
        state = {'available_tools': available,
                 'single_call_tools': sorted(set(available) & (CORE_NAMES | {'discover_tools'})),
                 'plan_status': (plan or {}).get('status', 'none'),
                 'current_step_tools': sorted(current['tools']) if current else [],
                 'loaded': sorted(self.loaded), 'missing_for_current_step': missing,
                 'discoveries_left': max(0, MAX_DISCOVERIES - self.calls)}
        if compact:
            state = {key: state[key] for key in ('available_tools', 'plan_status', 'current_step_tools')}
            return {'role': 'system', 'content':
                    '[本轮工具状态] 仅调用 available_tools；控制调用各自单独；计划执行中业务调用限 current_step_tools。\n' +
                    json.dumps(state, ensure_ascii=False, separators=(',', ':'))}
        return {'role': 'system', 'content':
                '[本轮工具状态] 本次 tool_calls 只能使用 available_tools；目录、计划或历史中出现的名字不代表可调用。'
                '需要尚未加载的工具时先单独 discover_tools，等下一次请求获得定义后调用；loaded 不重复发现。'
                'single_call_tools 每次只能选一个且不能与其他调用同批。计划执行中业务调用限 current_step_tools；无计划可直接处理或追问。\n' +
                json.dumps(state, ensure_ascii=False, separators=(',', ':'))}

    def permits(self, name):
        return name == 'discover_tools' or name in self.tools and (name in CORE_NAMES or name in self.loaded)

    def instructions(self):
        catalog = '\n'.join(f"{name} ({'MCP' if name in self.remote_names else '本地'}): " +
                            tool['function'].get('description', '')[:90]
                            for name, tool in self.tools.items() if name not in CORE_NAMES)
        return ('\n工具目录（仅能力）：先单独 discover_tools，names 填所需精确名称；下轮获得参数定义后调用。'
                '每轮最多发现3次，可批量选名。计划可直接引用目录名；发现不作业务步骤或证据。'
                '核心工具已加载，子 Agent 由服务器配职责工具。\n' + catalog + '\n')

    def discover(self, arguments):
        if self.calls >= MAX_DISCOVERIES:
            return {'error': '本轮工具发现已达3次上限，请使用已加载工具或说明缺少的能力。', 'error_code': 'discovery_limit'}
        self.calls += 1
        try:
            value = json.loads(arguments)
            if not isinstance(value, dict) or set(value) != {'names'}:
                raise ValueError('参数必须且只能包含 names。')
            names = value['names']
            if not isinstance(names, list) or not 1 <= len(names) <= 5 or any(type(name) is not str for name in names):
                raise ValueError('names 必须是1至5个工具名称。')
            if len(set(names)) != len(names) or any(name not in self.tools or name in CORE_NAMES for name in names):
                raise ValueError('只能选择目录中的业务工具，名称不能重复。')
        except (ValueError, TypeError) as error:
            result = {'error': f'工具发现参数无效：{error}', 'error_code': 'invalid_discovery'}
        else:
            added = sorted(set(names) - self.loaded)
            self.loaded.update(names)
            result = {'status': 'loaded' if added else 'already_loaded', 'newly_loaded': added, 'tools': [
                {'name': name, 'source': 'mcp' if name in self.remote_names else 'local'} for name in names],
                'message': '这些工具的参数定义已加载，后续请求可直接调用，不要重复发现；加载本身没有查询或执行业务。'}
        self.events.append({'sequence': self.calls, 'loaded_names': sorted(self.loaded), **deepcopy(result)})
        return result

    def snapshot(self):
        return {'version': 1, 'mode': 'mcp', 'calls': self.calls,
                'loaded': {name: definition_hash(self.tools[name]) for name in sorted(self.loaded)},
                'specialist': {name: definition_hash(self.tools[name]) for name in sorted(self.specialist)},
                'events': deepcopy(self.events)}
