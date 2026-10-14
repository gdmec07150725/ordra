"""Per-run discovery must actually withhold schemas and enforce its selection."""
from copy import deepcopy
import importlib
import importlib.util
import json
import unittest

from agent_service import TOOLS
from agent_plan import PLAN_TOOL
from knowledge_base import KNOWLEDGE_TOOL
from return_tools import RETURN_TOOLS


class ToolDiscoveryTests(unittest.TestCase):
    def test_current_request_contract_tracks_discovery_resume_and_plan(self):
        selection = self.selection()
        def state(selected, plan=None):
            return json.loads(selected.state_message(plan)['content'].split('\n', 1)[1])
        before = state(selection)
        self.assertEqual(set(before['available_tools']), self.names(selection))
        self.assertNotIn('get_order', before['available_tools'])
        self.assertEqual(before['single_call_tools'], ['discover_tools', 'update_plan'])
        selection.discover('{"names":["get_order","search_knowledge"]}')
        restored = self.selection(selection.snapshot())
        plan = {'status': 'in_progress', 'steps': [
            {'id': 'order', 'status': 'in_progress', 'tools': ['get_order']},
            {'id': 'policy', 'status': 'pending', 'tools': ['search_knowledge']}]}
        active = state(restored, plan)
        self.assertEqual(set(active['available_tools']), self.names(restored))
        self.assertEqual(active['current_step_tools'], ['get_order'])
        self.assertEqual(active['plan_status'], 'in_progress')
        compact = json.loads(restored.state_message(plan, compact=True)['content'].split('\n', 1)[1])
        for key in ('available_tools', 'plan_status', 'current_step_tools'):
            self.assertEqual(compact[key], active[key])
        self.assertFalse(self.selection().permits('get_order'))
        for step in plan['steps']:
            step['status'] = 'completed'
        plan['status'] = 'completed'
        self.assertEqual(state(restored, plan)['current_step_tools'], [])
        restored.discover('{"names":["get_order"]}')
        restored.discover('{"names":["get_order"]}')
        exhausted = state(restored)
        self.assertNotIn('discover_tools', exhausted['available_tools'])
        self.assertNotIn('discover_tools', exhausted['single_call_tools'])

    def test_loaded_tools_are_excluded_from_next_discovery_choices(self):
        selection = self.selection()
        selection.discover('{"names":["search_knowledge"]}')
        definition = next(t for t in selection.request_tools() if t['function']['name'] == 'discover_tools')
        choices = definition['function']['parameters']['properties']['names']['items'].get('enum')
        self.assertEqual(set(choices or []), {'get_order', 'get_return_request', 'prepare_return_request'})
        repeated = selection.discover('{"names":["search_knowledge"]}')
        self.assertEqual(repeated['status'], 'already_loaded')
        self.assertEqual(repeated['newly_loaded'], [])
        self.assertEqual(selection.calls, 2, '重复发现不能绕过调用次数限制')
        selection.discover('{"names":["get_order"]}')
        self.assertNotIn('discover_tools', self.names(selection))
        self.assertIn('get_order', self.names(selection))

    def selection(self, state=None, tools=None):
        self.assertIsNotNone(importlib.util.find_spec('tool_discovery'), '缺少按需工具选择实现')
        return importlib.import_module('tool_discovery').ToolDiscovery(
            tools or TOOLS + [KNOWLEDGE_TOOL] + RETURN_TOOLS + [PLAN_TOOL],
            remote_names={'get_order', 'search_knowledge'}, state=state)

    def names(self, selection):
        return {item['function']['name'] for item in selection.request_tools()}

    def test_definition_is_withheld_until_discovered(self):
        selection = self.selection()
        self.assertEqual(self.names(selection), {'update_plan', 'discover_tools'})
        self.assertFalse(selection.permits('get_order'))
        result = selection.discover('{"names":["get_order"]}')
        self.assertNotIn('error', result)
        self.assertEqual(self.names(selection), {'update_plan', 'discover_tools', 'get_order'})
        self.assertTrue(selection.permits('get_order'))
        self.assertFalse(selection.permits('prepare_return_request'))
        self.assertNotIn('parameters', json.dumps(result))

    def test_loaded_schema_is_exact_server_definition(self):
        remote = deepcopy(TOOLS[0])
        remote['function']['parameters']['properties']['order_id']['maxLength'] = 16
        selection = self.selection(tools=[remote, PLAN_TOOL])
        selection.discover('{"names":["get_order"]}')
        loaded = next(t for t in selection.request_tools() if t['function']['name'] == 'get_order')
        self.assertEqual(loaded['function']['parameters']['properties']['order_id']['maxLength'], 16)
        loaded['function']['parameters']['properties'].clear()
        self.assertIn('order_id', selection.request_tools()[-1]['function']['parameters']['properties'])

    def test_invalid_batch_cannot_partially_load_tools(self):
        for payload in ('{"names":["get_order","delete_all"]}', '{"names":["get_order"],"server":"other"}',
                        '{"names":"get_order"}', '{"names":[]}', '{"names":["update_plan"]}'):
            with self.subTest(payload=payload):
                selection = self.selection()
                self.assertIn('error', selection.discover(payload))
                self.assertFalse(selection.permits('get_order'))

    def test_discovery_budget_and_selection_survive_resume(self):
        selection = self.selection()
        for _ in range(3):
            selection.discover('{"names":["get_order"]}')
        restored = self.selection(json.loads(json.dumps(selection.snapshot())))
        self.assertTrue(restored.permits('get_order'))
        result = restored.discover('{"names":["search_knowledge"]}')
        self.assertEqual(result['error_code'], 'discovery_limit')
        self.assertFalse(restored.permits('search_knowledge'))
        self.assertFalse(self.selection().permits('get_order'))

    def test_changed_loaded_schema_cannot_be_silently_restored(self):
        selection = self.selection()
        selection.discover('{"names":["get_order"]}')
        changed = deepcopy(TOOLS + [KNOWLEDGE_TOOL] + RETURN_TOOLS + [PLAN_TOOL])
        changed[0]['function']['parameters']['required'] = ['order_id', 'customer_id']
        with self.assertRaisesRegex(ValueError, '定义.*变化'):
            self.selection(selection.snapshot(), changed)

    def test_catalog_does_not_expose_full_argument_schema(self):
        selection = self.selection()
        instructions = selection.instructions()
        self.assertIn('get_order', instructions)
        self.assertIn('search_knowledge', instructions)
        self.assertNotIn('additionalProperties', instructions)
        self.assertIn('MCP', instructions)


if __name__ == '__main__':
    unittest.main()
