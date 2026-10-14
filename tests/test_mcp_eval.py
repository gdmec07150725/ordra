from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import anyio


class MCPComparisonTests(unittest.TestCase):
    def evaluator(self):
        self.assertIsNotNone(importlib.util.find_spec('mcp_eval'), 'MCP comparison evaluator is missing')
        return importlib.import_module('mcp_eval')

    def order_trace(self):
        return {'status': 'completed', 'events': [
            {'kind': 'context_budget', 'data': {'role': 'actor', 'tool_names': ['discover_tools'], 'components': {'tools': 80}}},
            {'kind': 'tool_discovery', 'data': {'status': 'loaded', 'loaded_names': ['get_order'], 'tools': [{'name': 'get_order', 'source': 'mcp'}]}},
            {'kind': 'context_budget', 'data': {'role': 'actor', 'tool_names': ['discover_tools', 'get_order'], 'components': {'tools': 140}}},
            {'kind': 'tool_result', 'data': {'name': 'get_order', 'call_id': 'order', 'arguments': {'order_id': 'O1003'}, 'status': 'ok', 'attempts': 1, 'source': 'mcp', 'result': {'order_id': 'O1003', 'status': '已签收'}}},
        ]}

    def test_grade_rejects_missing_late_discovery_or_missing_request_schema(self):
        evaluator = self.evaluator()
        case = {'orders': ['O1003']}
        trace = self.order_trace()
        self.assertTrue(evaluator.grade_mode(case, 'mcp', trace)['passed'])
        mutations = []
        missing = deepcopy(trace)
        missing['events'].pop(1)
        mutations.append(missing)
        late = deepcopy(trace)
        late['events'].append(late['events'].pop(1))
        mutations.append(late)
        no_schema = deepcopy(trace)
        no_schema['events'][2]['data']['tool_names'] = ['discover_tools']
        mutations.append(no_schema)
        for bad in mutations:
            with self.subTest(events=bad['events']):
                self.assertIn('discovery_before_business', evaluator.grade_mode(case, 'mcp', bad)['failed_criteria'])

    def test_grade_rejects_local_fallback_wrong_order_and_empty_policy(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        trace['events'][-1]['data']['source'] = 'local'
        self.assertIn('business_transport', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])
        trace['events'][-1]['data']['result']['order_id'] = 'O9999'
        self.assertIn('order_evidence', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])
        self.assertIn('knowledge_evidence', evaluator.grade_mode({'knowledge': True}, 'mcp', trace)['failed_criteria'])

    def test_rejected_early_business_call_is_not_hidden_by_later_success(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        rejected = deepcopy(trace['events'][-1])
        rejected['data'].update(call_id='early', attempts=0, status='error', result={'error_code': 'tool_not_discovered', 'error': 'Tool was not loaded'})
        trace['events'].insert(1, rejected)
        self.assertIn('discovery_before_business', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])

    def test_initial_request_cannot_include_undeferred_local_business_schemas(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        trace['events'][0]['data']['tool_names'].append('prepare_return_request')
        self.assertIn('initial_schemas_deferred', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])

    def test_every_actor_request_rejects_schemas_exposed_before_discovery(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        trace['events'].insert(1, {'kind': 'context_budget', 'data': {'role': 'actor', 'tool_names': ['discover_tools', 'get_order']}})
        self.assertIn('actor_schema_whitelist', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])
        # Local business schemas are allowed after their own explicit discovery.
        trace = self.order_trace()
        trace['events'][1]['data']['tools'].append({'name': 'get_return_request', 'source': 'local'})
        trace['events'][2]['data']['tool_names'].append('get_return_request')
        self.assertTrue(evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['passed'])

    def test_specialist_requests_reject_extra_or_wrong_role_schemas(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        order = trace['events'][-1]
        order['data']['call_id'] = 'specialist_1:r1:order'
        trace['events'] = [trace['events'][0], {'kind': 'delegated_task', 'data': {'id': 'specialist_1', 'agent': 'orders'}},
                           {'kind': 'context_budget', 'data': {'role': 'specialist', 'agent': 'orders', 'tool_names': ['get_order', 'search_knowledge', 'prepare_return_request']}}, order]
        self.assertIn('specialist_schema_whitelist', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])
        trace['events'][2]['data']['tool_names'] = ['get_order']
        self.assertTrue(evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['passed'])
        trace['events'][2]['data']['agent'] = 'knowledge'
        self.assertIn('specialist_schema_whitelist', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace)['failed_criteria'])

    def test_actual_http_schema_names_must_match_prepared_request_observations(self):
        evaluator = self.evaluator()
        trace = self.order_trace()
        requests = [{'role': 'actor', 'sent': True, 'tool_names': ['discover_tools', 'get_order']},
                    {'role': 'actor', 'sent': True, 'tool_names': ['discover_tools', 'get_order']}]
        self.assertIn('observed_request_schemas', evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace, requests=requests)['failed_criteria'])
        requests[0]['tool_names'] = ['discover_tools']
        self.assertTrue(evaluator.grade_mode({'orders': ['O1003']}, 'mcp', trace, requests=requests)['passed'])

    def test_usage_coverage_counts_sent_requests_excluding_injected_presend_failure(self):
        evaluator = self.evaluator()
        trace = {'metrics': {'model_requests': 4, 'reported_requests': 3, 'prompt_tokens': 120, 'completion_tokens': 24, 'total_tokens': 144}}
        requests = [{'sent': True, 'role': 'actor', 'estimated_schema_tokens': 20}] * 3
        result = evaluator.metrics_for(trace, requests, 4, live=True)
        self.assertEqual((result['request_count'], result['reported_requests'], result['usage_status']), (3, 3, 'complete'))
        self.assertEqual(result['model_request_attempts'], 4)
        trace['metrics']['reported_requests'] = 1
        self.assertEqual(evaluator.metrics_for(trace, requests, 4, live=True)['usage_status'], 'partial')
        trace['metrics']['reported_requests'] = 0
        result = evaluator.metrics_for(trace, requests, 4, live=True)
        self.assertEqual(result['usage_status'], 'unreported')
        self.assertIsNone(result['total_tokens'])

    def test_pair_rejects_different_successful_business_results(self):
        evaluator = self.evaluator()
        local = {'passed': True, 'evidence': {'orders': [{'order_id': 'O1003', 'status': '已签收'}], 'knowledge': []}}
        remote = deepcopy(local)
        self.assertTrue(evaluator.grade_pair(local, remote)['passed'])
        remote['evidence']['orders'][0]['status'] = '已取消'
        self.assertIn('same_business_evidence', evaluator.grade_pair(local, remote)['failed_criteria'])

    def test_offline_real_stdio_comparison_and_recovery(self):
        evaluator = self.evaluator()
        with redirect_stdout(io.StringIO()):
            report = anyio.run(evaluator.evaluate_comparison)
        failures = {r['id']: {mode: {'failed': run['trajectory']['failed_criteria'], 'error': run['error']} for mode, run in r['modes'].items()} for r in report['results'] if not r['passed']}
        self.assertEqual((report['passed'], report['total']), (5, 5), json.dumps(failures, ensure_ascii=False))
        self.assertTrue(report['model_decisions_scripted'])
        for result in report['results']:
            self.assertEqual(set(result['modes']), {'local', 'mcp'})
            for mode, run in result['modes'].items():
                metrics = run['metrics']
                self.assertGreater(metrics['model_requests'], 0)
                sent = [r for r in run['requests'] if r['sent']]
                self.assertEqual(metrics['model_requests'], len(sent))
                self.assertEqual(metrics['estimated_schema_tokens'], sum(r['estimated_schema_tokens'] for r in sent))
                self.assertGreater(metrics['estimated_schema_tokens'], 0)
                if mode == 'mcp':
                    self.assertNotIn('get_order', sent[0]['tool_names'])
                    self.assertNotIn('search_knowledge', sent[0]['tool_names'])
                    calls = [e['data'] for e in run['trace']['events'] if e['kind'] == 'tool_result' and e['data']['name'] in ('get_order', 'search_knowledge')]
                    self.assertTrue(calls)
                    self.assertTrue(all(c['source'] == 'mcp' for c in calls))
            self.assertEqual(result['modes']['local']['evidence'], result['modes']['mcp']['evidence'])
            expected = {'local': 7, 'mcp': 7} if result['id'] == 'delegation_parallel' else {'local': 3, 'mcp': 4}
            self.assertEqual({mode: run['metrics']['model_requests'] for mode, run in result['modes'].items()}, expected)
        recovery = next(r for r in report['results'] if r['id'] == 'checkpoint_resume')
        for mode, run in recovery['modes'].items():
            self.assertTrue(run['observation']['resumed_same_run'])
            self.assertEqual(run['observation']['actual_order_calls'], 1)
            self.assertEqual(run['metrics']['model_request_attempts'], run['metrics']['model_requests'] + 1)
            if mode == 'mcp':
                self.assertIn('get_order', run['observation']['resume_tool_selection']['loaded'])

    def test_unfinished_timeout_is_reported_as_failure(self):
        evaluator = self.evaluator()
        with redirect_stdout(io.StringIO()):
            report = anyio.run(lambda: evaluator.evaluate_comparison([evaluator.CASES[0]], timeout_seconds=0.001))
        self.assertEqual(report['passed'], 0)
        self.assertEqual(report['total'], 1)
        for run in report['results'][0]['modes'].values():
            self.assertFalse(run['passed'])
            self.assertEqual(run['failure_kind'], 'timeout')
            self.assertTrue(run['error'])

    def test_cli_writes_json_and_escaped_readable_html(self):
        evaluator = self.evaluator()
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'comparison.json'
            process = subprocess.run([sys.executable, '-X', 'utf8', str(root / 'mcp_eval.py'), '--case', 'order_query', '--output', str(output)], cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual((report['passed'], report['total']), (1, 1))
            report['results'][0]['modes']['mcp']['answer'] = '<script>alert(1)</script>'
            rendered = evaluator.report_html(report)
            self.assertIn('&lt;script&gt;', rendered)
            self.assertNotIn('<script>', rendered)
            self.assertIn('tool_discovery', rendered)
            self.assertIn('估算', rendered)
            self.assertTrue(output.with_suffix('.html').exists())


if __name__ == '__main__':
    unittest.main()
