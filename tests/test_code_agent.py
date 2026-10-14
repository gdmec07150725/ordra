import copy
import unittest

from code_agent import run_analysis
from code_data import load_dataset


CORRECT_CODE = '''groups = {}
for row in rows:
    name = row[group_by]
    if name not in groups:
        groups[name] = {"name": name, "orders": 0, "refunded": 0, "refund_rate": 0.0}
    groups[name]["orders"] += 1
    if row["status"] == "refunded":
        groups[name]["refunded"] += 1
for item in groups.values():
    item["refund_rate"] = item["refunded"] / item["orders"]
{"groups": list(groups.values()), "total_orders": len(rows)}'''


class ScriptGenerator:
    def __init__(self, codes):
        self.codes, self.requests = list(codes), []

    async def __call__(self, messages):
        self.requests.append(copy.deepcopy(messages))
        return {'code': self.codes.pop(0), 'note': '脚本模型仅验证运行闭环。', 'usage': None}


class CodeAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_error_and_wrong_calculation_are_feedback_then_repaired(self):
        for bad in ('1 / 0', '{"groups": [], "total_orders": 0}'):
            with self.subTest(bad=bad):
                generator = ScriptGenerator([bad, CORRECT_CODE])
                events = []
                async def emit(event): events.append(event)
                result = await run_analysis(load_dataset('orders-edge'), 'product', .3, generator, emit=emit)
                self.assertEqual(result['status'], 'completed', result)
                self.assertEqual(result['attempts'], 2)
                self.assertEqual(result['metrics']['model_requests'], 2)
                self.assertEqual(result['metrics']['usage_status'], 'unavailable')
                self.assertTrue(result['verification']['passed'])
                self.assertIn('反馈', generator.requests[1][-1]['content'])
                self.assertIn('verify', [event['stage'] for event in events])

    async def test_three_failures_stop_without_a_report_and_without_all_data_in_prompt(self):
        generator = ScriptGenerator(['1 / 0'] * 3)
        result = await run_analysis(load_dataset('orders-demo'), 'product', .3, generator)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(generator.requests), 3)
        self.assertIsNone(result['report'])
        self.assertNotIn('D018', generator.requests[0][1]['content'])

    async def test_cancelled_execution_is_not_converted_to_success(self):
        import asyncio
        entered = asyncio.Event()
        async def cancelled(messages):
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(run_analysis(load_dataset('orders-demo'), 'channel', .3, cancelled))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_explicit_learning_injection_is_recorded_and_recovered(self):
        generator = ScriptGenerator([CORRECT_CODE, CORRECT_CODE])
        result = await run_analysis(load_dataset('orders-demo'), 'channel', .3, generator, exercise='runtime_error')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['history'][0]['injected'])
        self.assertNotEqual(result['history'][0]['code'], result['history'][0]['generated_code'])
        self.assertEqual(result['attempts'], 2)

    async def test_exercise_is_not_skipped_after_an_invalid_model_response(self):
        calls = 0
        async def generator(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {'error': '模型输出不完整。', 'sent': True}
            return {'code': CORRECT_CODE, 'note': '重新生成代码。', 'sent': True}
        result = await run_analysis(load_dataset('orders-demo'), 'product', .3, generator, exercise='wrong_result')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['attempts'], 3)
        self.assertTrue(result['history'][1]['injected'])
        self.assertFalse(result['history'][2]['injected'])
        self.assertEqual(result['metrics']['executions'], 2)
