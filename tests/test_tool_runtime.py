"""调度器行为测试：控制事件顺序，不依赖外部模型或真实业务数据。"""
import asyncio
import sqlite3
import unittest

import anyio
import httpx2

from tool_runtime import ToolJob, ToolPolicy, ToolRuntime


def job(index, execute, *, parallel=True, retry=True):
    return ToolJob(str(index), 'read', str(index), execute, parallel, retry)


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_limit_is_shared_between_specialist_batches(self):
        runtime = ToolRuntime(ToolPolicy(max_parallel=2))
        running, peak = 0, 0
        async def read():
            nonlocal running, peak
            running += 1; peak = max(peak, running)
            await asyncio.sleep(.02)
            running -= 1
            return {'ok': True}
        await asyncio.gather(runtime.run_batch([job(i, read) for i in range(3)]),
                             runtime.run_batch([job(i, read) for i in range(3, 6)]))
        self.assertEqual(peak, 2)

    async def test_parallel_limit_and_original_result_order(self):
        runtime = ToolRuntime(ToolPolicy(max_parallel=3))
        started, gate, seen = asyncio.Event(), asyncio.Event(), []

        async def read(index):
            seen.append(index)
            if len(seen) == 3:
                started.set()
            await gate.wait()
            await asyncio.sleep((5 - index) * .001)
            return {'value': index}

        task = asyncio.create_task(runtime.run_batch([job(i, lambda i=i: read(i)) for i in range(6)]))
        try:
            await asyncio.wait_for(started.wait(), 1)
            self.assertEqual(seen, [0, 1, 2])
        finally:
            gate.set()
        outcomes = await task
        self.assertEqual([item['result']['value'] for item in outcomes], list(range(6)))
        self.assertEqual(runtime.summary()['max_parallel'], 3)

    async def test_serial_tool_is_a_barrier_on_both_sides(self):
        events = []
        async def first():
            await asyncio.sleep(.005)
            events.append('first')
            return {}
        async def serial():
            self.assertEqual(events, ['first'])
            events.append('serial')
            return {}
        async def last():
            self.assertEqual(events, ['first', 'serial'])
            events.append('last')
            return {}
        await ToolRuntime().run_batch([job(0, first), job(1, serial, parallel=False, retry=False), job(2, last)])
        self.assertEqual(events, ['first', 'serial', 'last'])

    async def test_temporary_connection_failure_retries_once(self):
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError('private endpoint and credential must not be surfaced')
            return {'found': True}
        runtime = ToolRuntime(ToolPolicy(retry_delay_seconds=0))
        result = (await runtime.run_batch([job(0, read)]))[0]
        self.assertEqual(result['result'], {'found': True})
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(runtime.summary()['retries'], 1)
        self.assertNotIn('private', str(runtime.summary()))

    async def test_busy_exhaustion_returns_error_and_prevents_same_failure_loop(self):
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            raise sqlite3.OperationalError('database is locked')
        runtime = ToolRuntime(ToolPolicy(retry_delay_seconds=0))
        first = (await runtime.run_batch([job(0, read)]))[0]
        again = (await runtime.run_batch([job(0, read)]))[0]
        self.assertEqual(calls, 2)
        self.assertEqual(first['result']['error_code'], 'storage_busy')
        self.assertEqual(again['attempts'], 0)
        self.assertIn('不再重复', again['result']['error'])

    async def test_unsafe_operation_and_business_rejection_are_not_retried(self):
        counts = [0, 0]
        async def unsafe():
            counts[0] += 1
            raise ConnectionError('connection lost')
        async def rejected():
            counts[1] += 1
            return {'error': '参数无效，请补充订单号。'}
        await ToolRuntime().run_batch([job(0, unsafe, parallel=False, retry=False), job(1, rejected)])
        self.assertEqual(counts, [1, 1])

    async def test_timeout_cancels_wait_and_other_tool_still_succeeds(self):
        stopped = asyncio.Event()
        async def slow():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def fast():
            return {'ok': True}
        runtime = ToolRuntime(ToolPolicy(timeout_seconds=.025))
        outcomes = await asyncio.wait_for(runtime.run_batch([job(0, slow), job(1, fast)]), 1)
        self.assertTrue(stopped.is_set())
        self.assertEqual(outcomes[0]['attempts'], 1)
        self.assertEqual(outcomes[0]['result']['error_code'], 'tool_timeout')
        self.assertEqual(outcomes[1]['result'], {'ok': True})

    async def test_transport_timeout_does_not_overlap_a_retry(self):
        async def read():
            raise httpx2.ReadTimeout('timed out')
        outcome = (await ToolRuntime(ToolPolicy(retry_delay_seconds=0)).run_batch([job(0, read)]))[0]
        self.assertEqual(outcome['attempts'], 1)
        self.assertEqual(outcome['result']['error_code'], 'tool_timeout')

    async def test_timeout_budget_is_shared_by_attempts_and_retry_delay(self):
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            await anyio.sleep(.06)
            if calls == 1:
                raise ConnectionError('temporary')
            return {'found': True}
        outcome = (await ToolRuntime(ToolPolicy(timeout_seconds=.1, retry_delay_seconds=0)).run_batch([job(0, read)]))[0]
        self.assertEqual(outcome['result'].get('error_code'), 'tool_timeout')
        self.assertEqual(outcome['attempts'], 2)

        async def immediately_unavailable():
            raise ConnectionError('temporary')
        outcome = (await ToolRuntime(ToolPolicy(timeout_seconds=.025, retry_delay_seconds=1)).run_batch([job(0, immediately_unavailable)]))[0]
        self.assertEqual(outcome['result'].get('error_code'), 'tool_timeout')
        self.assertEqual(outcome['attempts'], 1)
        self.assertEqual(outcome['retries'], [], '尚未开始下一次尝试不能统计为已重试')

    async def test_fatal_database_error_cancels_siblings_and_preserves_exception_type(self):
        entered, stopped = asyncio.Event(), asyncio.Event()
        async def slow():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def corrupt():
            await entered.wait()
            raise sqlite3.DatabaseError('corrupt data')
        with self.assertRaisesRegex(sqlite3.DatabaseError, 'corrupt'):
            await ToolRuntime().run_batch([job(0, slow), job(1, corrupt)])
        self.assertTrue(stopped.is_set())

    async def test_external_cancellation_is_not_converted_into_error_or_retry(self):
        entered, stopped = asyncio.Event(), asyncio.Event()
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        runtime = ToolRuntime()
        task = asyncio.create_task(runtime.run_batch([job(0, read), job(1, read)]))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
        self.assertLessEqual(calls, 2)
        self.assertEqual(runtime.summary()['tools'], [])

    async def test_cancellation_during_retry_delay_does_not_start_next_attempt(self):
        entered = asyncio.Event()
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            entered.set()
            raise ConnectionError('temporary')
        task = asyncio.create_task(ToolRuntime(ToolPolicy(retry_delay_seconds=30)).run_batch([job(0, read)]))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(calls, 1)

    async def test_call_budget_spans_batches_and_failures_are_not_shared_between_turns(self):
        calls = 0
        async def read():
            nonlocal calls
            calls += 1
            return {'error': 'invalid'}
        runtime = ToolRuntime(ToolPolicy(max_calls=1))
        await runtime.run_batch([job(0, read)])
        result = (await runtime.run_batch([job(1, read)]))[0]
        self.assertEqual(result['result']['error_code'], 'tool_limit')
        await ToolRuntime().run_batch([job(0, read)])
        self.assertEqual(calls, 2)

    def test_invalid_policy_is_rejected(self):
        for kwargs in ({'timeout_seconds': 0}, {'timeout_seconds': float('nan')},
                       {'max_parallel': True}, {'max_calls': 0}, {'max_attempts': 1.5},
                       {'retry_delay_seconds': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ToolPolicy(**kwargs)
