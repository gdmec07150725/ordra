"""工具执行策略：有界并行、串行屏障、超时和有限重试。

模型选择工具及参数；parallel_safe / retry_safe 由服务器代码声明。
本模块不写聊天历史，也不重试模型生成或人工确认操作。
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import math
import sqlite3
from time import monotonic

import anyio
import httpx2


@dataclass(frozen=True)
class ToolPolicy:
    max_parallel: int = 3
    timeout_seconds: float = 8.0
    max_attempts: int = 2
    retry_delay_seconds: float = 0.2
    max_calls: int = 12

    def __post_init__(self):
        for value in (self.max_parallel, self.max_attempts, self.max_calls):
            if type(value) is not int or value < 1:
                raise ValueError('工具并行数、尝试次数和调用上限必须为正整数。')
        for value in (self.timeout_seconds, self.retry_delay_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('工具超时和退避时间必须为有限非负数。')
        if self.timeout_seconds == 0:
            raise ValueError('工具超时必须大于零。')


@dataclass(frozen=True)
class ToolJob:
    call_id: str
    name: str
    arguments: str
    execute: Callable[[], Awaitable[dict]]
    # 未声明的工具默认串行且不能自动重试。
    parallel_safe: bool = False
    retry_safe: bool = False
    cache_failures: bool = True
    # 仅服务器声明的计划控制工具豁免业务额度，仍受模型轮数/计划更新数限制。
    budgeted: bool = True


def transient_code(error: Exception) -> str | None:
    if isinstance(error, sqlite3.OperationalError):
        code = getattr(error, 'sqlite_errorcode', 0) & 255
        if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or str(error).lower() in ('database is locked', 'database table is locked'):
            return 'storage_busy'
    if isinstance(error, (ConnectionError, httpx2.TransportError)):
        return 'connection_error'
    return None


class ToolRuntime:
    """一个用户回合一个实例；失败缓存、调用预算不会跨会话共享。"""
    def __init__(self, policy: ToolPolicy | None = None):
        self.policy = policy or ToolPolicy()
        self.calls = 0
        self.running = 0
        self.peak_parallel = 0
        self.trace = []
        self.failures = {}
        self._limiter = None

    @staticmethod
    def _key(job: ToolJob):
        try:
            arguments = json.dumps(json.loads(job.arguments), sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            arguments = job.arguments
        return job.name, arguments

    async def _execute(self, job: ToolJob, allowed: bool) -> dict:
        started, attempts, retries = monotonic(), 0, []
        key = self._key(job)
        if not allowed:
            result = {'error': '本轮工具调用已达到上限，请根据已有结果回答。', 'error_code': 'tool_limit'}
        elif job.cache_failures and key in self.failures:
            result = {**self.failures[key], 'error': self.failures[key]['error'] + ' 本轮已失败，不再重复相同调用；请调整参数或如实说明。'}
        else:
            pending_retry = None
            try:
                # 一项工具的所有尝试及退避共用期限，重试不能重置超时预算。
                with anyio.fail_after(self.policy.timeout_seconds) as deadline_scope:
                    while True:
                        await anyio.lowlevel.checkpoint()
                        if anyio.current_time() >= deadline_scope.deadline:
                            raise TimeoutError
                        attempts += 1
                        if pending_retry is not None:
                            retries.append(pending_retry)
                        try:
                            result = await job.execute()
                            await anyio.lowlevel.checkpoint()
                            # 同步校验可能短暂阻塞循环，定时器尚未调度也不能接受迟到结果。
                            if anyio.current_time() >= deadline_scope.deadline:
                                raise TimeoutError
                            break
                        except (TimeoutError, httpx2.TimeoutException):
                            raise
                        except Exception as error:
                            code = transient_code(error)
                            if code is None:
                                # 数据库损坏、程序错误等继续由 API 终止本轮。
                                raise
                            if job.retry_safe and attempts < self.policy.max_attempts:
                                pending_retry = code
                                await anyio.sleep(self.policy.retry_delay_seconds * 2 ** (attempts - 1))
                                continue
                            result = {'error': '工具暂时不可用，无法确认结果。请根据已有依据回答，缺少依据时如实说明。',
                                      'error_code': code, 'retryable': False}
                            break
            except (TimeoutError, httpx2.TimeoutException):
                # 本地只读线程可能仍在结束，超时不立刻重试，避免叠加同一工作。
                result = {'error': '工具等待超时，无法确认结果。请如实说明暂时无法查询。',
                          'error_code': 'tool_timeout', 'retryable': False}
            if job.cache_failures and 'error' in result:
                self.failures[key] = result.copy()
        return {'call_id': job.call_id, 'name': job.name, 'result': result,
                'attempts': attempts, 'retries': retries,
                'status': 'skipped' if attempts == 0 else 'error' if 'error' in result else 'ok',
                'elapsed_ms': round((monotonic() - started) * 1000)}

    def reserve_batch(self, jobs: list[ToolJob]) -> list[bool]:
        """先预留额度，调用方可在真正执行前持久化消耗；恢复不退还预留额度。"""
        allowed = []
        for job in jobs:
            allowed.append(not job.budgeted or self.calls < self.policy.max_calls)
            if job.budgeted:
                self.calls += 1
        return allowed

    async def run_batch(self, jobs: list[ToolJob], *, on_result: Callable[[dict], None] | None = None, reserved: list[bool] | None = None) -> list[dict]:
        """安全只读工具分组并行；串行工具是屏障；结果始终按原调用顺序返回。"""
        results = [None] * len(jobs)
        allowed = self.reserve_batch(jobs) if reserved is None else reserved
        if len(allowed) != len(jobs):
            raise ValueError('工具额度与批次长度不一致。')
        if self._limiter is None:
            self._limiter = anyio.Semaphore(self.policy.max_parallel)
        limiter = self._limiter

        async def run_group(indices):
            errors = {}
            async with anyio.create_task_group() as group:
                async def run_one(index):
                    try:
                        async with limiter:
                            self.running += 1
                            self.peak_parallel = max(self.peak_parallel, self.running)
                            try:
                                results[index] = await self._execute(jobs[index], allowed[index])
                            finally:
                                self.running -= 1
                    except Exception as error:
                        errors[index] = error
                        group.cancel_scope.cancel()
                for index in indices:
                    group.start_soon(run_one, index)
            if errors:
                # 保持 sqlite3.Error 等原异常类型，供现有 API 错误边界处理。
                raise errors[min(errors)]
            if on_result is not None:
                # 只有执行器确认未超时的结果，才可更新本轮临时状态。
                # 在下一个串行屏障之前调用，便于后续工具检查已经完成的操作。
                for index in indices:
                    on_result(results[index])

        pending = []
        for index, job in enumerate(jobs):
            if job.parallel_safe:
                pending.append(index)
            else:
                if pending:
                    await run_group(pending)
                    pending = []
                await run_group([index])
        if pending:
            await run_group(pending)
        await anyio.lowlevel.checkpoint()
        self.trace.extend({key: value for key, value in item.items() if key != 'result'} for item in results)
        return results

    def summary(self) -> dict:
        return {'calls': len(self.trace), 'attempts': sum(item['attempts'] for item in self.trace),
                'retries': sum(len(item['retries']) for item in self.trace),
                'max_parallel': self.peak_parallel, 'tools': self.trace.copy()}
