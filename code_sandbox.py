"""无宿主能力的 Monty worker；每次执行使用全新会话，绝不回退到 Python exec。"""
import asyncio
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys
import sysconfig
from time import monotonic


MONTY_VERSION = '0.0.23'


@dataclass(frozen=True)
class SandboxPolicy:
    seconds: float = 2.0
    memory_bytes: int = 32 * 1024 * 1024
    code_chars: int = 12000
    print_chars: int = 4096
    output_bytes: int = 64 * 1024


class OutputLimitError(RuntimeError):
    pass


def sandbox_status():
    binary = Path(sysconfig.get_path('scripts')) / ('monty.exe' if sys.platform == 'win32' else 'monty')
    try:
        ready = binary.is_file() and all(version(name) == MONTY_VERSION for name in
                    ('pydantic-monty', 'pydantic-monty-client', 'pydantic-monty-runtime'))
    except PackageNotFoundError:
        ready = False
    return {'ready': ready, 'engine': 'Monty', 'version': MONTY_VERSION,
            'binary': str(binary), 'isolation': 'language_sandbox_with_worker_process',
            'message': '' if ready else '请在项目虚拟环境安装 requirements.txt 中固定版本的 pydantic-monty；不会使用普通 Python 代替沙箱。'}


async def run_code(code, inputs, *, policy=None):
    policy = policy or SandboxPolicy()
    started = monotonic()
    streams = {'stdout': '', 'stderr': ''}
    result = {'ok': False, 'value': None, 'error': '', 'error_type': ''}

    def capture(stream, text):
        remaining = policy.print_chars - sum(len(value) for value in streams.values())
        streams[stream] += text[:max(0, remaining)]
        if len(text) > remaining:
            raise OutputLimitError('打印输出超过 4096 字符上限。请返回简短统计结果。')

    try:
        if not isinstance(code, str) or not code.strip() or len(code) > policy.code_chars:
            raise ValueError('代码为空或超过 12000 字符上限。')
        code.encode('utf-8')
        if len(json.dumps(inputs, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 128 * 1024:
            raise ValueError('输入数据超过 128 KiB 上限。')
        status = sandbox_status()
        if not status['ready']:
            raise RuntimeError(status['message'])
        from pydantic_monty import AsyncMonty
        # Explicit binary ignores MONTY_BIN/PATH. No mounts, host objects/functions or OS handler.
        async with AsyncMonty(binary_path=status['binary'], min_processes=1, max_processes=1,
                              checkout_timeout=3, request_timeout=policy.seconds + 3,
                              max_checkouts_per_worker=1) as pool:
            async with pool.checkout(script_name='analysis.py', print_flush_interval=0,
                                     limits={'max_duration_secs': policy.seconds, 'max_memory': policy.memory_bytes,
                                             'max_recursion_depth': 100, 'max_suspensions': 32}) as session:
                value = await session.feed_run(code, inputs=inputs, external_lookup={}, print_callback=capture)
                encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
                if len(encoded) > policy.output_bytes:
                    raise OutputLimitError('结构化结果超过 64 KiB 上限。请只返回分组统计。')
                result.update(ok=True, value=json.loads(encoded))
    except asyncio.CancelledError:
        raise
    except Exception as error:
        result.update(error=str(error)[:1600], error_type=type(error).__name__)
    return {**result, **streams, 'elapsed_ms': round((monotonic() - started) * 1000)}
