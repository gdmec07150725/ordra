"""代码 Agent 教学评测：默认脚本模型＋真实沙箱，--live 显式调用模型。"""
import argparse
import asyncio
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic

from code_agent import ModelCodeGenerator, run_analysis
from code_data import load_dataset
from code_sandbox import sandbox_status
from context_memory import ContextPolicy


ROOT = Path(__file__).resolve().parent
CASES = [('normal', 'orders-demo', 'product'), ('edge', 'orders-edge', 'channel'),
         ('runtime_error', 'orders-demo', 'product'), ('wrong_result', 'orders-edge', 'product')]


async def evaluate(*, live=False, repeats=1, output):
    from openai import AsyncOpenAI
    client = None
    if live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', encoding='utf-8-sig')
        key = os.getenv('DEEPSEEK_API_KEY', '').strip()
        if not key:
            raise ValueError('真实模型评测需要 DEEPSEEK_API_KEY。')
        client = AsyncOpenAI(api_key=key, base_url='https://api.deepseek.com', max_retries=0, timeout=30)
    code = '' if live else (ROOT / 'examples/code-agent/reference.py').read_text(encoding='utf-8')
    model = os.getenv('DEEPSEEK_MODEL', 'deepseek-flash') if live else 'scripted'
    async def scripted(messages):
        return {'code': code, 'note': '固定脚本，仅验证工程链路。', 'usage': None, 'sent': False}
    report = {'mode': 'live' if live else 'scripted', 'model': model, 'thinking': 'disabled', 'sandbox': sandbox_status(),
              'complete': False, 'repeats': repeats, 'total': len(CASES) * repeats, 'results': [],
              'source_sha256': {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ('code_agent.py', 'code_data.py', 'code_sandbox.py')},
              'note': '错误恢复场景明确在首次代码执行时注入错误；离线模式无真实模型请求或token用量。固定小样本不是泛化能力结论。'}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report['passed'] = sum(item['passed'] for item in report['results'])
        output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    save()
    try:
        for repeat in range(1, repeats + 1):
            for case, dataset_id, group_by in CASES:
                started = monotonic()
                dataset = load_dataset(dataset_id)
                result = await run_analysis(dataset, group_by, .3,
                    ModelCodeGenerator(client, model, ContextPolicy.from_env()) if live else scripted,
                    exercise=case if case in ('runtime_error', 'wrong_result') else 'normal')
                passed = result['status'] == 'completed' and result['verification']['passed']
                if case in ('runtime_error', 'wrong_result'):
                    passed = passed and result['metrics']['executions'] >= 2 and any(item['injected'] for item in result['history'])
                item = {'case': case, 'repeat': repeat, 'dataset_id': dataset_id, 'dataset_sha256': dataset['sha256'],
                        'group_by': group_by, 'passed': bool(passed), 'elapsed_ms': round((monotonic()-started)*1000), 'result': result}
                report['results'].append(item)
                save()
                print(f'{"PASS" if passed else "FAIL"} {case}/{repeat}: {result["attempts"]} attempts, {result["metrics"]["model_requests"]} model requests, {item["elapsed_ms"]} ms', flush=True)
        report['complete'] = True
        save()
    finally:
        if client is not None:
            await client.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / '.local/evals' / f'code-agent-{datetime.now():%Y%m%d-%H%M%S}-{"live" if args.live else "offline"}.json'
    if output.suffix.lower() != '.json':
        parser.error('--output 需要 .json 文件。')
    report = asyncio.run(evaluate(live=args.live, repeats=args.repeats, output=output))
    print(f'{report["passed"]}/{report["total"]} passed · {output}')
    raise SystemExit(0 if report['passed'] == report['total'] else 1)
