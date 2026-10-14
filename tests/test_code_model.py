import json
import unittest

import httpx2
from openai import AsyncOpenAI

from code_agent import ModelCodeGenerator, run_analysis
from code_data import load_dataset
from context_memory import ContextPolicy
from test_code_agent import CORRECT_CODE


class CodeModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.responses, self.requests = [], []
        async def handle(request):
            self.requests.append(json.loads(request.content))
            return httpx2.Response(200, json=self.responses.pop(0))
        self.client = AsyncOpenAI(api_key='offline-only', base_url='https://model.invalid', max_retries=0,
                                 http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))

    async def asyncTearDown(self):
        await self.client.close()

    def response(self, content, reason='stop'):
        return {'id': 'offline', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': reason}],
                'usage': {'prompt_tokens': 50, 'completion_tokens': 20, 'total_tokens': 70}}

    async def test_invalid_and_truncated_responses_count_usage_and_can_recover(self):
        self.responses = [self.response('not JSON'), self.response('{"code":"unfinished', 'length'),
                          self.response(json.dumps({'code': CORRECT_CODE, 'note': '修正输出。'}))]
        result = await run_analysis(load_dataset('orders-demo'), 'product', .3, ModelCodeGenerator(self.client, 'offline'))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['metrics']['total_tokens'], 210)
        self.assertEqual(result['metrics']['model_requests'], 3)
        self.assertEqual(result['metrics']['executions'], 1)
        self.assertEqual(result['metrics']['usage_status'], 'complete')
        self.assertEqual(self.requests[0]['max_tokens'], 1800)
        self.assertEqual(self.requests[0]['response_format'], {'type': 'json_object'})
        self.assertEqual(self.requests[0].get('thinking'), {'type': 'disabled'})

    async def test_context_reserve_blocks_request_before_model(self):
        policy = ContextPolicy(max_tokens=2048, recent_tokens=128, summary_tokens=256)
        result = await run_analysis(load_dataset('orders-demo'), 'product', .3, ModelCodeGenerator(self.client, 'offline', policy))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['metrics']['model_requests'], 0)
        self.assertEqual(self.requests, [])
