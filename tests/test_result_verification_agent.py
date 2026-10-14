"""校验失败不发布草稿；修正和恢复共享计数及已完成工具。"""
import json
import unittest
import agent_service
import test_api_app as fixture
from test_stream_agent import chunk, text_stream
from test_tool_execution_api import tool_stream
from test_agent_plan_api import steps, update


class ResultVerificationAgentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.ApiAppTests.asyncSetUp
    asyncTearDown = fixture.ApiAppTests.asyncTearDown

    async def save(self, state):
        self.snapshot = json.loads(json.dumps(state))

    async def reviewer(self, **kwargs):
        self.reviews.append(kwargs)
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        return {'verdict': verdict, 'issues': ['订单状态与工具证据不一致，请核对后修正。'] if verdict == 'revise' else [],
                'evidence_ids': [], 'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}, 'elapsed_ms': 1}

    def run_agent(self, checkpoint=None):
        return agent_service.stream_agent(self.app.state.client, self.app.state.model, agent_service.new_conversation(), '查询 O1003',
                                          result_verifier=self.reviewer, checkpoint=checkpoint, save_checkpoint=self.save)

    async def test_bad_candidate_is_never_shown_or_saved_and_tools_are_reused(self):
        self.reviews, self.verdicts = [], ['revise', 'pass']
        self.responses = [tool_stream(('order', 'get_order', {'order_id': 'O1003'})), text_stream('错误草稿：O1003 已取消。'), text_stream('O1003 已签收。')]
        events = [e async for e in self.run_agent()]
        done = events[-1]
        self.assertEqual(done['answer'], 'O1003 已签收。')
        self.assertEqual(done['verification']['corrections'], 1)
        self.assertEqual(done['verification']['checks'], 2)
        self.assertEqual(done['verification']['status'], 'passed')
        self.assertEqual(done['execution']['calls'], 1)
        self.assertNotIn('错误草稿', ''.join(e.get('text', '') for e in events))
        self.assertNotIn('错误草稿', json.dumps(done['history'], ensure_ascii=False))
        self.assertIn('correcting', [e['event'] for e in events])

    async def test_two_corrections_end_with_honest_unverified_answer(self):
        self.reviews, self.verdicts = [], ['revise'] * 3
        self.responses = [text_stream('错误草稿') for _ in range(3)]
        done = [e async for e in self.run_agent()][-1]
        self.assertEqual(done['verification']['checks'], 3)
        self.assertEqual(done['verification']['corrections'], 2)
        self.assertEqual(done['verification']['status'], 'unverified')
        self.assertNotIn('错误草稿', done['answer'])
        self.assertIn('无法确认', done['answer'])
        self.assertEqual(len(self.requests), 3)

    def blocked_responses(self):
        plan = steps()[:1]
        self.responses = [update('start', plan)]
        plan[0]['status'] = 'blocked'
        self.responses.append(update('blocked', plan, '需要补充信息，具体说明见最终回复。'))

    async def test_rejected_blocked_note_can_be_rewritten_without_tools(self):
        self.reviews, self.verdicts = [], ['revise', 'pass']
        self.blocked_responses()
        self.responses.append(text_stream('请提供订单号，我才能查询订单状态。'))
        done = [e async for e in self.run_agent()][-1]
        self.assertEqual(done['answer'], '请提供订单号，我才能查询订单状态。')
        self.assertEqual(done['plan']['status'], 'blocked')
        self.assertEqual(done['verification']['corrections'], 1)
        self.assertEqual(len(self.requests), 3)
        self.assertNotIn('tools', self.requests[-1])
        self.assertIn('只改写答复', self.requests[-1]['messages'][-1]['content'])

    async def test_blocked_rewrite_rejects_unexpected_tool_calls_without_execution(self):
        self.reviews, self.verdicts = [], ['revise']
        self.blocked_responses()
        self.responses.append(tool_stream(('unexpected', 'get_order', {'order_id': 'O1003'})))
        done = [e async for e in self.run_agent()][-1]
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(done['plan']['status'], 'blocked')
        self.assertEqual(done['verification']['status'], 'unverified')
        self.assertFalse(any(m.get('tool_call_id') == 'unexpected' for m in done['history']))

    async def test_blocked_rewrite_has_one_attempt_even_after_restart(self):
        self.reviews, self.verdicts = [], ['revise', ValueError('校验中断'), 'revise']
        self.blocked_responses()
        self.responses.append(text_stream('仍然缺少具体信息。'))
        with self.assertRaisesRegex(ValueError, '校验中断'):
            _ = [e async for e in self.run_agent()]
        done = [e async for e in self.run_agent(self.snapshot)][-1]
        self.assertEqual(done['verification']['corrections'], 1)
        self.assertEqual(done['verification']['status'], 'unverified')
        self.assertEqual(len(self.requests), 3)

    async def test_interrupted_blocked_rewrite_does_not_request_another_after_restart(self):
        self.reviews, self.verdicts = [], ['revise', 'pass']
        self.blocked_responses()
        self.responses.extend([[chunk(content='未完成的改写'), ValueError('改写中断')], text_stream('请提供订单号。')])
        with self.assertRaisesRegex(ValueError, '改写中断'):
            _ = [e async for e in self.run_agent()]
        done = [e async for e in self.run_agent(self.snapshot)][-1]
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(done['verification']['status'], 'unverified')
        self.assertEqual(done['plan']['status'], 'blocked')
        self.assertNotIn('未完成的改写', done['answer'])

    async def test_reviewer_error_resume_rechecks_candidate_without_actor_request(self):
        self.reviews, self.verdicts = [], [ValueError('校验连接中断'), 'pass']
        self.responses = [text_stream('请提供订单号。')]
        with self.assertRaisesRegex(ValueError, '校验连接中断'):
            _ = [e async for e in self.run_agent()]
        self.assertEqual(self.snapshot['verification']['checks'], 1)
        self.assertEqual(self.snapshot['verification']['candidate'], '请提供订单号。')
        done = [e async for e in self.run_agent(self.snapshot)][-1]
        self.assertEqual(done['verification']['checks'], 2)
        self.assertEqual(len(self.requests), 1)
        # 最终提交失败后的重试不应重新请求执行模型或校验模型。
        repeated = [e async for e in self.run_agent(self.snapshot)][-1]
        self.assertEqual(repeated['answer'], done['answer'])
        self.assertEqual(len(self.reviews), 2)

    async def test_repeated_reviewer_failures_cannot_reset_budget_by_resume(self):
        self.reviews, self.verdicts = [], [ValueError('格式错误')] * 3
        self.responses = [text_stream('待核对的草稿')]
        checkpoint = None
        for _ in range(3):
            with self.assertRaises(ValueError):
                _ = [e async for e in self.run_agent(checkpoint)]
            checkpoint = self.snapshot
        done = [e async for e in self.run_agent(checkpoint)][-1]
        self.assertEqual(done['verification']['status'], 'unverified')
        self.assertEqual(done['verification']['checks'], 3)
        self.assertEqual(len(self.requests), 1)

    async def test_planned_budget_exhaustion_finishes_locally_after_restart(self):
        self.reviews, self.verdicts = [], [ValueError('中断')]
        plan = steps()[:1]
        self.responses = [update('start', plan), tool_stream(('order', 'get_order', {'order_id': 'O1003'}))]
        plan[0].update(status='completed', evidence=['order'])
        self.responses.extend([update('end', plan), text_stream('待核对')])
        with self.assertRaises(ValueError):
            _ = [e async for e in self.run_agent()]
        self.snapshot['rounds'] = 12
        self.snapshot['verification'].update(status='correcting', candidate=None, checks=2, corrections=2)
        events = [e async for e in self.run_agent(self.snapshot)]
        self.assertEqual(events[-1]['verification']['status'], 'unverified')
        self.assertEqual(len(self.requests), 4)
