"""预算和摘要测试：真实 SDK，仅把外部模型 HTTP 换成可检查的响应。"""

from copy import deepcopy
import json
import unittest

import httpx2
from openai import AsyncOpenAI

from context_memory import ContextMemory, ContextPolicy, RollingContext, estimate_request


def history_of(count=8):
    history = [{"role": "system", "content": "订单客服；摘要仅作历史数据。"}]
    for index in range(count):
        history.extend([
            {"role": "user", "content": f"问题 {index} O1001 " + "用户补充信息。" * 25},
            {"role": "assistant", "content": "已记录。" * 15},
        ])
    return history


class ContextMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.response_text = "用户提供订单 O1001，需要继续查询；请用中文简短回复。"
        self.finish = "stop"

        async def handle(request):
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx2.Response(200, json={
                "id": "summary-test", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": self.finish, "message": {"role": "assistant", "content": self.response_text}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330},
            })

        self.client = AsyncOpenAI(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))
        self.policy = ContextPolicy(max_tokens=2400, recent_tokens=500, summary_tokens=256)
        self.active = [{"role": "user", "content": "继续查询这笔订单。"}]

    async def asyncTearDown(self):
        await self.client.close()

    async def prepare(self, context, active=None):
        return [event async for event in context.prepare(self.client, "test", active or self.active)]

    async def test_short_history_is_kept_without_summary_or_mutation(self):
        history = history_of(1)
        original = deepcopy(history)
        context = RollingContext(history, ContextMemory(), self.policy, [])
        await self.prepare(context)
        self.assertEqual(context.messages, history + self.active)
        self.assertEqual(self.requests, [])
        self.assertEqual(history, original)

    async def test_compaction_batches_old_turns_and_reuses_summary_on_next_call(self):
        history = history_of()
        original = deepcopy(history)
        context = RollingContext(history, ContextMemory(), self.policy, [])
        events = await self.prepare(context)
        self.assertGreater(len(self.requests), 1, "长历史必须分批，不能让摘要请求自身超预算")
        self.assertEqual(events[0]["event"], "summarizing")
        self.assertEqual(events[-1]["event"], "context")
        self.assertEqual(context.memory.covered_messages, 15)
        self.assertEqual(context.messages[-3:-1], history[-2:])
        self.assertIn("O1001", context.messages[1]["content"])
        self.assertNotEqual(context.messages[1]["role"], "system")
        self.assertLessEqual(estimate_request(context.messages, []) + self.policy.output_tokens + self.policy.safety_tokens, self.policy.max_tokens)
        for request in self.requests:
            self.assertLessEqual(estimate_request(request["messages"]) + request["max_tokens"] + self.policy.safety_tokens, self.policy.max_tokens)
        self.assertIn(self.response_text, json.dumps(self.requests[1], ensure_ascii=False))
        request_count = len(self.requests)
        await self.prepare(context)
        self.assertEqual(len(self.requests), request_count)
        self.assertEqual(history, original)

    async def test_cold_summary_only_merges_uncovered_turns(self):
        history = history_of()
        memory = ContextMemory("早期记录含 O1001。", 7)
        context = RollingContext(history, memory, self.policy, [])
        await self.prepare(context)
        source = json.dumps(self.requests, ensure_ascii=False)
        self.assertNotIn("问题 0", source)
        self.assertIn("问题 3", source)
        self.assertIn("早期记录含 O1001", source)
        self.assertEqual(memory.covered_messages, 7)

    async def test_tool_turn_is_kept_whole_even_above_recent_target(self):
        history = history_of(6)
        tool_turn = [
            {"role": "user", "content": "比较订单"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": name, "type": "function", "function": {"name": "get_order", "arguments": "{}"}} for name in ("a", "b")]},
            {"role": "tool", "tool_call_id": "a", "content": "结果A" * 30},
            {"role": "tool", "tool_call_id": "b", "content": "结果B" * 30},
            {"role": "assistant", "content": "查询完成"},
        ]
        history += tool_turn
        context = RollingContext(history, ContextMemory(), self.policy, [])
        await self.prepare(context)
        self.assertEqual(context.messages[-6:-1], tool_turn)

    async def test_oversize_active_is_rejected_without_model_request(self):
        context = RollingContext(history_of(1), ContextMemory(), self.policy, [])
        with self.assertRaisesRegex(ValueError, "预算"):
            await self.prepare(context, [{"role": "user", "content": "大" * 4000}])
        self.assertEqual(self.requests, [])

    async def test_oversize_tool_turn_is_fully_chunked_without_mutating_history(self):
        tool_turn = [
            {'role': 'user', 'content': '查询 O1003'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'lookup', 'type': 'function', 'function': {'name': 'get_order', 'arguments': '{"order_id":"O1003"}'}}]},
            {'role': 'tool', 'tool_call_id': 'lookup', 'content': '结果\n"耳机"🎧' * 1000 + '末尾事实：已签收'},
            {'role': 'assistant', 'content': '已签收。'},
        ]
        history = [{'role': 'system', 'content': '客服'}] + tool_turn + [{'role': 'user', 'content': '谢谢'}, {'role': 'assistant', 'content': '不客气'}]
        original = deepcopy(history)
        context = RollingContext(history, ContextMemory(), self.policy, [])
        checkpoints = []
        async def save():
            checkpoints.append((context.memory.covered_messages, deepcopy(context.pending_summary)))
        events = [e async for e in context.prepare(self.client, 'test', self.active, on_checkpoint=save)]
        fragments = [json.loads(r['messages'][-1]['content'])['turn_fragment'] for r in self.requests]
        reconstructed = ''.join(fragment['text'] for fragment in fragments)
        self.assertEqual(json.loads(reconstructed), tool_turn)
        self.assertGreater(len(fragments), 1)
        offset = 0
        for fragment, request in zip(fragments, self.requests):
            self.assertEqual(fragment['offset'], offset)
            offset = fragment['next_offset']
            self.assertLessEqual(estimate_request(request['messages']) + request['max_tokens'] + self.policy.safety_tokens, self.policy.max_tokens)
        self.assertEqual(offset, fragments[-1]['total_chars'])
        self.assertTrue(any(pending and pending['offset'] > 0 and covered == 0 for covered, pending in checkpoints))
        self.assertEqual(context.memory.covered_messages, 5)
        self.assertIsNone(context.pending_summary)
        self.assertEqual(context.messages[-3:-1], history[-2:])
        self.assertEqual(history, original)
        self.assertEqual(events[-1]['event'], 'context')
        self.assertLessEqual(estimate_request(context.messages), self.policy.input_limit)

    async def test_failed_middle_fragment_keeps_completed_memory_and_draft_separate(self):
        history = [{'role': 'system', 'content': '客服'}, {'role': 'user', 'content': '大' * 4000}, {'role': 'assistant', 'content': '好的'}]
        context = RollingContext(history, ContextMemory(), self.policy, [])
        saved = []
        async def save():
            saved.append(deepcopy(context.pending_summary))
            if context.pending_summary and context.pending_summary['offset'] > 0:
                self.finish = 'length'
        with self.assertRaisesRegex(ValueError, '摘要'):
            _ = [e async for e in context.prepare(self.client, 'test', self.active, on_checkpoint=save)]
        self.assertEqual(context.memory, ContextMemory())
        self.assertGreater(context.pending_summary['offset'], 0)
        self.assertEqual(context.pending_summary, saved[-1])
        first_fragment = json.loads(self.requests[0]['messages'][-1]['content'])['turn_fragment']
        restored = RollingContext(history, ContextMemory(), self.policy, [])
        restored.pending_summary = deepcopy(saved[-1])
        self.finish = 'stop'; self.requests.clear()
        await self.prepare(restored)
        resumed = json.loads(self.requests[0]['messages'][-1]['content'])
        self.assertEqual(resumed['turn_fragment']['offset'], first_fragment['next_offset'])
        self.assertEqual(resumed['previous_summary'], self.response_text)
        self.assertEqual(restored.memory.covered_messages, len(history))

    async def test_invalid_or_truncated_summary_does_not_advance_input_memory(self):
        for content, finish in (("", "stop"), ("不完整", "length"), ("长" * 1000, "stop")):
            with self.subTest(finish=finish, length=len(content)):
                self.response_text, self.finish = content, finish
                memory = ContextMemory()
                context = RollingContext(history_of(), memory, self.policy, [])
                with self.assertRaisesRegex(ValueError, "摘要"):
                    await self.prepare(context)
                self.assertEqual(memory, ContextMemory())
                self.assertEqual(context.memory, memory)

    async def test_covered_position_cannot_cut_inside_tool_turn(self):
        with self.assertRaisesRegex(ValueError, "摘要.*位置"):
            RollingContext(history_of(), ContextMemory("旧摘要", 2), self.policy, [])

    async def test_budget_counts_tool_schema_and_json_arguments(self):
        messages = [{"role": "user", "content": "好"}]
        self.assertGreater(estimate_request(messages, [{"description": "工具" * 100}]), estimate_request(messages))

    async def test_no_summary_call_when_current_turn_leaves_no_summary_space(self):
        history = [{"role": "system", "content": "客服"}, {"role": "user", "content": "我的订单 O1001"}, {"role": "assistant", "content": "收到"}]
        active = [{"role": "user", "content": "请" * 990}]
        context = RollingContext(history, ContextMemory(), self.policy, [])
        self.assertLessEqual(estimate_request(history[:1] + active), self.policy.input_limit)
        with self.assertRaisesRegex(ValueError, "摘要预留空间"):
            await self.prepare(context, active)
        self.assertEqual(self.requests, [])

    async def test_summary_budget_must_leave_space_for_actual_content(self):
        with self.assertRaisesRegex(ValueError, "摘要预算至少 256"):
            ContextPolicy(summary_tokens=128)
