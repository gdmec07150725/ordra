"""第八课：token 预算、滚动摘要、近期完整回合；默认离线，--live 才调用模型。

仅构造演示历史，不读取或修改网页数据库。离线摘要是固定演示数据。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx2
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

from agent_service import TOOLS, describe_error, new_conversation, stream_agent
from context_memory import ContextMemory, ContextPolicy, RollingContext, estimate_request
from orders import get_order


def build_demo_history() -> list:
    """订单只在最早一轮出现，后面的近期回合不重复订单号。"""
    order = get_order("O1001")
    history = new_conversation() + [
        {"role": "user", "content": "请查一下我的订单 O1001，之后请用中文简短回答。"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "demo-order", "type": "function", "function": {"name": "get_order", "arguments": '{"order_id":"O1001"}'}}]},
        {"role": "tool", "tool_call_id": "demo-order", "content": json.dumps(order, ensure_ascii=False)},
        {"role": "assistant", "content": "订单已发货。"},
    ]
    for index in range(9):
        history.extend([
            {"role": "user", "content": f"练习 {index + 1}：" + "我在学习对话历史与上下文的区别，请保留之前的订单信息，之后还要继续追问。" * 4},
            {"role": "assistant", "content": "历史记录完整保存。模型每次看到的是本次选择的上下文；较早内容可以整理成摘要。" * 2},
        ])
    return history


class OfflineSummaryClient:
    """固定演示摘要，只验证预算算法。它不理解输入，也不代表模型能力。"""
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="用户提供订单 O1001，希望中文简短回答。正在学习上下文管理，之后会继续追问该订单；具体状态需重新查工具。"))],
        )


def show_context(info: dict) -> None:
    print(f"已完成 {info['total_turns']} 轮：摘要覆盖 {info['summarized_turns']} 轮，近期完整保留 {info['kept_turns']} 轮。")
    print(f"本次请求估算 {info['estimated_input_tokens']}/{info['input_limit']} 输入 tokens；总预算 {info['max_tokens']} 已扣除回答预留及安全余量。")
    print(f"本轮摘要请求 {info['summary_calls']} 次；摘要实际报告用量：{info['summary_usage']}")


async def demo(history: list, policy: ContextPolicy, live: bool) -> None:
    question = "这笔订单买的是什么？"
    if not live:
        print("离线模式：使用明确写死的演示摘要，不访问模型，不消耗额度。")
        context = RollingContext(history, ContextMemory(), policy, TOOLS)
        async for event in context.prepare(OfflineSummaryClient(), "offline", [{"role": "user", "content": question}]):
            if event["event"] == "summarizing":
                print("正在把较早完整回合合入演示摘要…")
        show_context(context.info)
        print("演示摘要：", context.memory.text)
        print("请求角色：", " → ".join(message["role"] for message in context.messages))
        print(f"原始历史仍有 {len(history)} 条；--live 才能验证模型实际摘要和追问。")
        return

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False, encoding="utf-8-sig")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("请先在项目 .env 中配置 DEEPSEEK_API_KEY。")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash"
    try:
        async with AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0, max_retries=0) as client:
            memory = ContextMemory()
            for question in (question, "它现在到哪里了？"):
                print("\n提问：", question, flush=True)
                async for event in stream_agent(client, model, history, question, policy=policy, memory=memory):
                    if event["event"] == "summarizing":
                        print("正在生成滚动摘要…", flush=True)
                    elif event["event"] == "complete":
                        print("回答：", event["answer"])
                        show_context(event["context"])
                        print("回答阶段实际报告用量：", event["usage"])
                        print("真实摘要：", event["memory"].text)
                        history, memory = event["history"], event["memory"]
    except (APIError, httpx2.HTTPError, ValueError) as error:
        raise ValueError(describe_error(error, api_key)) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="第八课：按预算触发增量摘要，完整历史不删除。")
    parser.add_argument("--budget", type=int, default=4300, help="演示总预算，默认 4300；网页默认 8192")
    parser.add_argument("--recent", type=int, default=500, help="压缩后近期历史目标，默认 500；网页默认 2048")
    parser.add_argument("--live", action="store_true", help="用真实 DeepSeek 摘要并连续追问两次，会消耗额度")
    args = parser.parse_args()
    try:
        policy = ContextPolicy(max_tokens=args.budget, recent_tokens=args.recent)
        history = build_demo_history()
        print(f"完整历史 {len(history)} 条；仅历史与工具定义的估算输入 {estimate_request(history, TOOLS)} tokens。")
        asyncio.run(demo(history, policy, args.live))
    except ValueError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
