"""第九课：从 SQLite 找回摘要遗漏的原话。默认离线，--live 才请求模型。

仅使用临时数据库和虚构会话，不读取或修改网页已有聊天记录。
为了稳定复现“摘要漏掉细节”，本例直接放入一份故意不含收货时间的摘要。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import anyio
import httpx2
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

from agent_service import describe_error, new_conversation, stream_agent
from context_memory import ContextMemory
from session_store import SessionStore


def seed_demo(store: SessionStore) -> None:
    store.initialize()
    history = new_conversation() + [
        {"role": "user", "content": "O1001 的收货时间要求：请周五18点之后再送达。"},
        {"role": "assistant", "content": "好的，我了解你的收货时间要求。"},
        {"role": "user", "content": "我还在学习对话的上下文管理。"},
        {"role": "assistant", "content": "可以继续提问。"},
    ]
    store.create_session("demo", history)
    store.replace_history("demo", history, ContextMemory("用户查询过 O1001，具体收货要求未记录。", 3))
    # 相似关键词出现在另一个会话，也不应该被当前会话查到。
    store.create_session("other", new_conversation() + [
        {"role": "user", "content": "O1001 请周一上午送达，发票抬头写另一家公司。"},
        {"role": "assistant", "content": "已了解。"},
    ])


async def demo_live(store: SessionStore) -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False, encoding="utf-8-sig")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("请先在项目 .env 中配置 DEEPSEEK_API_KEY。")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash"

    async def search_current_session(query: str, order: str) -> dict:
        return await anyio.to_thread.run_sync(store.search_history, "demo", query, order)

    try:
        async with AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0, max_retries=0) as client:
            for question in ("我最开始对 O1001 的收货时间有什么要求？请核对原话。", "我之前说过的发票抬头是什么？"):
                # 每轮从磁盘重新加载，观察摘要与原始历史分别起什么作用。
                stored = store.load_session("demo")
                print("\n提问：", question, flush=True)
                async for event in stream_agent(client, model, stored["history"], question, memory=stored["memory"], search_history=search_current_session):
                    if event["event"] == "tool_start":
                        print("模型请求工具：", event["name"], flush=True)
                    elif event["event"] == "tool_result" and event["name"] == "search_history":
                        print("实际检索结果：", json.dumps(event["result"], ensure_ascii=False), flush=True)
                    elif event["event"] == "complete":
                        if not event["retrieval"]["calls"]:
                            raise ValueError("这次模型没有调用历史检索；演示未通过，请检查提示词或重新运行。")
                        store.replace_history("demo", event["history"], event["memory"])
                        print("回答：", event["answer"])
                        print("检索统计：", event["retrieval"])
                        print("回答阶段实际用量：", event["usage"])
                        print("最近请求估算输入：", event["context"]["estimated_input_tokens"])
    except (APIError, httpx2.HTTPError, ValueError) as error:
        raise ValueError(describe_error(error, api_key)) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="第九课：历史检索与摘要遗漏恢复。")
    parser.add_argument("--live", action="store_true", help="请求真实 DeepSeek 验证工具选择与两次回答，会消耗额度")
    args = parser.parse_args()
    try:
        with TemporaryDirectory(prefix="history-demo-") as directory:
            store = SessionStore(Path(directory) / "chat.db")
            seed_demo(store)
            stored = store.load_session("demo")
            print("预先准备的摘要（故意遗漏时间）：", stored["memory"].text)
            print("近期原文：", stored["history"][stored["memory"].covered_messages:])
            if args.live:
                asyncio.run(demo_live(store))
            else:
                print("\n离线模式：下面直接调用 Python 检索函数，不代表模型自行选择工具。")
                for query in ("O1001 收货", "发票 抬头"):
                    print(f"\n搜索 {query!r}：")
                    print(json.dumps(store.search_history("demo", query, "oldest"), ensure_ascii=False, indent=2))
                print("\n添加 --live 可观察：模型请求工具 → Python 搜索 → 模型根据原文回答。")
    except ValueError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
