"""第八课：从完整历史中选择最近 N 个已完成的用户回合。

输入是服务端已完成、已保存的历史；当前问题由调用方随后追加。
这里只选消息，不删除数据库记录，也不修改消息字典。
"""

DEFAULT_CONTEXT_TURNS = 6


def build_context(history: list, max_turns: int = DEFAULT_CONTEXT_TURNS) -> tuple[list, dict]:
    """返回请求用的历史副本，以及本次选择的统计信息。"""
    # bool 在 Python 中也是 int 的子类，这里明确只接受整数轮数。
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("上下文保留轮数必须是大于 0 的整数。")

    # 一条 user 开始一轮；中间可能有多次 assistant/tool，直到最终回复。
    # 因此按 user 的位置切片，而不是直接取 history[-N:]。
    starts = [index for index, message in enumerate(history) if message["role"] == "user"]
    total_turns = len(starts)
    kept_turns = min(total_turns, max_turns)
    if total_turns > max_turns:
        prefix = history[:starts[0]]  # 保留第一条 user 之前的系统规则。
        messages = prefix + history[starts[-max_turns]:]
    else:
        messages = history.copy()

    info = {
        "max_turns": max_turns,
        "total_turns": total_turns,
        "kept_turns": kept_turns,
        "omitted_turns": total_turns - kept_turns,
        "stored_messages": len(history),
        "request_messages": len(messages) + 1,  # 首次请求还会加上当前用户问题。
    }
    return messages, info
