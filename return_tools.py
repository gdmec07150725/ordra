"""模型只提出退货预览；用户确认创建的接口不属于模型工具。"""
import json


RETURN_TOOLS = [
    {"type": "function", "function": {
        "name": "prepare_return_request", "description": "根据用户明确提供的订单和原因准备本地模拟退货预览，尚未提交。",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "maxLength": 50},
            "reason": {"type": "string", "minLength": 1, "maxLength": 500, "description": "用户明确提供的退货原因，不能猜测。"}},
            "required": ["order_id", "reason"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_return_request", "description": "查询当前会话中指定订单已提交的本地模拟退货申请。",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "maxLength": 50}},
                       "required": ["order_id"], "additionalProperties": False}}},
]

RETURN_INSTRUCTIONS = (
    "申请退货先查询订单和售后资料，缺少订单号或退货原因先询问，不得猜测原因或自动判定已批准。"
    "信息齐全后用 prepare_return_request 生成预览，每轮一份，告诉用户核对下方卡片并点击确认提交。"
    "预览不是申请成功；聊天中的确认文字不能提交，必须点击卡片。后续对话会使旧未确认预览失效。"
    "已提交的申请用 get_return_request 查当前状态和审核原因，不能只依据旧消息。所有申请均为本地模拟，人工审核通过也不等于已退款，不能承诺完成时间。"
)


def validate_return_arguments(name, arguments):
    expected = {"order_id", "reason"} if name == "prepare_return_request" else {"order_id"}
    if not isinstance(arguments, dict) or set(arguments) != expected:
        raise ValueError("只能提供工具要求的订单号和原因，确认必须通过网页卡片。")
    order_id = arguments['order_id']
    if not isinstance(order_id, str) or not 1 <= len(order_id.strip()) <= 50:
        raise ValueError("请提供有效订单号。")
    if 'reason' in arguments and (not isinstance(arguments['reason'], str) or not 1 <= len(arguments['reason'].strip()) <= 500):
        raise ValueError("请提供 1–500 字的退货原因。")
    return arguments


async def execute_return_tool(name, arguments_json, handler):
    if handler is None:
        return {"error": "当前入口未启用退货申请。"}
    try:
        arguments = validate_return_arguments(name, json.loads(arguments_json))
        return await handler(name, arguments)
    except (ValueError, TypeError, LookupError) as error:
        return {"error": str(error)}
