"""模拟订单数据和查询函数：后续可以把这个函数接成 Agent 工具。

当前只是普通 Python 函数，没有模型调用，也没有数据库连接。
本文件不在顶层打印或执行查询，因此 import 它不会自动运行演示。
"""

# 大写变量名是“按约定当作常量使用”的风格，并不强制禁止修改。
# 外层字典的键是订单号，值也是一个字典，用来保存订单详情。
ORDERS = {
    "O1001": {
        "status": "已发货",
        "logistics": "包裹已到达武汉转运中心",
        "product_name": "无线键盘",
    },
    "O1002": {
        "status": "待发货",
        "logistics": "商家正在备货，暂时没有物流轨迹",
    },
    "O1003": {
    "status": "已签收",
    "logistics": "客户已签收",
},
}


# def 定义函数。冒号之后缩进的内容是函数体。
# order_id: str 表示“期望参数是字符串”，-> dict 表示“期望返回字典”。
# 类型标注帮助阅读和编辑器提示；Python 不会仅凭标注自动校验输入。
def get_order(order_id: str) -> dict:
    """根据订单号返回模拟订单；未找到时返回带 error 字段的字典。

    参数必须是非空字符串，否则抛出异常，由调用方决定怎样提示用户。
    示例：get_order("O1001")。
    """

    # isinstance() 在实际运行时检查类型；传入 1001 等数字会被拒绝。
    if not isinstance(order_id, str):
        raise TypeError("订单号必须是字符串，例如 'O1001'。")

    # strip() 去掉两端空白，upper() 转为大写，让 ' o1001 ' 也能查询。
    normalized_id = order_id.strip().upper()

    # 空字符串在 if 中被视为 False，所以 not normalized_id 表示没有内容。
    if not normalized_id:
        raise ValueError("订单号不能为空，请输入 O1001 或 O1002。")

    # .get() 在订单不存在时返回 None，而不会直接抛出 KeyError。
    order = ORDERS.get(normalized_id)

    # 对 None 的判断使用 is None，这也是 Python 中常用的写法。
    if order is None:
        return {"error": f"未找到订单 {normalized_id}，请核对订单号。"}

    # copy() 创建一份浅拷贝，避免下面添加字段时直接修改原订单字典。
    # 本例字段值全部是字符串，浅拷贝足够；它不会递归复制嵌套对象。
    result = order.copy()
    result["order_id"] = normalized_id

    # return 把结果交给调用者；它不会自动在终端打印。
    return result
