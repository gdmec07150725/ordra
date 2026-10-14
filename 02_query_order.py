"""第二个可运行示例：从终端读取订单号，调用函数，打印中文 JSON。

在项目目录中运行：./.venv/Scripts/python.exe -X utf8 ./02_query_order.py O1001
不提供订单号时默认查询 O1001；订单编号的第一个字符是字母 O。
"""

import json  # 标准库：处理 JSON。
import sys  # 标准库：读取启动参数、设置程序退出码。

# 从同目录的 orders.py 引入 get_order 函数；这里不写 .py 后缀。
from orders import get_order


def main() -> int:
    """程序的主要流程。返回 0 表示正常结束，非 0 表示本次输入有问题。"""

    # sys.argv 是命令行参数列表：第 0 项是脚本路径，第 1 项才是订单号。
    # 我们只接受一个订单号，避免误把多段输入当作一笔订单。
    if len(sys.argv) > 2:
        print("请只提供一个订单号，例如：02_query_order.py O1001")
        return 2

    if len(sys.argv) == 2:
        order_id = sys.argv[1]
    else:
        order_id = "O1001"

    print("正在查询教学用模拟订单：", order_id)

    try:
        # 这里是程序明确调用函数，还不是由模型决定调用工具。
        result = get_order(order_id)
    except (TypeError, ValueError) as error:
        # 圆括号内列出两种预期异常，不会掩盖其他未知编程错误。
        print("输入有问题：", error)
        return 1

    # “订单不存在”是函数正常返回的业务结果，用 error 字段表达。
    # 它与上面“参数类型或内容不合法”的异常是两种处理路径。
    if "error" in result:
        print("查询结果：没有找到这笔订单。")
    else:
        print("查询结果：")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# 直接运行本文件时，Python 会把 __name__ 设为 "__main__"。
# 如果其他文件 import 本模块，这个条件不成立，不会自动启动查询。
if __name__ == "__main__":
    # main() 先执行，sys.exit() 再把它的返回值作为退出码交给终端。
    sys.exit(main())
