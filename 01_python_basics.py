"""第一课：用模拟客服数据认识 Python 基础语法。

三引号可以包住多行字符串；文件开头这样的文字叫“模块文档字符串”。
本文件会按从上到下的顺序执行。所有订单都是教学用的模拟数据。
第一遍先理解前四部分；JSON 和异常处理可以在看完订单查询后再读。
"""

# import 用来引入模块，类似 TypeScript 的 import。
# json 是 Python 自带的标准库，本节不需要 pip 安装第三方包。
import json


print("=== 1. 变量：给数据起名字 ===")

# Python 给变量赋值时不写 let / const，也不需要行末分号。
# 这里的变量名使用 snake_case（小写单词之间加下划线）。
customer_name = "小林"  # str：字符串。
order_count = 2  # int：整数。
is_vip = False  # bool：布尔值，Python 写 True / False，首字母大写。
selected_order = None  # None 表示暂时没有值，可类比这里使用的 null。

# f 开头的是格式化字符串，花括号里的表达式会被替换为实际值。
print(f"客户：{customer_name}，订单数量：{order_count}")
print("是否会员：", is_vip, "当前选中的订单：", selected_order)


print("\n=== 2. dict：把一笔订单的字段放在一起 ===")

# dict 是字典，适合表示接口中的一条结构化记录。
# 对于这种字符串键的数据，你可以先类比 TypeScript 的普通对象。
order = {
    "order_id": "O1001",
    "status": "已发货",
    "logistics": "包裹已到达武汉转运中心",
}

# 通过键访问字段；Python 字典使用 order["status"]，不是 order.status。
print("订单状态：", order["status"])

# .get() 在字段不存在时返回指定默认值，避免直接访问触发 KeyError。
print("预计送达：", order.get("estimated_delivery", "暂未提供"))


print("\n=== 3. list 和 for：保存并遍历多条消息 ===")

# list 类似数组。这里的每一项又是一个 dict，形成“列表里放字典”。
# 后续调用模型时，会用到类似结构的消息列表；此处仅演示数据结构。
messages = [
    {"role": "user", "content": "帮我查询 O1001。"},
    {"role": "assistant", "content": "以下是订单查询结果。"},
]

# for 的这一行以冒号结束，下面缩进 4 个空格的代码属于循环体。
# Python 使用缩进划分代码块，不使用 JavaScript 的大括号。
for message in messages:
    print(f"{message['role']}：{message['content']}")

print("消息数量：", len(messages))  # len() 类似读取数组的 length。


print("\n=== 4. if / else：根据数据选择下一步 ===")

# Python 中用 == 比较值；= 用于赋值。
if order["status"] == "已发货":
    print("可以展示物流信息。")
else:
    print("这笔订单暂时没有发货物流。")


print("\n=== 5. JSON：在 Python 对象和文本之间转换 ===")

# dumps() 将 Python 字典变成 JSON 字符串，可类比 JSON.stringify()。
# ensure_ascii=False 让中文直接显示；indent=2 让输出按两格缩进排版。
json_text = json.dumps(order, ensure_ascii=False, indent=2)
print(json_text)

# loads() 将 JSON 字符串转回 Python 对象，可类比 JSON.parse()。
# “s”可以记作处理字符串；这里不涉及网络请求或文件保存。
restored_order = json.loads(json_text)
print("转换回字典后读取订单号：", restored_order["order_id"])


print("\n=== 6. try / except：处理可能失败的操作 ===")

# 模拟收到格式错误的数据，演示如何捕获异常。不会修改任何文件。
invalid_json = '{"order_id": "O1001",}'

try:
    # 标准 JSON 不接受对象末尾多出的逗号，所以这一行会产生异常。
    json.loads(invalid_json)
except json.JSONDecodeError as error:
    # except 类似 catch。这里只捕获我们预期的 JSON 解析异常。
    print("已捕获预期的 JSON 格式错误：", error.msg)

# 异常已经被上面处理，因此程序能继续执行到这里。
print("\n基础演示完成。下一步阅读 orders.py，再运行 02_query_order.py。")
