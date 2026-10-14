# 第六课：用 FastAPI 把 Agent 接到网页上

> 本文按课程阶段讲解；请在仓库根目录执行命令。安装与当前功能以[仓库 README](../README.md)为准，早期章节中的简化流程不代表最终实现。

> 版本说明：本篇记录第六课的内存会话设计。项目的共享服务现已按[第七课](07_SQLite会话持久化.md)升级为 SQLite 持久化：普通重启不会清空已保存历史，并且先保存成功、再返回完成信号。下文关于内存会话和完成时机的说明用于理解第六课，会话保存行为以第七课为准，上下文选择以[第八课](08_上下文管理.md)的预算与滚动摘要为准。启动命令和原六个接口仍然沿用，另增加会话列表和删除接口；网页“新对话”现为创建独立会话，可在历史列表中切换或删除。

上一课在终端逐段显示模型回复。这一课让同一个订单 Agent 接收网页请求，并把文字片段送回浏览器。

本课新增代码仍在 `.`，关键位置有中文注释。先运行、再看接口主流程，最后理解异步清理；不用一次记住所有新语法。

## 1. 先打开页面

当前机器已安装依赖。每次需要启动服务时，在 PowerShell 执行：

```powershell
# 在仓库根目录运行
.\.venv\Scripts\python.exe -X utf8 .\07_api_server.py
```

保持终端运行，然后打开：

- 聊天页面：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- 接口文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 服务状态：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

如果页面已经能打开，说明服务正在运行，无须再次启动。出现“地址已被使用”时，先检查原来的服务窗口；不要同时启动两个相同端口的服务。

本课继续读取原有 `.env` 中的 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`。密钥由 Python 读取，页面不需要填写。修改 Python、订单数据或 `.env` 后需重启服务；修改网页文件后刷新即可。本课没有开启自动重载。

自己在终端启动的服务可用 `Ctrl+C` 关闭。会话仅存在当前服务进程的内存中，服务重启后清空；单纯刷新网页可以恢复仍在运行的服务中已完成的对话。

## 2. 先试三个动作

1. 输入“我的 O1001 订单到哪里了？”，观察“查询订单”和逐段出现的回答。
2. 接着输入“这笔订单买的是什么？”，观察模型怎样利用上一轮的订单号。
3. 点击“新对话”，再输入“帮我查订单”，这次应先询问订单号。

O1001 开头是英文字母 **O**。当前模拟数据中，O1001 包含“无线键盘”，物流为“包裹已到达武汉转运中心”；页面会使用你在 `orders.py` 中保存的数据。

Enter 发送，Shift + Enter 换行。生成期间暂停输入，完成后可以继续追问。

## 3. HTTP 层的作用

```text
浏览器 web/app.js
    │ POST /api/chat/stream
    │ {session_id, message}
    ▼
FastAPI：api_app.py
    │ 校验请求 → 找到会话 → 取得历史
    ▼
Agent：agent_service.py
    │ 请求 DeepSeek → 收齐工具参数 → get_order()
    │ 把工具结果发回 DeepSeek → 接收回复片段
    ▼
FastAPI：把事件编码为 SSE，完整结束后保存历史
    │ delta / done / error 等事件
    ▼
浏览器：逐段更新聊天文字
```

模型与工具的往返过程沿用上一课。主要变化是输入来自 HTTP 请求，输出交给 HTTP 响应。

为了方便理解接口，这次页面使用原生 `fetch`。以后换成 React 时，仍然发送同样的 JSON、读取同样的 SSE；把页面节点更新换成 React state 即可。

## 4. 按这个顺序阅读代码

| 文件 | 负责什么 | 第一遍重点看哪里 |
|---|---|---|
| `07_api_server.py` | 用 Uvicorn 启动服务 | `uvicorn.run(...)` |
| `api_app.py` | 接口、请求校验、会话和响应 | `ChatRequest`、`create_session`、`chat`、`chat_stream` |
| `agent_service.py` | 异步模型与工具循环 | `stream_agent()` |
| `web/app.js` | 发送问题、解析 SSE、显示回复 | 表单 submit 回调与 `readEvents()` |
| `web/index.html` | 聊天页面结构 | form、textarea 和消息容器 |
| `web/styles.css` | 页面样式与窄屏布局 | 有需要时再看 |
| `tests/test_api_app.py` | 离线验证接口行为 | 学完主流程后再读 |

`01` 至 `06` 的独立脚本仍然可以照常运行。新的 `agent_service.py` 使用普通模块名，方便 `api_app.py` 导入。

## 5. FastAPI 的路由是什么

在 `api_app.py` 中可以看到：

```python
@app.get("/api/health")
async def health():
    return {"status": "ok", "model": app.state.model, "configured": bool(app.state.api_key)}
```

`@app.get(...)` 是装饰器。这里把下面的函数登记为一个 GET 接口：浏览器请求这个路径，FastAPI 就执行这个函数，并把返回的字典转成 JSON 响应。

你可以把它和前端熟悉的“给某个事件注册处理函数”联系起来：这里注册的是 URL 和 HTTP 方法。

`uvicorn.run("api_app:app", ...)` 中的 `api_app:app` 表示“导入 `api_app.py`，找到里面名为 `app` 的应用对象”。Uvicorn 负责接收网络连接，FastAPI 负责找到对应的处理函数。

本课的接口如下：

| 方法与路径 | 用途 |
|---|---|
| `GET /api/health` | 检查服务、模型名称和是否配置密钥 |
| `POST /api/sessions` | 创建会话，返回 `session_id` |
| `GET /api/sessions/{session_id}` | 查看已完成历史和忙碌状态 |
| `POST /api/sessions/{session_id}/reset` | 清空一个空闲会话 |
| `POST /api/chat` | 等完整回答后返回一次 JSON |
| `POST /api/chat/stream` | 逐个返回 SSE 事件 |

## 6. 请求类型会在运行时校验

两个聊天接口都接收：

```json
{
  "session_id": "创建会话时返回的编号",
  "message": "我的 O1001 订单到哪里了？"
}
```

对应的 Python 类型是：

```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)
```

`class` 定义类，`BaseModel` 是 Pydantic 提供的基类。这里先把它理解为“带检查规则的请求结构”。

它看起来类似 TypeScript 的请求类型，但多做了一件事：**真正检查网络传入的数据**。普通 Python 类型标注和 TS 类型声明都不会自动拦截错误 JSON；FastAPI 配合 Pydantic 会执行这一步检查。

- `strict=True`：问题必须真的是字符串，不把数字自动改成字符串。
- `str_strip_whitespace=True`：去掉字符串首尾的空白。
- `Field(...)`：规定长度，空问题或超过 2000 字符会被拒绝。
- `extra="forbid"`：拒绝未声明的字段，例如浏览器擅自传入完整 `messages`。

失败时 FastAPI 返回 HTTP 422，Agent 不会请求模型。路由参数写成 `body: ChatRequest` 后，FastAPI 会将请求 JSON 解析为这个类型；代码中用 `body.message` 取得字段。

## 7. session_id 怎么保留上下文

第四、五课只有一个终端会话，用一个 `history` 列表即可。HTTP 每次都是一次独立请求，所以需要明确“这句话属于哪段对话”。

本课在 `app.state.sessions` 中保存一个字典：

```text
会话 A 的编号 → Session(history=A 的消息列表, busy=False)
会话 B 的编号 → Session(history=B 的消息列表, busy=False)
```

`@dataclass` 帮我们为这个数据类生成初始化等基础方法。`field(default_factory=new_conversation)` 表示每创建一个 Session，都调用函数得到新的列表。这样不同会话不会共享同一个历史列表。

浏览器的 `sessionStorage` 只记住会话编号。发送下一句话时仍带上这个编号，后端就能找到对应历史，追加问题，再发给模型。

网页刷新后会查询服务端历史。历史包含 `system`、`user`、带工具请求的 `assistant`、`tool` 和最终 `assistant`；网页聊天区只显示普通问答。保存历史接口便于在 `/docs` 中观察这些角色如何配对。

这只是本机教学服务的会话隔离。编号不等同于登录身份；本课仅监听 `127.0.0.1`，使用单进程内存，不包含用户登录或数据库持久化。

## 8. 为什么从 OpenAI 改成 AsyncOpenAI

上一课同步执行请求时，终端本来就是等这一条结果。这次服务需要在等待模型时，继续响应其他请求，因此改用异步客户端：

```python
stream = await client.chat.completions.create(..., stream=True)
try:
    async for chunk in stream:
        ...
finally:
    await stream.close()
```

这里省略了清理时的取消保护，完整代码使用 `CancelScope(shield=True)` 保证关闭连接的等待可以完成。

和 TypeScript 对照：

| Python 写法 | 可以怎样理解 |
|---|---|
| `async def` | 定义异步函数，类似 `async function` |
| `await` | 等待结果期间允许其他异步任务继续运行 |
| `async for` | 逐个等待异步到来的元素，类似 `for await...of` |
| `async with` | 进入资源上下文，退出时执行异步清理 |
| `yield` | 交出一个值并暂停，下一次迭代再继续 |

只在函数名前加 `async` 并不会让里面所有同步操作自动变成异步。本课的模型网络请求使用 `AsyncOpenAI`；`get_order` 只是一次很短的本地字典查询，所以仍使用普通函数。

`stream_agent()` 同时含有 `async def` 和 `yield`，因此是**异步生成器**：它会陆续交出事件，而不是最后只返回一个值。调用方这样接收：

```python
async for event in stream_agent(client, model, history, question):
    # 每次得到一个字典，例如 {"event": "delta", "text": "已发货"}
    ...
```

## 9. 为什么同时提供 JSON 和 SSE

两个接口使用同一个 `stream_agent()`，区别在于怎样把事件交给浏览器。

普通接口 `/api/chat` 收集事件，等到 `complete` 后一次返回：

```json
{
  "session_id": "...",
  "answer": "订单 O1001 已发货。",
  "usage": {"prompt_tokens": 400, "completion_tokens": 20, "total_tokens": 420, "reported_rounds": 1}
}
```

上面的数字只是格式示例。实际用量来自模型返回值，`reported_rounds` 表示有多少次模型请求报告了用量。

流式接口 `/api/chat/stream` 则使用 `StreamingResponse` 逐个发送事件，例如：

```text
event: delta
data: {"text": "订单 O1001"}

event: delta
data: {"text": " 已发货。"}

event: done
data: {"answer": "订单 O1001 已发货。", "session_id": "...", "usage": {"total_tokens": 420, "reported_rounds": 1}}

```

这里展示了事件格式，省略了部分用量字段。事件之间的**空行**是分隔符；整份响应不是单个 JSON，不能直接 `response.json()`。

SSE 是服务器向浏览器单向发送事件的格式。我们要用 POST 提交问题，所以页面用 `fetch` 读取响应体流并解析事件，而没有使用浏览器的 `EventSource` 接口。

本课的事件约定：

| 事件 | 页面如何处理 |
|---|---|
| `round` | 开始下一次模型请求，清理本轮临时文字 |
| `delta` | 把新文字追加到当前回复 |
| `tool_start` | 显示正在查询订单 |
| `tool_result` | 显示查询结束，准备整理回复 |
| `usage` | 携带目前累计的用量；页面在 done 时统一显示 |
| `done` | 用完整正文确认最终回复，可以继续提问 |
| `error` | 显示错误，本次不作为成功回复 |

Agent 内部的 `complete` 还携带候选历史。接口将它转换成 `done`，等响应层接受这帧的发送后保存候选历史；不把内部完整历史夹在这个事件里发送。

模型服务的 chunk、后端 SSE 事件、浏览器一次 `reader.read()`，是三种不同的分片边界。浏览器可能一次读到半个中文字符、半个事件或多个事件。所以 `readEvents()` 同时使用：

1. `TextDecoder(..., {stream: true})` 保留未收齐的 UTF-8 字节。
2. `buffer` 保留尚未出现空行分隔符的文本。
3. 收齐事件后才 `JSON.parse`，再根据事件名更新页面。

## 10. 完成、失败和断连分别怎么处理

每个问题从成功历史的副本开始。只有模型正常完成、正文有效时才产生 `complete`，接口才会保存这轮历史；SSE 还要等完成事件交给响应层发送，发送失败时不保存。

工具参数仍然要按调用序号拼接，并检查结束状态、调用 ID 和名称；不在收到半个参数时执行函数。每个问题最多请求模型 4 次。

同一会话正在生成时，`busy=True`，新的提问或清空请求返回 409。检查 busy 与设置 busy 之间没有 `await`，在本课的单进程事件循环中这两步不会被另一个请求插入。

| 情况 | 返回或处理方式 |
|---|---|
| 请求字段不符合要求 | HTTP 422 |
| 会话不存在 | HTTP 404，刷新页面可重新创建 |
| 会话正在生成 | HTTP 409 |
| 服务端尚未配置密钥 | HTTP 503 |
| JSON 接口遇到模型服务错误 | HTTP 502 和错误说明 |
| SSE 已经开始后遇到模型错误 | `error` 事件；已发出的 HTTP 状态不能再修改 |
| SSE 浏览器断连 | 取消当前等待、关闭模型流、释放 busy |

流中已显示的文字不代表这一轮已经完成。遇到 `length`、缺少结束状态或连接异常时，服务端不保存半轮历史；错误信息会隐藏当前配置的 API Key。

第一遍可以先跳过 `run_until_disconnect` 和 `SessionStreamResponse` 的细节。它们负责同时处理请求、监听断连，并在 `finally` 中结束任务、关闭生成器和释放会话；`aclosing()` 帮助确保异步生成器提前退出时也能关闭。`CancelScope(shield=True)` 让清理过程有机会完成。

普通 JSON 和 SSE 接口都使用独立的断连监听任务，因此等待模型出字期间也可以及时取消。

服务端完成保存后，最后的网络传输仍可能失败。因此浏览器如果拿不到 `done`，会标记未完成，刷新后以服务端已保存的历史为准。这一课没有加入请求重试与去重机制。

## 11. 在 /docs 里练普通 JSON 接口

1. 打开 `/docs`，展开 `POST /api/sessions`，点击 **Try it out → Execute**。
2. 复制响应里的 `session_id`。
3. 展开 `POST /api/chat`，点击 **Try it out**，填写这个编号和问题后执行。
4. 保持同一个编号，再问“这笔订单是什么商品？”。
5. 用 `GET /api/sessions/{session_id}` 查看完整消息角色及工具结果。

`/docs` 是本课的接口调试入口。想观察逐段显示，使用聊天页面；在浏览器开发者工具的 Network 中也可以检查 `/api/chat/stream` 的响应。

## 12. 两个小练习

**练习一：解释一次请求经过的函数。** 从 `web/app.js` 的 submit 回调开始，依次指出请求在哪校验、history 在哪找到、工具在哪执行、完整历史在哪保存。

**练习二：缩短问题长度限制。** 将 `ChatRequest.message` 的 `max_length` 临时改成 20，重启服务，再从 `/docs` 发送一个超过 20 字符的问题。观察 HTTP 422 与页面上的长度限制分别在哪一层生效，最后改回 2000。

完成这两点就掌握了本课主线。后续可以把内存会话存入 SQLite，学习数据库读写，让服务重启后仍能继续原来的对话；也可以先将当前页面换成 React，沿用现有接口。

## 13. 离线验证

接口测试使用真实 SDK 与模拟模型传输，覆盖 JSON/SSE、会话隔离、取消及事务保存边界。

运行离线验证：

```powershell
# 在仓库根目录运行
.\.venv\Scripts\python.exe -X utf8 -W error -m unittest discover -s tests -v
```

官方延伸阅读：[FastAPI 的异步处理](https://fastapi.tiangolo.com/async/) 解释 `async def` 与等待网络的关系；[StreamingResponse 文档](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) 介绍如何使用生成器返回响应流。
