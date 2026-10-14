# 第二课：向 DeepSeek 发出第一次模型请求

> 本文按课程阶段讲解；请在仓库根目录执行命令。安装与当前功能以[仓库 README](../README.md)为准，早期章节中的简化流程不代表最终实现。

本课的目标是看清一条完整路径：**终端输入 → Python 组织消息 → DeepSeek API → Python 读取回答 → 终端显示**。主文件是 `03_model_chat.py`，每段关键代码都加了中文注释。

运行前请按[仓库 README](../README.md)安装依赖，复制配置示例并填写自己的 API Key。

## 1. 运行和观察

在 PowerShell 执行：

```powershell
# 在仓库根目录运行
.\.venv\Scripts\python.exe -X utf8 .\03_model_chat.py
```

不传参数时，默认问题为“请用一句话介绍你自己。”；也可以传入自己的问题：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\03_model_chat.py '请向有 TypeScript 基础的人解释 Python 的字典。'
```

模型回复和 token 用量会随问题、模型版本与生成结果变化。脚本会打印实际响应；不要把某次输出当成固定答案。

脚本每次运行发送一个问题，不保存上一次的聊天历史。代码关闭了自动重试；需要再次请求时，由你重新运行。

## 2. 新增的文件各有什么用

| 文件 | 用途 | 现在是否需要读 |
|---|---|---|
| `03_model_chat.py` | 读取问题、发送请求、显示回答 | 重点阅读 |
| `.env` | 保存本机 DeepSeek 密钥与模型名 | 首次使用时填写 |
| `.env.example` | 展示配置字段，不含实际密钥 | 配置前参考 |
| `requirements.txt` | 记录使用的第三方库与版本 | 理解它类似依赖清单即可 |
| `tests/test_model_chat.py` | 模拟接口响应，验证成功和失败情况 | 熟悉主流程后再读 |

`.env` 已加入 `.gitignore`，代码通过环境变量读取密钥。首次使用时复制 `.env.example` 为 `.env` 并填写自己的模型配置；已有配置无需覆盖。若终端本身设置了同名环境变量，代码中的 `override=False` 会让终端配置优先。

本课使用 `openai` 兼容客户端和 `python-dotenv` 配置加载库。DeepSeek 官方接口采用兼容格式，官方 Python 示例也使用 `OpenAI` 客户端；请求地址配置为 `https://api.deepseek.com`，密钥来自 DeepSeek。[DeepSeek 首次调用文档](https://api-docs.deepseek.com/zh-cn/)

`requirements.txt` 记录了本机验证过的直接依赖版本。以后重建 `.venv` 时，在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
```

首次使用时请先按仓库 README 创建虚拟环境并安装依赖。

## 3. 按这四段理解主代码

### 第一段：拿到配置和问题

```python
# os.getenv 读取环境变量；第二个参数是变量不存在时的默认值。
api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

# sys.argv[1:] 取出脚本名之后的参数，再将它们连接成一个问题。
question = " ".join(sys.argv[1:]).strip()
```

实际脚本还通过 `load_dotenv()` 加载与脚本放在一起的 `.env`，并处理默认问题与空输入。可以把这一段理解为前端发送接口请求前，先取得配置和用户输入。

### 第二段：把问题放进 messages

```python
messages = [
    # system 提供这次对话的总体要求。
    {"role": "system", "content": "你是一位耐心的中文助手，请简洁、清楚地回答问题。"},
    # user 是用户本次提出的问题。
    {"role": "user", "content": question},
]
```

这里正好用到第一课的 `list` 和 `dict`：外层是消息列表，每条消息是一个字典。模型只能依据请求中提供的消息等上下文作答；它不会因为脚本与 `orders.py` 放在同一目录，就自动读取订单文件。

### 第三段：创建客户端并发送请求

```python
# with 在退出代码块时关闭客户端连接。
with OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=30.0,  # 网络操作的超时配置，单位为秒。
    max_retries=0,  # 本课由用户决定是否重试。
) as client:
    response = client.chat.completions.create(
        model=model,  # 默认使用 deepseek-flash，可通过 .env 配置。
        messages=messages,
        stream=False,  # 等完整响应返回后再处理。
        max_tokens=512,  # 限制输出长度，Token 数不等于字符数。
        extra_body={"thinking": {"type": "disabled"}},  # 本课关闭思考模式。
    )
```

SDK 帮你完成请求序列化、HTTP 通信和响应解析，与前端封装好的 API 客户端相似。本课使用同步调用，执行到这里会等待结果；后续学习流式输出与并发时，再引入相应处理。

本课显式设置 `stream=False` 与关闭思考模式，便于先观察一次普通问答。`extra_body` 用来传入服务提供方的扩展参数；这些字段的含义以 [DeepSeek 创建对话补全文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/) 为准。

### 第四段：读取回答和用量

```python
# choices 是候选回答列表；[0] 取第一个候选。
answer = response.choices[0].message.content
print(answer)
```

`response` 是 SDK 解析出的对象，因此使用点号读取属性；它与第一课直接通过 `result["status"]` 访问字典不同。SDK 已处理 JSON 转换，这里不用自己写 `json.loads()`。

真实脚本会先检查候选列表、回答内容和结束原因，再判断是否成功；同时打印 `usage` 中的输入、输出和总 Token 数。用量是模型处理文本的计量信息，并非字数。回答被长度上限截断时，脚本会提示并返回退出码 2，避免把部分回答当作正常完成。

## 4. 为什么需要异常处理

远程调用可能遇到配置错误、网络超时或服务端限流。脚本以非零退出码报告失败，方便你区分“模型正常回答”和“请求失败”。

服务端返回 HTTP 错误时，脚本显示状态码和原始响应正文，包括其中的具体错误描述；正文为空时，使用 `str(error)` 显示 SDK 异常信息。如果错误正文包含当前 API Key，会将它替换为 `[已隐藏 API Key]`。网络超时、连接失败仍由各自的分支给出提示。

先读成功路径，再看 `except` 分支即可。现在不要求记住所有异常类，也不用自己实现重试或复杂日志。

离线验证覆盖请求地址与消息格式、缺少密钥、空问题、认证失败时的详细错误及密钥隐藏、纯文本错误、空错误正文、超时、空回答、回答截断，共 9 项。它们使用模拟响应，不读取本机密钥，不调用真实服务。需要复查时运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

## 5. 你可以完成的三个小练习

1. 保持代码不变，在终端换一个问题，观察模型回答。
2. 把 `system` 内容改为“你是一位中文客服助手，请用两句话以内回答”，再运行相同问题，观察要求是否影响表达。
3. 对照终端显示的 Token 用量，找出代码从 `response` 读取它们的位置，说出 `messages` 与 `response` 分别是什么。

改动提示词可能影响回复，但不能把自然语言要求当作权限校验或可靠业务规则。

下一步把 `get_order` 提供给模型：模型提出调用请求，Python 检查参数并执行本机函数，再把查询结果发回模型，让它回答“我的 O1001 订单到哪里了？”。这是从普通模型问答进入工具调用 Agent 的关键一步。对应示例已在 [第三课：订单查询 Agent](03_订单查询Agent_工具调用.md) 中完成。
