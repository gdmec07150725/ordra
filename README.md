# Ordra · 订单客服 Agent

Ordra 是基于 **Python、FastAPI、LangGraph 和 MCP** 的电商客服演示项目。围绕模拟订单查询、政策检索和退货申请，实现多轮对话、工具调用、人工审批、检查点恢复与离线评测。

订单、业务政策和分析 CSV 均为示例数据。退货流程只创建本地模拟申请，没有连接真实订单系统或支付退款接口。

![Ordra 订单客服界面](docs/images/order-agent.jpg)

## 功能

- **订单与政策查询**：通过本地 MCP 服务查询订单、检索知识库；回答保留实际检索来源。
- **可恢复的执行流程**：LangGraph 组织上下文、模型决策、工具、委派、审批和结果校验；SQLite 保存会话及检查点。
- **人工审批与确认**：生成退货预览前可批准、修改参数或拒绝；预览另需用户确认，才创建模拟申请。
- **上下文管理**：token 预算、滚动摘要、当前任务压缩和会话原文检索，保留完整持久化历史。
- **按需加载与委派**：工具发现、Skills 主指引及引用资料按需读取；订单与知识子 Agent 使用独立上下文并共享预算。
- **执行记录与评测**：记录工具证据、审批、恢复、请求数与已报告用量；提供单轮、多轮、MCP、Skills 和策略对照评测。
- **代码分析实验**：对模拟订单 CSV 生成并运行 Python 子集代码，由宿主独立重算结果，通过后导出 Markdown/CSV。

## 架构

~~~mermaid
flowchart TD
    UI["网页 · fetch / SSE"] --> API["FastAPI · 会话与任务接口"]
    API --> G["LangGraph 主聊天图"]
    G --> C["上下文与计划"]
    G --> M["模型决策"]
    M --> D["工具发现 / Skills / 子 Agent"]
    D --> T["工具运行时 · 预算 / 并发 / 重试"]
    T --> MCP["本地 MCP stdio 服务"]
    MCP --> O["模拟订单"]
    MCP --> K["示例知识库"]
    T --> A["人工审批 → 退货预览"]
    A --> CONF["用户确认 → 本地模拟申请"]
    G --> V["答复与证据校验"]
    G <--> DB[("SQLite · 会话 / 检查点 / 轨迹")]
    V --> UI
~~~

| 入口或模块 | 作用 |
| --- | --- |
| `/`、`api_app.py`、`full_agent_graph.py` | 多轮主聊天与 LangGraph 执行 |
| `mcp_tool_client.py`、`mcp_tool_server.py` | 固定的本地 stdio 工具服务与调用 |
| `human_approval.py`、`checkpoint_store.py` | 审批决定、归属校验与恢复 |
| `knowledge_base.py`、`local_embeddings.py` | 关键词 / 本地向量混合检索 |
| `/runs` | 执行轨迹与上下文预算 |
| `/graph` | 单问题 LangGraph 对照实验 |
| `/code` | Monty 代码分析实验 |
| `tests/`、`*_eval.py` | 回归测试与场景评测 |

## 快速开始

建议 **Python 3.12+**。当前开发环境已在 **Python 3.14** 验证；其他版本及全新依赖安装仍需在目标环境验证。依赖固定在 `requirements.txt`，前端无需 Node.js 构建。

克隆仓库并进入 Ordra 项目目录后，创建环境并安装依赖：

~~~powershell
git clone https://github.com/gdmec07150725/ordra.git
cd ordra
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`，并将 `DEEPSEEK_MODEL` 替换为账号可用的模型名称。随后启动：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 07_api_server.py
~~~

打开 [聊天页面](http://127.0.0.1:8000)、[健康检查](http://127.0.0.1:8000/api/health) 或 [接口文档](http://127.0.0.1:8000/docs)。服务默认监听 `127.0.0.1:8000`，使用单进程；首次启动会自动创建 `data/` 中的数据库。

macOS / Linux 使用 `.venv/bin/python` 替换上述解释器路径，复制配置使用 `cp .env.example .env`。

可以尝试：

> 查询 O1003 的状态和退货政策。
> O1003 的耳机左侧没有声音，请帮我准备退货申请。

生成预览前，页面会要求人工审批；批准后再核对预览并确认提交。没有模型密钥时可运行下面的离线评测与基础订单查询：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 02_query_order.py O1001
~~~

## 配置

环境变量优先于 `.env`；修改配置后重启服务。`.env.example` 只包含占位字段和演示设置。

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 自行填写；离线评测不需要 |
| `DEEPSEEK_MODEL` | 自行填写可用模型名；应用通过 DeepSeek 兼容 API 调用 |
| `AGENT_TOOL_MODE` | `mcp` 为默认模式；`local` 用于本地直接调用对照 |
| `AGENT_HISTORY_SEARCH` | `keywords` 只用关键词；`hybrid` 启用本地向量混合检索 |
| `AGENT_CONTEXT_TOKENS` | 总上下文预算，示例为 `32768` |
| `AGENT_RECENT_TOKENS` | 近期完整历史目标预算，示例为 `2048` |
| `AGENT_SUMMARY_TOKENS` | 滚动摘要预算，示例为 `512` |

示例配置使用关键词检索，无需下载向量模型。需要混合检索时，先运行下面的命令下载模型，再将 `AGENT_HISTORY_SEARCH` 改为 `hybrid` 并重启：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 local_embeddings.py
~~~

模型文件保存在 `.local/embeddings/`。服务只读取本地模型，缺失或不可用时回退到关键词检索。知识库位于 `knowledge/`，技能位于 `skills/`；被选中的知识片段和技能内容会进入聊天模型上下文。

## 离线评测

以下命令用脚本模型或模拟传输替代外部模型 API，业务工具、临时 SQLite 数据库和对应的本地 MCP / 沙箱仍实际运行：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 agent_eval.py --suite all
.\.venv\Scripts\python.exe -X utf8 mcp_eval.py
.\.venv\Scripts\python.exe -X utf8 langgraph_eval.py
.\.venv\Scripts\python.exe -X utf8 code_eval.py
~~~

场景评测在 `.local/evals/` 生成 JSON 和 HTML 报告；失败时返回非零退出码。按需运行 `decision_eval.py` 比较主管直查、子任务委派等策略。

只有显式加入 `--live` 的场景评测才调用配置模型并产生用量。离线通过率验证的是执行流程与约束，不能当作真实模型准确率、生产可用率或成本收益。

## 实现边界

- 当前是本地教学与工程验证项目，没有用户登录、真实订单权限系统或公网多租户部署方案。
- 已保存的工具结果可在恢复时复用；尚未保存的只读工作可能重做。恢复不重置预算。
- 结果校验包含模型判断与程序约束，仍可能漏掉语义错误；模拟申请不代表退款成功。
- 工具按需发现可能增加模型请求，小工具目录不保证降低总 token 或耗时。
- Monty 支持 Python 子集，未提供任意依赖安装；代码分析任务中断后需要重新运行。
- 模型微调仅有[实施方案](docs/本项目如何做模型微调.md)，尚未训练模型。

## 文档

从[学习指南](docs/learning-guide.md)了解从 Python 函数到完整 Agent 的演进，或直接阅读：

- [FastAPI 与网页聊天](docs/06_FastAPI接口与网页聊天.md)
- [上下文管理](docs/08_上下文管理.md)与[RAG](docs/10_业务知识库RAG使用.md)
- [子 Agent 委派](docs/11_子Agent接入与任务委派.md)
- [MCP 与工具发现](docs/12_MCP接入与工具按需发现.md)
- [Skills 按需加载](docs/13_Skills接入与按需加载.md)
- [代码执行与沙箱](docs/14_代码执行Agent与沙箱隔离.md)
- [执行策略实验](docs/Agent执行策略实验.md)

## 许可证

[MIT](LICENSE) · Copyright 2026 Liao Nan
