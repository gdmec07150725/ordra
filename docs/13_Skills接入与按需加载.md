# 第十三课：Skills 接入与按需加载

> 本文按课程阶段讲解；请在仓库根目录执行命令。安装与当前功能以[仓库 README](../README.md)为准，早期章节中的简化流程不代表最终实现。

本课对应正式聊天入口 `/` 的实际实现。项目使用 LangGraph、工具调用、MCP、RAG、子 Agent、审批和检查点；Skills 提供可复用的处理指引，以及让模型按任务需要读取指引的机制。

## 1. Skills 提供什么

| 能力 | 当前实现 |
| --- | --- |
| 技能定义 | 本地 `skills/<名称>/SKILL.md` |
| 初始发现 | 系统提示词只携带名称、简介；文件版本保存在任务快照 |
| 正文读取 | 主模型请求 `load_skill({"name":"..."})` |
| 引用读取 | 主模型按需请求 `read_skill_resource({"name":"...","path":"references/...md"})` |
| 上下文管理 | 正文接受前检查预算，接受后计入每次模型请求 |
| 恢复 | 检查点保存已加载的正文、版本、次数 |
| 观察 | `/runs` 查看加载事件、版本和技能上下文用量 |

内置 `return-handling`（退货处理）和 `order-policy-check`（订单与政策核对）两个技能。

本项目采用 LangChain 文档描述的按需读取技能模式，工具循环继续运行在现有 LangGraph 中。这里的加载器是本项目的 Python 实现，没有为了这个功能再接入一套 LangChain Agent 或创建新的子 Agent。

## 2. Skill、RAG、MCP、子 Agent 分别做什么

以“查询 O1003，看看能否退货，并生成预览”为例：

| 机制 | 负责的问题 | 示例 |
| --- | --- | --- |
| Skill | 这个任务应该怎样处理 | 先核对订单和政策，补齐原因，再生成预览并等待确认 |
| RAG | 相关规则和知识是什么 | 检索适用的退货政策原文及版本 |
| 业务工具 | 当前业务事实、操作结果是什么 | 查询 O1003 状态、生成实际预览 |
| MCP | 以统一协议提供工具 | 提供 `get_order`、`search_knowledge` 的定义和调用入口 |
| 子 Agent | 谁独立处理一项子任务 | orders 查订单，knowledge 查政策 |
| LangGraph | 如何流转、暂停、保存和恢复 | 执行工具后回到模型，审批时暂停，断线后继续 |

Skill 文档里可以包含知识，但这两个技能只承载流程。会变化的政策继续放在知识库，实时订单仍通过工具读取，避免同一规则在多个地方出现不同版本。

## 3. 完整执行过程

```text
用户提出任务
  ↓
程序扫描本地技能，固定本次任务的名称、简介、版本
  ↓
主模型看到任务 + 技能目录 + load_skill 的参数定义
  ↓ 根据任务选择；简单查询也可以不加载
load_skill({"name":"return-handling"})
  ↓
校验名称、路径、版本、文件大小、次数和请求预算
  ↓
正文作为 tool 结果进入本轮上下文，同时保存检查点
  ↓
主模型读取指引，选择 discover_tools / 检索 / 计划 / 委派
  ↓
按现有权限调用业务工具，等待必要的人工审批
  ↓
核对实际证据 → 生成最终答复
```

“按需”的核心是正文什么时候交给模型。目录扫描为了校验格式和计算版本会在本地读取文件，但第一轮请求不发送正文。

模型并非天然知道你的技能列表。程序把目录与 `load_skill` 的工具定义发给它，模型根据描述、用户任务和已有上下文提出调用。是否选得合适仍取决于模型与技能描述；描述不清楚，模型可能不加载或选错，需要通过运行记录和评测改进。

Skill 也不是固定执行脚本。文档给处理顺序与约束，模型可以根据缺失信息、查询结果选择下一步。真正的审批、允许调用的工具和预算由程序强制执行。

## 4. SKILL.md 怎么写

实际文件位于 `skills/return-handling/SKILL.md`，结构如下。以下为简化示例，完整流程见该文件：

```markdown
---
name: return-handling
description: 用户需要退货、退款或售后申请时，指导核对订单与政策、补充原因、生成预览和等待确认。
---
# 退货处理

1. 提取用户明确提供的订单号和具体退货原因，缺失时询问。
2. 查询订单并检索适用政策，不能把流程说明当作业务事实。
3. 信息齐全后生成预览，遵守平台审批。
4. 请用户核对并确认，只有实际提交成功才告知申请编号。
```

`name` 必须与目录名一致，使用小写字母、数字和单个连字符，最长 64 个字符。`description` 必须是非空字符串，最长 1024 个字符。文件需要 YAML 前置元数据与非空 Markdown 正文。

描述应写清适用场景与能力，例如“综合订单状态与售后政策判断能否退换”，避免只有“订单技能”这样的宽泛名称。正文适合包含处理顺序、需要询问的信息、停止条件、证据要求和小例子。

不要把 API Key、实时订单信息或未经核实的退款天数写进技能。

## 5. 关键代码在哪里

### 5.1 目录、版本和加载器

`skill_runtime.py` 中的 `SkillRuntime` 负责扫描、读取和保存加载状态：

```python
runtime = SkillRuntime(PROJECT_DIR / "skills")
catalog_prompt = runtime.instructions()  # 只有目录和加载规则，没有正文

result = runtime.load(
    '{"name":"return-handling"}',
    accept=budget_check,  # 由主图提供实际请求预算检查
)
checkpoint_data = runtime.snapshot()
```

`_read()` 校验文件，再计算完整文件内容的 SHA-256 作为版本。版本不是文件名里的 V1/V2，也不是模型产生的值；修改元数据或正文都会改变版本。

成功时内部结果类似下面的结构，版本值在这里省略：

```json
{
  "status": "loaded",
  "name": "return-handling",
  "version": "完整文件的 SHA-256",
  "estimated_tokens": 900,
  "instructions": "# 退货处理\n\n1. ..."
}
```

`estimated_tokens` 在示例里仅为示意值。实际采用项目已有的 UTF-8 保守估算，不是模型服务的精确分词或计费数据。

### 5.2 模型怎么调用

`SKILL_TOOL` 注册名为 `load_skill` 的函数工具，只允许 `name` 参数。模型响应中的关键结构类似：

```json
{
  "tool_calls": [{
    "id": "call_skill_1",
    "type": "function",
    "function": {
      "name": "load_skill",
      "arguments": "{\"name\":\"return-handling\"}"
    }
  }]
}
```

模型只提出调用。真正读取磁盘的是 `full_agent_graph.py` 的 `make_job()` 中 `load_skill` 分支，然后主图将结果追加为带同一 `tool_call_id` 的 tool 消息。下一次请求模型才看见正文。

MCP 模式下，`tool_discovery.py` 把 `load_skill` 列为核心工具，所以无需先发现“读取技能”的工具。但 Skill 正文提到的业务工具仍需要按原规则发现，不能通过技能增加工具权限。

### 5.3 如何控制预算

加载前，主图构造包含候选正文的请求视图，主要判断为：

```python
essential = system_prefix + current_messages_with_skill + feedback
future_tools = all_discoverable_schemas_and_core_tools
reserve = summary_reserve + 1024
fits = estimate_request(essential, future_tools) + reserve <= policy.input_limit
```

这是解释用的缩写；准确实现见 `make_job()` 内的 `fits()`。完整正文、本轮消息以及本任务可能发现的全部工具定义一起计算，避免“加载技能时放得下，发现业务工具后立即超限”。旧回合可通过原有滚动摘要腾出空间，所以接受前为必要摘要保留额度；另预留 1024 个估算 tokens 给后续操作。

放不下时返回 `skill_budget_exceeded`，正文不进入上下文、不算已加载。模型可以继续用已有工具或说明限制。不会把流程截成一半冒充完整技能。

这个预留量不保证任意后续工具结果都能放下。加载后的每次模型请求仍经过统一预算检查，检索仍按现有规则裁剪，历史仍可压缩。

`context_observation.py` 在请求预算里单列“技能指引”。目录计入系统提示词，load_skill 定义计入工具定义；已加载的 tool 消息计入技能项，三者不会重复计数。

### 5.4 如何保存和恢复

`full_agent_graph.py` 在原有快照中增加 `skills`：

```text
skills
  version       快照结构版本
  catalog       本次任务固定的名称、简介、文件版本
  loaded        已接受的完整正文与版本
  calls         已接受的加载尝试次数
  events        仅含元数据的加载事件
```

加载成功后断线，继续执行直接使用检查点里的正文，不重新读取、不再占一次加载次数。即使本地文件后来修改或删除，这个已加载任务仍使用原版本。

如果任务开始时只看到了目录，还没加载该技能，而文件已经变化，则返回 `skill_version_changed`。重新发起一轮任务才能看到新版目录；不要把同一任务里的旧简介和新正文混在一起。

新一轮对话重新选择技能。`historical_skill_view()` 只在请求副本中移除上一轮技能正文，保留加载记录；数据库完整原文保持不变。升级前的旧检查点保持原有工具集合，继续执行不会强行插入新技能。

### 5.5 为什么加载不等于执行成功

`load_skill` 是控制操作：不占业务工具额度，不产生订单事实，不完成计划步骤。首次加载新技能不计为停滞，重复/失败加载仍可能触发原有防空转机制。

`result_verification.py` 去掉技能正文，并从可引用业务证据 ID 中排除加载结果。这样“读了退货指南”不会被误认为“已经核实订单”或“已经办理退货”。

## 6. 如何新增一个技能

1. 在 `skills/` 下创建一个新目录，例如 `delivery-inquiry/`。
2. 新建 `SKILL.md`，填写匹配目录的 name、明确描述和步骤。
3. 步骤只使用项目实际支持的工具；需要新能力时先实现和注册工具。
4. 发起新的聊天请求，然后在执行记录中检查模型是否加载了正确技能。

修改已有技能也在下一轮新任务生效，通常无需重启服务。已开始的任务使用自己的版本快照。`/graph` 是独立实验页，技能加载功能接在正式 `/` 聊天流程。

当前实现边界：

- 最多 8 个技能进入目录，目录元数据还有 1536 个估算 tokens 的上限，按目录名排序选择；不符合格式或超限的技能会被跳过。
- 单个文件最多 24000 字节，正文加载还必须符合当前请求预算。
- 每个任务最多 3 次加载尝试；成功、重复、参数错误等被接受的工具结果都累计次数。超时未接受的结果不会提交技能状态。
- 支持本地 SKILL.md 正文及其明确链接的 `references/` Markdown 文件；引用由模型单独选择读取，不自动跟随链接。`scripts/` 不会执行，`allowed-tools` 不会提升权限。
- 每个技能最多声明 8 个引用；引用单文件最多 24000 字节，每任务最多 4 次读取尝试，另受模型总轮次、防空转和上下文预算约束。
- 主 Agent 使用技能；现有两个子 Agent 不自动复制这些正文。

目录异常可在项目根目录执行以下命令查看名称与解析错误类型，不需要模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from skill_runtime import SkillRuntime; r=SkillRuntime('skills'); print(r.catalog); print(r.errors)"
```

## 7. 怎么手动测试

先刷新 [聊天页面](http://127.0.0.1:8000/)；在健康接口确认 `skills_enabled: true`。已有未完成任务按旧检查点继续，验证新功能请新建对话。

### 场景 A：明确要求加载

发送：

> 请先加载 return-handling 技能，再处理 O1003 的退货。原因是耳机左侧没有声音，先查询订单和政策并生成预览，等我确认。

预期聊天状态出现“正在读取处理指引”，随后查询和预览沿用原有流程。需要审批时会暂停；测试加载无需确认提交实际模拟申请。

打开 [执行记录](http://127.0.0.1:8000/runs)，选择刚才的会话和任务：

- “技能按需加载”显示技能名、版本、结果。
- 加载前的模型预算“技能指引”为 0；加载后的请求出现该项用量。
- 目录在系统提示词内，工具定义单独统计。
- 后续订单/知识查询仍产生各自真实工具结果。

### 场景 B：观察模型自主选择

新建对话，发送：

> 帮我核对 O1003 的订单状态与退货、退款政策，说明适用条件和还需要补充什么。

观察它是否选择 `order-policy-check`。这是决策质量测试，模型也可能直接调用工具；一次未加载不代表程序失效。对比场景 A 可以区分“加载链路有问题”和“模型没有选择加载”。

### 场景 C：中断恢复

让模型加载技能并处理退货，在等待审批时刷新页面，再继续操作。观察同一任务的加载次数没有增加、版本没有变化，审批仍需决定。

### 场景 D：简单查询与新回合

新建对话发送“O1001 到哪里了？”；一般可以直接查订单，无需先加载技能。随后问另一个问题，前轮的完整技能正文不会因为历史存在就自动加入新请求。

### 离线自动验证

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_skill*.py" -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_result_verification.py -k skill -v
```

这些测试使用临时目录、临时数据库和模拟模型响应，覆盖完整图与 HTTP 接口，不调用真实聊天模型，也不改你的历史会话。它们证明加载、预算、审批、恢复等程序行为；真实模型是否选对技能、是否恰当地遵循指引，仍需场景 B 这类评测。

## 8. 常见问题

“读取技能需要再调用模型或 Embedding 吗？”——本地读取、版本计算和预算估算不需要；选择技能和读完正文后的下一步决策需要原有聊天模型请求，所以整个任务可能增加模型轮次与输入 tokens。

“为什么没有加载我新增的文件？”——检查目录名与 name、YAML 格式、简介、文件大小、目录总量；确保开启的是新任务，再看模型是否提出了 load_skill。

“为什么 load_skill 和 get_order 不能同时返回一批？”——模型在这一批生成 get_order 参数时还没看到技能正文。程序要求控制操作独立一批，先获取指引，下一次决策再使用。

“这是否完整实现了 Agent Skills 规范的所有能力？”——没有。目前支持目录、主正文、显式 Markdown 引用三层，不包括脚本执行、任意文件访问、远程安装等完整生态功能。

## 9. 建议阅读的代码顺序与官方资料

先读两个 SKILL.md，再读 `skill_runtime.py`，随后看 `full_agent_graph.py` 的工具注册、`load_skill` 分支与 `snapshot()`；最后看 `context_observation.py`、`result_verification.py` 和 `trace_store.py`，理解成本、事实边界与可观测性。

设计参考 [LangChain Skills 模式](https://docs.langchain.com/oss/python/langchain/multi-agent/skills) 和 [Agent Skills 格式规范](https://agentskills.io/specification)。具体支持范围以本教程和当前代码为准。

## 10. 第三层引用如何按需读取

现在内置退货技能的目录为：

```text
skills/return-handling/
├── SKILL.md
└── references/
    ├── missing-information.md
    └── exception-handling.md
```

主文件保留简短流程，细节放进引用。例如在 `SKILL.md` 正文中声明：

```markdown
需要细化询问缺失信息时，读取[缺失信息询问指引](references/missing-information.md)。
```

本实现识别这种直接 Markdown 链接，不扫描未声明文件；仅支持 `references/` 内的 UTF-8 `.md`，路径使用 `/`，不带空格、URL 编码、锚点、绝对路径或 `..`。引用文件中的链接不会继续递归展开。

完整调用顺序是：

```text
模型看到技能名和简介
  → load_skill({"name":"return-handling"})
  → 返回主指引及 references 文件清单（没有引用正文）
  → 模型判断是否需要某份细则
  → read_skill_resource({
      "name":"return-handling",
      "path":"references/missing-information.md"
    })
  → 返回该引用全文
  → 下一次模型请求才依据它询问用户或调用业务工具
```

初始扫描会在本地校验已声明文件并计算 SHA-256，模型最初仍只接收名称和简介。文件清单随主指引返回，引用全文只在单独调用读取工具并通过预算后进入上下文。

引用读取本身不请求模型或 Embedding；模型决定读取，以及读完后决定下一步，仍各属于正常模型决策轮次。把主文件拆小不保证总成本一定下降，只有任务确实不需要全部细节时才可能节省输入。

### 10.1 关键代码

`skill_resources.py` 负责显式引用清单、路径边界、文件大小和内容版本。`skill_runtime.py` 的 `read_resource()` 负责加载状态、次数、重复读取和预算接受。核心分支为：

```python
current, body = read_reference(self.root, name, path)
if current['version'] != metadata['version']:
    result = {'error': '引用文件已更新，请新建任务后读取新版本。',
              'error_code': 'skill_resource_version_changed'}
else:
    result = {'status': 'loaded', 'name': name, **current, 'content': body,
              'estimated_tokens': estimate_request([{'role': 'tool', 'content': body}])}
    if accept(result):
        self.resources[key] = deepcopy(result)
```

这是主要分支；完整错误处理与次数限制见源码。`full_agent_graph.py` 共用主指引的预算接受逻辑，将候选引用、当前任务、可能加载的工具定义和后续操作余量一起计算。放不下会返回 `skill_resource_budget_exceeded`，不截断流程，不保存为已加载。

引用首次读取属于准备工作，不算新业务事实、不完成计划步骤，也不增加业务工具调用数。重复读取返回 `already_loaded` 和元数据，不重复注入正文；重复和错误读取仍消耗尝试次数，持续无进展会触发原有停止机制。工具调用必须单独成批，不能与查询或审批操作混在同一批。

引用次数在执行文件读取前预留并保存，超时或中断不会返还次数；晚到的正文不会被接受。若恰好在预留后、实际读取前中断，该次也保守计入尝试。执行记录中的引用事件记录已结束的读取结果，中途断开的尝试仍计入总次数。

版本和恢复行为：

- 任务开始时固定引用路径和文件版本。开始后修改尚未读取的文件，读取会拒绝混用版本，需新建任务。
- 已读取内容保存在检查点；随后修改或删除磁盘文件，同一任务恢复使用原内容，次数也保留。
- 新一轮重新选择技能及引用；仅请求副本移除旧正文，完整会话历史仍保存原文。
- 升级前的 Skills 检查点保持原两层能力，恢复不会加入新工具；新任务才启用引用读取。
- 文件打不开、过大或编码无效时，主指引仍可加载，引用标为不可用；修正文件后从新任务开始。

### 10.2 手动测试

在正式聊天首页刷新后新建对话，发送：

> 请先加载 return-handling，再读取 references/missing-information.md。我想退 O1003，但还没说明原因，请按指引询问。

打开“执行记录”，预期依次看到“技能按需加载”和“技能引用读取”。后者包含技能名、相对路径、版本和内容估算 tokens；后续模型请求的“技能指引”预算增加。回复应询问具体原因，不能自行补造原因或直接生成退货预览。

再新建对话测试自主选择：

> 我想退 O1003，但还没说明原因。请使用适合的处理流程及缺失信息询问细则，告诉我需要补充什么。

此时未指定技能名和文件名，由模型决定。未读取引用不一定表示程序失败，可能已有主指引足够回答；专项评测将其记录为“未按该场景预期选择”，与回答是否正确分别展示。

普通查询对照：

> O1003 当前是什么订单状态？

通常直接查询订单即可，无需读取技能及引用。执行记录是验证实际动作的依据，不能只根据模型声称“已读指引”来判断。

### 10.3 自动评测与成本对照

```powershell
# 离线：脚本模拟模型决策，真实加载器、API、数据库和本地 MCP
.\.venv\Scripts\python.exe -X utf8 agent_eval.py --suite skills

# 真实模型：自主选择，并自动增加一次关闭 Skills 的同题对照
.\.venv\Scripts\python.exe -X utf8 agent_eval.py --live --case skill_auto

# 只验证明确读取
.\.venv\Scripts\python.exe -X utf8 agent_eval.py --live --case skill_required
```

默认 7 个场景：明确加载、自主选择、无需技能、错误路径纠正、重复读取、超预算、中断恢复。自主选择和无需技能各附带一次关闭 Skills 的同题对照，共 9 次执行。`--suite all` 也包含此套件。预算和中断场景明确使用超大引用或注入连接错误；仅操作临时技能副本和临时数据库，不修改真实技能、订单和历史。

JSON 和 HTML 报告写到 `.local/evals/`，可使用 `--output <路径>.json` 指定文件。报告展示实际回答、规则检查、独立结果校验、模型请求数、已报告 token 用量及耗时。离线 token 用量由脚本模拟，不能据此判断真实节省；真实模型对照也只是该次样本，不是稳定提升的证明。回答规则与结果校验都有局限，不能作为全面语义正确率。

`skill_eval.py` 负责场景和实际观测评分，`agent_eval.py` 复用原有报告入口。引用的文件读取和恢复机制由单元与图测试验证，真实模型是否选择合适则用 `--live` 观察。
