# 旅行规划 Agent 学习项目总设计

> 本文是整个项目的学习地图，不是一次性实现清单。每个阶段都先建立一个可观察的最小机制，再把它接入旅行规划助手。阶段完成的标准不是“代码能运行”，而是能沿数据流解释代码、验证成功路径和失败路径，并能用 Git 说明这次改动的边界。

## 1. 项目定位

### 1.1 最终要做什么

我们要逐步构建一个旅行规划助手：

```text
用户输入旅行需求
  -> Agent 解析当前任务
  -> 模型决定直接回答或请求能力
  -> Runtime 校验并执行工具/检索/记忆操作
  -> 结果作为新的观察回到上下文
  -> 模型继续规划
  -> 输出有依据的旅行计划和可观察轨迹
```

最终案例：

> “我想下周去上海玩三天，预算 2000 元，喜欢历史景点。”

系统逐渐学会：查询天气、搜索旅行资料、记住用户偏好、并发查询多个能力、在失败或超时后安全停止、恢复中断运行，并在需要时把任务交给其他 Agent。

### 1.2 明确不做什么

- 不直接改写 Hermes 源码；先用小项目建立自己的参照实现。
- 不把 LangChain/LangGraph 当作第一步；先手写控制流，再做框架映射。
- 不在没有测试需求时接入真实天气、地图、酒店或机票业务 API。
- 不把所有名词都实现成生产级系统；分布式调度、容器沙箱、复杂计费和云部署只在需要时扩展。
- 不把 API key 写进源码、README、测试、Git 历史或命令输出。

## 2. 学习方式与固定节奏

每个阶段都遵循同一个闭环：

```text
本阶段要解决的问题
  -> 贯穿案例和数据流
  -> 写一个预期行为测试
  -> 看到测试因功能缺失而失败
  -> 写最少实现
  -> 运行成功和失败路径
  -> 阅读 diff 和 Git 状态
  -> 做一个小改动或故障注入
  -> 用参考答案自查理解
```

用户不需要每次把自测题回复给我；本项目会提供问题和参考答案。只有遇到不理解、运行失败或需要验收时，再反馈具体位置。

每阶段固定产物：

1. 一条可运行的数据流；
2. 一组行为测试；
3. 一份阶段说明或实验记录；
4. 一次 Git diff 检查；
5. 一份“主线必须掌握 / 进阶了解 / 暂不展开”清单。

## 3. 总体架构

### 3.1 当前已经存在的组件

```text
app/domain.py
  Message       发给模型的消息
  ToolCall      模型提出的结构化动作
  AgentEvent    Runtime 事件轨迹

app/model.py
  ModelProvider          模型能力协议
  FakeModel              确定性模型
  OpenAICompatibleModel  OpenAI-compatible HTTP 适配器

app/tools.py
  ToolDefinition  工具描述
  ToolRegistry    注册、schema、分发和权限边界

app/agent.py
  TravelAgent     最小 Agent Loop

app/config.py
  ModelConfig     provider、API key、base URL、model 的配置边界

app/cli.py
  CLI             用户入口和模型选择
```

### 3.2 预计演进后的组件

```text
app/
  domain.py          稳定的数据结构和事件类型
  config.py          环境变量与配置校验
  model.py           Fake、OpenAI-compatible、故障模型
  agent.py           Agent Loop 与停止条件
  tools.py           工具契约、schema、注册表、权限
  context.py         Prompt、历史和上下文预算
  storage.py         SQLite 会话与事件持久化
  rag.py             文档、切分、关键词检索、证据
  memory.py          用户偏好和长期记忆
  queue.py           工具任务队列、并行、取消、屏障
  harness.py         运行时控制面
  evaluation.py      评测任务、轨迹断言、回归集
  mcp_client.py      MCP Client 和远端工具适配
  multi_agent.py     supervisor、worker、handoff
  service.py         FastAPI 服务入口
```

### 3.3 两条必须始终分开的边界

```text
模型 Provider：把内部请求翻译成供应商协议，再把响应翻译回来
Runtime：决定是否继续、是否执行工具、何时停止、失败如何传播
```

```text
RAG：从外部资料中找证据
Memory：保存和回忆用户/Agent 的长期信息
```

## 4. 阶段总览

| 阶段 | 名称 | 新增核心能力 | 预计主要文件 | 结果 |
| --- | --- | --- | --- | --- |
| 0 | 项目骨架与 Git 起步 | CLI、Message、FakeModel、事件轨迹 | `domain.py`、`model.py`、`agent.py`、`cli.py` | 一次纯文本闭环 |
| 1 | 最小 Agent Loop | ToolCall、一次工具往返、停止条件 | `domain.py`、`agent.py` | `model -> action -> observation -> final` |
| 2 | 工具系统 | Definition、Registry、schema、分发错误 | `tools.py` | 多工具可注册、可校验 |
| 3 | 模型适配与真实 DeepSeek | Provider、请求转换、响应解析、配置 | `model.py`、`config.py`、`cli.py` | Fake/DeepSeek 可切换 |
| 4 | Prompt 与上下文 | system prompt、历史、工具目录、预算 | `context.py`、`agent.py` | 能解释“实际发给模型什么” |
| 5 | SQLite 会话持久化 | sessions、messages、events、tool calls | `storage.py` | 中断后恢复对话 |
| 6 | 最小 RAG | 文档、切分、关键词召回、引用 | `rag.py` | 依据来自资料而不是猜测 |
| 7 | Memory | 偏好写入、搜索、召回、过期 | `memory.py` | 长期信息和 RAG 分离 |
| 8 | 多工具队列 | 串行、并行、取消、失败策略 | `queue.py`、`agent.py` | 多个能力可控执行 |
| 9 | Runtime Harness | 超时、预算、重试、中断、checkpoint | `harness.py` | 运行可控、可恢复 |
| 10 | Evaluation Harness | 任务集、轨迹断言、故障注入、回归 | `evaluation.py` | 能重复评价 Agent |
| 11 | MCP | Server、Client、能力发现、远程工具 | `mcp_client.py`、`mcp_server.py` | 协议工具接入 Registry |
| 12 | Multi-Agent | supervisor、worker、handoff、失败传播 | `multi_agent.py` | 判断何时拆 Agent |
| 13 | Skill/Plugin/Delegation | 扩展点和能力组合 | `extensions.py` | 理解 Hermes 扩展机制 |
| 14 | FastAPI 工程整合 | HTTP 服务、流式事件、鉴权、部署边界 | `service.py` | 从 CLI 到后端服务 |
| 15 | Hermes 对照阅读 | 按调用链映射成熟项目 | `docs/comparison.md` | 能定位入口和边界 |

下面每个阶段都使用同一套模板：目的、数据流、实现增量、失败实验、验收证据和 Hermes 对照。

## 5. 分阶段详细路线

### 阶段 0：项目骨架与 Git 起步

**目的**：先建立一个能观察的项目，而不是先追求“聪明”。

**数据流**：

```text
CLI 文本
  -> TravelAgent
  -> FakeModel
  -> AgentEvent
  -> 最终文本
```

**实现增量**：

- 建立独立 Git 仓库，确认仓库边界；
- 定义 `Message`、`AgentEvent`、`AgentResult`；
- 实现 `FakeModel.complete()`；
- 实现一次普通文本请求；
- CLI 打印事件和最终答案；
- 编写测试并练习 `status`、`diff`、`add`、`restore --staged`、`log`。

**必须理解**：数据结构、依赖注入、不可变结果、事件与最终答案的差异。

**失败实验**：删掉事件记录；观察只剩答案后无法解释系统中间发生了什么。

**验收证据**：能沿 `user_input -> model_request -> model_response -> run_finished` 解释每个事件。

**Hermes 对照**：先只看 CLI 入口和一次请求的外围，不追求逐文件阅读。

### 阶段 1：最小 Agent Loop

**目的**：理解 Agent 与普通 LLM 调用的区别：Agent 可以根据模型动作继续运行。

**数据流**：

```text
用户消息
  -> 模型返回 ToolCall
  -> Runtime 校验并执行工具
  -> tool observation 加回 messages
  -> 模型返回最终文本
```

**实现增量**：

- 增加 `ToolCall`；
- 让 FakeModel 可以返回 `str | ToolCall`；
- 在 Agent 中追加 assistant/tool 消息；
- 增加一次工具轮数上限；
- 记录 `tool_requested` 和 `tool_result` 事件。

**必须理解**：动作和观察、消息历史、停止条件、一次工具往返。

**失败实验**：允许 FakeModel 连续返回 ToolCall；观察为什么必须有最大轮数。

**验收证据**：手工画出两次模型请求中 `messages` 的差异。

**Hermes 对照**：对应模型响应接收、工具轮和最终化等控制流概念。

### 阶段 2：工具系统

**目的**：把工具从 Agent 内部的条件分支中抽离出来，形成可注册的权限边界。

**实现增量**：

- `ToolDefinition`：名称、描述、参数 schema、handler；
- `ToolRegistry.register()`：拒绝重复工具名；
- `ToolRegistry.dispatch()`：未知工具、缺失参数、多余参数、类型错误；
- `model_schemas()`：生成模型可读工具目录；
- 自定义工具注入测试。

**必须理解**：契约、注册表、schema、显式允许列表、模型不能直接执行 Python。

**失败实验**：根据模型返回的字符串动态执行函数；解释安全和可调试性问题。

**验收证据**：同一个 Agent 不改主流程即可换成 `echo_city` 工具。

**Hermes 对照**：对应工具注册、工具执行器和工具权限边界。

### 阶段 3：模型适配与真实 DeepSeek

**目的**：理解“Agent 控制流”和“供应商 API”不是一回事，并能安全使用真实模型。

**当前实现**：

- `ModelProvider` 协议；
- `OpenAICompatibleModel`；
- `Message -> provider payload`；
- 简化 schema -> JSON Schema；
- 文本响应 -> `str`；
- tool call 响应 -> `ToolCall`；
- `DEEPSEEK_API_KEY` 环境变量配置；
- CLI `--provider fake|deepseek`。
- DeepSeek thinking/tool-call 模式的 `reasoning_content` 保留与回传；

**推荐运行方式**：

```powershell
$env:DEEPSEEK_API_KEY = "你的 API key"
$env:DEEPSEEK_MODEL = "deepseek-flash"
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider deepseek `
  "我想下周去上海玩三天，预算 2000 元，喜欢历史景点"
```

API key 只写入当前 PowerShell 进程环境。也可以写入用户级环境变量，但不要写入源码或 `.env` 并提交。

**默认配置**：

- base URL：`https://api.deepseek.com`；
- model：`deepseek-flash`；
- 可用 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 覆盖。

**必须理解**：HTTP 请求、认证头、供应商响应、工具调用 ID、适配层错误。

**失败实验**：缺少 API key；供应商返回空 `choices`；tool arguments 不是 JSON。

**验收证据**：使用注入 transport 测试请求 payload，并让真实 Provider 驱动已有工具循环。

**Hermes 对照**：对应 model provider、模型响应标准化和 Runtime 与外部服务的边界。

### 阶段 4：Prompt 与上下文

**目的**：回答“数据库里有的历史”和“这一次真正发给模型的上下文”是否相同。

**实现增量**：

- 把 system prompt 独立为配置；
- 统一组装消息、工具目录和运行元数据；
- 引入 token/消息数量预算的教学版估算；
- 截断旧历史并记录事件；
- 对比完整历史与发送上下文。

**必须理解**：上下文不是数据库，Prompt 不是记忆，工具定义也是上下文的一部分。

**失败实验**：无限追加历史，观察上下文预算和旧信息干扰。

**验收证据**：打印“保存历史”和“实际请求”两份不同视图。

**Hermes 对照**：对应上下文构造、消息压缩和请求前处理。

### 阶段 5：SQLite 会话持久化

**目的**：让 Agent 进程退出后仍能恢复会话和运行证据。

**实现增量**：

- `sessions`：会话元数据；
- `messages`：对话消息；
- `events`：运行轨迹；
- `tool_calls`：工具请求和结果；
- `session_id` 贯穿一次运行；
- 中断后重新加载消息继续执行。

**必须理解**：持久化历史、单次运行状态、事件日志三者不同。

**失败实验**：只保存最终答案，尝试恢复工具调用上下文，说明为什么失败。

**验收证据**：运行中断、重新启动、读取数据库、继续请求。

**Hermes 对照**：对应 Session DB、session persistence、turn finalizer。

### 阶段 6：最小 RAG

**目的**：让旅行回答引用本地资料，而不是让模型凭记忆猜测。

**实现增量**：

- 文档和文档片段数据结构；
- 按段落切分；
- 中文关键词检索；
- top-k 结果；
- 片段 ID 和来源引用；
- 检索结果注入上下文。

**必须理解**：切分、索引、召回、证据、未命中；RAG 是找资料，不是保存用户偏好。

**失败实验**：没有命中时强行拼接最相近内容；观察幻觉风险。

**验收证据**：固定文档的命中、未命中和引用测试。

**后置增强**：向量 embedding、向量数据库、混合检索和 rerank。

### 阶段 7：Memory

**目的**：让 Agent 能记住“用户喜欢历史景点、不喜欢早起”等长期信息。

**实现增量**：

- `MemoryItem`：内容、用户、来源、时间、置信度；
- 写入策略：显式偏好与模型推断分开；
- 关键词搜索和召回；
- 记忆进入 prompt；
- 过期、冲突和删除。

**必须理解**：Memory 与 RAG 的来源、生命周期和错误代价不同。

**失败实验**：写入错误偏好，观察后续计划如何被持续污染。

**验收证据**：同一用户两次会话中偏好被召回；另一个用户不能看到它。

### 阶段 8：多工具队列

**目的**：从一次工具调用升级到多个工具任务的排序、并行和取消。

**实现增量**：

- `ToolTask` 和任务状态；
- 串行执行；
- 无依赖任务并行执行；
- sequential barrier；
- 单任务失败策略；
- 取消和资源清理。

**必须理解**：并发不是简单地把 for 循环换成线程；必须表达依赖、顺序和失败传播。

**失败实验**：并行执行有依赖的任务，观察结果不稳定。

**验收证据**：天气、景点、预算三个任务的事件顺序和失败路径可重复。

### 阶段 9：Runtime Harness

**目的**：给 Agent 加上“不能无限运行、失败要安全停止、可以恢复”的运行控制面。

**实现增量**：

- 最大模型轮数；
- 最大工具调用数；
- 超时；
- 重试和退避；
- 取消信号；
- 工具权限；
- 资源清理；
- checkpoint/replay；
- 人工确认节点；
- 成本和调用预算。

**必须理解**：Harness 控制运行，Agent 做决策；不要把所有控制逻辑塞进 Agent 主循环。

**验收证据**：故障注入、超时、中断恢复和预算耗尽各有测试。

### 阶段 10：Evaluation Harness

**目的**：从“程序跑了”升级为“行为可以重复评价”。

**实现增量**：

- 固定任务集；
- 轨迹快照；
- 工具调用断言；
- 最终答案规则检查；
- 故障注入模型/工具；
- 回归比较；
- 结果报告。

**必须理解**：运行 Harness 和评测 Harness 是两套责任，不要混成一个 `run()`。

**验收证据**：修改 Prompt 后能发现轨迹或结果回归。

### 阶段 11：MCP

**目的**：理解本地函数工具和协议工具的差异。

**实现增量**：

- 极小 MCP Server：天气、笔记搜索；
- MCP Client 连接和能力发现；
- 把远端工具映射为 `ToolDefinition`；
- 参数和结果转换；
- 连接失败、超时和信任边界。

**必须理解**：Host、Client、Server、协议、能力发现、远程执行。

**失败实验**：远端工具断开时，不能把空结果伪装成正常观察。

**验收证据**：同一 Agent 可以无感调用本地工具和 MCP 工具，但事件标明来源。

### 阶段 12：Multi-Agent 编排

**目的**：理解什么时候需要拆 Agent，什么时候普通工具就足够。

**实现增量**：

- supervisor；
- research worker；
- budget worker；
- handoff 数据；
- 独立上下文；
- 共享状态最小化；
- worker 失败、取消和超时传播；
- 可选并行 worker。

**必须理解**：handoff 不是普通工具调用；子 Agent 有自己的上下文、目标和失败边界。

**失败实验**：为了查询一个天气工具拆一个 Agent，计算额外复杂度。

**验收证据**：必需 worker 失败时 supervisor 不生成伪造最终计划。

### 阶段 13：Skill、Plugin、Delegation

**目的**：把能力扩展机制放在已有 Runtime 之上，而不是重新发明 Agent Loop。

**实现增量**：

- Skill：可加载的任务说明和工具组合；
- Plugin hook：运行前、工具前后、运行结束；
- Delegation：向子任务发送明确输入和返回契约；
- 版本、权限和失败边界。

**必须理解**：扩展点解决的是真实变化点，不是为了增加抽象层。

### 阶段 14：FastAPI 工程整合

**目的**：把 CLI Agent 包装为可调用服务，同时保留 Runtime 可测试性。

**实现增量**：

- 请求/响应 schema；
- session API；
- 流式事件；
- 异步任务；
- 鉴权和限流边界；
- 数据库迁移；
- 服务级测试；
- Docker/部署只做最小闭环。

**必须理解**：HTTP 生命周期、后台任务、会话并发和 Agent 运行状态不能混为一谈。

### 阶段 15：Hermes/EvoAgent 对照阅读

**目的**：回到成熟项目，用自己的实现作为索引，而不是漫无目的逐文件阅读。

**阅读方法**：

1. 选一条具体入口：CLI 请求、工具调用或恢复请求；
2. 找入口函数和主循环；
3. 标记消息/状态对象；
4. 找模型 Provider 边界；
5. 找工具注册和执行器；
6. 找 session/memory；
7. 找 harness、checkpoint、评测和多 Agent；
8. 将每一段映射回本项目的文件和测试。

**最终产物**：`docs/comparison.md`，只记录调用链和边界，不做全仓逐行摘要。

## 6. Git/GitHub 学习路线

### 6.1 本地 Git

按项目阶段练习：

1. `git status`：工作区发生了什么；
2. `git diff`：代码具体改了什么；
3. `git add`：哪些改动进入暂存区；
4. `git restore --staged`：如何撤回暂存；
5. `git commit`：创建可回看的阶段节点；
6. `git log`：定位历史决策；
7. 分支：隔离实验；
8. 合并冲突：理解双方修改；
9. `git diff HEAD~1`：审查一次提交完整内容。

AI 不自动 commit 或 push。每次提交前先展示变更摘要，由用户决定是否提交。

### 6.2 GitHub

在本地 Git 熟悉后再学习：

- remote、clone、fetch、pull、push；
- Pull Request；
- review comment；
- CI 测试；
- 分支保护；
- secret 管理。

真实 API key 不允许进入 GitHub。使用仓库 Secrets 或本地环境变量。

## 7. 当前进度与下一步

当前已完成：

- 阶段 0：项目骨架、CLI、FakeModel、事件轨迹；
- 阶段 1：一次工具往返；
- 阶段 2：工具注册表和 schema 校验；
- 阶段 3 的 Provider 核心：`ModelProvider`、OpenAI-compatible 转换和测试；
- DeepSeek 配置入口：`--provider deepseek`、`DEEPSEEK_API_KEY`、默认 `deepseek-flash`。

当前正在完成：

- 由学习者填入 API key 后，使用真实 DeepSeek 运行一次旅行规划请求；
- 观察真实模型的文本响应和工具调用响应；
- 记录真实 API 的错误形状、调用成本和模型行为，不把它们混入 FakeModel 的确定性测试。

下一阶段：

- Prompt 与上下文模块；
- 把 system prompt、历史、工具目录和实际发送 payload 明确分层；
- 再进入 SQLite 会话持久化。

## 8. 每阶段验收总标准

阶段只有同时满足以下条件，才从“学习中”进入“已掌握”：

1. 能用自己的话说明核心机制、状态变化和一个失败边界；
2. 成功路径和失败路径都有自动化测试；
3. 能运行一次真实或 Fake 轨迹并指出每个状态变化；
4. 能阅读 `git diff`，说明哪些文件属于本阶段；
5. 能完成一个小改动，而不是只复制命令。

如果代码能跑但这些证据不足，阶段仍然保持“学习中”，不把运行成功误认为理解完成。
