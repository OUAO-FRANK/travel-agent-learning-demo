# Travel Agent Lab

这是一个用于学习 Agent Runtime 的旅行规划助手。项目从普通 Python 的最小实现开始，逐步加入工具、RAG、Memory、Harness、MCP、工具队列和多 Agent。

## 当前阶段

当前实现确定性的 Agent Loop，并已加入上下文窗口、最小 RAG、最小 Memory
和串行多工具队列：

```text
CLI 输入
  -> TravelAgent
  -> FakeModel 返回文本或 get_weather 请求
  -> 执行天气工具并把 observation 放回消息
  -> FakeModel 返回最终文本
  -> 事件轨迹和最终答案
```

FakeModel 还可以用 `--demo-tool-batch` 一次返回两个天气工具调用，观察
`tool_queue_started`、两个 `tool_requested/tool_result` 对和
`tool_queue_finished`。当前队列是串行、fail-fast 的；并发、依赖关系、取消和重试
仍留到后续课程。

Runtime Harness 当前先提供运行预算：`--max-model-requests` 限制一次运行的模型请求数，
`--max-tool-calls` 限制工具调用数。预算耗尽时 Agent 返回 `status=stopped`，并记录
`harness_limit_reached`，不会继续调用模型或工具。

`ContextBuilder` 会把 system prompt、最近的完整对话轮次和工具目录组装成
模型本轮真正收到的请求；Agent 内部的完整历史与这个请求窗口是两个不同概念。
当前用消息数量做教学版预算，工具调用请求和工具结果会作为一轮整体保留。

现在还可以给 Agent 接入本地旅行资料。使用 `--knowledge-base` 后，Agent 会先按
关键词检索资料片段，再把带来源的证据作为临时上下文交给模型。检索证据不会写入
SQLite 的 `messages`，也不等同于用户的长期 Memory。

现在还支持最小长期 Memory。使用 `--memory-db` 和 `--memory-user` 后，明确以
“请记住……”开头的输入会写入独立的 `memories` 表；后续同一用户的请求会按关键词
召回相关记忆，并把它作为临时上下文交给模型。普通的“我喜欢历史景点”不会自动写入，
这样可以先看清显式写入、用户隔离和召回的基本机制。

现在还支持一个可观察的上下文压缩实验。使用 `--max-messages 4 --compress-context` 时，
超出预算的较早完整轮次会被整理成一条摘要消息。Fake 模式使用本地提取规则，
DeepSeek 模式会额外调用一次 LLM 生成摘要；最近轮次保持原文。SQLite 中的原始
`messages` 不会被删除或改写。压缩摘要正文会作为 `context_compressed` 事件证据保存在
`events` 表，但不会被当作新的对话消息写入 `messages` 表。

阶段 5 增加了可选的 `SQLiteSessionStore`。传入 `session_store` 和 `session_id`
后，Agent 会保存 `sessions`、`messages`、`events` 和 `tool_calls`，重新创建
Store 后可以继续同一会话的历史。CLI 通过 `--db` 和 `--session-id` 开启持久化；
两项必须同时提供。运行前会打印恢复的历史消息、角色和内容，运行后会重新从数据库读取
消息、事件和工具调用统计，方便观察“本轮发生了什么”和“已保存什么”。默认 CLI
仍使用内存模式，避免不指定会话时把不同用户请求混在一起。

当前版本支持两种模式：

- `fake`：默认模式，不访问网络、不需要 API Key，结果确定；
- `deepseek`：使用 DeepSeek OpenAI-compatible API，需要环境变量中的 API Key。

默认注册表只启用固定数据的 `get_weather` 工具，但 Agent 也支持注入其他工具。

当前已经有最小的通用工具注册表、参数 schema 和多工具分发能力：

- `ToolDefinition` 描述工具名称、说明、参数类型和 handler；
- `ToolRegistry` 负责注册、查找、参数校验和分发；
- `TravelAgent` 可以注入自定义注册表，因此不需要在 Agent 内部增加
  `if tool_name == ...` 分支。

当前 schema 在注册表内部仍是教学用的简化形式；Provider 会在发送请求前把它转换成
JSON Schema。

当前模型边界也已经抽象为 `ModelProvider`：Agent 每次请求都会把消息和工具目录
一起交给模型。项目现在同时提供：

- `FakeModel`：不联网、可重复、用于理解和测试；
- `OpenAICompatibleModel`：负责把内部消息和工具说明转换为兼容 Chat Completions
  的 HTTP 请求，并解析文本或工具调用响应。

Provider 的 HTTP 传输函数可以注入测试替身，因此测试不需要 API Key 或网络，Agent
主循环也不需要知道供应商协议细节。

## 运行

在 Windows PowerShell 中，如果终端默认编码导致中文显示异常，可以先执行：

```powershell
$env:PYTHONUTF8 = "1"
```

然后运行：

```powershell
cd E:\agent_learn\projects\travel-agent
D:\miniforge3\envs\fastapi-learn\python.exe -m app "我想下周去上海玩三天，预算 2000 元，喜欢历史景点"
```

程序会先输出事件轨迹，再输出最终答案。

使用真实 DeepSeek 模型时，在同一个 PowerShell 窗口中先设置 API key：

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API key"
$env:DEEPSEEK_MODEL = "deepseek-flash"
```

然后运行：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider deepseek `
  "我想下周去上海玩三天，预算 2000 元，喜欢历史景点"
```

如果要观察 SQLite 会话持久化，先选一个数据库文件位置。下面把数据库放在用户数据目录，
避免把本地实验数据放进 Git 仓库：

```powershell
$db = Join-Path $env:LOCALAPPDATA "TravelAgentLab\travel-agent.db"
```

第一次运行会新建数据库和会话：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider deepseek `
  --db $db `
  --session-id shanghai-demo `
  "我想下周去上海玩三天，喜欢历史景点"
```

再次使用相同的 `$db` 和 `--session-id`，程序会把前一轮消息恢复到模型上下文：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider deepseek `
  --db $db `
  --session-id shanghai-demo `
  "预算是 2000 元，请把路线安排得具体一些"
```

第二次运行的 `history_before_count` 应大于 0，`history_before_roles` 会显示已恢复的
角色顺序。`persisted_message_count` 和 `persisted_event_count` 则是运行后重新从数据库
读取的统计。要开始一个全新会话，换一个 `--session-id`；要从头开始整个实验，换一个 `$db`。
使用 `--provider fake` 也能验证保存和恢复，但 FakeModel 的答案是预设文本，不会根据历史改变；
想观察模型如何利用上一轮内容，请使用 DeepSeek 模式。

要直接观察上下文压缩，可以使用一个小预算连续运行三次：

```powershell
$compressionDb = Join-Path $env:LOCALAPPDATA "TravelAgentLab\compression-demo.db"

D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --db $compressionDb `
  --session-id compression-demo `
  --max-messages 4 `
  --compress-context `
  "我想去上海三天"

D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --db $compressionDb `
  --session-id compression-demo `
  --max-messages 4 `
  --compress-context `
  "预算是 2000 元"

D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --db $compressionDb `
  --session-id compression-demo `
  --max-messages 4 `
  --compress-context `
  "请把路线按天安排"
```

第三次运行会出现 `context_compression_started`、`context_compressed` 两个事件，
并在 `model_request` 中看到类似 `roles=['system', 'assistant', 'user']`：system 是应用规则，
assistant 是运行时生成的历史摘要。数据库仍会显示 `persisted_message_count=6`，说明压缩
只改变本次请求窗口，没有抹掉原始对话；摘要正文可在 `events` 事件证据中查看。

要运行最小 RAG：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider deepseek `
  --knowledge-base .\data\travel_notes.md `
  --rag-top-k 2 `
  "我想安排上海外滩夜景和历史景点"
```

输出中重点观察 `rag_search_started`、`rag_search_completed`，以及
`model_request` 的 `retrieved_sources`。模型会看到类似 `[travel_notes.md:0]` 的来源标记；
没有命中时，系统会明确告诉模型资料库没有直接证据。

要观察 Memory，可以先使用 FakeModel 做两次运行。两次命令必须使用同一个
`--memory-db` 和 `--memory-user`：

```powershell
$memoryDb = Join-Path $env:LOCALAPPDATA "TravelAgentLab\memories-demo.db"

D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider fake `
  --memory-db $memoryDb `
  --memory-user alice `
  "请记住我喜欢历史景点"

D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider fake `
  --memory-db $memoryDb `
  --memory-user alice `
  "帮我规划上海历史路线"
```

第一次重点观察 `memory_written`；第二次重点观察 `memory_search_completed`、
`memory_ids` 和 `model_request` 中的 `recalled_memory_ids`。如果把用户改成 `bob`，
第二次不会看到 `alice` 的记忆。FakeModel 的答案固定不变，但请求轨迹会证明记忆是否
进入了模型上下文；换成 `--provider deepseek` 才能观察真实模型如何利用它。

要直接观察串行多工具队列：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider fake `
  --demo-tool-batch `
  "同时查上海和北京天气"
```

第二次 `model_request` 会包含一条 assistant 批量工具请求和两个 tool observation；
事件中的 `call-1-1`、`call-1-2` 表示队列按顺序执行了两个任务。

要观察 Runtime Harness 截止模型请求：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m app `
  --provider fake `
  --demo-tool-batch `
  --max-model-requests 1 `
  "同时查上海和北京天气"
```

第一次模型请求会返回工具批次并执行；工具结果回来后，第二次模型请求被 Harness
拒绝。重点观察 `harness_limit_reached` 的 `reason` 和最后的 `status=stopped`。

如果你要显式使用旧名称 `deepseek-v4-flash`，只需改为：

```powershell
$env:DEEPSEEK_MODEL = "deepseek-v4-flash"
```

当前官方模型列表使用 `deepseek-flash`；旧名称仍可能被兼容接受，因此默认采用官方当前标识。
也可以设置 `DEEPSEEK_BASE_URL` 覆盖默认地址 `https://api.deepseek.com`。

不要把 API key 写入源码、README、测试文件或 Git 提交。`.env` 已被 `.gitignore` 忽略，但当前项目默认不自动读取 `.env`，最直观的方式是使用上面的 PowerShell 环境变量。

## 测试

```powershell
cd E:\agent_learn\projects\travel-agent
D:\miniforge3\envs\fastapi-learn\python.exe -m unittest discover -v
```

测试重点不是验证“模型是否聪明”，而是验证：

- 消息是否按预期发送；
- FakeModel 是否按顺序返回；
- 回复耗尽时是否显式失败；
- 事件是否按正确顺序产生；
- CLI 是否把运行轨迹展示出来；
- 工具注册表是否拒绝未知工具、重复名称和非法参数；
- Agent 是否通过注入的注册表执行自定义工具。
- 上下文窗口是否保留 system prompt、完整工具回合，并正确记录被截断的历史。
- 上下文压缩是否只替换请求窗口中的旧轮次、保留最近轮次和数据库原文。
- RAG 是否只召回匹配片段、保留来源，并把证据作为临时上下文注入模型请求。
- SQLite 重新打开后是否能恢复消息、事件和工具调用证据。
- CLI 是否能在重复使用同一数据库和 session ID 时恢复历史并显示持久化统计。
