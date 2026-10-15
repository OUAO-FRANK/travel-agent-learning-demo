# Travel Agent Lab

这是一个用于学习 Agent Runtime 的旅行规划助手。项目从普通 Python 的最小实现开始，逐步加入工具、RAG、Memory、Harness、MCP、工具队列和多 Agent。

## 当前阶段

当前实现确定性的单工具 Agent Loop，并已加入第一版上下文窗口：

```text
CLI 输入
  -> TravelAgent
  -> FakeModel 返回文本或 get_weather 请求
  -> 执行天气工具并把 observation 放回消息
  -> FakeModel 返回最终文本
  -> 事件轨迹和最终答案
```

`ContextBuilder` 会把 system prompt、最近的完整对话轮次和工具目录组装成
模型本轮真正收到的请求；Agent 内部的完整历史与这个请求窗口是两个不同概念。
当前用消息数量做教学版预算，工具调用请求和工具结果会作为一轮整体保留。

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
