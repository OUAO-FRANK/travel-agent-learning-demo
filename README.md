# Travel Agent Lab

这是一个用于学习 Agent Runtime 的旅行规划助手。项目从普通 Python 的最小实现开始，逐步加入工具、RAG、Memory、Harness、MCP、工具队列和多 Agent。

## 当前阶段

当前实现确定性的单工具 Agent Loop：

```text
CLI 输入
  -> TravelAgent
  -> FakeModel 返回文本或 get_weather 请求
  -> 执行天气工具并把 observation 放回消息
  -> FakeModel 返回最终文本
  -> 事件轨迹和最终答案
```

当前版本：

- 不访问网络；
- 不调用真实 LLM；
- 不需要 API Key；
- 默认注册表只启用固定数据的 `get_weather` 工具，但 Agent 也支持注入其他工具；
- 每次使用固定的 Fake Model 回复。

当前已经有最小的通用工具注册表、参数 schema 和多工具分发能力：

- `ToolDefinition` 描述工具名称、说明、参数类型和 handler；
- `ToolRegistry` 负责注册、查找、参数校验和分发；
- `TravelAgent` 可以注入自定义注册表，因此不需要在 Agent 内部增加
  `if tool_name == ...` 分支。

当前 schema 是教学用的简化形式，完整 JSON Schema 和真实模型适配仍在后续阶段加入。

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
