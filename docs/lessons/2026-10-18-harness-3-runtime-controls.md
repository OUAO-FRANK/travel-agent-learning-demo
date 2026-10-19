# Harness 3：运行边界与失败控制

## 一句话本质

本课给 Agent 加一个“运行管理员”：它不替模型规划上海路线，只负责决定当前请求还能不能继续、失败要不要重试、工具有没有权限执行，以及什么时候安全停止。

## 贯穿案例

用户说：

> “下周去上海三天，预算 2000 元，先查天气再给路线。”

模型第一次请求 `get_weather`，工具返回天气；模型第二次请求继续查询。没有 Harness 时，模型可能不断请求工具，网络暂时失败时直接报错，或者一个批次中第一个工具已经执行后才发现第二个工具没有权限。

本课的正确流程是：

```text
模型请求
  -> Harness 消耗一次模型预算
  -> Provider 暂时失败？只对可重试错误退避重试
  -> 模型提出工具批次
  -> Harness 消耗工具轮预算
  -> 整批预检白名单/审批
  -> 全部允许后才进入 handler
  -> 每个安全边界检查取消和截止时间
  -> 完成或记录停止原因
```

## 核心组件

### `CancellationToken`

这是一个共享的取消信号。`cancel()` 只设置一个标志，不能强行杀掉正在运行的同步 Python 函数。Agent 会在模型请求和工具 handler 真正开始前检查它，并在同步操作返回后、重试等待前后再次检查。

有一个容易忽略的安全窗口：Agent 先记录 `model_request` / `tool_requested` 事件，再进入 Provider 或工具 handler。事件可能需要写 SQLite；如果恰好在这段时间收到取消，只在写事件前检查就不够，因此事件记录后还会再检查一次。模型尝试/工具调用预算已经预留，取消不会把预留预算退回，但外部 Provider/handler 不会启动。

这仍然是协作式检查，不是原子“禁止启动”锁：如果取消恰好发生在最后一次检查之后，Python 仍可能刚好进入同步函数。当前保证是每个安全边界都立即检查，并且同步调用返回后不继续下一步；强行中断同步函数要用异步取消、子进程或沙箱等不同执行模型。

代码位置：`app/harness.py`。

```python
from app.agent import TravelAgent
from app.harness import CancellationToken, RuntimeHarness
from app.model import FakeModel

token = CancellationToken()
harness = RuntimeHarness(cancellation_token=token)
model = FakeModel(["这条预设回复不会被消费"])
agent = TravelAgent(model, harness=harness)

token.cancel()  # 发出协作式停止请求；不是杀掉正在运行的函数。
result = agent.run("规划上海旅行")
print(result.status)  # stopped
print(result.events[-1].payload["reason"])  # run_cancelled
```

结果是 `status=stopped` 和 `reason=run_cancelled`，模型不会被调用。

### `max_tool_rounds`

“工具调用次数”和“工具轮数”不是同一个东西：一个工具轮可以包含多个并列工具调用。`max_tool_rounds=1` 允许模型提出一批工具，但不允许拿到结果后再次发起第二批。

`RuntimeHarness.consume_tool_round()` 在模型返回 `ToolCall` 或 `ToolCallBatch` 后调用；超限返回 `max_tool_rounds_exceeded`，Agent 将它转换为可观察的停止结果。

### `RetryPolicy`

只重试 `RetryableModelProviderError`。当前 Provider 将连接错误、超时、HTTP 408/429/5xx 标成可重试；HTTP 4xx、JSON/响应结构错误和本地程序错误不重试。

每次模型尝试都会先预留一次 `max_model_requests`。所以第一次失败、第二次成功时，模型请求计数是 2，而不是 1。若第一次失败已经用尽模型请求预算，Harness 会直接停止，不会先排队、记录重试事件或等待退避。摘要也分两类：LLM 摘要请求与主模型一样消耗预算；默认本地提取摘要不调用模型、不消耗模型请求预算，但仍检查取消和总截止时间。自定义摘要器必须声明 `consumes_model_budget`；如果未声明，Agent 会保守地按“可能会请求模型”计预算。默认退避为指数增长：

```text
第 1 次重试：backoff_seconds
第 2 次重试：2 * backoff_seconds
```

还要注意错误优先级：请求失败后先检查本次运行是否已经取消或超时，再检查重试上限。如果最后一次同步调用“运行到超时以后才返回一个网络错误”，终态应该说明超时；不能只说重试次数耗尽。

### `ToolPolicy`

工具策略在任何 handler 执行前检查整批调用：

```python
ToolPolicy(allowed_tools=frozenset({"get_weather"}))
```

如果批次是 `get_weather + book_hotel`，而白名单只有 `get_weather`，两个工具都不会执行。这样避免已经产生部分副作用后才发现权限不足。需要审批的工具当前只安全停止并记录 `tool_approval_required`；审批决定写入 SQLite 属于 Harness 4。

### `run_stopped` 与停止事件

停止时现在会记录两层证据：

1. 原因专属事件，例如 `harness_cancelled`、`harness_timeout_reached`、`harness_limit_reached`、`harness_retry_exhausted`；
2. 统一的 `run_stopped`，包含 `reason`、`model_requests`、`tool_calls`、`tool_rounds`。

最终 `run_finished` 仍然保留，表示本次运行已经结束；它不是停止原因的替代品。

## 资源清理边界

- `app/model.py` 的 HTTP response 使用 `with urlopen(...)`，成功和 JSON 解析失败都会关闭 response。
- `app/storage.py` 的 SQLite 连接由 `_connect()` 上下文管理器显式关闭。
- 工具 handler 自己创建的文件、网络连接由 handler 自己关闭。
- Harness 不关闭调用方注入的长生命周期模型或 Store，因为它不拥有这些资源。

## 代码阅读路线

建议按这条数据流阅读源文件，不需要先读测试文件：

| 顺序 | 文件与入口 | 阅读时看什么 |
| --- | --- | --- |
| 1 | [`app/harness.py`](../../app/harness.py)：`CancellationToken` | 取消只是共享标志；`wait()` 让重试等待可以被唤醒，不会强杀同步函数。 |
| 2 | [`app/harness.py`](../../app/harness.py)：`RuntimeHarness.execute_model()` | 每次尝试预留请求预算、执行事件回调、复查取消/截止时间、调用 Provider；瞬时失败才进入有限退避。 |
| 3 | [`app/agent.py`](../../app/agent.py)：`TravelAgent.run()` 中的模型/工具分支 | Agent 决定“想问模型什么/想调用哪个工具”，Harness 只回答“是否允许继续”。 |
| 4 | [`app/agent.py`](../../app/agent.py)：`execute_one()` 和 `_stop_run()` | 工具 handler 前再次检查取消；停止时把原因和计数写成事件。 |
| 5 | [`app/model.py`](../../app/model.py)：`RetryableModelProviderError`、`_urllib_transport()` | Provider 将网络状态映射成可重试/不可重试错误，并在离开 HTTP 边界时关闭响应。 |
| 6 | [`app/summarizer.py`](../../app/summarizer.py)：`consumes_model_budget` | 只有 LLM 摘要消耗模型预算；本地字符串摘要不冒充模型调用。 |
| 7 | [`app/cli.py`](../../app/cli.py)：参数定义和 `RuntimeHarness(...)` 装配 | CLI 把命令行值接到 Harness；业务逻辑不解析终端参数。 |

学习时可在 `execute_model()` 里先追踪一条“失败后成功”的路径：第一次进入循环时请求数是 1；异常类型匹配 `RetryableModelProviderError` 后安排等待；第二次循环请求数变成 2；成功后 Agent 才继续。普通 `ModelProviderError` 不匹配该异常分支，所以不会重试。

## 官方参考

- [Python 3.12 `threading.Event`](https://docs.python.org/3.12/library/threading.html#event-objects)：`set()` / `clear()` / `wait()` 的协作式信号语义。
- [Python 3.12 `urllib.error`](https://docs.python.org/3.12/library/urllib.error.html)：`HTTPError` 作为 HTTP 响应错误的含义。
- [Python 3.12 `sqlite3`](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-the-connection-context-manager)：连接上下文管理器和事务的区别；项目的 `_connect()` 额外负责显式关闭连接。

## 直接运行演示

在项目目录 `E:\agent_learn\projects\travel-agent` 执行：

```powershell
# 下面使用本项目当前环境的 Python；如果你的环境路径不同，只替换这一行。
$python = 'D:\miniforge3\envs\fastapi-learn\python.exe'

# 1. 取消在模型请求前生效：不会调用 FakeModel
& $python -m app --cancel-before-run "规划上海三日游"

# 2. FakeModel 连续提出两轮工具，但 Harness 只允许一轮
& $python -m app --demo-tool-rounds --max-tool-rounds 1 "查询上海天气并规划行程"

# 3. 白名单不包含 get_weather，整批工具不会进入 handler
& $python -m app --demo-tool-batch --allow-tool book_hotel "同时查上海和北京天气"

# 4. 模拟一次临时模型故障，观察退避后第二次成功
& $python -m app --demo-retry --max-model-retries 1 --retry-backoff-seconds 0 "规划上海旅行"

# 5. 真实 DeepSeek：只在确实需要时设置 API key；本课的重试参数也可观察
& $python -m app --provider deepseek --max-model-retries 2 --retry-backoff-seconds 1 "我想下周去上海玩三天"
```

第 4 条命令不会把 API key 写入仓库；配置层仍从 `DEEPSEEK_API_KEY` 环境变量读取。

## 主线必须掌握

- Agent 决定“想调用什么”，Harness 决定“现在是否允许调用”。
- 取消是协作式边界，不是同步函数的强制杀死。
- 重试必须按异常类型分类，不能把所有 `ModelProviderError` 都重试。
- 每次模型尝试都会预留预算；只有真实 LLM 摘要才消耗模型请求预算，本地提取摘要不算模型请求。
- 工具批次先整批预检，再执行任何 handler。
- 事件轨迹解释停止原因，最终答案本身不能替代运行证据。

## 进阶了解

- 生产环境可以把同步模型调用放入子进程或异步客户端，从而实现更强的超时隔离。
- 工具需要声明幂等性后，才可能设计有限的工具重试。
- `approval_required` 的决定必须绑定 `run_id`、工具调用 ID、工具名和规范化参数摘要；这在下一课持久化 checkpoint 时实现。

## 暂不展开

SQLite checkpoint/resume、崩溃中的 `unknown_tool`、人工审批持久化、token/成本预算和并发工具队列不属于本课。

## 自测问题与参考答案

### 1. 为什么取消不能直接杀掉正在执行的同步工具？

参考答案：当前 handler 是同步 Python 函数，Harness 没有安全的线程杀死语义。强行终止可能留下半写文件、半提交请求或锁未释放，所以本课只在调用前后和等待边界协作式停止。

### 2. 为什么一个工具批次要整体预检？

参考答案：如果按顺序执行，前面的工具可能已经产生副作用，后面的工具才发现没有权限，运行就变成半完成状态。整批预检保证“要么整批有资格执行，要么一个都不执行”。

### 3. `max_tool_rounds` 和 `max_tool_calls` 有什么区别？

参考答案：轮数限制模型最多发起多少批工具请求；调用数限制所有批次中最多实际执行多少个工具。一个批次可以包含多个调用，因此两个预算必须分开记录。

### 4. 哪些模型错误不应自动重试？

参考答案：HTTP 4xx（除 408/429）、非法 JSON、缺少 `choices`、工具参数格式错误和本地程序错误都不应重试；它们通常需要修正请求或代码，而不是等待后重复发送。

## 小练习

在 `tests/test_agent.py` 增加一个自定义工具批次测试：第一个工具名为 `read_note`，第二个为 `send_email`；创建只允许 `read_note` 的 `ToolPolicy`，两个 handler 都把自己的名字追加到列表。运行 Agent 后要求：

- `result.status == "stopped"`；
- 事件包含 `tool_permission_denied`；
- 记录列表仍然是空列表；
- 两个 handler 都没有被调用。

参考现有测试 `test_tool_policy_checks_a_batch_before_calling_any_handler` 自查即可，不需要修改生产代码。
