# Runtime Harness 剩余能力设计规格

**日期：** 2026-10-18  
**项目：** Travel Agent Lab   
**状态：** 待用户审阅

## 1. 目标

在现有“模型请求/工具调用预算 + 协作式截止时间”的基础上，补齐本项目 Runtime
Harness 的剩余主线能力：

1. 运行轮数上限、可取消运行、有限重试和资源收尾；
2. 工具白名单、人工审批和安全停止；
3. 写入 SQLite 的 checkpoint，以及进程重启后的安全恢复；
4. 模型 usage、token/成本预算和可观察的运行终态。

这些能力分为三课实现，而不是把所有字段和分支一次塞进 `RuntimeHarness`：

| 课程 | 内容 | 完成后能观察什么 |
| --- | --- | --- |
| Harness 3 | 运行轮数、取消、重试、权限、资源清理 | 为什么一次运行会停止，以及失败是否可重试 |
| Harness 4 | SQLite checkpoint/resume、审批状态、不确定工具调用 | 进程重启后从哪个安全边界继续 |
| Harness 5 | usage、token 上限、成本预算、终态汇总 | 一次运行消耗了什么，为什么因成本停止 |

## 2. 不变的边界

### 2.1 Harness 和 Agent 的职责

```text
Agent：决定当前任务需要什么模型请求和工具调用
Harness：决定这次请求是否还允许发生、失败是否能重试、运行是否必须停止
Store：保存已经发生的事实和可恢复状态
```

Harness 不替模型决定旅行路线，也不直接寻找工具；它只提供运行控制决策。

### 2.2 协作式控制

当前项目的模型和工具接口是同步 Python 函数。取消和截止时间只在调用前、调用后
以及重试等待前检查，不能安全地强行杀死一个已经运行的同步函数。真正的强制超时
需要异步客户端、子进程或沙箱，暂不放进本项目主线。

### 2.3 副作用和“恰好一次”

模型请求通常可以重试；工具可能写文件、发消息或扣费，不能因为进程重启就自动重放。
工具执行开始前先持久化状态，进程若在执行中崩溃，恢复时将其标记为 `unknown`，要求
人工确认结果或明确决定是否重试。项目不会声称所有外部工具都能做到 exactly-once。

## 3. 贯穿案例

用户请求：

> “下周去上海三天，预算 2000 元；查天气后再给我路线。”

模型先请求 `get_weather`。运行可能经历四种状态：

```text
模型暂时失败
  -> 在重试上限内等待并重试
  -> 成功后继续

工具需要审批
  -> 保存 approval_pending checkpoint
  -> 进程退出
  -> 用户批准后恢复

工具执行中进程崩溃
  -> 数据库留下 running
  -> 恢复时变成 unknown
  -> 不自动重放，等待人工处理

token 或费用超过预算
  -> 保存预算事实
  -> 记录停止事件
  -> 返回 stopped，而不是伪造最终旅行计划
```

错误做法是只在内存里保留 `messages` 和一个布尔值。这样进程重启后无法判断工具
是否已经执行，容易重复副作用；也无法解释运行为什么停止。

## 4. 运行状态机

```text
created
  -> model_pending
  -> tool_pending
  -> approval_pending
  -> tool_running
  -> model_pending       （工具完成，继续下一轮）
  -> completed            （模型返回最终文本）

任意安全边界 -> stopped     （预算/超时/取消/权限/重试耗尽）
tool_running + 进程重启 -> unknown_tool
unknown_tool -> tool_completed 或 tool_pending（必须人工选择）
```

状态含义：

| 状态 | 含义 | 可否自动继续 |
| --- | --- | --- |
| `created` | 已创建运行但还没有模型结果 | 可以 |
| `model_pending` | 下一步是无副作用的模型请求 | 可以 |
| `tool_pending` | 已保存模型工具请求，尚未开始执行 | 仅在权限允许时可以 |
| `approval_pending` | 工具等待人工批准 | 不可以，需批准/拒绝 |
| `tool_running` | 工具已标记开始，结果尚未持久化 | 不可以，恢复时转 `unknown_tool` |
| `unknown_tool` | 外部副作用是否发生未知 | 不可以，需人工查证并提交结果 |
| `completed` | 已产生最终文本 | 不再继续 |
| `stopped` | Harness 安全停止 | 只有明确 resume 语义时继续 |

checkpoint 只在安全边界写入。保存的最小状态包括 `run_id`、`session_id`、原始用户
输入、当前阶段、工具轮数、待执行工具调用、消息历史引用、预算计数、usage 汇总、
审批决定和最后停止原因。`state_json` 必须包含 `state_version`，以后状态结构变化时由显式
迁移处理，不能让旧 checkpoint 被新代码静默误读。对话消息和事件仍写入已有表，不把
大段消息重复塞进状态 JSON。

## 5. 课程三：运行边界与失败控制

### 5.1 新增机制

- `max_tool_rounds`：限制模型发起工具往返的轮数，替代 Agent 内部不可配置常量；
- `CancellationToken`：用 `threading.Event` 表示取消信号，提供 `cancel()` 和只读检查；
- `retry_policy`：只包装模型请求，限制最大重试次数并支持可注入 sleep；
- `ToolPolicy`：允许工具名称集合，以及需要人工审批的工具集合；
- 资源收尾：资源由创建它的组件负责关闭。Provider 的 HTTP response、SQLite 连接使用
  上下文管理器；工具 handler 负责关闭它自己打开的资源。Harness 不关闭调用方注入的
  长生命周期对象，也不新增没有实际所有者的通用 cleanup registry。

### 5.2 重试规则

```text
调用模型
  -> ModelProviderError
  -> 未取消、未超时、attempt < max_retries ?
       是：记录 model_retry，等待退避后重试
       否：记录 harness_retry_exhausted，安全停止
```

只有 `RetryableModelProviderError` 可以重试，它只包装连接错误、请求超时、HTTP 429 和
HTTP 5xx。其他 HTTP 4xx、响应 JSON/结构错误、模型返回格式错误和本地程序错误继续使用
不可重试异常，不得仅凭“都是 `ModelProviderError`”就重试。
工具错误不自动重试。原因是当前工具契约没有幂等声明，盲目重试可能重复写操作。
模型每一次尝试在记录请求事件前预留一次 `max_model_requests`；如果事件记录期间收到取消，Provider 不会被调用，但预留预算不退回，以免请求事件和预算状态不一致。只有确实发起模型调用的 LLM 摘要器消耗此预算；本地提取摘要只检查取消和截止时间。重试前先确认还有请求预算，否则直接停止而不排队或等待退避。退避前和退避后都检查取消及总截止时间，测试通过注入的 sleep 代替真实等待。该计数覆盖主模型请求和 LLM 摘要请求。

工具策略在任何 handler 执行前完成“整批预检”：先确认这一轮全部工具都在白名单内，
再确认全部需要审批的调用已经批准。只要其中一个调用仍待审批，本轮就不执行任何工具，
避免出现“前两个工具已经产生副作用，第三个才发现没有权限”的半完成状态。

### 5.3 事件

新增可观察事件：

- `harness_cancelled`；
- `model_retry_scheduled`；
- `harness_retry_exhausted`；
- `tool_permission_denied`；
- `tool_approval_required`；
- `run_stopped`（统一携带 `reason` 和计数）。

## 6. 课程四：SQLite checkpoint/resume

### 6.1 数据库变化

在当前 `sessions/messages/events/tool_calls` 之外增加：

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

`state_json` 只保存恢复控制所需的结构化小状态；每次 checkpoint 使用同一个 `run_id`
更新，避免用多份快照猜测哪份是最新状态。Store 为每个状态迁移提供一个原子写入口，
在同一 SQLite 事务内写入该迁移关联的消息、事件、工具事实和 run 状态；事务提交成功后
Agent 才能进入下一阶段，失败则全部回滚，不能出现“工具结果已保存但 run 仍是 running”
这类撕裂状态。

### 6.2 恢复规则

1. `completed` 和已拒绝的 `stopped` 不重复执行；
2. `model_pending` 可以重新发模型请求，模型请求没有工具副作用；
3. `tool_pending` 先重新检查权限和审批，再决定执行；
4. `tool_running` 在新进程加载时转为 `unknown_tool`，不自动重放；
5. `unknown_tool` 必须通过显式命令提交人工查证结果：标记 `completed` 并填写工具结果，
   或确认“工具没有产生副作用”后改回 `tool_pending`；
6. 工具完成事件、消息和 checkpoint 必须按“结果事实先持久化，再进入下一轮”顺序写入。

### 6.3 CLI 入口

```powershell
# 创建并运行一个可恢复的任务
python -m app --db $db --session-id shanghai --run-id run-001 "规划上海三日游"

# 用户批准待处理工具后恢复
python -m app --db $db --session-id shanghai --resume-run run-001 --approve-call call-1

# 对不确定工具调用做人工决策；默认不自动重放
python -m app --db $db --resume-run run-001 --resolve-unknown call-1 `
  --unknown-outcome completed --tool-result "预订系统确认未扣款，查询结果已补录"
```

`--resolve-unknown` 必须同时指定 `--unknown-outcome completed|not_executed`；选择
`completed` 时还必须提供 `--tool-result`，选择 `not_executed` 才允许恢复执行。命令只在
数据库中写入明确决定，不把“用户没有输入”解释成批准。审批绑定 `run_id`、`call_id`、
工具名和规范化参数摘要；任何参数变化都需要重新审批。同一会话同一时间只允许有一个
未完成 run，避免另一个 run 改写恢复所依赖的消息顺序。

审批拒绝也必须显式持久化。拒绝后不执行 handler，而是给该工具调用写入一条
`"Approval denied by user."` 的工具观察，并将 run 恢复为 `model_pending`，让模型有机会
改用安全方案或解释无法继续；它不是静默批准，也不是直接伪造成功结果。

## 7. 课程五：usage、token 和成本预算

### 7.1 Provider 契约

模型适配器需要保留本次响应的可选 `ModelUsage`：

```python
@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

DeepSeek Chat Completions 响应包含 `prompt_tokens`、`completion_tokens` 和
`total_tokens`；Provider 负责解析，Harness 只负责累计和决策。usage 包含摘要调用和主
Agent 调用。FakeModel 使用 `usage=None`，不能把字符数冒充 token 数。

### 7.2 成本规则

价格不写死在代码中，由运行配置传入：

```text
estimated_cost_usd =
  prompt_tokens / 1_000_000 * input_price_per_million
  + completion_tokens / 1_000_000 * output_price_per_million
```

当供应商没有返回 usage 时，成本状态为 `unknown`，而不是 `0`。如果配置了
`max_total_tokens` 或 `max_cost_usd`，每次响应后检查；超过时记录
`harness_budget_exceeded` 并停止下一次模型请求。

### 7.3 运行汇总

完成或停止事件统一包含：模型尝试次数、成功请求数、工具调用数、已知 token 总数、
成本是否可估算、停止原因和最终状态。汇总是证据，不是替代原始事件。

## 8. 测试与验收

每课都先写失败测试，再实现最少代码。最低行为覆盖：

### Harness 3

- 模型临时失败后在重试上限内成功；
- 重试会消耗模型请求预算；
- 工具错误不会自动重试；
- 取消在下一安全边界停止；
- 未授权工具不会进入 handler；
- Provider HTTP response 和 SQLite 连接在成功、异常、取消三条路径都关闭；工具 handler
  对自身资源使用上下文管理器，不由 Harness 越权关闭。

### Harness 4

- checkpoint 可从新建 Store 重新加载；
- `model_pending` 可以恢复并继续；
- `approval_pending` 没有批准时不会调用工具；
- `tool_running` 重启后变成 `unknown_tool`；
- 已完成工具不会被恢复流程重复执行；
- 拒绝/解决命令会产生独立事件和终态。

### Harness 5

- Provider usage 正确解析并累计；
- FakeModel 的未知 usage 不会生成假成本；
- token/成本超过上限后不再发起下一次模型请求；
- 运行终态包含停止原因和资源汇总。

CLI 至少提供一条 FakeModel 可重复演示；真实 DeepSeek 只用于验证 Provider 返回 usage，
不把 API key 写入仓库或测试。

## 9. 明确不在本组课程内

- 工具队列并行、依赖图和队列级取消；
- 强制终止同步 Python 函数；
- 分布式锁、跨进程 worker 调度和生产级沙箱；
- Evaluation Harness、MCP、Multi-Agent、Skill/Plugin；
- 由 LLM 自动提取记忆和向量语义检索；
- 对任意外部副作用提供 exactly-once 保证。

## 10. 参考资料

- Python `time.monotonic` 官方文档：<https://docs.python.org/3/library/time.html>，检查日期 2026-10-18；
- Python `threading.Event` 官方文档：<https://docs.python.org/3/library/threading.html>，检查日期 2026-10-18；
- SQLite Transactions 官方文档：<https://sqlite.org/lang_transaction.html>，检查日期 2026-10-18；
- DeepSeek Chat Completions 官方文档：<https://api-docs.deepseek.com/api/create-chat-completion/>，检查日期 2026-10-18。
