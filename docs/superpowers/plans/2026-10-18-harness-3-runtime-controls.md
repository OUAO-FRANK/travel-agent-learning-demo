# Harness 3 运行时控制实现计划

> **面向 AI 代理的工作者：** 本计划按 TDD 执行。每个行为先写测试并确认失败，再写最小实现；不自动执行 Git commit。

**目标：** 为旅行 Agent 增加可解释的运行轮数、协作式取消、模型有限重试、工具策略和资源边界控制。

**架构：** `RuntimeHarness` 保存本次运行的控制状态；`TravelAgent` 在模型/工具安全边界调用 Harness。模型重试只处理显式可重试异常，工具调用不自动重试。工具策略先对整批调用做权限预检，避免部分副作用后才发现权限问题。

**技术栈：** Python 3.12、`threading.Event`、标准库 `unittest`、现有 `FakeModel`/`ToolRegistry`/`ToolQueue`。

---

### 任务 1：运行控制原语

**文件：**
- 修改：`app/harness.py`
- 测试：`tests/test_harness.py`

- [x] 添加失败测试：取消信号可被设置、查询；`start_run()` 不隐式清除调用方发出的取消信号，调用方可显式 `reset()`；`max_tool_rounds` 非法值被拒绝；轮数耗尽返回稳定原因。
- [x] 运行 `python -m unittest tests.test_harness -v`，确认新测试因接口不存在或行为缺失失败。
- [x] 实现 `CancellationToken`、`max_tool_rounds` 和 `consume_tool_round()`；保留既有模型/工具预算与截止时间行为。
- [x] 运行同一测试文件，确认全部通过。

### 任务 2：Agent 接入轮数和取消

**文件：**
- 修改：`app/agent.py`
- 修改：`app/harness.py`
- 测试：`tests/test_agent.py`

- [x] 添加失败测试：第二次工具请求在 `max_tool_rounds=1` 时安全停止；运行开始前已取消时不调用模型；模型返回后取消时不执行工具。
- [x] 运行定向测试，确认因 Agent 尚未检查取消或仍使用固定 `MAX_TOOL_ROUNDS` 而失败。
- [x] 在模型请求前、模型返回后、工具执行前使用 Harness 安全边界检查；把固定常量替换为 `consume_tool_round()`；停止事件统一记录 `reason`。
- [x] 运行定向测试和现有 Harness/Agent 测试，确认通过且旧事件行为不回归。

### 任务 3：模型错误分类与有限重试

**文件：**
- 修改：`app/model.py`
- 修改：`app/summarizer.py`
- 修改：`app/agent.py`
- 修改：`app/harness.py`
- 测试：`tests/test_provider.py`
- 测试：`tests/test_agent.py`

- [x] 添加失败测试：连接/超时/429/5xx 可重试；其他 4xx、响应结构错误不重试；重试次数计入模型请求预算；重试前取消会停止；无剩余模型预算时不先安排退避。
- [x] 运行定向测试，确认异常分类和重试行为尚未存在。
- [x] 添加 `RetryableModelProviderError`；HTTP transport 只将 408/429/5xx 和网络失败映射为可重试异常；Provider 保持结构错误不可重试。
- [x] 在 Agent 中用注入的 retry policy/sleep 重试主模型和 LLM 摘要请求；每次尝试先消费模型预算，并在退避前后检查取消和截止时间；本地摘要不消耗模型预算。
- [x] 运行 Provider、Agent 和项目可用的全量测试，确认本课相关测试通过。

### 任务 4：工具白名单与整批预检

**文件：**
- 修改：`app/harness.py`
- 修改：`app/agent.py`
- 测试：`tests/test_agent.py`

- [x] 添加失败测试：未授权工具不会进入 handler；批量调用中任意一个未授权时，整批都不执行；授权工具正常执行。
- [x] 运行定向测试确认失败原因是缺少 ToolPolicy。
- [x] 实现 `ToolPolicy`，支持 allowed tool names 和 approval-required names；本课只实现未授权安全停止，approval-required 留给 Harness 4 的持久化审批。
- [x] Agent 执行工具前完成整批预检，记录 `tool_permission_denied`，不执行任何 handler。
- [x] 运行 Harness、Agent 和工具相关测试，确认通过。

### 任务 5：可观察事件、CLI 参数和课程说明

**文件：**
- 修改：`app/agent.py`
- 修改：`app/cli.py`
- 修改：`tests/test_cli.py`
- 创建：`docs/lessons/2026-10-18-harness-3-runtime-controls.md`

- [x] 添加失败测试：CLI 能设置 `--max-tool-rounds`、`--cancel-before-run` 和工具白名单演示参数，并打印停止/拒绝事件；增加有限重试演示。
- [x] 运行 CLI 定向测试确认参数不存在或输出缺失。
- [x] 实现最小 CLI 接线；`--cancel-before-run` 只用于确定性演示，不模拟强制中断同步函数。
- [x] 写课程说明：一句话本质、上海案例、数据流、代码职责、错误对比、运行命令、参考答案和练习。
- [x] 运行本课相关测试、`py_compile`、`git diff --check`，并手工运行 FakeModel CLI 演示。全量 discovery 另有原有 `ToolQueue.execute_parallel` 测试失败，不属于本课范围。

## 范围边界

本计划不实现 SQLite checkpoint/resume、持久化人工审批、token/成本预算、工具并发、强制终止同步函数或工具自动重试；这些分别属于 Harness 4/5 或后续队列课程。

## 计划自检

- 规格覆盖：轮数、取消、重试、权限和资源边界分别映射到任务 1-5。
- 失败路径：每个新增行为先有定向失败测试。
- 依赖顺序：Harness 原语 -> Agent 接入 -> Provider 错误分类 -> 工具策略 -> CLI/课程。
- 未自动提交 Git；用户可在每一步查看 `git diff`。
