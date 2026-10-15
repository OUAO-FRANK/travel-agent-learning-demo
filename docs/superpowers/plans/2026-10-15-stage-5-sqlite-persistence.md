# 阶段 5：SQLite 会话持久化实现计划

> **面向学习者：** 本计划用来沿数据流学习，不自动执行 `git commit` 或 `git push`。

**目标：** 为旅行规划 Agent 增加最小 SQLite 持久化，使消息历史、运行事件和工具调用证据可以在重新创建 Store 后恢复。

**架构：** `TravelAgent` 继续负责模型循环；`SQLiteSessionStore` 只负责会话生命周期、消息/事件/工具调用的读写和 JSON 序列化。持久化是可选依赖，不传 Store 时保持原有的纯内存行为。每个数据库操作使用独立短事务，避免测试和进程退出时留下打开的文件句柄。

**技术栈：** Python 3 标准库 `sqlite3`、`json`、`dataclasses`、`unittest`；不新增第三方依赖。

---

## 文件职责

- 创建 `app/storage.py`：SQLite schema、会话状态、消息/事件/工具调用的序列化与恢复。
- 修改 `app/agent.py`：可选注入 Store；运行开始加载历史；运行过程中追加消息和事件；失败时保留已经发生的证据。
- 创建 `tests/test_storage.py`：锁定四张表、消息恢复、事件恢复和工具调用证据行为。
- 修改 `tests/test_agent.py`：锁定重新创建 Agent 后可以继续使用同一会话历史。
- 修改 `app/cli.py`：增加成对的 `--db`、`--session-id` 参数和持久化观测信息。
- 修改 `tests/test_cli.py`：锁定 CLI 首次运行、继续会话和参数校验行为。
- 修改 `docs/design.md`：记录阶段 5 的完成范围和下一阶段。
- 修改 `README.md`：说明 SQLite 持久化入口和当前边界。

## 执行步骤

1. 先写 `tests/test_storage.py` 的失败测试：消息往返、事件与工具证据、四张表存在。
2. 运行定向测试，确认失败原因是 `app.storage` 尚不存在。
3. 实现最小 `SQLiteSessionStore` 和四张表：`sessions`、`messages`、`events`、`tool_calls`。
4. 重新运行存储测试，确认数据库文件可关闭、临时目录可清理，且消息/事件顺序稳定。
5. 为 `TravelAgent` 增加可选 `session_store` 和 `session_id`，保存新增消息、事件及最终 assistant 答案。
6. 增加 Agent 恢复测试，使用两个不同的 FakeModel 验证第二次请求能看到第一轮历史。
7. 为真实 CLI 增加 SQLite 参数；通过运行前后的数据库读取展示恢复历史和持久化统计。
8. 确保同一持久化会话的工具调用 ID 跨多次 `run` 持续递增。
9. 运行全量测试、字节码编译和 `git diff --check`。

## 验收标准

- 新建 Store 后数据库包含四张学习阶段所需的表。
- 关闭并重新创建 Store 后，`Message`、`AgentEvent` 和工具调用证据顺序不变。
- 第二个 Agent 通过同一个 `session_id` 能读取上一轮用户消息和 assistant 最终回答。
- CLI 使用同一数据库和会话 ID 再运行时，能够展示已恢复的消息和最新数据库统计。
- 默认不传数据库参数的现有 CLI 行为不变。
- 同一会话跨多次 Agent 运行产生的工具调用 ID 不冲突。
- 失败运行不会伪造完成答案，数据库仍保留已经发生的消息和事件。
