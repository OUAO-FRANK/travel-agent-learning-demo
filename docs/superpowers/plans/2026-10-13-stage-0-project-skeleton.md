# 阶段 0：Notebook 驱动的项目骨架实现计划

> **面向 AI 代理的工作者：** 实现阶段应按任务顺序执行，每次只推进一个可运行的小步骤；代码主要由学习者完成，助手先解释机制、提供失败测试、检查差异并验收，不一次性生成完整工程。

**目标：** 在独立的旅行 Agent 项目中，通过可反复运行的教学 Notebook，理解并跑通一条可观察、可测试的 Fake Model 文本回答路径，同时练习最基本的 Git 工作区、暂存区和提交差异。

**架构：** Notebook 负责解释、实验、运行输出和练习；`app/` 保存可复用实现；`tests/` 保存行为契约。CLI 将用户文本交给 Agent；Agent 维护本轮消息并调用 Fake Model；Agent 通过结构化事件报告输入、模型请求、模型响应和最终状态。阶段 0 只处理普通文本回答，工具调用往返属于阶段 1。

**技术栈：** Python 3、标准库 `dataclasses`、`argparse`、`unittest`；不新增依赖，不要求 API Key。

---

## 文件清单

- 创建 `projects/travel-agent/.gitignore`：排除 Python 缓存、虚拟环境、本地环境变量文件和测试缓存。
- 创建 `projects/travel-agent/README.md`：记录项目目的、环境要求、运行和测试命令、阶段边界。
- 创建 `projects/travel-agent/app/__init__.py`：标记 Python 包。
- 创建 `projects/travel-agent/app/domain.py`：定义 `Message` 和 `AgentEvent` 两个最小数据结构。
- 创建 `projects/travel-agent/app/model.py`：定义 Fake Model 的最小 `complete(messages)` 行为。
- 创建 `projects/travel-agent/app/agent.py`：构造本轮消息、调用模型、产生事件并返回最终回答。
- 创建 `projects/travel-agent/app/cli.py`：读取命令行问题并把结构化事件打印为可读轨迹。
- 创建 `projects/travel-agent/app/__main__.py`：支持 `python -m app` 启动 CLI。
- 创建 `projects/travel-agent/tests/__init__.py`：标记测试包。
- 创建 `projects/travel-agent/tests/test_agent.py`：锁定消息输入、最终结果与事件顺序。
- 创建 `projects/travel-agent/notebooks/00_stage_0_agent_skeleton.ipynb`：阶段 0 的讲解、实验、练习和验收记录。
- 修改 `projects/travel-agent/docs/design.md`：无；设计规格保持不变。

## 执行前检查

- [ ]  确认项目路径是 `E:\agent_learn\projects\travel-agent`，不得在 `E:\agent_learn` 根目录运行 Git 写操作。
- [ ]  执行 `git -C E:\agent_learn\projects\travel-agent rev-parse --show-toplevel`。如果仍被上层空 `.git` 目录阻挡，先检查 Git 版本与报错；不得删除或重命名现有 `E:\agent_learn\.git`。只有在确认新 `.git` 创建于项目目录内部、不会改动上层目录后，才执行 `git -C E:\agent_learn\projects\travel-agent init`。
- [ ]  记录 `D:\miniforge3\envs\fastapi-learn\python.exe --version` 和 `D:\miniforge3\envs\fastapi-learn\python.exe -m unittest --help` 是否可用；无需安装依赖。
- [ ]  阅读并解释工作区、暂存区、提交各自表示什么；此时不提交。

## 教学执行规则

- Notebook 是学习入口，不是唯一实现位置。每个核心机制先在 Notebook 中用最小代码观察，再迁移到 `app/` 并由 `tests/` 固化行为。
- 每个 Notebook 小节包含目标、前置概念、可运行代码、预期观察、常见错误和一个小练习。
- Notebook 开头显式设置项目路径并导入模块；单元不依赖隐藏的上次执行状态。
- 命令注明工作目录；涉及 `app` 导入时，从 `E:\agent_learn\projects\travel-agent` 启动 Jupyter 内核或 Python。
- Notebook 执行通过不等于掌握；验收仍需要学习者解释数据流、完成小改动并运行项目测试。

## Notebook 小课顺序

1. **项目与运行环境**：确认项目目录、Python 模块搜索路径、Git 仓库边界和测试命令。
2. **消息与事件**：用旅行输入构造 `Message` 和 `AgentEvent`，观察职责差异、冻结字段和可变 payload 边界。
3. **Fake Model**：实现确定性的 `complete(messages)`，观察模型读取消息列表、保存请求和返回文本。
4. **最小 Agent Loop**：组合用户输入、模型请求、模型响应和完成事件，追踪一轮状态变化。
5. **阶段验收与 Git**：运行测试、回顾 Notebook 输出、检查工作区/暂存区差异；提交由学习者决定。

Notebook 采用教学模板：先给目标与路线，再用“短 Markdown + 小代码单元”推进；至少包含一个练习、一个常见错误和一个可选扩展。Notebook 是可执行讲义，`app/` 和 `tests/` 是项目工程资产。

## 任务 1：建立目录和忽略本地生成文件

**文件：**

- 创建：`projects/travel-agent/.gitignore`
- 创建：`projects/travel-agent/app/__init__.py`
- 创建：`projects/travel-agent/tests/__init__.py`

- [X]  **步骤 1：创建空包标记并检查路径**

创建两个空文件：`app/__init__.py` 和 `tests/__init__.py`。随后运行：

```powershell
Get-ChildItem E:\agent_learn\projects\travel-agent -Force
```

预期：文件仅出现在旅行 Agent 项目目录下。

- [X]  **步骤 2：添加最小 `.gitignore`**

添加以下规则：

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
.pytest_cache/
```

说明：`.env` 只加入忽略规则，不创建、不读取、不修改任何密钥文件。

- [X]  **步骤 3：检查工作区路径和候选变更**

运行：

```powershell
git -C E:\agent_learn\projects\travel-agent status --short --branch
git -C E:\agent_learn\projects\travel-agent diff --no-index -- NUL E:\agent_learn\projects\travel-agent\.gitignore
```

预期：状态只涉及旅行 Agent 项目；若仓库尚未初始化，按“执行前检查”处理；`diff --no-index` 会因发现新增文件返回非零码，这是预期。不得用 `git add` 或 `git commit` 代替理解这次差异。

## 任务 2：定义消息和事件

**文件：**

- 修改：`projects/travel-agent/app/domain.py`
- 测试：`projects/travel-agent/tests/test_agent.py`
- 教学：`projects/travel-agent/notebooks/00_stage_0_agent_skeleton.ipynb`

- [X]  **步骤 1：写数据结构的最小测试**

在 `tests/test_agent.py` 中写入：

```python
import unittest

from app.domain import AgentEvent, Message


class DomainTests(unittest.TestCase):
    def test_message_keeps_role_and_content(self):
        message = Message(role="user", content="计划上海三日游")

        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "计划上海三日游")

    def test_event_keeps_kind_and_payload(self):
        event = AgentEvent(kind="user_input", payload={"content": "你好"})

        self.assertEqual(event.kind, "user_input")
        self.assertEqual(event.payload, {"content": "你好"})


if __name__ == "__main__":
    unittest.main()
```

- [X]  **步骤 2：运行测试确认失败**

运行：

```powershell
cd E:\agent_learn\projects\travel-agent
D:\miniforge3\envs\fastapi-learn\python.exe -m unittest tests.test_agent -v
```

预期：失败，因为 `app.domain` 和两个数据结构尚未实现。

- [X]  **步骤 3：学习者实现两个数据结构**

学习者阅读失败信息后，在 `domain.py` 用 `dataclasses.dataclass` 定义：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    # role 表示消息由谁产生；阶段 0 只使用 system 和 user 两种角色。
    role: str
    # content 是模型真正看到的文本；当前不引入多模态内容块。
    content: str


@dataclass(frozen=True)
class AgentEvent:
    # kind 区分 user_input、model_response 等事件类型。
    kind: str
    # payload 保存该事件的附加数据；具体字段由 kind 决定。
    payload: dict[str, object]
```

实现前解释 `frozen=True` 表示什么、为什么本阶段希望消息/事件创建后不被意外改写，以及它不代表对整个 Python 对象图做深度不可变保证。文件顶部需从 `dataclasses` 导入 `dataclass`。

- [X]  **步骤 4：重新运行测试**

运行同一条 `unittest` 命令。

预期：2 项测试通过。

- [ ] **步骤 5：在 Notebook 中完成概念实验**

Notebook 必须展示并解释以下可运行实验：

```python
from app.domain import AgentEvent, Message

user_message = Message(role="user", content="我想去上海旅游")
user_event = AgentEvent(
    kind="user_input",
    payload={"content": user_message.content, "source": "notebook"},
)

print(user_message)
print(user_event)
```

随后实验重新绑定 `user_message.role` 的失败，以及修改 `user_event.payload` 内部字典的边界。Notebook Markdown 必须解释：哪一个对象会发给模型、哪一个对象用于观察运行，以及为什么两者不能合并。

## 任务 3：建立 Fake Model 和 Agent 文本闭环

**文件：**

- 创建：`projects/travel-agent/app/model.py`
- 创建：`projects/travel-agent/app/agent.py`
- 修改：`projects/travel-agent/tests/test_agent.py`
- 教学：`projects/travel-agent/notebooks/00_stage_0_agent_skeleton.ipynb`

- [ ]  **步骤 1：先为 Agent 的可观察行为写测试**

在 `test_agent.py` 增加以下测试代码：

```python
from app.agent import TravelAgent
from app.model import FakeModel


class AgentTests(unittest.TestCase):
    def test_run_sends_system_and_user_messages_to_model(self):
        model = FakeModel(["这是一个上海三日游计划。"])
        agent = TravelAgent(model)

        result = agent.run("计划上海三日游")

        self.assertEqual(result.answer, "这是一个上海三日游计划。")
        self.assertEqual(
            model.requests[0],
            [
                Message(role="system", content="你是旅行规划助手。"),
                Message(role="user", content="计划上海三日游"),
            ],
        )

    def test_run_emits_ordered_events(self):
        model = FakeModel(["先安排外滩，再参观历史建筑。"])
        agent = TravelAgent(model)

        result = agent.run("我喜欢历史景点")

        self.assertEqual(
            [event.kind for event in result.events],
            ["user_input", "model_request", "model_response", "run_finished"],
        )
        self.assertEqual(result.events[0].payload["content"], "我喜欢历史景点")
        self.assertEqual(
            result.events[-1].payload["status"],
            "completed",
        )
```

这里的 `TravelAgent`、`FakeModel` 和结果类型尚不存在，第一次运行必须得到预期失败，而不是跳过红灯直接补实现。

- [ ]  **步骤 2：运行测试确认失败**

运行：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m unittest tests.test_agent -v
```

预期：新增 Agent 测试因目标模块/类型尚不存在而失败。

- [ ]  **步骤 3：学习者实现 Fake Model**

在 `model.py` 中声明接受回复列表的 `FakeModel`。它应维护 `requests` 列表；每次 `complete(messages)` 先保存本次消息副本，再按顺序返回下一条预设文本。

实现约束：

- 不调用网络或真实 LLM。
- 不修改调用者持有的 `messages` 列表。
- 本阶段只有一轮模型调用；回复列表耗尽时明确报错，不能静默重用最后一条回复。

- [ ]  **步骤 4：学习者实现 Agent 闭环**

在 `agent.py` 实现最小 `TravelAgent.run(user_text)`。每次运行都新建本轮消息列表，内容为 system 消息和 user 消息；调用 `FakeModel.complete(messages)`；记录有序的 `user_input`、`model_request`、`model_response`、`run_finished` 事件；返回含 `answer` 与 `events` 的结果对象。

实现约束：

- system 文本固定为 `你是旅行规划助手。`。
- `model_request` 事件可以只携带消息条数和各角色，避免初始日志重复输出全部隐私文本；测试仍直接检查 Fake Model 收到的完整消息。
- `model_response` 携带返回文本；`run_finished` 状态为 `completed`。
- 不加工具调用分支、自动重试或会话持久化。

- [ ]  **步骤 5：运行所有测试确认通过**

运行：

```powershell
D:\miniforge3\envs\fastapi-learn\python.exe -m unittest discover -v
```

预期：Domain 和 Agent 测试全部通过；Fake Model 只记录一条请求。

- [ ] **步骤 6：在 Notebook 中追踪一轮完整数据流**

Notebook 必须打印或表格化展示：用户文本、发送给模型的 `messages`、Fake Model 返回值、事件顺序和最终 `answer`，并让学习者先预测输出再运行。

## 任务 4：增加 CLI 和人工可读轨迹

**文件：**

- 创建：`projects/travel-agent/app/cli.py`
- 创建：`projects/travel-agent/app/__main__.py`
- 创建：`projects/travel-agent/README.md`
- 修改：`projects/travel-agent/notebooks/00_stage_0_agent_skeleton.ipynb`

- [ ]  **步骤 1：CLI 先保持单次输入**

CLI 从位置参数接收用户问题，构造一条固定 Fake Model 回复，调用 `TravelAgent.run()`，然后按事件顺序输出事件名称，最后打印最终回答。使用 `argparse`，不实现交互式多轮输入。

- [ ]  **步骤 2：加入模块启动入口**

`__main__.py` 只负责导入并调用 `app.cli.main()`，使以下命令可用：

```powershell
python -m app "我想下周去上海玩三天，预算 2000 元，喜欢历史景点"
```

- [ ]  **步骤 3：补充 README**

README 说明：这是教学用确定性 Agent；当前不访问网络、不调用真实 LLM；给出上面的运行命令和 `python -m unittest discover -v` 测试命令；注明工具往返在阶段 1 实现。

- [ ]  **步骤 4：运行 CLI 验证可见轨迹**

运行：

```powershell
cd E:\agent_learn\projects\travel-agent
D:\miniforge3\envs\fastapi-learn\python.exe -m app "我想下周去上海玩三天，预算 2000 元，喜欢历史景点"
```

预期：依次出现用户输入、模型请求、模型回答、运行完成事件，最后显示假模型给出的固定文本；无需网络连接。

- [ ]  **步骤 5：在 Notebook 中连接 CLI 和 Agent 轨迹**

Notebook 展示与 CLI 相同的一轮 Agent 运行，比较结构化事件和面向人的终端输出。确保代码单元可从头运行，不依赖此前手工执行的隐藏状态。

## 任务 5：Git 差异阅读与阶段验收

**文件：**

- 检查：`projects/travel-agent/`
- 修改：`projects/travel-agent/notebooks/00_stage_0_agent_skeleton.ipynb`（补充阶段验收练习和记录）

- [ ]  **步骤 1：查看工作区状态**

运行：

```powershell
git -C E:\agent_learn\projects\travel-agent status --short --branch
```

解释哪些路径是未跟踪、哪些已修改，以及当前分支名。

- [ ]  **步骤 2：查看未暂存差异**

运行：

```powershell
git -C E:\agent_learn\projects\travel-agent diff
git -C E:\agent_learn\projects\travel-agent diff --no-index -- NUL E:\agent_learn\projects\travel-agent\app\agent.py
```

解释为什么普通 `git diff` 默认看不到未跟踪文件；第二条命令只用于学习新增文件的差异表示，非零退出码是预期。

- [ ]  **步骤 3：学习者选择性暂存并检查暂存差异**

学习者根据解释自行执行 `git add`（可按文件选择），随后运行：

```powershell
git -C E:\agent_learn\projects\travel-agent diff --cached --stat
git -C E:\agent_learn\projects\travel-agent diff --cached
git -C E:\agent_learn\projects\travel-agent status --short
```

必须确认没有 `.env`、SQLite 数据库、缓存或其他机器本地产物进入暂存区。若暂存了错误文件，先讲解并使用 `git restore --staged <path>` 撤出，不删除文件。

- [ ]  **步骤 4：口述验收**

学习者不看文档解释：一条用户输入如何成为模型消息、Fake Model 如何返回文本、Runtime 如何产生事件、Agent 如何判断本轮结束；再解释工作区、暂存区、提交的区别。

- [ ]  **步骤 5：Notebook 从头执行并完成自测**

重启 Notebook 内核后从第一格运行到最后一格；完成课程中的预测题，并用自己的话说明一条输入经过 Agent 后如何成为最终回答。Notebook 的成功执行只证明示例可复现，不单独作为掌握证明。

- [ ]  **步骤 6：由学习者决定是否提交**

助手先总结暂存区中的变更文件与测试结果。只有学习者明确决定提交并检查摘要后，才由学习者执行类似以下命令：

```powershell
git -C E:\agent_learn\projects\travel-agent commit -m "feat: add observable fake agent loop"
```

不自动 commit、push 或创建 GitHub 仓库。

## 计划自检

- 阶段 0 目标均有对应任务：项目边界/Git、CLI、消息、Fake Model、Agent 文本循环、事件轨迹、测试和运行命令。
- 阶段 1 的工具往返未被提前塞入阶段 0。
- Notebook 负责教学与实验；计划文件负责实现顺序与验收索引；源码和测试仍是项目的正式实现与行为契约。
- Notebook 从项目根目录可重现执行，不要求在 `E:\\agent_learn` 工作区根目录导入 `app`。
- 测试预期与 `Message`、`AgentEvent`、`FakeModel.requests`、`TravelAgent.run()` 和结果字段一致。
- Git 初始化仅针对 `projects/travel-agent`；上层空 `.git` 保留不动。
- 计划不自动提交，也不要求配置 API Key。
