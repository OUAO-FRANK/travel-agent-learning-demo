"""旅行规划 Agent 的第一版运行时。

阶段 0 只实现了一次最小文本闭环；阶段 1 增加一次工具往返：

    用户文本
      -> system/user 消息
      -> FakeModel 返回文本或 ToolCall
      -> 如果是 ToolCall，则执行工具并把 observation 放回消息
      -> FakeModel 返回最终文本回答

这里已经加入了最小工具调用，但暂时没有加入自动重试、记忆、RAG 或持久化。
这些能力会在后续阶段逐一加入。先把最短控制流写清楚，才能知道后面
新增的每个分支到底改变了什么。
"""

from dataclasses import dataclass

from app.domain import AgentEvent, Message, ToolCall
from app.model import ModelProvider
from app.tools import ToolRegistry, build_default_registry


@dataclass(frozen=True)
class AgentResult:
    """一次 Agent 运行的结果。

    ``answer`` 是最终展示给用户的文本；``events`` 是本次运行的完整
    观察轨迹。把两者一起返回，而不是只返回字符串，是为了让调用方
    同时拥有“用户结果”和“系统证据”。
    """

    # 最终答案只保存模型返回的文本。
    answer: str

    # events 按发生顺序保存本次运行的事件。
    # 阶段 0 使用 tuple，避免结果创建后调用方追加或删除事件。
    events: tuple[AgentEvent, ...]


class TravelAgent:
    """负责组织一次旅行规划请求的最小 Agent。

    当前 Agent 的决策规则是：收到用户文本后，构造固定的 system/user
    消息；如果模型返回 ToolCall，就执行一次允许的工具，把工具结果
    加回消息列表，再请求模型生成最终文本。

    这里的 ``model`` 通过构造函数传入，而不是在类内部直接创建
    ``FakeModel``。这叫依赖注入：Agent 只依赖“有 complete 方法的模型”，
    后续可以传入真实模型、测试模型或故障模型，而不用修改 Agent 控制流。
    """

    # system 消息是应用给模型的稳定角色说明。
    # 阶段 0 先固定它，避免 Prompt 组装本身成为额外学习负担。
    SYSTEM_PROMPT = "你是旅行规划助手。"

    # 阶段 1 只允许一次工具往返，防止还没学习停止条件就引入无限循环。
    MAX_TOOL_ROUNDS = 1

    def __init__(self, model: ModelProvider, registry: ToolRegistry | None = None) -> None:
        # 保存模型依赖，但不复制或修改模型对象。
        # Agent 负责流程，具体的 ModelProvider 实现负责产生模型回复。
        self.model = model

        # 允许调用方注入一组受限工具；不传时使用默认天气工具。
        # 这样测试和不同部署环境可以拥有不同能力集合。
        # 显式判断 None，而不是使用 ``registry or ...``。
        # 这样即使未来 ToolRegistry 定义了真假值规则，调用方传入的
        # 自定义注册表也不会被错误替换成默认注册表。
        self.registry = (
            build_default_registry() if registry is None else registry
        )

    def run(self, user_text: str) -> AgentResult:
        """执行一次最小 Agent 运行并返回答案与事件轨迹。

        数据流按以下顺序发生：

        1. 创建事件列表，并记录原始用户输入；
        2. 组装本次发给模型的消息；
        3. 调用模型；
        4. 如果得到 ToolCall，执行工具并追加 assistant/tool 消息；
        5. 再次请求模型，直到得到最终文本；
        6. 记录模型响应和运行完成事件；
        7. 返回不可变的 AgentResult。

        当前方法最多执行一轮工具。FakeModel 如果没有可用回复，会把
        异常传出；我们暂时不自动重试，因为重试策略要到可靠执行阶段
        再单独设计和测试。
        """

        # 事件列表是本次 run 的局部状态。
        # 每次调用 run 都新建一份，避免两次用户请求共享历史事件。
        events: list[AgentEvent] = []

        # 先记录输入，方便从轨迹开头还原用户请求。
        events.append(
            AgentEvent(
                kind="user_input",
                payload={"content": user_text},
            )
        )

        # 消息列表是“发给模型的上下文”。
        # system 消息提供应用规则，user 消息承载当前任务。
        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_text),
        ]

        tool_rounds = 0

        while True:
            # 每次请求都重新生成工具目录，确保模型看到的是当前注册表
            # 允许使用的能力，而不是 Agent 内部写死的一份列表。
            tool_schemas = self.registry.model_schemas()

            # 每次请求模型前都记录消息摘要。
            # 第二次请求会包含 assistant 的 ToolCall 和 tool observation，
            # 这正是 Agent Loop 与阶段 0 单轮调用的关键区别。
            events.append(
                AgentEvent(
                    kind="model_request",
                    payload={
                        "message_count": len(messages),
                        "roles": [message.role for message in messages],
                        "tool_count": len(tool_schemas),
                    },
                )
            )

            # 模型可能直接回答，也可能提出工具调用。
            # Agent 只依赖 ModelProvider 契约，不关心具体供应商实现。
            model_output = self.model.complete(messages, tools=tool_schemas)

            if isinstance(model_output, str):
                # 只有文本回复可以成为最终答案。
                events.append(
                    AgentEvent(
                        kind="model_response",
                        payload={"text": model_output},
                    )
                )
                events.append(
                    AgentEvent(
                        kind="run_finished",
                        payload={"status": "completed"},
                    )
                )
                return AgentResult(
                    answer=model_output,
                    events=tuple(events),
                )

            if not isinstance(model_output, ToolCall):
                # FakeModel 当前只允许 str 或 ToolCall；其他类型说明模型
                # 适配器违反了契约，不能把它当作正常答案继续传播。
                raise TypeError("model output must be text or ToolCall")

            if tool_rounds >= self.MAX_TOOL_ROUNDS:
                # 工具调用次数是运行边界的一部分，防止模型无限请求工具。
                raise RuntimeError("maximum tool rounds exceeded")

            tool_rounds += 1
            call = model_output

            # 把模型的动作也放进消息历史。下一轮模型需要知道自己刚才
            # 请求了什么工具，否则工具结果会失去上下文。
            messages.append(
                Message(
                    role="assistant",
                    content="",
                    tool_call=call,
                )
            )
            events.append(
                AgentEvent(
                    kind="tool_requested",
                    payload={
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                )
            )

            # Runtime 执行工具，而不是让模型直接执行 Python 函数。
            observation = self.registry.dispatch(call)
            messages.append(Message(role="tool", content=observation))
            events.append(
                AgentEvent(
                    kind="tool_result",
                    payload={
                        "name": call.name,
                        "content": observation,
                    },
                )
            )
