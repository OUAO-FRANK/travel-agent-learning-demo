"""Agent 的最小领域数据结构。

这个模块只描述“系统里流动的数据长什么样”，暂时不负责：

* 调用模型；
* 执行工具；
* 保存数据库；
* 决定 Agent 下一步做什么。

把数据结构单独放在这里，是为了让后面的 Model、Runtime 和 CLI
共享同一套简单的数据契约，而不是每个模块各自用字典拼装数据。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """模型提出的一次工具调用请求。

    工具调用不是工具执行结果，而是模型告诉 Runtime：

        “我希望调用名为 ``name`` 的工具，并传入 ``arguments``。”

    Runtime 仍然必须检查工具是否存在、参数是否合法以及调用是否被允许。
    不能因为模型返回了一个工具名，就直接执行任意 Python 函数。
    """

    # name 用于从工具注册表中找到对应的能力。
    name: str

    # arguments 是工具的结构化参数。
    # 当前阶段使用字典；参数 schema 和严格校验留到工具系统阶段。
    arguments: dict[str, object]

    # DeepSeek thinking/tool-call 模式可能返回 reasoning_content。
    # Runtime 不把它展示给用户，但下一轮请求需要原样带回供应商。
    reasoning_content: str | None = None


@dataclass(frozen=True)
class ToolCallBatch:
    """模型在一次响应中提出的多个独立工具调用。"""

    # 使用 tuple 固定调用顺序；队列阶段会按这个顺序执行。
    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        # 空批次没有任何可执行工作，通常说明 Provider 解析出了异常响应。
        if not self.calls:
            raise ValueError("tool call batch must not be empty")


@dataclass(frozen=True)
class Message:
    """一条发给模型的消息。

    在阶段 0 中，我们只需要知道两件事：

    1. ``role``：这条消息是谁产生的，例如 ``system`` 或 ``user``；
    2. ``content``：模型实际接收到的文本。

    当前只加入了支持工具往返所需的 ``tool_call`` 字段；时间戳、消息 ID、
    图片、多工具调用列表等更复杂的字段仍然后置，避免协议过早膨胀。
    """

    # role 决定消息在对话中的身份。
    # system 通常用于应用规则，user 表示用户请求。
    role: str

    # content 是消息中的文本。
    # 工具请求消息可以暂时使用空字符串，因为动作放在 tool_call 字段。
    content: str

    # tool_call 只有 assistant 请求工具时才有值；普通文本消息为 None。
    # 使用可选字段让旧的 system/user 文本消息继续保持简单。
    tool_call: ToolCall | None = None

    # 新版模型可能在一条 assistant 消息中请求多个工具。
    # 保留 tool_call 字段是为了兼容前面课程已经实现的单工具消息。
    tool_calls: tuple[ToolCall, ...] | None = None

    def __post_init__(self) -> None:
        if self.tool_call is not None and self.tool_calls is not None:
            raise ValueError("message cannot contain tool_call and tool_calls together")
        if self.tool_calls is not None and not self.tool_calls:
            raise ValueError("message tool_calls must not be empty")


@dataclass(frozen=True)
class AgentEvent:
    """Agent 运行过程中的一条可观察事件。

    ``Message`` 描述“发给模型的数据”，而 ``AgentEvent`` 描述“系统
    运行时发生了什么”。例如：

    * ``user_input``：用户输入了什么；
    * ``model_request``：Runtime 准备向模型发送请求；
    * ``model_response``：模型返回了什么；
    * ``run_finished``：本轮运行以什么状态结束。

    事件是调试和学习 Agent 控制流的观察窗口。没有事件时，我们只能
    看到最终答案，很难知道中间到底发生了什么。
    """

    # kind 是事件类型。它让读取日志的一方知道 payload 应该如何解释。
    kind: str

    # payload 保存这个事件专属的附加数据。
    # 使用字典是因为不同事件需要的字段不同：输入事件需要 content，
    # 结束事件需要 status。阶段 0 先保持这个结构简单，后续再考虑更强的 schema。
    payload: dict[str, object]
