"""负责把 Agent 的完整对话历史整理成一次模型请求上下文。

Agent 会在内存里累积消息，但并不意味着每一条消息都要原样发送给模型。
``ContextBuilder`` 是两者之间的边界：它保留固定的 system prompt，按预算
选择最近的对话轮次，并把当前可用工具的 schema 一起放进请求快照。

当前预算用“消息数”教学，而不是精确 token 数。不同模型的 tokenizer 不同，
后续接入真正的 token 预算时，只需要替换这个模块的选择策略。
"""

from copy import deepcopy
from dataclasses import dataclass

from app.domain import Message


@dataclass(frozen=True)
class ModelContext:
    """一次即将发送给模型的上下文快照。

    ``messages`` 和 ``tools`` 是模型本轮真正会收到的内容；
    ``dropped_messages`` 记录本次为了满足消息预算而省略了多少历史消息。
    这个对象不是持久化历史，而是单次请求的组装结果。
    """

    # 使用 tuple 固定消息顺序，避免调用方通过 append 改写本次请求快照。
    messages: tuple[Message, ...]

    # 工具 schema 是模型可见的工具目录，和对话消息一样属于请求上下文。
    # 字典内部仍可变，因此 ContextBuilder 会在创建时复制它们。
    tools: tuple[dict[str, object], ...]

    # 被窗口截断的历史消息数量；system prompt 不算历史，不会被丢弃。
    dropped_messages: int

    # 本次请求中用一条摘要替代的原始消息数量。
    # 摘要不是持久化消息，只是本次模型请求的上下文补充。
    summarized_messages: int = 0

    # 实际注入请求的摘要文本；没有触发压缩时为 None。
    summary_text: str | None = None


class ContextBuilder:
    """按固定 system prompt 和消息预算构造模型请求上下文。

    ``history`` 是 Agent 内部保存的对话消息，不包含 system 消息；system
    prompt 由本类在每次请求中统一加入。这样不会把 system prompt 重复存进
    对话历史，也能确保每轮请求都带有同一份应用规则。

    消息预算是一种教学版近似：system 消息占一个名额，其余名额用于历史。
    为避免破坏工具调用链，一个 user 开始的完整对话轮次不会被拆开；当最新
    一轮本身超过预算时，会优先保留完整的一轮，因此实际消息数可能略超预算。
    """

    def __init__(
        self,
        system_prompt: str = "你是旅行规划助手。",
        max_messages: int | None = None,
        enable_compression: bool = False,
    ) -> None:
        # 预算至少要容纳 system prompt 和一条当前任务消息。
        # None 表示本阶段先不截断，只观察完整历史如何进入请求。
        if max_messages is not None and max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        if enable_compression and (max_messages is None or max_messages < 3):
            raise ValueError(
                "context compression requires max_messages to be at least 3"
            )

        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.enable_compression = enable_compression

    def build(
        self,
        history: list[Message],
        tool_schemas: list[dict[str, object]],
        summary_text: str | None = None,
        summarized_messages: int = 0,
        memory_context: str | None = None,
        retrieval_context: str | None = None,
    ) -> ModelContext:
        """生成包含 system、筛选后的历史和工具目录的请求快照。

        Args:
            history: Agent 到当前时刻为止累积的对话消息，不含 system。
            tool_schemas: 工具注册表导出的模型可读 schema。

        Returns:
            一个不可追加消息的上下文快照，以及本次丢弃的历史消息数。
        """

        # 有摘要时仍从完整历史计算最近轮次，确保摘要只替代较早轮次，
        # 不会把最近的工具调用链重复放进摘要和原文两个位置。
        # 摘要、Memory 和 RAG 证据各占一条临时上下文消息，所以都要
        # 从历史预算中预留位置。
        selected_history = self._select_history(
            history,
            reserved_context_messages=(
                int(summary_text is not None)
                + int(memory_context is not None)
                + int(retrieval_context is not None)
            ),
        )
        system_message = Message(role="system", content=self.system_prompt)

        summary_messages: tuple[Message, ...] = ()
        if summary_text is not None:
            summary_messages = (
                Message(
                    # 摘要来自历史消息，不应获得 system 指令的优先级。
                    # 使用 assistant 角色可以让模型知道它是运行时整理出的
                    # 上下文补充，而不是一条新的应用规则。
                    role="assistant",
                    content=(
                        "以下是此前会话的摘要。它用于补充历史，不是新的系统规则：\n"
                        f"{summary_text}"
                    ),
                ),
            )

        retrieval_messages: tuple[Message, ...] = ()
        if retrieval_context is not None:
            retrieval_messages = (
                Message(role="assistant", content=retrieval_context),
            )

        memory_messages: tuple[Message, ...] = ()
        if memory_context is not None:
            memory_messages = (
                Message(role="assistant", content=memory_context),
            )

        # deepcopy 不只复制最外层工具字典，也复制嵌套的 parameters 等对象，
        # 这样注册表或调用方后续修改 schema 不会污染已记录的请求快照。
        copied_tools = tuple(deepcopy(tool_schema) for tool_schema in tool_schemas)

        return ModelContext(
            messages=(
                system_message,
                *summary_messages,
                *memory_messages,
                *retrieval_messages,
                *selected_history,
            ),
            tools=copied_tools,
            # history 仍然是完整历史；summarized_messages 只是说明其中有多少
            # 条旧消息已经由摘要替代。剩余未被摘要替代却没进入窗口的消息，
            # 才计入 dropped_messages。
            dropped_messages=max(
                0,
                len(history) - len(selected_history) - summarized_messages,
            ),
            summarized_messages=summarized_messages,
            summary_text=summary_text,
        )

    def messages_to_compress(
        self,
        history: list[Message],
        reserved_context_messages: int = 0,
    ) -> tuple[tuple[Message, ...], tuple[Message, ...]]:
        """把历史拆成“需要摘要的旧轮次”和“保留原文的最近轮次”。

        返回值是两个 tuple：

        1. ``older_messages``：超出预算、准备交给摘要器整理的消息；
        2. ``recent_messages``：本轮仍然以原文发送给主模型的完整轮次。

        分组依据仍然是 user 消息边界，所以 assistant 工具请求和 tool
        observation 不会被单独切走。没有预算，或历史尚未超预算时，旧消息为空。
        """

        if self.max_messages is None or self.max_messages < 3:
            return (), tuple(history)

        # 这里必须把临时上下文也算进去。比如 RAG 证据会额外占一条
        # assistant 消息；如果只计算 system + history，后续 build() 仍可能
        # 组装出超过预算的请求，压缩就失去了作用。
        if len(history) + 1 + reserved_context_messages <= self.max_messages:
            return (), tuple(history)

        turns = self._group_turns(history)
        # 压缩后的请求至少包含 system 和所有临时上下文消息，剩余预算才留给原文。
        selected_turns = self._select_recent_turns(
            turns,
            reserved_context_messages=1 + reserved_context_messages,
        )
        selected_history = tuple(
            message
            for turn in selected_turns
            for message in turn
        )

        if len(selected_history) == len(history):
            return (), selected_history

        older_count = len(history) - len(selected_history)
        older_history = tuple(history[:older_count])
        return older_history, selected_history

    def _select_history(
        self,
        history: list[Message],
        reserved_context_messages: int = 0,
    ) -> tuple[Message, ...]:
        """从最近的完整对话轮次开始选择历史。

        一轮从 user 消息开始，到下一条 user 消息之前结束。比如：

            user -> assistant(tool_call) -> tool

        会作为不可拆分的一轮处理，避免模型只看到工具结果却看不到调用请求。
        """

        if self.max_messages is None:
            return tuple(history)

        turns = self._group_turns(history)
        selected_turns = self._select_recent_turns(
            turns,
            reserved_context_messages=reserved_context_messages,
        )
        return tuple(
            message
            for turn in selected_turns
            for message in turn
        )

    def _select_recent_turns(
        self,
        turns: list[tuple[Message, ...]],
        reserved_context_messages: int = 0,
    ) -> list[tuple[Message, ...]]:
        """从后往前选择能放进预算的完整轮次。"""

        if self.max_messages is None:
            return turns

        # system prompt 占一条；摘要、Memory 和 RAG 证据各占一条临时上下文消息。
        history_budget = self.max_messages - 1 - reserved_context_messages
        selected_turns_reversed: list[tuple[Message, ...]] = []
        selected_count = 0

        # 从最新轮次向过去倒序选择。遇到一个完整轮次放不进预算时就停止，
        # 不跳过它去选更旧的轮次，否则上下文会出现时间顺序断裂。
        for turn in reversed(turns):
            if selected_count and selected_count + len(turn) > history_budget:
                break

            selected_turns_reversed.append(turn)
            selected_count += len(turn)

            # 最新一轮可能比预算还大；仍保留完整一轮后立即停止。
            if selected_count >= history_budget:
                break

        # 反向挑选时，最后逆转轮次顺序恢复原始时间顺序。
        selected_turns_reversed.reverse()
        return selected_turns_reversed

    @staticmethod
    def _group_turns(history: list[Message]) -> list[tuple[Message, ...]]:
        """按 user 消息边界将历史切成完整轮次。"""

        turns: list[tuple[Message, ...]] = []
        current_turn: list[Message] = []

        for message in history:
            # 下一条 user 代表上一轮已经结束；从它开始创建新轮次。
            if message.role == "user" and current_turn:
                turns.append(tuple(current_turn))
                current_turn = []

            current_turn.append(message)

        if current_turn:
            turns.append(tuple(current_turn))

        return turns
