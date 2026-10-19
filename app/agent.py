"""旅行规划 Agent 的第一版运行时。

阶段 0 只实现了一次最小文本闭环；阶段 1 增加一次工具往返：

    用户文本
      -> system/user 消息
      -> FakeModel 返回文本或 ToolCall
      -> 如果是 ToolCall，则执行工具并把 observation 放回消息
      -> FakeModel 返回最终文本回答

这里已经加入了工具队列、SQLite 会话持久化、Memory、RAG 和 Runtime Harness。
Harness 当前负责运行预算、协作式截止时间、取消、模型有限重试、工具轮数和工具
权限预检；SQLite checkpoint、人工审批持久化和成本预算仍在后续课程实现。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from app.context import ContextBuilder
from app.domain import AgentEvent, Message, ToolCall, ToolCallBatch
from app.harness import HarnessStop, RuntimeHarness, ToolPolicy
from app.model import ModelProvider
from app.memory import (
    SQLiteMemoryStore,
    extract_explicit_memory,
    format_memory_context,
)
from app.rag import KeywordRetriever, format_retrieval_context
from app.storage import SQLiteSessionStore
from app.summarizer import ConversationSummarizer, ExtractiveSummarizer
from app.tools import ToolRegistry, build_default_registry
from app.tool_queue import ToolQueue


T = TypeVar("T")


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

    # completed 表示正常生成答案；stopped 表示 Harness 主动停止运行。
    status: str = "completed"


class TravelAgent:
    """负责组织一次旅行规划请求的最小 Agent。

    当前 Agent 的决策规则是：收到用户文本后，加载可选历史，构造固定的
    system/user 消息；如果模型返回 ToolCall 或 ToolCallBatch，就执行允许的
    工具，把工具结果加回消息列表，再请求模型生成最终文本，并按需记录证据。

    这里的 ``model`` 通过构造函数传入，而不是在类内部直接创建
    ``FakeModel``。这叫依赖注入：Agent 只依赖“有 complete 方法的模型”，
    后续可以传入真实模型、测试模型或故障模型，而不用修改 Agent 控制流。
    """

    # system 消息是应用给模型的稳定角色说明。
    # 阶段 0 先固定它，避免 Prompt 组装本身成为额外学习负担。
    SYSTEM_PROMPT = "你是旅行规划助手。"

    def __init__(
        self,
        model: ModelProvider,
        registry: ToolRegistry | None = None,
        context_builder: ContextBuilder | None = None,
        session_store: SQLiteSessionStore | None = None,
        summarizer: ConversationSummarizer | None = None,
        retriever: KeywordRetriever | None = None,
        rag_top_k: int = 3,
        memory_store: SQLiteMemoryStore | None = None,
        memory_user_id: str | None = None,
        harness: RuntimeHarness | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
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

        # ContextBuilder 独立负责“哪些内容进入这次请求”。Agent 仍保存完整
        # 对话历史，但不再自己决定 system prompt 和历史窗口如何组装。
        self.context_builder = (
            ContextBuilder(system_prompt=self.SYSTEM_PROMPT)
            if context_builder is None
            else context_builder
        )

        # session_store 是可选依赖。默认保持纯内存运行，避免早期阶段的
        # FakeModel 测试和 CLI 示例必须创建数据库；传入它才开启持久化。
        self.session_store = session_store

        # 摘要器也通过依赖注入传入。默认使用本地规则，保证早期测试不需要
        # 网络；CLI 在真实 DeepSeek 模式下会显式注入 LLM 摘要器。
        self.summarizer = (
            ExtractiveSummarizer() if summarizer is None else summarizer
        )

        if rag_top_k <= 0:
            raise ValueError("rag_top_k must be greater than 0")
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.tool_queue = ToolQueue()
        self.harness = RuntimeHarness() if harness is None else harness
        self.tool_policy = ToolPolicy() if tool_policy is None else tool_policy

        # Memory 必须同时知道“存到哪里”和“属于哪个用户”。如果只传其中
        # 一个，记忆可能丢失或错误地串到别的用户，因此启动时直接拒绝。
        if (memory_store is None) != (memory_user_id is None):
            raise ValueError("memory_store and memory_user_id must be provided together")
        self.memory_store = memory_store
        self.memory_user_id = memory_user_id

    def run(
        self,
        user_text: str,
        session_id: str | None = None,
    ) -> AgentResult:
        """执行一次最小 Agent 运行并返回答案与事件轨迹。

        数据流按以下顺序发生：

        1. 创建事件列表，并记录原始用户输入；
        2. 读取历史，必要时生成仅供本轮请求使用的摘要；
        3. 组装本次发给模型的消息并调用模型；
        4. 如果得到 ToolCall，执行工具并追加 assistant/tool 消息；
        5. 再次请求模型，直到得到最终文本；
        6. 记录模型响应和运行完成事件；
        7. 返回不可变的 AgentResult。

        工具轮数、取消和模型重试由注入的 RuntimeHarness 控制。同步模型或工具
        正在执行时不能被安全强杀，取消会在下一个运行边界生效。FakeModel 如果
        没有可用回复，仍会把普通 RuntimeError 原样暴露，便于发现脚本配置错误。
        """

        if self.session_store is not None and session_id is None:
            # 存储多个会话时必须有稳定键；静默使用固定名称会把不同用户
            # 的旅行请求混在一起，因此明确要求调用方传入 session_id。
            raise ValueError("session_id is required when session_store is enabled")

        # 预算属于一次 run，而不是 Agent 实例的永久状态；同一个 Agent
        # 连续服务两个请求时，第二次应从零开始计算预算。
        self.harness.start_run()

        # 事件列表是本次 run 的局部状态。
        # 每次调用 run 都新建一份，避免两次用户请求共享历史事件。
        events: list[AgentEvent] = []

        # 先记录输入，方便从轨迹开头还原用户请求。
        self._record_event(
            events,
            AgentEvent(
                kind="user_input",
                payload={"content": user_text},
            ),
            session_id,
        )

        # 在写入消息、Memory 或 RAG 结果前先检查一次运行信号。预取消的
        # 请求仍保留 user_input 轨迹，但不应产生任何业务状态副作用。
        initial_runtime_decision = self.harness.check_runtime()
        if not initial_runtime_decision.allowed:
            return self._stop_run(
                events,
                session_id,
                initial_runtime_decision.reason or "runtime_stopped",
            )

        # history 是 Agent 内部累积的对话历史，不直接等同于模型请求。
        # system prompt 在每轮由 ContextBuilder 加入，避免混入可截断历史。
        messages = (
            self.session_store.load_messages(session_id)
            if self.session_store is not None and session_id is not None
            else []
        )
        user_message = Message(role="user", content=user_text)
        messages.append(user_message)
        self._record_message(user_message, session_id)

        # 压缩只改变“本轮发给模型的窗口”，不修改 messages，也不删除
        # SQLite 中的原始历史。摘要策略由注入的 summarizer 决定：测试可用
        # 本地规则，真实运行可用 LLM，因此 Runtime 不写死供应商或摘要算法。
        #
        # 这两个变量跨模型请求保留：Agent 可能先请求工具，工具结果又会
        # 增加消息数量，所以第二次模型请求前可能需要扩大摘要覆盖范围。
        summary_text: str | None = None
        summarized_messages = 0
        memory_context: str | None = None
        recalled_memory_ids: tuple[int, ...] = ()
        retrieval_context: str | None = None
        retrieved_sources: tuple[str, ...] = ()

        if self.memory_store is not None and self.memory_user_id is not None:
            explicit_memory = extract_explicit_memory(user_text)
            if explicit_memory is not None:
                memory_item = self.memory_store.add(
                    self.memory_user_id,
                    explicit_memory,
                    source="explicit",
                )
                self._record_event(
                    events,
                    AgentEvent(
                        kind="memory_written",
                        payload={
                            "memory_id": memory_item.id,
                            "user_id": self.memory_user_id,
                            "content": memory_item.content,
                            "source": memory_item.source,
                        },
                    ),
                    session_id,
                )

            self._record_event(
                events,
                AgentEvent(
                    kind="memory_search_started",
                    payload={"user_id": self.memory_user_id, "query": user_text},
                ),
                session_id,
            )
            memory_results = self.memory_store.search(
                self.memory_user_id,
                user_text,
            )
            memory_context = format_memory_context(memory_results)
            recalled_memory_ids = tuple(item.id for item in memory_results)
            self._record_event(
                events,
                AgentEvent(
                    kind="memory_search_completed",
                    payload={
                        "user_id": self.memory_user_id,
                        "query": user_text,
                        "result_count": len(memory_results),
                        "memory_ids": list(recalled_memory_ids),
                    },
                ),
                session_id,
            )

        if self.retriever is not None:
            self._record_event(
                events,
                AgentEvent(
                    kind="rag_search_started",
                    payload={"query": user_text, "top_k": self.rag_top_k},
                ),
                session_id,
            )
            retrieval_results = self.retriever.search(user_text, top_k=self.rag_top_k)
            retrieval_context = format_retrieval_context(retrieval_results)
            retrieved_sources = tuple(result.chunk_id for result in retrieval_results)
            self._record_event(
                events,
                AgentEvent(
                    kind="rag_search_completed",
                    payload={
                        "query": user_text,
                        "result_count": len(retrieval_results),
                        "sources": list(retrieved_sources),
                    },
                ),
                session_id,
            )

        while True:
            # 每次模型请求前都重新检查预算，而不是只在 run() 开始时检查。
            # 原因是一次工具往返会追加 assistant(tool_call) 和 tool 两条消息；
            # 第一轮尚未超预算，不代表拿到工具结果后的第二轮仍未超预算。
            if self.context_builder.enable_compression:
                older_messages, _ = self.context_builder.messages_to_compress(
                    messages,
                    reserved_context_messages=(
                        int(memory_context is not None)
                        + int(retrieval_context is not None)
                    ),
                )
                # 只在摘要覆盖范围扩大时重新生成，避免同一轮重复记录压缩事件。
                if len(older_messages) > summarized_messages:
                    summarized_messages = len(older_messages)
                    source_characters = sum(
                        len(message.content) for message in older_messages
                    )
                    self._record_event(
                        events,
                        AgentEvent(
                            kind="context_compression_started",
                            payload={
                                "compressed_messages": summarized_messages,
                                "max_messages": self.context_builder.max_messages,
                                "source_characters": source_characters,
                            },
                        ),
                        session_id,
                    )
                    self._record_event(
                        events,
                        AgentEvent(
                            kind="context_summary_requested",
                            payload={
                                "compressed_messages": summarized_messages,
                                "source_characters": source_characters,
                            },
                        ),
                        session_id,
                    )
                    try:
                        summary_operation = lambda: self.summarizer.summarize(
                            older_messages
                        )
                        if getattr(
                            self.summarizer,
                            "consumes_model_budget",
                            True,
                        ):
                            # 只有真的请求模型的摘要器才消耗模型请求预算，并
                            # 进入有限重试；默认本地摘要显式声明不消耗预算。
                            # 未声明策略的自定义摘要器按“可能访问模型”处理，
                            # 避免漏掉它实际发出的供应商请求。
                            summary_text = self._execute_model(
                                events,
                                session_id,
                                summary_operation,
                                {
                                    "purpose": "context_summary",
                                    "message_count": len(older_messages) + 1,
                                    "compressed_messages": summarized_messages,
                                },
                            )
                        else:
                            # 本地摘要没有供应商请求，但仍遵守运行总超时和取消，
                            # 避免一次耗时的自定义摘要后继续调用主模型。
                            before_summary = self.harness.check_runtime()
                            if not before_summary.allowed:
                                raise HarnessStop(
                                    before_summary.reason or "runtime_stopped"
                                )
                            summary_text = summary_operation()
                            after_summary = self.harness.check_runtime()
                            if not after_summary.allowed:
                                raise HarnessStop(
                                    after_summary.reason or "runtime_stopped"
                                )
                    except HarnessStop as stop:
                        return self._stop_run(events, session_id, stop.reason)
                    self._record_event(
                        events,
                        AgentEvent(
                            kind="context_compressed",
                            payload={
                                "compressed_messages": summarized_messages,
                                "max_messages": self.context_builder.max_messages,
                                "source_characters": source_characters,
                                "summary_characters": len(summary_text),
                                "summary": summary_text,
                            },
                        ),
                        session_id,
                    )

            # 每次请求都从当前注册表获取工具目录，再与对话历史一起交给
            # ContextBuilder。ContextBuilder 的输出才是实际发送给模型的窗口。
            tool_schemas = self.registry.model_schemas()
            context = self.context_builder.build(
                messages,
                tool_schemas,
                summary_text=summary_text,
                summarized_messages=summarized_messages,
                memory_context=memory_context,
                retrieval_context=retrieval_context,
            )

            # 每次真正的供应商尝试都会记录 model_request 并消耗一次预算，
            # 包括下面可能发生的重试。
            try:
                model_output = self._execute_model(
                    events,
                    session_id,
                    lambda: self.model.complete(
                        list(context.messages),
                        tools=list(context.tools),
                    ),
                    {
                        "purpose": "agent",
                        "message_count": len(context.messages),
                        "roles": [message.role for message in context.messages],
                        "tool_count": len(context.tools),
                        "dropped_messages": context.dropped_messages,
                        "summarized_messages": context.summarized_messages,
                        "recalled_memory_ids": list(recalled_memory_ids),
                        "retrieved_sources": list(retrieved_sources),
                    },
                )
            except HarnessStop as stop:
                return self._stop_run(events, session_id, stop.reason)

            if isinstance(model_output, str):
                # 只有文本回复可以成为最终答案。
                assistant_message = Message(
                    role="assistant",
                    content=model_output,
                )
                messages.append(assistant_message)
                self._record_message(assistant_message, session_id)
                self._record_event(
                    events,
                    AgentEvent(
                        kind="model_response",
                        payload={"text": model_output},
                    ),
                    session_id,
                )
                self._record_event(
                    events,
                    AgentEvent(
                        kind="run_finished",
                        payload={"status": "completed"},
                    ),
                    session_id,
                )
                return AgentResult(
                    answer=model_output,
                    events=tuple(events),
                    status="completed",
                )

            if not isinstance(model_output, (ToolCall, ToolCallBatch)):
                # 当前只允许 str、ToolCall 或 ToolCallBatch；其他类型说明模型
                # 适配器违反了契约，不能把它当作正常答案继续传播。
                raise TypeError("model output must be text, ToolCall, or ToolCallBatch")

            calls = (
                (model_output,)
                if isinstance(model_output, ToolCall)
                else model_output.calls
            )

            round_decision = self.harness.consume_tool_round()
            if not round_decision.allowed:
                return self._stop_run(
                    events,
                    session_id,
                    round_decision.reason or "max_tool_rounds_exceeded",
                )

            # 批次先整体过权限检查，再创建队列或调用任何 handler。
            # 这样批次里即使前几个工具合法，后面一个不允许也不会留下
            # “做了一半”的外部副作用。
            policy_decision = self.tool_policy.check(calls)
            if not policy_decision.allowed:
                permission_kind = (
                    "tool_approval_required"
                    if policy_decision.reason
                    and policy_decision.reason.startswith("tool_approval_required:")
                    else "tool_permission_denied"
                )
                self._record_event(
                    events,
                    AgentEvent(
                        kind=permission_kind,
                        payload={
                            "reason": policy_decision.reason,
                            "tools": [call.name for call in calls],
                        },
                    ),
                    session_id,
                )
                return self._stop_run(
                    events,
                    session_id,
                    policy_decision.reason or "tool_permission_denied",
                )

            is_batch = len(calls) > 1

            # 把模型的动作也放进消息历史。下一轮模型需要知道自己刚才
            # 请求了什么工具，否则工具结果会失去上下文。
            assistant_message = Message(
                role="assistant",
                content="",
                tool_call=calls[0] if len(calls) == 1 else None,
                tool_calls=calls if len(calls) > 1 else None,
            )
            messages.append(assistant_message)
            self._record_message(assistant_message, session_id)
            if is_batch:
                self._record_event(
                    events,
                    AgentEvent(
                        kind="tool_queue_started",
                        payload={"count": len(calls)},
                    ),
                    session_id,
                )

            call_counter = 0

            def execute_one(call: ToolCall) -> str:
                """执行队列中的一个调用，并记录完整的可观察轨迹。"""

                nonlocal call_counter
                call_counter += 1
                tool_decision = self.harness.consume_tool_call()
                if not tool_decision.allowed:
                    raise HarnessStop(
                        tool_decision.reason or "runtime_budget_exceeded"
                    )
                call_id = (
                    self.session_store.next_tool_call_id(session_id)
                    if self.session_store is not None and session_id is not None
                    else (
                        f"call-{self.harness.tool_rounds}-{call_counter}"
                        if is_batch
                        else f"call-{self.harness.tool_rounds}"
                    )
                )
                self._record_event(
                    events,
                    AgentEvent(
                        kind="tool_requested",
                        payload={
                            "call_id": call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    ),
                    session_id,
                )

                # tool_requested 事件可能被同步持久化，也可能在持久化期间收到
                # 取消；在进入 handler 前再次检查，确保取消后不启动副作用。
                tool_runtime_decision = self.harness.check_runtime()
                if not tool_runtime_decision.allowed:
                    raise HarnessStop(
                        tool_runtime_decision.reason or "runtime_stopped"
                    )

                # Runtime 执行工具，而不是让模型直接执行 Python 函数。
                observation = self.registry.dispatch(call)
                tool_message = Message(role="tool", content=observation)
                messages.append(tool_message)
                self._record_message(tool_message, session_id)
                self._record_event(
                    events,
                    AgentEvent(
                        kind="tool_result",
                        payload={
                            "call_id": call_id,
                            "name": call.name,
                            "content": observation,
                        },
                    ),
                    session_id,
                )
                # 工具已经执行并返回，因此先记录结果，再判断它是否耗尽了
                # 本次运行时间。即使超时，轨迹也不会丢掉已发生的事实。
                runtime_decision = self.harness.check_runtime()
                if not runtime_decision.allowed:
                    raise HarnessStop(
                        runtime_decision.reason or "runtime_stopped"
                    )
                return observation

            try:
                self.tool_queue.execute(calls, execute_one)
            except HarnessStop as stop:
                return self._stop_run(events, session_id, stop.reason)
            if is_batch:
                self._record_event(
                    events,
                    AgentEvent(
                        kind="tool_queue_finished",
                        payload={"count": len(calls)},
                    ),
                    session_id,
                )

    def _record_event(
        self,
        events: list[AgentEvent],
        event: AgentEvent,
        session_id: str | None,
    ) -> None:
        """同时写入本次结果和持久化事件表。"""

        events.append(event)
        if self.session_store is not None and session_id is not None:
            self.session_store.append_event(session_id, event)

    def _execute_model(
        self,
        events: list[AgentEvent],
        session_id: str | None,
        operation: Callable[[], T],
        request_payload: dict[str, object],
    ) -> T:
        """通过 Harness 执行一次模型操作，并记录每次尝试与重试。

        ``operation`` 可以是 Agent 主请求，也可以是 LLM 摘要请求。统一经过
        这里，保证重试尝试会逐次消耗同一份模型预算，且不会漏掉摘要费用。
        """

        def record_attempt(attempt: int) -> None:
            # 每次回调对应一次真实 Provider 调用，而不是逻辑上的一个请求。
            self._record_event(
                events,
                AgentEvent(
                    kind="model_request",
                    payload={**request_payload, "attempt": attempt},
                ),
                session_id,
            )

        def record_retry(attempt, error, delay_seconds: float) -> None:
            self._record_event(
                events,
                AgentEvent(
                    kind="model_retry_scheduled",
                    payload={
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "delay_seconds": delay_seconds,
                    },
                ),
                session_id,
            )

        def record_exhaustion(attempt, error) -> None:
            self._record_event(
                events,
                AgentEvent(
                    kind="harness_retry_exhausted",
                    payload={
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                    },
                ),
                session_id,
            )

        return self.harness.execute_model(
            operation,
            on_attempt=record_attempt,
            on_retry=record_retry,
            on_exhausted=record_exhaustion,
        )

    def _record_message(self, message: Message, session_id: str | None) -> None:
        """在启用会话时保存一条新增消息。"""

        if self.session_store is not None and session_id is not None:
            self.session_store.append_message(session_id, message)

    def _stop_run(
        self,
        events: list[AgentEvent],
        session_id: str | None,
        reason: str,
    ) -> AgentResult:
        """把 Harness 的停止决定转换成可观察的 AgentResult。"""

        timeout_reached = reason == "runtime_timeout_exceeded"
        stop_payload: dict[str, object] = {
            "reason": reason,
            "model_requests": self.harness.model_requests,
            "tool_calls": self.harness.tool_calls,
            "tool_rounds": self.harness.tool_rounds,
        }
        if timeout_reached:
            stop_payload["timeout_seconds"] = self.harness.timeout_seconds

        stop_event_kind = (
            "harness_timeout_reached"
            if timeout_reached
            else (
                "harness_cancelled"
                if reason == "run_cancelled"
                else (
                    "harness_limit_reached"
                    if reason
                    in {
                        "max_model_requests_exceeded",
                        "max_tool_calls_exceeded",
                        "max_tool_rounds_exceeded",
                    }
                    else None
                )
            )
        )
        if stop_event_kind is not None:
            # 只在确实命中时间、取消或计数边界时生成对应 Harness 事件。
            # 权限拒绝和重试耗尽已有各自专属事件，不能错误地伪装成预算耗尽。
            self._record_event(
                events,
                AgentEvent(kind=stop_event_kind, payload=stop_payload),
                session_id,
            )
        self._record_event(
            events,
            AgentEvent(
                kind="run_stopped",
                payload={**stop_payload, "status": "stopped"},
            ),
            session_id,
        )
        self._record_event(
            events,
            AgentEvent(
                kind="run_finished",
                payload={"status": "stopped", "reason": reason},
            ),
            session_id,
        )
        return AgentResult(
            answer=f"运行已停止：{reason}",
            events=tuple(events),
            status="stopped",
        )
