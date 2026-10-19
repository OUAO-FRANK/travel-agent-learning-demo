"""Agent Runtime Harness 的当前层：运行预算与截止时间。

Harness 不决定旅行计划，也不决定调用哪个工具。它只负责回答一个运行时问题：
“本次运行还允许继续消耗多少模型请求和工具调用、还能运行多久？”
把这个策略独立出来后，后续可以在不重写 Agent 决策逻辑的情况下加入取消、重试和
成本预算等运行控制。
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from threading import Event
from typing import TypeVar

from app.domain import ToolCall
from app.model import RetryableModelProviderError



@dataclass(frozen=True)
class HarnessDecision:
    """一次预算检查的结果。"""

    allowed: bool
    reason: str | None = None


class HarnessStop(RuntimeError):
    """控制流异常：Harness 要求当前运行在安全边界停止。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RetryPolicy:
    """模型请求的有限重试配置。

    ``max_retries`` 表示第一次失败后最多追加多少次尝试，因此总请求数是
    ``1 + max_retries``。可注入 sleep 函数让测试不必真的等待；默认使用
    CancellationToken.wait，因此等待期间也能响应取消。
    """

    max_retries: int = 0
    backoff_seconds: float = 0.0
    sleep: Callable[[float], None] | None = None

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")


@dataclass(frozen=True)
class ToolPolicy:
    """定义 Runtime 允许进入工具 handler 的边界。"""

    allowed_tools: frozenset[str] | None = None
    approval_required: frozenset[str] = frozenset()

    def check(self, calls: Iterable[ToolCall]) -> HarnessDecision:
        """对整批调用做预检，任何一个不允许都会拒绝整批。"""

        for call in calls:
            if self.allowed_tools is not None and call.name not in self.allowed_tools:
                return HarnessDecision(False, f"tool_not_allowed:{call.name}")
            if call.name in self.approval_required:
                return HarnessDecision(False, f"tool_approval_required:{call.name}")
        return HarnessDecision(True)


T = TypeVar("T")


class CancellationToken:
    """一个可在线程之间共享的协作式取消信号。

    ``Event`` 只表示“是否有人发出了取消请求”，它不会强行终止已经在运行
    的同步 Python 函数。Harness 会在模型请求、工具调用和重试等待等安全边界
    检查这个信号。
    """

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """发出取消请求；重复调用是安全的。"""

        self._event.set()

    def reset(self) -> None:
        """为下一次运行清除取消请求。"""

        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        """返回当前是否已经收到取消请求。"""

        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """等待指定时间或直到收到取消；收到取消时提前返回 True。"""

        return self._event.wait(timeout)


class RuntimeHarness:
    """记录一次运行的消耗，并在操作边界检查可选的截止时间。

    ``clock`` 默认使用单调时钟。与日历时间不同，单调时钟不会因为系统时间校准而倒退，因此适合计算“运行了多久”。测试可以传入一个可控时钟，不必真的等待几秒就能验证超时行为。
    """

    def __init__(
        self,
        max_model_requests: int = 8,
        max_tool_calls: int = 8,
        timeout_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
        max_tool_rounds: int = 1,
        cancellation_token: CancellationToken | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if max_model_requests <= 0:
            raise ValueError("max_model_requests must be greater than 0")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than 0")
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be greater than 0")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")

        self.max_model_requests = max_model_requests
        self.max_tool_calls = max_tool_calls
        self.max_tool_rounds = max_tool_rounds
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self.cancellation_token = (
            CancellationToken()
            if cancellation_token is None
            else cancellation_token
        )
        self.retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self.model_requests = 0
        self.tool_calls = 0
        self.tool_rounds = 0
        self._deadline: float | None = None

    def start_run(self) -> None:
        """开始一次新运行，并清空本次运行的消耗计数。"""

        self.model_requests = 0
        self.tool_calls = 0
        self.tool_rounds = 0
        # 保存绝对截止点，而不是每次检查时重新累计时长。
        # 这样所有阶段共享同一份总运行时间预算。
        self._deadline = (
            None
            if self.timeout_seconds is None
            else self._clock() + self.timeout_seconds
        )

    def check_deadline(self) -> HarnessDecision:
        """检查总运行时间是否已经到期，不会消耗请求或工具预算。"""

        if self._deadline is not None and self._clock() >= self._deadline:
            return HarnessDecision(False, "runtime_timeout_exceeded")
        return HarnessDecision(True)

    def check_cancelled(self) -> HarnessDecision:
        """检查协作式取消信号，不会消耗任何预算。"""

        if self.cancellation_token.is_cancelled:
            return HarnessDecision(False, "run_cancelled")
        return HarnessDecision(True)

    def check_runtime(self) -> HarnessDecision:
        """按取消优先、截止时间其次的顺序检查运行是否还能继续。"""

        cancellation_decision = self.check_cancelled()
        if not cancellation_decision.allowed:
            return cancellation_decision
        return self.check_deadline()

    def cancel(self) -> None:
        """请求当前运行在下一个安全边界停止。"""

        self.cancellation_token.cancel()

    def consume_model_request(self) -> HarnessDecision:
        """为一次模型尝试预留预算；每次重试也会单独预留一次。"""

        deadline_decision = self.check_runtime()
        if not deadline_decision.allowed:
            return deadline_decision

        if self.model_requests >= self.max_model_requests:
            return HarnessDecision(False, "max_model_requests_exceeded")
        self.model_requests += 1
        return HarnessDecision(True)

    def execute_model(
        self,
        operation: Callable[[], T],
        on_attempt: Callable[[int], None] | None = None,
        on_retry: Callable[[int, RetryableModelProviderError, float], None]
        | None = None,
        on_exhausted: Callable[[int, RetryableModelProviderError], None]
        | None = None,
    ) -> T:
        """执行一次模型操作，并按策略重试可恢复的供应商错误。

        每次循环先预留一次请求预算，再记录 attempt，最后才进入 Provider。
        这样预算、事件轨迹和实际调用顺序一致；取消可能在事件回调期间到达，
        因此回调之后还要检查一次运行状态。
        """

        retry_count = 0
        while True:
            request_decision = self.consume_model_request()
            if not request_decision.allowed:
                raise HarnessStop(request_decision.reason or "runtime_budget_exceeded")
            attempt_number = retry_count + 1
            if on_attempt is not None:
                on_attempt(attempt_number)

            # 记录/持久化 attempt 事件本身可能耗时；这段间隙也可能收到取消。
            # 在真正进入 Provider 前再查一次，避免已取消的请求继续发往外部。
            attempt_decision = self.check_runtime()
            if not attempt_decision.allowed:
                raise HarnessStop(attempt_decision.reason or "runtime_stopped")

            try:
                result = operation()
            except RetryableModelProviderError as error:
                # 最后一次调用也可能在返回错误前遇到取消或截止时间。
                # 先判断运行状态，再判断是否达到重试上限，才能报告真正的
                # 停止原因，而不是把已取消的运行误报成 retry exhausted。
                failed_request_runtime = self.check_runtime()
                if not failed_request_runtime.allowed:
                    raise HarnessStop(
                        failed_request_runtime.reason or "runtime_stopped"
                    )

                if retry_count >= self.retry_policy.max_retries:
                    if on_exhausted is not None:
                        on_exhausted(attempt_number, error)
                    raise HarnessStop("harness_retry_exhausted")

                # 重试本身也要占一份模型请求预算。若预算已被刚才失败的请求
                # 用尽，就直接停止，不先记录“已安排重试”或做无意义的退避等待。
                if self.model_requests >= self.max_model_requests:
                    raise HarnessStop("max_model_requests_exceeded")

                retry_count += 1

                before_sleep = self.check_runtime()
                if not before_sleep.allowed:
                    raise HarnessStop(before_sleep.reason or "runtime_stopped")
                delay = self.retry_policy.backoff_seconds * (2 ** (retry_count - 1))
                if on_retry is not None:
                    on_retry(retry_count, error, delay)

                # 重试事件可能需要持久化；回调期间若收到取消或跨过截止点，
                # 不应再进入退避等待。默认 Event.wait 能被取消唤醒，但注入的
                # sleep 可能是不可中断的，因此这里必须在 sleep 前主动检查。
                before_wait = self.check_runtime()
                if not before_wait.allowed:
                    raise HarnessStop(before_wait.reason or "runtime_stopped")

                if self.retry_policy.sleep is None:
                    self.cancellation_token.wait(delay)
                else:
                    self.retry_policy.sleep(delay)
                after_sleep = self.check_runtime()
                if not after_sleep.allowed:
                    raise HarnessStop(after_sleep.reason or "runtime_stopped")
            else:
                after_request = self.check_runtime()
                if not after_request.allowed:
                    raise HarnessStop(after_request.reason or "runtime_stopped")
                return result

    def consume_tool_call(self) -> HarnessDecision:
        """尝试消耗一次工具调用预算。"""

        deadline_decision = self.check_runtime()
        if not deadline_decision.allowed:
            return deadline_decision

        if self.tool_calls >= self.max_tool_calls:
            return HarnessDecision(False, "max_tool_calls_exceeded")
        self.tool_calls += 1
        return HarnessDecision(True)

    def consume_tool_round(self) -> HarnessDecision:
        """尝试消耗一次模型发起的工具往返预算。"""

        runtime_decision = self.check_runtime()
        if not runtime_decision.allowed:
            return runtime_decision

        if self.tool_rounds >= self.max_tool_rounds:
            return HarnessDecision(False, "max_tool_rounds_exceeded")
        self.tool_rounds += 1
        return HarnessDecision(True)
