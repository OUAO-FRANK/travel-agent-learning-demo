"""Agent Runtime Harness 的当前层：运行预算与截止时间。

Harness 不决定旅行计划，也不决定调用哪个工具。它只负责回答一个运行时问题：
“本次运行还允许继续消耗多少模型请求和工具调用、还能运行多久？”
把这个策略独立出来后，后续可以在不重写 Agent 决策逻辑的情况下加入取消、重试和
成本预算等运行控制。
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class HarnessDecision:
    """一次预算检查的结果。"""

    allowed: bool
    reason: str | None = None


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
    ) -> None:
        if max_model_requests <= 0:
            raise ValueError("max_model_requests must be greater than 0")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than 0")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")

        self.max_model_requests = max_model_requests
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self.model_requests = 0
        self.tool_calls = 0
        self._deadline: float | None = None

    def start_run(self) -> None:
        """开始一次新运行，并清空本次运行的消耗计数。"""

        self.model_requests = 0
        self.tool_calls = 0
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

    def consume_model_request(self) -> HarnessDecision:
        """尝试消耗一次模型请求预算。"""

        deadline_decision = self.check_deadline()
        if not deadline_decision.allowed:
            return deadline_decision

        if self.model_requests >= self.max_model_requests:
            return HarnessDecision(False, "max_model_requests_exceeded")
        self.model_requests += 1
        return HarnessDecision(True)

    def consume_tool_call(self) -> HarnessDecision:
        """尝试消耗一次工具调用预算。"""

        deadline_decision = self.check_deadline()
        if not deadline_decision.allowed:
            return deadline_decision

        if self.tool_calls >= self.max_tool_calls:
            return HarnessDecision(False, "max_tool_calls_exceeded")
        self.tool_calls += 1
        return HarnessDecision(True)
