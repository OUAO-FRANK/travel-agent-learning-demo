"""Agent Runtime Harness 的第一层：运行预算。

Harness 不决定旅行计划，也不决定调用哪个工具。它只负责回答一个运行时问题：
“本次运行还允许继续消耗多少模型请求和工具调用？”
把这个策略独立出来后，后续可以在不重写 Agent 决策逻辑的情况下加入超时、取消、
重试和成本预算。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessDecision:
    """一次预算检查的结果。"""

    allowed: bool
    reason: str | None = None


class RuntimeHarness:
    """记录一次 Agent 运行的模型请求数和工具调用数。"""

    def __init__(
        self,
        max_model_requests: int = 8,
        max_tool_calls: int = 8,
    ) -> None:
        if max_model_requests <= 0:
            raise ValueError("max_model_requests must be greater than 0")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than 0")

        self.max_model_requests = max_model_requests
        self.max_tool_calls = max_tool_calls
        self.model_requests = 0
        self.tool_calls = 0

    def start_run(self) -> None:
        """开始一次新运行，并清空本次运行的消耗计数。"""

        self.model_requests = 0
        self.tool_calls = 0

    def consume_model_request(self) -> HarnessDecision:
        """尝试消耗一次模型请求预算。"""

        if self.model_requests >= self.max_model_requests:
            return HarnessDecision(False, "max_model_requests_exceeded")
        self.model_requests += 1
        return HarnessDecision(True)

    def consume_tool_call(self) -> HarnessDecision:
        """尝试消耗一次工具调用预算。"""

        if self.tool_calls >= self.max_tool_calls:
            return HarnessDecision(False, "max_tool_calls_exceeded")
        self.tool_calls += 1
        return HarnessDecision(True)
