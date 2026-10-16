"""多工具调用的最小串行队列。

本阶段只解决一个问题：当模型一次提出多个互相独立的工具调用时，Runtime
如何按模型给出的顺序逐个执行，并保留每个调用的结果。并发、依赖关系、取消
和重试属于后续 Harness 课程，不在这里提前混入。
"""

from dataclasses import dataclass
from collections.abc import Callable, Iterable

from app.domain import ToolCall


@dataclass(frozen=True)
class ToolExecution:
    """队列中一次已经完成的工具执行。"""

    call: ToolCall
    result: str


class ToolQueue:
    """按输入顺序串行执行工具调用。"""

    def execute(
        self,
        calls: Iterable[ToolCall],
        dispatch: Callable[[ToolCall], str],
    ) -> tuple[ToolExecution, ...]:
        """执行所有调用；第一个异常会停止队列并继续向上抛出。"""

        executions: list[ToolExecution] = []
        for call in calls:
            # 不捕获异常：本阶段采用 fail-fast 策略，避免把后续结果伪装成
            # 完整批次。后续再单独增加 continue-on-error 和重试策略。
            result = dispatch(call)
            executions.append(ToolExecution(call=call, result=result))
        return tuple(executions)
