"""模型适配层的教学用实现。

真实 Agent 通常不会直接在 Runtime 里写死某家模型供应商的 HTTP 调用，
而是把“给模型一组消息，得到一个回复”抽象成一个小接口。本阶段先用
FakeModel 实现这个接口，等 Agent Loop 逻辑稳定后，再替换成真实模型适配器。

FakeModel 的价值不是模拟真实模型的智能程度，而是提供：

* 确定性：同样的预设回复会产生同样的结果；
* 可观察性：可以检查 Runtime 实际发给模型的消息；
* 可测试性：不需要网络、API Key 或模型服务。
"""

from app.domain import Message, ToolCall


class FakeModel:
    """按照预设顺序返回文本的模型替身。

    参数：
        replies：本次 FakeModel 运行允许返回的文本列表。

    例如：

    ``FakeModel(["第一轮回答", "第二轮回答"])``

    第一次调用 ``complete`` 返回第一条文本，第二次调用返回第二条文本。
    这正好可以模拟后续 Agent Loop 中的多轮模型交互。

    重要边界：
        回复列表耗尽后会抛出 ``RuntimeError``，而不是重复最后一条回复。
        如果自动重复结果，测试可能看起来通过，但实际上 Agent 可能已经
        多调用了一轮模型；明确失败更容易暴露控制流错误。
    """

    def __init__(self, replies: list[str | ToolCall]) -> None:
        # list(replies) 创建一份独立列表。
        # 这样调用方后来修改自己的 replies 列表，不会改变 FakeModel 的脚本。
        self._replies = list(replies)

        # 这个游标表示下一次 complete 应该消费哪一条预设回复。
        # 它是 FakeModel 的运行状态，而不是模型消息的一部分。
        self._next_reply_index = 0

        # requests 用于观测 Runtime 发给模型的每一次消息列表。
        # 外部测试可以据此检查 system/user 消息是否正确组装。
        self.requests: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str | ToolCall:
        """记录一次模型请求，并返回下一条预设文本。

        输入：
            messages：本次发送给模型的消息列表。

        输出：
            一条预设的文本回复或一个工具调用请求。

        注意：这里保存的是 ``list(messages)``，不是原始 ``messages`` 引用。
        如果直接保存原始引用，调用方之后对消息列表的修改会回头改变历史，
        导致调试轨迹和实际请求不一致。
        """

        # 只复制列表容器，不复制 Message 对象。
        # 当前 Message 使用 frozen=True，字段不能被重新绑定，因此浅拷贝
        # 已经足以保护这层“请求列表结构”。后续若消息内部包含可变内容块，
        # 再根据需要引入深拷贝或不可变内容结构。
        self.requests.append(list(messages))

        if self._next_reply_index >= len(self._replies):
            # 不允许静默重用最后一条回复，因为那会隐藏多调用模型的 bug。
            raise RuntimeError("FakeModel has no remaining replies")

        reply = self._replies[self._next_reply_index]
        self._next_reply_index += 1
        return reply
