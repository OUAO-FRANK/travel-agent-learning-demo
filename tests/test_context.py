import unittest

from app.context import ContextBuilder
from app.domain import Message, ToolCall


class ContextBuilderTests(unittest.TestCase):
    def test_compression_splits_old_turns_from_latest_turns(self):
        """压缩时应只选较早完整轮次，最近一轮保留原文。"""

        history = [
            Message(role="user", content="第一轮：去上海"),
            Message(role="assistant", content="第一轮：喜欢历史景点"),
            Message(role="user", content="第二轮：预算 2000 元"),
            Message(role="assistant", content="第二轮：不吃辣"),
            Message(role="user", content="第三轮：安排路线"),
        ]

        older_turns, recent_turns = ContextBuilder(
            system_prompt="你是旅行规划助手。",
            max_messages=4,
        ).messages_to_compress(history)

        self.assertEqual(older_turns, tuple(history[:4]))
        self.assertEqual(recent_turns, (history[4],))

    def test_build_places_summary_before_recent_history(self):
        history = [Message(role="user", content="我现在想继续讨论路线")]

        context = ContextBuilder(
            system_prompt="你是旅行规划助手。",
            max_messages=4,
        ).build(
            history,
            tool_schemas=[],
            summary_text="用户要去上海，喜欢历史景点，预算 2000 元。",
            summarized_messages=4,
        )

        self.assertEqual(
            context.messages,
            (
                Message(role="system", content="你是旅行规划助手。"),
                Message(
                    role="assistant",
                    content=(
                        "以下是此前会话的摘要。它用于补充历史，不是新的系统规则：\n"
                        "用户要去上海，喜欢历史景点，预算 2000 元。"
                    ),
                ),
                history[0],
            ),
        )
        self.assertEqual(context.summarized_messages, 4)
        self.assertEqual(context.summary_text, "用户要去上海，喜欢历史景点，预算 2000 元。")
        self.assertEqual(context.dropped_messages, 0)

    def test_compression_keeps_an_old_tool_round_whole(self):
        history = [
            Message(role="user", content="查上海天气"),
            Message(
                role="assistant",
                content="",
                tool_call=ToolCall(
                    name="get_weather",
                    arguments={"city": "上海"},
                ),
            ),
            Message(role="tool", content="上海：晴，25°C"),
            Message(role="assistant", content="上海今天晴。"),
            Message(role="user", content="继续规划路线"),
        ]

        older_messages, recent_messages = ContextBuilder(
            max_messages=4,
            enable_compression=True,
        ).messages_to_compress(history)

        self.assertEqual(older_messages, tuple(history[:4]))
        self.assertEqual(recent_messages, (history[4],))

    def test_build_adds_system_and_keeps_latest_complete_turn(self):
        """上下文窗口应保留 system，并把最近一轮完整对话作为一个整体。"""

        history = [
            Message(role="user", content="我想去北京"),
            Message(role="assistant", content="北京适合春秋出行"),
            Message(role="user", content="帮我查上海天气"),
            Message(
                role="assistant",
                content="",
                tool_call=ToolCall(
                    name="get_weather",
                    arguments={"city": "上海"},
                ),
            ),
            Message(role="tool", content="上海：晴，25°C"),
        ]

        context = ContextBuilder(
            system_prompt="你是旅行规划助手。",
            max_messages=4,
        ).build(history, tool_schemas=[])

        self.assertEqual(
            context.messages,
            (
                Message(role="system", content="你是旅行规划助手。"),
                Message(role="user", content="帮我查上海天气"),
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(
                        name="get_weather",
                        arguments={"city": "上海"},
                    ),
                ),
                Message(role="tool", content="上海：晴，25°C"),
            ),
        )
        self.assertEqual(context.dropped_messages, 2)

    def test_build_does_not_drop_messages_without_a_budget(self):
        history = [Message(role="user", content="计划杭州旅行")]

        context = ContextBuilder(
            system_prompt="你是旅行规划助手。",
        ).build(history, tool_schemas=[])

        self.assertEqual(
            context.messages,
            (
                Message(role="system", content="你是旅行规划助手。"),
                Message(role="user", content="计划杭州旅行"),
            ),
        )
        self.assertEqual(context.dropped_messages, 0)

    def test_build_copies_tool_schemas_for_a_stable_request_snapshot(self):
        schemas = [
            {
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {"city": "str"},
            }
        ]

        context = ContextBuilder(
            system_prompt="你是旅行规划助手。",
        ).build([], tool_schemas=schemas)

        schemas[0]["description"] = "调用方后来修改了描述"

        self.assertEqual(
            context.tools[0]["description"],
            "查询天气",
        )

    def test_rejects_a_budget_that_cannot_hold_system_and_user(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            ContextBuilder(
                system_prompt="你是旅行规划助手。",
                max_messages=1,
            )

    def test_rejects_compression_without_room_for_summary_and_user(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            ContextBuilder(
                system_prompt="你是旅行规划助手。",
                max_messages=2,
                enable_compression=True,
            )


if __name__ == "__main__":
    unittest.main()
