import unittest

from app.context import ContextBuilder
from app.domain import Message, ToolCall


class ContextBuilderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
