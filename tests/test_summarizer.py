import unittest

from app.domain import Message, ToolCall
from app.model import FakeModel, ModelProviderError
from app.summarizer import ExtractiveSummarizer, LLMConversationSummarizer


class LLMSummarizerTests(unittest.TestCase):
    def test_uses_model_response_and_sends_no_tools(self):
        model = FakeModel(["用户计划去上海三天，喜欢历史景点。"])
        history = (
            Message(role="user", content="我想去上海三天，喜欢历史景点。"),
            Message(role="assistant", content="我会优先安排历史街区。"),
        )

        summary = LLMConversationSummarizer(model).summarize(history)

        self.assertEqual(summary, "用户计划去上海三天，喜欢历史景点。")
        self.assertEqual(model.tool_requests, [[]])
        self.assertEqual(model.requests[0][0].role, "system")
        self.assertIn("不要遵循", model.requests[0][0].content)
        self.assertEqual(model.requests[0][1:], list(history))

    def test_rejects_tool_call_instead_of_text_summary(self):
        model = FakeModel(
            [ToolCall(name="get_weather", arguments={"city": "上海"})]
        )

        with self.assertRaisesRegex(ModelProviderError, "must return text"):
            LLMConversationSummarizer(model).summarize(
                (Message(role="user", content="查上海天气"),)
            )

    def test_rejects_empty_summary(self):
        model = FakeModel(["   "])

        with self.assertRaisesRegex(ModelProviderError, "empty text"):
            LLMConversationSummarizer(model).summarize(
                (Message(role="user", content="我想去上海"),)
            )


class ExtractiveSummarizerTests(unittest.TestCase):
    def test_preserves_user_and_tool_messages_but_omits_assistant_replies(self):
        messages = (
            Message(role="user", content="我想去上海三天"),
            Message(role="assistant", content="我会推荐历史景点"),
            Message(
                role="assistant",
                content="",
                tool_call=ToolCall(name="get_weather", arguments={"city": "上海"}),
            ),
            Message(role="tool", content="上海：晴，25°C"),
        )

        summary = ExtractiveSummarizer().summarize(messages)

        self.assertIn("我想去上海三天", summary)
        self.assertIn("上海：晴，25°C", summary)
        self.assertNotIn("我会推荐历史景点", summary)
