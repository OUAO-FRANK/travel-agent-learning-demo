import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

from app.agent import TravelAgent
from app.domain import Message, ToolCall, ToolCallBatch
from app.model import (
    ModelProviderError,
    OpenAICompatibleModel,
    RetryableModelProviderError,
    _urllib_transport,
)


class OpenAICompatibleModelTests(unittest.TestCase):
    def test_http_transient_errors_are_marked_retryable(self):
        for status in (408, 429, 500, 503):
            error = HTTPError(
                "https://example.test/chat/completions",
                status,
                "temporary",
                {},
                BytesIO(b"{}"),
            )
            with self.subTest(status=status), patch(
                "app.model.urlopen",
                side_effect=error,
            ):
                with self.assertRaises(RetryableModelProviderError):
                    _urllib_transport("https://example.test", {}, {})

    def test_http_client_errors_are_not_marked_retryable(self):
        error = HTTPError(
            "https://example.test/chat/completions",
            400,
            "bad request",
            {},
            BytesIO(b"{}"),
        )

        with patch("app.model.urlopen", side_effect=error):
            with self.assertRaises(ModelProviderError) as context:
                _urllib_transport("https://example.test", {}, {})

        self.assertNotIsInstance(context.exception, RetryableModelProviderError)

    def test_http_response_is_closed_when_json_parsing_fails(self):
        class Response:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.closed = True

            def read(self):
                return b"not-json"

        response = Response()
        with patch("app.model.urlopen", return_value=response):
            with self.assertRaises(ModelProviderError):
                _urllib_transport("https://example.test", {}, {})

        self.assertTrue(response.closed)

    def test_http_error_body_is_closed_when_status_is_not_successful(self):
        class TrackedHTTPError(HTTPError):
            closed = False

            def close(self):
                self.closed = True
                super().close()

        error = TrackedHTTPError(
            "https://example.test/chat/completions",
            503,
            "service unavailable",
            {},
            BytesIO(b"temporary error"),
        )

        with patch("app.model.urlopen", side_effect=error):
            with self.assertRaises(RetryableModelProviderError):
                _urllib_transport("https://example.test", {}, {})

        self.assertTrue(error.closed)

    def test_complete_builds_request_and_parses_text_response(self):
        captured: dict[str, object] = {}

        def transport(endpoint, headers, payload):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["payload"] = payload
            return {
                "choices": [
                    {"message": {"content": "上海适合安排三天行程。"}}
                ]
            }

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        answer = provider.complete(
            [
                Message(role="system", content="你是旅行助手。"),
                Message(role="user", content="我想去上海三天。"),
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {"city": "str"},
                }
            ],
        )

        self.assertEqual(answer, "上海适合安排三天行程。")
        self.assertEqual(captured["endpoint"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured["headers"], {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        })
        self.assertEqual(
            captured["payload"],
            {
                "model": "demo-model",
                "messages": [
                    {"role": "system", "content": "你是旅行助手。"},
                    {"role": "user", "content": "我想去上海三天。"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "查询天气",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
        )

    def test_complete_parses_tool_call_response(self):
        def transport(endpoint, headers, payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "先查询上海天气。",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "上海"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        result = provider.complete(
            [Message(role="user", content="查天气")],
            tools=[],
        )

        self.assertEqual(
            result,
            ToolCall(
                name="get_weather",
                arguments={
                    "city": "上海",
                },
                reasoning_content="先查询上海天气。",
            ),
        )

    def test_complete_parses_multiple_tool_calls_as_a_batch(self):
        def transport(endpoint, headers, payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "上海"}',
                                    }
                                },
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "北京"}',
                                    }
                                },
                            ],
                        }
                    }
                ]
            }

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        self.assertEqual(
            provider.complete([Message(role="user", content="查两个城市")], tools=[]),
            ToolCallBatch(
                calls=(
                    ToolCall(name="get_weather", arguments={"city": "上海"}),
                    ToolCall(name="get_weather", arguments={"city": "北京"}),
                )
            ),
        )

    def test_complete_converts_tool_round_messages(self):
        captured: dict[str, object] = {}

        def transport(endpoint, headers, payload):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "天气已查询。"}}]}

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        provider.complete(
            [
                Message(role="user", content="查上海天气"),
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(
                        name="get_weather",
                        arguments={"city": "上海"},
                        reasoning_content="先查询上海天气。",
                    ),
                ),
                Message(role="tool", content="上海：晴，25°C"),
            ],
            tools=[],
        )

        self.assertEqual(
            captured["payload"]["messages"],
            [
                {"role": "user", "content": "查上海天气"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "先查询上海天气。",
                    "tool_calls": [
                        {
                            "id": "call-0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "上海"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "上海：晴，25°C",
                    "tool_call_id": "call-0",
                },
            ],
        )

    def test_complete_rejects_malformed_provider_response(self):
        def transport(endpoint, headers, payload):
            return {"choices": []}

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        with self.assertRaisesRegex(ModelProviderError, "missing choices"):
            provider.complete([Message(role="user", content="你好")])

    def test_provider_can_run_the_existing_agent_tool_loop(self):
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "上海"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "上海今天晴。"}}]},
        ]

        def transport(endpoint, headers, payload):
            return responses.pop(0)

        provider = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="demo-model",
            transport=transport,
        )

        result = TravelAgent(provider).run("上海天气怎么样？")

        self.assertEqual(result.answer, "上海今天晴。")
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "tool_requested",
                "tool_result",
                "model_request",
                "model_response",
                "run_finished",
            ],
        )


if __name__ == "__main__":
    unittest.main()
