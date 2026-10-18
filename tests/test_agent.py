import unittest
import tempfile
from pathlib import Path

from app.domain import AgentEvent, Message, ToolCall, ToolCallBatch
from app.agent import TravelAgent
from app.context import ContextBuilder
from app.model import FakeModel
from app.memory import SQLiteMemoryStore
from app.harness import RuntimeHarness
from app.rag import KeywordRetriever
from app.summarizer import LLMConversationSummarizer
from app.tools import ToolDefinition, ToolRegistry
from app.storage import SQLiteSessionStore


class DomainTests(unittest.TestCase):
    def test_message_keeps_role_and_content(self):
        message = Message(role="user", content="计划上海三日游")

        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "计划上海三日游")

    def test_event_keeps_kind_and_payload(self):
        event = AgentEvent(kind="user_input", payload={"content": "你好"})

        self.assertEqual(event.kind, "user_input")
        self.assertEqual(event.payload, {"content": "你好"})

    def test_tool_call_keeps_name_and_arguments(self):
        call = ToolCall(name="get_weather", arguments={"city": "上海"})

        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"city": "上海"})


class FakeModelTests(unittest.TestCase):
    def test_complete_returns_replies_in_order_and_keeps_request_snapshots(self):
        model = FakeModel(["第一版计划", "补充预算说明"])
        first_request = [Message(role="user", content="计划上海三日游")]
        second_request = [
            *first_request,
            Message(role="assistant", content="第一版计划"),
        ]

        first_answer = model.complete(first_request)
        second_answer = model.complete(second_request)

        self.assertEqual(first_answer, "第一版计划")
        self.assertEqual(second_answer, "补充预算说明")
        self.assertEqual(model.requests, [first_request, second_request])

        # 调用方后来修改自己的列表，不应该改写 FakeModel 已经记录的请求。
        first_request.append(Message(role="user", content="后来追加的问题"))
        self.assertEqual(len(model.requests[0]), 1)

    def test_complete_raises_when_replies_are_exhausted(self):
        model = FakeModel(["唯一回复"])
        request = [Message(role="user", content="你好")]

        model.complete(request)

        with self.assertRaisesRegex(RuntimeError, "no remaining replies"):
            model.complete(request)


class TravelAgentTests(unittest.TestCase):
    def test_harness_stops_after_a_model_call_returns_past_the_deadline(self):
        now = [0.0]

        class SlowModel:
            def complete(self, messages, tools=()):
                now[0] = 6.0
                return "这份回答已经超时"

        agent = TravelAgent(
            SlowModel(),
            harness=RuntimeHarness(
                timeout_seconds=5,
                clock=lambda: now[0],
            ),
        )

        result = agent.run("规划旅行")

        self.assertEqual(result.status, "stopped")
        self.assertIn("runtime_timeout_exceeded", result.answer)
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "harness_timeout_reached",
                "run_finished",
            ],
        )

    def test_harness_records_a_tool_result_then_stops_if_tool_exceeded_deadline(self):
        now = [0.0]

        def slow_weather_tool(arguments):
            now[0] = 6.0
            return f"{arguments['city']}晴朗"

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="get_weather",
                description="查询城市天气",
                schema={"city": str},
                handler=slow_weather_tool,
            )
        )
        model = FakeModel(
            [
                ToolCall(name="get_weather", arguments={"city": "上海"}),
                "不会被请求到的最终答案",
            ]
        )
        agent = TravelAgent(
            model,
            registry=registry,
            harness=RuntimeHarness(
                timeout_seconds=5,
                clock=lambda: now[0],
            ),
        )

        result = agent.run("查上海天气")

        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "tool_requested",
                "tool_result",
                "harness_timeout_reached",
                "run_finished",
            ],
        )

    def test_harness_stops_before_a_second_model_request_when_budget_is_exhausted(self):
        model = FakeModel(
            [
                ToolCall(name="get_weather", arguments={"city": "上海"}),
                "不会被请求到的最终答案",
            ]
        )
        agent = TravelAgent(
            model,
            harness=RuntimeHarness(max_model_requests=1),
        )

        result = agent.run("查上海天气")

        self.assertEqual(result.status, "stopped")
        self.assertIn("max_model_requests_exceeded", result.answer)
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "tool_requested",
                "tool_result",
                "harness_limit_reached",
                "run_finished",
            ],
        )

    def test_harness_stops_a_tool_batch_before_exceeding_tool_budget(self):
        model = FakeModel(
            [
                ToolCallBatch(
                    calls=(
                        ToolCall(name="get_weather", arguments={"city": "上海"}),
                        ToolCall(name="get_weather", arguments={"city": "北京"}),
                    )
                ),
                "不会被请求到的最终答案",
            ]
        )
        agent = TravelAgent(
            model,
            harness=RuntimeHarness(max_tool_calls=1),
        )

        result = agent.run("同时查上海和北京天气")

        self.assertEqual(result.status, "stopped")
        self.assertIn("max_tool_calls_exceeded", result.answer)
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "tool_queue_started",
                "tool_requested",
                "tool_result",
                "harness_limit_reached",
                "run_finished",
            ],
        )
    def test_run_executes_a_tool_batch_in_order_before_next_model_request(self):
        model = FakeModel(
            [
                ToolCallBatch(
                    calls=(
                        ToolCall(name="get_weather", arguments={"city": "上海"}),
                        ToolCall(name="get_weather", arguments={"city": "北京"}),
                    )
                ),
                "上海晴朗，北京多云。",
            ]
        )

        result = TravelAgent(model).run("同时查上海和北京天气")

        self.assertEqual(result.answer, "上海晴朗，北京多云。")
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "model_request",
                "tool_queue_started",
                "tool_requested",
                "tool_result",
                "tool_requested",
                "tool_result",
                "tool_queue_finished",
                "model_request",
                "model_response",
                "run_finished",
            ],
        )
        self.assertEqual(
            [message.content for message in model.requests[1][-2:]],
            ["上海：晴，25°C", "北京：多云，20°C"],
        )
        self.assertEqual(
            len(model.requests[1][-3].tool_calls or ()),
            2,
        )
    def test_run_writes_explicit_memory_and_recalls_it_for_later_request(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_store = SQLiteMemoryStore(Path(directory) / "travel-agent.db")
            first_model = FakeModel(["好的，我记住了。"])
            first_agent = TravelAgent(
                first_model,
                memory_store=memory_store,
                memory_user_id="alice",
            )

            first_result = first_agent.run("请记住我喜欢历史景点")

            second_model = FakeModel(["我会优先安排历史景点。"])
            second_agent = TravelAgent(
                second_model,
                memory_store=memory_store,
                memory_user_id="alice",
            )
            result = second_agent.run("帮我规划上海历史路线")

            self.assertIn("memory_written", [event.kind for event in first_result.events])
            self.assertIn("memory_search_completed", [event.kind for event in result.events])
            self.assertTrue(
                any(
                    message.role == "assistant"
                    and "[memory:1] 我喜欢历史景点" in message.content
                    for message in second_model.requests[0]
                )
            )

    def test_run_retrieves_evidence_and_injects_it_into_model_context(self):
        model = FakeModel(["根据资料，上海外滩适合安排在晚上。"])
        retriever = KeywordRetriever.from_text(
            "上海外滩适合夜景，建议晚上前往。\n\n杭州西湖适合散步。",
            source="travel-notes.md",
        )

        result = TravelAgent(model, retriever=retriever).run("上海夜景怎么安排")

        self.assertEqual(result.answer, "根据资料，上海外滩适合安排在晚上。")
        self.assertEqual(
            [event.kind for event in result.events],
            [
                "user_input",
                "rag_search_started",
                "rag_search_completed",
                "model_request",
                "model_response",
                "run_finished",
            ],
        )
        self.assertEqual(
            [message.role for message in model.requests[0]],
            ["system", "assistant", "user"],
        )
        self.assertIn("travel-notes.md:0", model.requests[0][1].content)
        self.assertIn("上海外滩适合夜景", model.requests[0][1].content)

    def test_run_summarizes_old_turns_and_keeps_raw_database_messages(self):
        """Agent 压缩模型上下文，但不删除 SQLite 中的原始消息。"""

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")

            # 两轮旧历史先照常写进数据库；这两次运行尚未启用消息预算。
            for user_text, answer in [
                (
                    "我计划下周去上海玩三天，特别喜欢历史景点，尽量不要安排太早的行程。",
                    "收到，我会安排一个轻松的路线。",
                ),
                (
                    "整个旅行预算控制在 2000 元以内，我不吃辣，优先选择地铁方便的路线。",
                    "记下预算和饮食偏好。",
                ),
            ]:
                TravelAgent(FakeModel([answer]), session_store=store).run(
                    user_text,
                    session_id="trip-1",
                )

            model = FakeModel(
                [
                    "这是加入历史摘要后的路线建议。",
                ]
            )
            summary_model = FakeModel(
                [
                    "用户计划下周去上海三天，喜欢历史景点，预算 2000 元；不吃辣，偏好地铁方便的路线。",
                ]
            )
            agent = TravelAgent(
                model,
                context_builder=ContextBuilder(
                    max_messages=4,
                    enable_compression=True,
                ),
                session_store=store,
                summarizer=LLMConversationSummarizer(summary_model),
            )

            result = agent.run("请把路线按天安排", session_id="trip-1")

            self.assertEqual(
                [event.kind for event in result.events],
                [
                    "user_input",
                    "context_compression_started",
                    "context_summary_requested",
                    "context_compressed",
                    "model_request",
                    "model_response",
                    "run_finished",
                ],
            )
            self.assertEqual(len(model.requests), 1)
            self.assertEqual(len(summary_model.requests), 1)
            self.assertEqual(
                [message.role for message in summary_model.requests[0]][0],
                "system",
            )
            self.assertEqual(summary_model.tool_requests, [[]])
            self.assertEqual(
                [message.role for message in model.requests[0]],
                ["system", "assistant", "user"],
            )
            self.assertIn("喜欢历史景点", model.requests[0][1].content)
            self.assertIn("2000 元", model.requests[0][1].content)
            self.assertEqual(
                len(store.load_messages("trip-1")),
                6,
                "SQLite 仍保存原始对话，不保存被压缩后的替代内容",
            )

    def test_tool_round_can_trigger_compression_before_second_model_request(self):
        """工具结果增加消息后，第二次模型请求也应重新检查预算。"""

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")
            TravelAgent(
                FakeModel(["上一轮已记录上海行程。"]),
                session_store=store,
            ).run("我准备去上海", session_id="trip-1")

            model = FakeModel(
                [
                    ToolCall(name="get_weather", arguments={"city": "上海"}),
                    "今天适合游览历史街区。",
                ]
            )
            summary_model = FakeModel(["用户准备去上海；工具结果显示上海晴，25°C。"])
            agent = TravelAgent(
                model,
                context_builder=ContextBuilder(
                    max_messages=4,
                    enable_compression=True,
                ),
                session_store=store,
                summarizer=LLMConversationSummarizer(summary_model),
            )

            result = agent.run("请查上海天气", session_id="trip-1")

            self.assertEqual(
                [event.kind for event in result.events],
                [
                    "user_input",
                    "model_request",
                    "tool_requested",
                    "tool_result",
                    "context_compression_started",
                    "context_summary_requested",
                    "context_compressed",
                    "model_request",
                    "model_response",
                    "run_finished",
                ],
            )
            self.assertEqual(
                [message.role for message in model.requests[1]],
                ["system", "assistant", "user", "assistant", "tool"],
            )
            self.assertIn("准备去上海", model.requests[1][1].content)
            self.assertEqual(len(summary_model.requests), 1)
            self.assertEqual(model.requests[1][-1].content, "上海：晴，25°C")

    def test_run_can_resume_messages_from_a_persisted_session(self):
        """第二个 Agent 进程应看到第一个 Agent 已保存的对话历史。"""

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")

            first_model = FakeModel(["第一轮计划已经完成。"])
            TravelAgent(first_model, session_store=store).run(
                "我想去上海",
                session_id="trip-1",
            )

            second_model = FakeModel(["第二轮补充了预算。"])
            result = TravelAgent(second_model, session_store=store).run(
                "预算是 2000 元",
                session_id="trip-1",
            )

            self.assertEqual(result.answer, "第二轮补充了预算。")
            self.assertEqual(
                second_model.requests[0],
                [
                    Message(role="system", content="你是旅行规划助手。"),
                    Message(role="user", content="我想去上海"),
                    Message(role="assistant", content="第一轮计划已经完成。"),
                    Message(role="user", content="预算是 2000 元"),
                ],
            )

    def test_repeated_persisted_tool_runs_get_distinct_call_ids(self):
        """同一会话的多次工具调用不能复用同一个数据库主键。"""

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")

            first_model = FakeModel(
                [
                    ToolCall(name="get_weather", arguments={"city": "上海"}),
                    "第一轮完成",
                ]
            )
            TravelAgent(first_model, session_store=store).run(
                "查上海天气",
                session_id="trip-1",
            )

            second_model = FakeModel(
                [
                    ToolCall(name="get_weather", arguments={"city": "北京"}),
                    "第二轮完成",
                ]
            )
            TravelAgent(second_model, session_store=store).run(
                "再查北京天气",
                session_id="trip-1",
            )

            self.assertEqual(
                [call["call_id"] for call in store.load_tool_calls("trip-1")],
                ["call-1", "call-2"],
            )

    def test_run_uses_context_builder_for_model_request_metadata(self):
        """Agent 事件应记录实际上下文窗口，而不是内部完整历史。"""

        model = FakeModel(["这是一个上海三日游计划。"])
        agent = TravelAgent(
            model,
            context_builder=ContextBuilder(
                system_prompt="你是旅行规划助手。",
                max_messages=2,
            ),
        )

        result = agent.run("计划上海三日游")

        request_event = result.events[1]
        self.assertEqual(request_event.kind, "model_request")
        self.assertEqual(request_event.payload["message_count"], 2)
        self.assertEqual(request_event.payload["dropped_messages"], 0)

    def test_run_sends_system_and_user_messages_to_model(self):
        model = FakeModel(["这是一个上海三日游计划。"])
        agent = TravelAgent(model)

        result = agent.run("计划上海三日游")

        self.assertEqual(result.answer, "这是一个上海三日游计划。")
        self.assertEqual(
            model.requests[0],
            [
                Message(role="system", content="你是旅行规划助手。"),
                Message(role="user", content="计划上海三日游"),
            ],
        )

    def test_run_sends_registered_tool_schemas_to_model(self):
        """模型请求必须包含当前注册表的工具说明。"""

        model = FakeModel(["这是一个上海三日游计划。"])
        agent = TravelAgent(model)

        agent.run("计划上海三日游")

        self.assertEqual(
            model.tool_requests[0],
            [
                {
                    "name": "get_weather",
                    "description": "查询固定城市的天气示例数据",
                    "parameters": {"city": "str"},
                }
            ],
        )

    def test_run_emits_ordered_events(self):
        model = FakeModel(["先安排外滩，再参观历史建筑。"])
        agent = TravelAgent(model)

        result = agent.run("我喜欢历史景点")

        self.assertEqual(
            [event.kind for event in result.events],
            ["user_input", "model_request", "model_response", "run_finished"],
        )
        self.assertEqual(result.events[0].payload["content"], "我喜欢历史景点")
        self.assertEqual(result.events[-1].payload["status"], "completed")

    def test_run_executes_tool_and_sends_observation_back_to_model(self):
        model = FakeModel(
            [
                ToolCall(name="get_weather", arguments={"city": "上海"}),
                "上海今天晴，最高温度 25°C。",
            ]
        )
        agent = TravelAgent(model)

        result = agent.run("我周末去上海，天气怎么样？")

        self.assertEqual(result.answer, "上海今天晴，最高温度 25°C。")
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
        self.assertEqual(
            model.requests[1][-2],
            Message(
                role="assistant",
                content="",
                tool_call=ToolCall(
                    name="get_weather",
                    arguments={"city": "上海"},
                ),
            ),
        )
        self.assertEqual(
            model.requests[1][-1],
            Message(role="tool", content="上海：晴，25°C"),
        )

    def test_run_rejects_unknown_tool(self):
        model = FakeModel(
            [ToolCall(name="book_flight", arguments={"city": "上海"})]
        )
        agent = TravelAgent(model)

        with self.assertRaisesRegex(ValueError, "unknown tool"):
            agent.run("帮我订机票")

    def test_run_uses_injected_registry_for_custom_tool(self):
        """验证 Agent 依赖注册表，而不是在内部硬编码工具分支。"""

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="echo_city",
                description="返回城市名称",
                schema={"city": str},
                handler=lambda arguments: f"城市：{arguments['city']}",
            )
        )
        model = FakeModel(
            [
                ToolCall(name="echo_city", arguments={"city": "杭州"}),
                "已记录目的地：杭州。",
            ]
        )

        result = TravelAgent(model, registry=registry).run("我想去杭州")

        self.assertEqual(result.answer, "已记录目的地：杭州。")
        self.assertEqual(model.requests[1][-1], Message(role="tool", content="城市：杭州"))


if __name__ == "__main__":
    unittest.main()
