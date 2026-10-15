import unittest

from app.domain import AgentEvent, Message, ToolCall
from app.agent import TravelAgent
from app.model import FakeModel
from app.tools import ToolDefinition, ToolRegistry


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
