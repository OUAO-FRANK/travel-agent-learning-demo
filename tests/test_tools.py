import unittest

from app.domain import ToolCall
from app.tools import ToolDefinition, ToolRegistry, build_default_registry


class ToolRegistryTests(unittest.TestCase):
    def test_register_rejects_duplicate_tool_names(self):
        registry = ToolRegistry()
        definition = ToolDefinition(
            name="echo_city",
            description="返回城市名称",
            schema={"city": str},
            handler=lambda arguments: f"城市：{arguments['city']}",
        )

        registry.register(definition)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(definition)

    def test_dispatch_validates_schema_and_calls_registered_handler(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="echo_city",
                description="返回城市名称",
                schema={"city": str},
                handler=lambda arguments: f"城市：{arguments['city']}",
            )
        )

        result = registry.dispatch(
            ToolCall(name="echo_city", arguments={"city": "上海"})
        )

        self.assertEqual(result, "城市：上海")

    def test_dispatch_rejects_unknown_tool(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "unknown tool"):
            registry.dispatch(ToolCall(name="not_registered", arguments={}))

    def test_dispatch_rejects_missing_extra_and_wrong_type_arguments(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="echo_city",
                description="返回城市名称",
                schema={"city": str},
                handler=lambda arguments: f"城市：{arguments['city']}",
            )
        )

        with self.assertRaisesRegex(ValueError, "missing required argument"):
            registry.dispatch(ToolCall(name="echo_city", arguments={}))

        with self.assertRaisesRegex(ValueError, "unexpected argument"):
            registry.dispatch(
                ToolCall(
                    name="echo_city",
                    arguments={"city": "上海", "country": "中国"},
                )
            )

        with self.assertRaisesRegex(ValueError, "must be str"):
            registry.dispatch(ToolCall(name="echo_city", arguments={"city": 123}))

    def test_default_registry_contains_weather_tool(self):
        registry = build_default_registry()

        self.assertEqual(
            registry.dispatch(
                ToolCall(name="get_weather", arguments={"city": "上海"})
            ),
            "上海：晴，25°C",
        )

        schema = registry.model_schemas()[0]
        self.assertEqual(schema["name"], "get_weather")
        self.assertEqual(schema["parameters"], {"city": "str"})


if __name__ == "__main__":
    unittest.main()
