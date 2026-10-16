import unittest

from app.domain import ToolCall
from app.tool_queue import ToolQueue


class ToolQueueTests(unittest.TestCase):
    def test_execute_runs_calls_in_order_and_returns_each_result(self):
        calls = (
            ToolCall(name="first", arguments={}),
            ToolCall(name="second", arguments={}),
        )
        execution_order: list[str] = []

        def dispatch(call: ToolCall) -> str:
            execution_order.append(call.name)
            return f"result:{call.name}"

        results = ToolQueue().execute(calls, dispatch)

        self.assertEqual(execution_order, ["first", "second"])
        self.assertEqual(
            [(item.call.name, item.result) for item in results],
            [("first", "result:first"), ("second", "result:second")],
        )

    def test_execute_stops_at_first_failure(self):
        calls = (
            ToolCall(name="first", arguments={}),
            ToolCall(name="second", arguments={}),
        )
        execution_order: list[str] = []

        def dispatch(call: ToolCall) -> str:
            execution_order.append(call.name)
            raise ValueError("first failed")

        with self.assertRaisesRegex(ValueError, "first failed"):
            ToolQueue().execute(calls, dispatch)

        self.assertEqual(execution_order, ["first"])


if __name__ == "__main__":
    unittest.main()
