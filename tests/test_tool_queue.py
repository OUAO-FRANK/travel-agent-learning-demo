import unittest
import threading
import time

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

    def test_execute_parallel_runs_independent_calls_at_the_same_time(self):
        calls = (
            ToolCall(name="first", arguments={}),
            ToolCall(name="second", arguments={}),
        )
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def dispatch(call: ToolCall) -> str:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return f"result:{call.name}"

        results = ToolQueue().execute_parallel(calls, dispatch)

        self.assertGreaterEqual(maximum_active, 2)
        self.assertEqual(
            [(item.call.name, item.result) for item in results],
            [("first", "result:first"), ("second", "result:second")],
        )


if __name__ == "__main__":
    unittest.main()
