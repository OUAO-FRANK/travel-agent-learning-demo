import unittest

from app.harness import RuntimeHarness


class RuntimeHarnessTests(unittest.TestCase):
    def test_model_and_tool_budgets_are_counted_per_run(self):
        harness = RuntimeHarness(max_model_requests=2, max_tool_calls=1)

        harness.start_run()

        self.assertTrue(harness.consume_model_request().allowed)
        self.assertTrue(harness.consume_model_request().allowed)
        model_decision = harness.consume_model_request()
        self.assertFalse(model_decision.allowed)
        self.assertEqual(model_decision.reason, "max_model_requests_exceeded")

        self.assertTrue(harness.consume_tool_call().allowed)
        tool_decision = harness.consume_tool_call()
        self.assertFalse(tool_decision.allowed)
        self.assertEqual(tool_decision.reason, "max_tool_calls_exceeded")

        harness.start_run()
        self.assertTrue(harness.consume_model_request().allowed)
        self.assertTrue(harness.consume_tool_call().allowed)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeHarness(max_model_requests=0)
        with self.assertRaises(ValueError):
            RuntimeHarness(max_tool_calls=0)


if __name__ == "__main__":
    unittest.main()
