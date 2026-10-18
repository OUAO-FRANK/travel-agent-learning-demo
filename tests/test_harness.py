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

    def test_deadline_is_checked_before_consuming_a_budget(self):
        now = [100.0]
        harness = RuntimeHarness(
            timeout_seconds=5,
            clock=lambda: now[0],
        )

        harness.start_run()
        now[0] = 104.9
        self.assertTrue(harness.consume_model_request().allowed)

        now[0] = 105.0
        decision = harness.consume_tool_call()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "runtime_timeout_exceeded")
        self.assertEqual(harness.tool_calls, 0)

    def test_starting_a_new_run_resets_the_deadline(self):
        now = [10.0]
        harness = RuntimeHarness(timeout_seconds=2, clock=lambda: now[0])

        harness.start_run()
        now[0] = 12.0
        self.assertFalse(harness.check_deadline().allowed)

        now[0] = 20.0
        harness.start_run()
        self.assertTrue(harness.check_deadline().allowed)

    def test_zero_timeout_expires_immediately(self):
        harness = RuntimeHarness(timeout_seconds=0)

        harness.start_run()

        decision = harness.check_deadline()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "runtime_timeout_exceeded")

    def test_negative_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeHarness(timeout_seconds=-1)


if __name__ == "__main__":
    unittest.main()
