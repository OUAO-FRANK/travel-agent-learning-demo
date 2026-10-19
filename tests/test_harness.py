import unittest

from app.harness import (
    CancellationToken,
    HarnessStop,
    RetryPolicy,
    RuntimeHarness,
    ToolPolicy,
)
from app.domain import ToolCall
from app.model import ModelProviderError, RetryableModelProviderError


class RuntimeHarnessTests(unittest.TestCase):
    def test_retry_policy_validates_limits(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_retries=-1)
        with self.assertRaises(ValueError):
            RetryPolicy(backoff_seconds=-0.1)

    def test_tool_policy_checks_every_call_before_execution(self):
        policy = ToolPolicy(allowed_tools=frozenset({"get_weather"}))

        allowed = policy.check(
            [ToolCall(name="get_weather", arguments={"city": "上海"})]
        )
        denied = policy.check(
            [
                ToolCall(name="get_weather", arguments={"city": "上海"}),
                ToolCall(name="book_hotel", arguments={"city": "上海"}),
            ]
        )

        self.assertTrue(allowed.allowed)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason, "tool_not_allowed:book_hotel")

    def test_tool_policy_marks_approval_required_calls(self):
        policy = ToolPolicy(approval_required=frozenset({"book_hotel"}))

        decision = policy.check(
            [ToolCall(name="book_hotel", arguments={"city": "上海"})]
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "tool_approval_required:book_hotel")

    def test_model_call_retries_retryable_provider_error(self):
        attempts = [0]
        sleeps = []
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(
                max_retries=2,
                backoff_seconds=0.25,
                sleep=sleeps.append,
            )
        )

        def operation():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RetryableModelProviderError("temporary failure")
            return "ok"

        harness.start_run()
        scheduled = []
        result = harness.execute_model(
            operation,
            on_attempt=lambda attempt: None,
            on_retry=lambda attempt, error, delay: scheduled.append(
                (attempt, str(error), delay)
            ),
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts[0], 3)
        self.assertEqual(harness.model_requests, 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(
            scheduled,
            [
                (1, "temporary failure", 0.25),
                (2, "temporary failure", 0.5),
            ],
        )

    def test_model_call_does_not_retry_non_retryable_provider_error(self):
        attempts = [0]
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(max_retries=3, sleep=lambda _: None)
        )

        def operation():
            attempts[0] += 1
            raise ModelProviderError("invalid response")

        harness.start_run()
        with self.assertRaises(ModelProviderError):
            harness.execute_model(operation, on_attempt=lambda attempt: None)

        self.assertEqual(attempts[0], 1)
        self.assertEqual(harness.model_requests, 1)

    def test_retry_exhaustion_reports_cancellation_before_retry_limit(self):
        exhausted = []
        harness = RuntimeHarness(retry_policy=RetryPolicy(max_retries=0))

        def operation():
            harness.cancel()
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "run_cancelled"):
            harness.execute_model(
                operation,
                on_exhausted=lambda attempt, error: exhausted.append(attempt),
            )

        self.assertEqual(exhausted, [])

    def test_retry_exhaustion_reports_deadline_before_retry_limit(self):
        now = [0.0]
        harness = RuntimeHarness(
            timeout_seconds=5,
            clock=lambda: now[0],
            retry_policy=RetryPolicy(max_retries=0),
        )

        def operation():
            now[0] = 5.0
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "runtime_timeout_exceeded"):
            harness.execute_model(operation)

    def test_model_call_stops_when_retry_is_cancelled(self):
        attempts = [0]
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(max_retries=2, sleep=lambda _: None)
        )

        def operation():
            attempts[0] += 1
            harness.cancel()
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "run_cancelled"):
            harness.execute_model(operation, on_attempt=lambda attempt: None)

        self.assertEqual(attempts[0], 1)
        self.assertEqual(harness.model_requests, 1)

    def test_model_call_stops_when_cancelled_during_backoff(self):
        attempts = [0]
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(max_retries=2, sleep=lambda _: harness.cancel())
        )

        def operation():
            attempts[0] += 1
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "run_cancelled"):
            harness.execute_model(operation)

        self.assertEqual(attempts[0], 1)
        self.assertEqual(harness.model_requests, 1)

    def test_model_call_checks_runtime_after_retry_event_before_sleep(self):
        attempts = [0]
        sleeps = []
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff_seconds=10,
                sleep=sleeps.append,
            )
        )

        def operation():
            attempts[0] += 1
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "run_cancelled"):
            harness.execute_model(
                operation,
                on_retry=lambda attempt, error, delay: harness.cancel(),
            )

        self.assertEqual(attempts[0], 1)
        self.assertEqual(sleeps, [])

    def test_model_call_checks_cancellation_after_attempt_event(self):
        harness = RuntimeHarness()
        called = []

        def operation():
            called.append(True)
            return "不应请求"

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "run_cancelled"):
            harness.execute_model(
                operation,
                on_attempt=lambda _: harness.cancel(),
            )

        self.assertEqual(called, [])
        # 本次模型尝试的预算已预留并记入轨迹，但取消检查阻止了真正的调用。
        self.assertEqual(harness.model_requests, 1)

    def test_retry_does_not_sleep_when_no_model_budget_remains(self):
        sleeps = []
        scheduled = []
        harness = RuntimeHarness(
            max_model_requests=1,
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff_seconds=10,
                sleep=sleeps.append,
            ),
        )

        def operation():
            raise RetryableModelProviderError("temporary failure")

        harness.start_run()
        with self.assertRaisesRegex(HarnessStop, "max_model_requests_exceeded"):
            harness.execute_model(
                operation,
                on_retry=lambda attempt, error, delay: scheduled.append(
                    (attempt, delay)
                ),
            )

        self.assertEqual(sleeps, [])
        self.assertEqual(scheduled, [])
        self.assertEqual(harness.model_requests, 1)

    def test_model_call_reports_scheduled_retries(self):
        attempts = [0]
        scheduled = []
        harness = RuntimeHarness(
            retry_policy=RetryPolicy(max_retries=1, sleep=lambda _: None)
        )

        def operation():
            attempts[0] += 1
            if attempts[0] == 1:
                raise RetryableModelProviderError("temporary failure")
            return "ok"

        harness.start_run()
        result = harness.execute_model(
            operation,
            on_attempt=lambda attempt: None,
            on_retry=lambda attempt, error, delay: scheduled.append(
                (attempt, str(error), delay)
            ),
        )

        self.assertEqual(result, "ok")
        self.assertEqual(scheduled, [(1, "temporary failure", 0.0)])

    def test_cancellation_token_is_not_cleared_by_starting_a_run(self):
        token = CancellationToken()
        harness = RuntimeHarness(cancellation_token=token)

        self.assertTrue(harness.check_cancelled().allowed)

        token.cancel()
        decision = harness.check_cancelled()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "run_cancelled")

        harness.start_run()
        self.assertFalse(harness.check_cancelled().allowed)

        # 取消请求属于调用方；只有调用方显式 reset 才能清除。
        token.reset()
        self.assertTrue(harness.check_cancelled().allowed)

    def test_tool_round_budget_is_counted_per_run(self):
        harness = RuntimeHarness(max_tool_rounds=2)

        harness.start_run()

        self.assertTrue(harness.consume_tool_round().allowed)
        self.assertTrue(harness.consume_tool_round().allowed)
        decision = harness.consume_tool_round()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_tool_rounds_exceeded")

        harness.start_run()
        self.assertTrue(harness.consume_tool_round().allowed)

    def test_invalid_tool_round_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeHarness(max_tool_rounds=0)

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
