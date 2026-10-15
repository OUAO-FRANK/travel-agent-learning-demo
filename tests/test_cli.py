import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from app.cli import main


class CliTests(unittest.TestCase):
    def test_main_prints_event_kinds_and_final_answer(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["我喜欢历史景点"])

        printed = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("event=user_input", printed)
        self.assertIn("event=model_request", printed)
        self.assertIn("event=model_response", printed)
        self.assertIn("event=run_finished", printed)
        self.assertIn("answer=这是一个确定性的旅行规划示例。", printed)

    def test_fake_provider_is_the_default(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["我喜欢历史景点"])

        self.assertEqual(exit_code, 0)
        self.assertIn("answer=这是一个确定性的旅行规划示例。", output.getvalue())

    def test_deepseek_provider_reports_missing_api_key(self):
        output = io.StringIO()
        error_output = io.StringIO()

        with redirect_stdout(output), redirect_stderr(error_output):
            exit_code = main(
                ["我喜欢历史景点", "--provider", "deepseek"],
                env={},
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("DEEPSEEK_API_KEY", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
