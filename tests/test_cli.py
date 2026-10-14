import io
import unittest
from contextlib import redirect_stdout

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


if __name__ == "__main__":
    unittest.main()
