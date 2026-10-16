import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

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

    def test_main_can_show_persisted_storage_summary(self):
        """CLI 使用 --db 和 --session-id 时应展示持久化结果。"""

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            database_path = Path(directory) / "travel-agent.db"

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--db",
                        str(database_path),
                        "--session-id",
                        "trip-1",
                        "我想去上海",
                    ]
                )

            printed = output.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn(f"storage_path={database_path}", printed)
            self.assertIn("persisted_message_count=2", printed)
            self.assertIn("persisted_event_count=4", printed)
            self.assertIn("persisted_tool_call_count=0", printed)

    def test_main_can_enable_context_compression(self):
        """CLI 应允许设置小消息预算来直接观察上下文压缩。"""

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            output = io.StringIO()
            common_arguments = [
                "--db",
                str(database_path),
                "--session-id",
                "trip-1",
                "--max-messages",
                "4",
                "--compress-context",
            ]

            with redirect_stdout(output):
                first_exit_code = main([*common_arguments, "我想去上海三天"])
                second_exit_code = main([*common_arguments, "预算是 2000 元"])
                third_exit_code = main([*common_arguments, "请把路线按天安排"])

            printed = output.getvalue()

            self.assertEqual((first_exit_code, second_exit_code, third_exit_code), (0, 0, 0))
            self.assertIn("event=context_compression_started", printed)
            self.assertIn("event=context_compressed", printed)
            self.assertIn("'compressed_messages': 4", printed)
            self.assertIn("'max_messages': 4", printed)

    def test_main_can_enable_rag_from_a_knowledge_base(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge_base = Path(directory) / "travel-notes.md"
            knowledge_base.write_text(
                "上海外滩适合夜景，建议晚上前往。\n\n杭州西湖适合散步。",
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--knowledge-base",
                        str(knowledge_base),
                        "--rag-top-k",
                        "1",
                        "上海夜景怎么安排",
                    ]
                )

            printed = output.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn("event=rag_search_started", printed)
            self.assertIn("event=rag_search_completed", printed)
            self.assertIn("travel-notes.md:0", printed)

    def test_main_can_write_and_recall_explicit_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_db = Path(directory) / "memories.db"
            first_output = io.StringIO()
            second_output = io.StringIO()
            common = ["--memory-db", str(memory_db), "--memory-user", "alice"]

            with redirect_stdout(first_output):
                first_exit_code = main([*common, "请记住我喜欢历史景点"])
            with redirect_stdout(second_output):
                second_exit_code = main([*common, "帮我规划上海历史路线"])

            self.assertEqual((first_exit_code, second_exit_code), (0, 0))
            self.assertIn("event=memory_written", first_output.getvalue())
            self.assertIn("event=memory_search_completed", second_output.getvalue())
            self.assertIn("memory_ids': [1]", second_output.getvalue())

    def test_main_can_run_the_deterministic_multi_tool_demo(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "--provider",
                    "fake",
                    "--demo-tool-batch",
                    "同时查上海和北京天气",
                ]
            )

        printed = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("event=tool_queue_started", printed)
        self.assertIn("'count': 2", printed)
        self.assertIn("event=tool_queue_finished", printed)
        self.assertIn("call-1-1", printed)
        self.assertIn("call-1-2", printed)

    def test_main_restores_history_when_reusing_session_id(self):
        """第二次 CLI 运行应从数据库恢复第一次运行的消息。"""

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            output = io.StringIO()
            common_arguments = [
                "--db",
                str(database_path),
                "--session-id",
                "trip-1",
            ]

            with redirect_stdout(output):
                first_exit_code = main([*common_arguments, "我想去上海"])
                second_exit_code = main([*common_arguments, "预算 2000 元"])

            printed = output.getvalue()

            self.assertEqual(first_exit_code, 0)
            self.assertEqual(second_exit_code, 0)
            self.assertIn("history_before_count=2", printed)
            self.assertIn("history_before_roles=['user', 'assistant']", printed)
            self.assertIn("history_before[1] role=user content='我想去上海'", printed)
            self.assertIn(
                "history_before[2] role=assistant content='这是一个确定性的旅行规划示例。'",
                printed,
            )
            self.assertIn("persisted_message_count=4", printed)

    def test_main_rejects_only_one_persistence_argument(self):
        with self.assertRaises(SystemExit):
            main(["--db", "trip.db", "我想去上海"])


if __name__ == "__main__":
    unittest.main()
