import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.domain import AgentEvent, Message, ToolCall, ToolCallBatch
from app.storage import SQLiteSessionStore


class SQLiteSessionStoreTests(unittest.TestCase):
    def test_connection_is_closed_when_initial_setup_fails(self):
        class BrokenConnection:
            closed = False
            rolled_back = False

            def execute(self, statement, parameters=()):
                raise RuntimeError("sqlite setup failed")

            def rollback(self):
                self.rolled_back = True

            def commit(self):
                raise AssertionError("a failed connection must not commit")

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")
            connection = BrokenConnection()
            with patch("app.storage.sqlite3.connect", return_value=connection):
                with self.assertRaisesRegex(RuntimeError, "sqlite setup failed"):
                    store.ensure_session("trip-1")

        self.assertTrue(connection.rolled_back)
        self.assertTrue(connection.closed)

    def test_messages_round_trip_after_opening_a_new_store(self):
        """进程重启后，消息应能从 SQLite 恢复为领域对象。"""

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            first_store = SQLiteSessionStore(database_path)
            first_store.ensure_session("trip-1")
            first_store.append_message(
                "trip-1",
                Message(role="user", content="我想去上海"),
            )
            first_store.append_message(
                "trip-1",
                Message(
                    role="assistant",
                    content="",
                    tool_call=ToolCall(
                        name="get_weather",
                        arguments={"city": "上海"},
                        reasoning_content="需要先查询天气",
                    ),
                ),
            )

            restarted_store = SQLiteSessionStore(database_path)

            self.assertEqual(
                restarted_store.load_messages("trip-1"),
                [
                    Message(role="user", content="我想去上海"),
                    Message(
                        role="assistant",
                        content="",
                        tool_call=ToolCall(
                            name="get_weather",
                            arguments={"city": "上海"},
                            reasoning_content="需要先查询天气",
                        ),
                    ),
                ],
            )

    def test_events_and_tool_call_evidence_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "travel-agent.db")
            store.ensure_session("trip-1")

            store.append_event(
                "trip-1",
                AgentEvent(
                    kind="tool_requested",
                    payload={
                        "call_id": "call-1",
                        "name": "get_weather",
                        "arguments": {"city": "上海"},
                    },
                ),
            )
            store.append_event(
                "trip-1",
                AgentEvent(
                    kind="tool_result",
                    payload={
                        "call_id": "call-1",
                        "name": "get_weather",
                        "content": "上海：晴，25°C",
                    },
                ),
            )

            self.assertEqual(
                [event.kind for event in store.load_events("trip-1")],
                ["tool_requested", "tool_result"],
            )
            self.assertEqual(
                store.load_tool_calls("trip-1"),
                [
                    {
                        "call_id": "call-1",
                        "name": "get_weather",
                        "arguments": {"city": "上海"},
                        "status": "completed",
                        "result": "上海：晴，25°C",
                    }
                ],
            )

    def test_schema_contains_the_four_learning_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            SQLiteSessionStore(database_path)

            connection = sqlite3.connect(database_path)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()

            self.assertTrue(
                {"sessions", "messages", "events", "tool_calls"}.issubset(tables)
            )

    def test_batch_tool_calls_round_trip_after_reopening_store(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            store = SQLiteSessionStore(database_path)
            store.append_message(
                "trip-1",
                Message(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ToolCall(name="get_weather", arguments={"city": "上海"}),
                        ToolCall(name="get_weather", arguments={"city": "北京"}),
                    ),
                ),
            )

            restarted_store = SQLiteSessionStore(database_path)

            self.assertEqual(
                restarted_store.load_messages("trip-1"),
                [
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=(
                            ToolCall(name="get_weather", arguments={"city": "上海"}),
                            ToolCall(name="get_weather", arguments={"city": "北京"}),
                        ),
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
