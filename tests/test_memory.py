import tempfile
import unittest
from pathlib import Path

from app.memory import MemoryItem, SQLiteMemoryStore, extract_explicit_memory


class MemoryTests(unittest.TestCase):
    def test_explicit_memory_text_is_extracted_from_remember_command(self):
        self.assertEqual(
            extract_explicit_memory("请记住我喜欢历史景点"),
            "我喜欢历史景点",
        )
        self.assertIsNone(extract_explicit_memory("我喜欢历史景点"))

    def test_memory_is_persisted_and_recalled_for_the_same_user(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "travel-agent.db"
            store = SQLiteMemoryStore(database_path)

            created = store.add("alice", "我喜欢历史景点", source="explicit")
            restarted_store = SQLiteMemoryStore(database_path)

            self.assertIsInstance(created, MemoryItem)
            self.assertEqual(restarted_store.list("alice"), [created])
            self.assertEqual(
                [item.content for item in restarted_store.search("alice", "上海历史路线")],
                ["我喜欢历史景点"],
            )

    def test_memory_isolated_between_users(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "travel-agent.db")
            store.add("alice", "我喜欢历史景点")

            self.assertEqual(store.search("bob", "历史景点"), [])


if __name__ == "__main__":
    unittest.main()
