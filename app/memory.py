"""旅行 Agent 的最小长期记忆组件。

Memory 和 RAG 的职责不同：RAG 从外部资料中找事实，Memory 保存某个用户
明确告诉 Agent 的长期信息。本阶段只实现显式写入和关键词召回，不让模型
自动猜测用户偏好，避免把错误推断永久保存。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class MemoryItem:
    """一条可长期保存的用户记忆。"""

    # id 是数据库分配的稳定标识，方便事件日志引用具体记忆。
    id: int

    # user_id 把记忆归属到用户，而不是归属到某一次会话。
    user_id: str

    # content 是记忆的自然语言内容，例如“我喜欢历史景点”。
    content: str

    # source 记录写入来源；本阶段显式输入固定使用 explicit。
    source: str

    # created_at 使用 UTC ISO 字符串，避免不同机器的本地时间混杂。
    created_at: str


class SQLiteMemoryStore:
    """把长期记忆保存到 SQLite，并提供简单关键词搜索。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if self.database_path.parent != Path(""):
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def add(self, user_id: str, content: str, source: str = "explicit") -> MemoryItem:
        """为用户新增一条记忆，并返回写入后的完整对象。"""

        _validate_non_empty(user_id, "user_id")
        _validate_non_empty(content, "content")
        _validate_non_empty(source, "source")
        created_at = _utc_now()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (user_id, content, source, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, content, source, created_at),
            )
            memory_id = int(cursor.lastrowid)

        return MemoryItem(memory_id, user_id, content, source, created_at)

    def list(self, user_id: str) -> list[MemoryItem]:
        """按写入顺序返回某个用户的全部记忆。"""

        _validate_non_empty(user_id, "user_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, content, source, created_at
                FROM memories
                WHERE user_id = ?
                ORDER BY id ASC
                """,
                (user_id,),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def search(self, user_id: str, query: str, limit: int = 3) -> list[MemoryItem]:
        """按关键词重叠返回最相关的用户记忆。"""

        _validate_non_empty(user_id, "user_id")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        query_terms = _terms(query)
        if not query_terms:
            return []

        # 记忆量很小时先全部读出再评分，代码更直观；后续规模变大时
        # 再把检索换成 SQLite FTS 或向量索引，不改变 Agent 的接口。
        scored: list[tuple[int, int, MemoryItem]] = []
        for item in self.list(user_id):
            score = len(query_terms & _terms(item.content))
            if score > 0:
                scored.append((score, item.id, item))

        # 分数高的优先；分数相同时使用稳定的数据库 id，保证实验可复现。
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [item for _, _, item in scored[:limit]]

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()


def extract_explicit_memory(user_text: str) -> str | None:
    """从明确的“记住……”命令中提取记忆内容。

    这里只识别显式前缀，不从普通陈述中自动推断。这样“我喜欢历史景点”
    不会悄悄写入长期存储，用户必须明确说“请记住我喜欢历史景点”。
    """

    normalized = user_text.strip()
    prefixes = ("请记住", "记住")
    for prefix in prefixes:
        if normalized.startswith(prefix):
            content = normalized[len(prefix):].strip(" ：:，,")
            return content or None
    return None


def format_memory_context(items: list[MemoryItem]) -> str | None:
    """把召回的记忆格式化成只供本轮模型使用的上下文消息。"""

    if not items:
        return None
    lines = [
        "以下是与当前请求相关的用户长期记忆。它们是用户过去明确提供的信息，"
        "只能作为偏好参考，不要把它们当成外部事实："
    ]
    lines.extend(f"[memory:{item.id}] {item.content}" for item in items)
    return "\n".join(lines)


def _memory_from_row(row: tuple[object, ...]) -> MemoryItem:
    return MemoryItem(
        id=int(row[0]),
        user_id=str(row[1]),
        content=str(row[2]),
        source=str(row[3]),
        created_at=str(row[4]),
    )


def _terms(text: str) -> set[str]:
    """提取教学版关键词：英文单词和中文相邻二字片段。"""

    import re

    terms = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    terms.update(chinese[index:index + 2] for index in range(len(chinese) - 1))
    return terms


def _validate_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
