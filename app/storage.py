"""SQLite 会话持久化层。

本模块只负责把 Agent 的领域数据写入和读出 SQLite，不负责决定 Agent
下一步做什么。这样可以清楚区分三件事：

* ``TravelAgent``：运行控制流；
* ``SQLiteSessionStore``：保存消息、事件和工具证据；
* SQLite：进程退出后仍然存在的文件数据。

当前实现使用 Python 标准库 ``sqlite3``，不引入 ORM。教学阶段先看清 SQL
表和序列化边界，后续如果需要 PostgreSQL 或 SQLAlchemy，可以替换这一层。
"""

# 让类型注解可以延迟求值（PEP 563）。
# 好处：可以在注解里写还没定义的类型名，也不会在运行时立刻求值，避免循环导入等问题。
from __future__ import annotations

import json          # 用于把对象序列化成 JSON 字符串，以及反序列化回来
import sqlite3       # Python 标准库自带的 SQLite 驱动，无需额外安装
from contextlib import contextmanager   # 把「打开-使用-关闭」封装成 with 语句
from dataclasses import asdict          # 把 dataclass 实例转成普通 dict
from datetime import datetime, timezone # 生成带时区的 UTC 时间
from pathlib import Path                # 统一处理路径
from typing import Iterator             # 用于类型注解

from app.domain import AgentEvent, Message, ToolCall
# 从领域层导入三个核心数据结构：
# - Message：一条消息（role + content，可选 tool_call）
# - AgentEvent：一个事件（kind + payload）
# - ToolCall：一次工具调用（name + arguments + 可选 reasoning_content）


class SQLiteSessionStore:
    """保存一个或多个旅行会话的最小 SQLite 存储。

    每个公开方法都短暂打开一个连接并在方法结束时关闭。这样不会把一个
    长连接生命周期藏在 Agent 里，也能让“重新创建 Store 就像进程重启”
    的行为在测试中清晰可见。
    """

    def __init__(self, database_path: str | Path) -> None:
        # 把传入的路径统一转成 Path 对象。
        # 好处：Windows 的 "data\\a.db" 和 Linux 的 "data/a.db" 都能统一处理，
        # 也方便调用 .parent / .mkdir 等方法。
        self.database_path = Path(database_path)

        # 如果数据库路径有父目录（比如 "data/sessions.db" 的父目录是 "data"），
        # 就确保这个目录存在。
        # Path("") 表示「当前目录」，此时不需要创建父目录，所以用 != 排除。
        # parents=True：递归创建多层目录；
        # exist_ok=True：目录已存在也不报错。
        if self.database_path.parent != Path(""):
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化表结构（4 张表）。
        # 使用 CREATE TABLE IF NOT EXISTS，所以重复调用是安全的，
        # 也不会覆盖已有数据。
        self._initialize_schema()

    def ensure_session(self, session_id: str) -> None:
        """确保会话元数据存在；已存在的会话不会被覆盖。"""

        # 先校验 session_id 不是空字符串。
        self._validate_session_id(session_id)

        # 用 _connect() 拿到一个短连接。
        # with 结束时，_connect() 会自动 commit 并 close。
        with self._connect() as connection:
            # INSERT OR IGNORE：
            # - 如果 session_id 不存在，插入一行；
            # - 如果已存在（主键冲突），静默忽略，不报错、不覆盖。
            # 这正是「确保存在」的语义。
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, created_at)
                VALUES (?, ?)
                """,
                # 参数用 ? 占位，由 sqlite3 负责转义，避免 SQL 注入。
                (session_id, _utc_now()),
            )

    def append_message(self, session_id: str, message: Message) -> None:
        """按会话内递增序号保存一条消息。"""

        # 先确保会话存在（否则外键约束会失败）。
        self.ensure_session(session_id)

        # Message 里可能带一个 ToolCall 对象。
        # SQLite 不能直接存 Python 对象，所以这里：
        # 1. asdict(...) 把 dataclass 转成普通 dict；
        # 2. json.dumps(...) 把 dict 转成 JSON 字符串。
        # 如果 message.tool_call 是 None，就存 NULL。
        if message.tool_calls is not None:
            # 批量调用使用一个带 calls 字段的 JSON 包装，兼容旧数据库中
            # 直接保存单个 ToolCall 字典的格式。
            tool_call_json = json.dumps(
                {"calls": [asdict(call) for call in message.tool_calls]},
                ensure_ascii=False,
            )
        else:
            tool_call_json = (
                json.dumps(asdict(message.tool_call), ensure_ascii=False)
                if message.tool_call is not None
                else None
            )

        with self._connect() as connection:
            # 计算本会话 messages 表里的下一个序号。
            sequence = self._next_sequence(connection, "messages", session_id)

            # 插入一条消息记录。
            # 注意：这里只存 tool_call_json（字符串），不存对象本身。
            connection.execute(
                """
                INSERT INTO messages
                    (session_id, sequence, role, content, tool_call_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sequence,
                    message.role,
                    message.content,
                    tool_call_json,
                ),
            )

    def load_messages(self, session_id: str) -> list[Message]:
        """按原始顺序恢复一个会话的消息历史。"""

        self._validate_session_id(session_id)

        with self._connect() as connection:
            # 按 sequence 升序读取，保证恢复出来的顺序和写入顺序一致。
            rows = connection.execute(
                """
                SELECT role, content, tool_call_json
                FROM messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
        # 注意：连接在 with 块结束时已经关闭，rows 是已经取回内存的数据，
        # 所以下面继续用 rows 是安全的。

        # 把每一行数据库记录还原成 Message 对象。
        # row[2] 是 tool_call_json：
        # - 如果不是 NULL，用 _tool_call_from_json 反序列化成 ToolCall；
        # - 如果是 NULL，就保持 None。
        return [_message_from_row(row) for row in rows]

    def append_event(self, session_id: str, event: AgentEvent) -> None:
        """保存事件，并同步维护工具调用证据表。"""

        self.ensure_session(session_id)

        # event.payload 是一个 dict，需要转成 JSON 字符串才能存进 TEXT 列。
        # ensure_ascii=False 让中文原样存储，而不是被转成 \uXXXX。
        payload_json = json.dumps(event.payload, ensure_ascii=False)

        with self._connect() as connection:
            # 计算 events 表里本会话的下一个序号。
            sequence = self._next_sequence(connection, "events", session_id)

            # 插入事件记录。
            connection.execute(
                """
                INSERT INTO events (session_id, sequence, kind, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, sequence, event.kind, payload_json),
            )

            # 关键：除了存原始事件，还要同步维护 tool_calls 表。
            # 这样「工具调用」这件事就有一条独立、可直接查询的记录，
            # 而不用每次去 events 里筛选、拼接两条事件。
            self._sync_tool_call_evidence(connection, session_id, event)

    def load_events(self, session_id: str) -> list[AgentEvent]:
        """恢复一个会话的完整事件轨迹。"""

        self._validate_session_id(session_id)

        with self._connect() as connection:
            # 按 sequence 升序读回所有事件。
            rows = connection.execute(
                """
                SELECT kind, payload_json
                FROM events
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        # 把 payload_json 字符串反序列化回 dict，重新构造 AgentEvent。
        return [
            AgentEvent(kind=row[0], payload=json.loads(row[1]))
            for row in rows
        ]

    def load_tool_calls(self, session_id: str) -> list[dict[str, object]]:
        """恢复工具调用的请求、参数、状态和结果。"""

        self._validate_session_id(session_id)

        with self._connect() as connection:
            # 按 sequence 升序读回所有工具调用记录。
            rows = connection.execute(
                """
                SELECT call_id, name, arguments_json, status, result
                FROM tool_calls
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        # 把 arguments_json 反序列化成 dict，其余字段直接放回。
        # 返回的是普通 dict 列表，而不是某个 dataclass。
        return [
            {
                "call_id": row[0],
                "name": row[1],
                "arguments": json.loads(row[2]),
                "status": row[3],
                "result": row[4],
            }
            for row in rows
        ]

    def next_tool_call_id(self, session_id: str) -> str:
        """为串行运行的同一会话生成下一个工具调用 ID。

        ``TravelAgent`` 的轮数只在一次 ``run`` 内计数；如果用户退出程序后
        再继续同一会话，轮数会重新从 1 开始。因此持久化模式下必须从数据库
        已保存的工具调用继续编号，而不能只使用当前运行的局部计数器。
        同一会话的并发写入还需要原子分配 ID，将在工具队列阶段处理。
        """

        self._validate_session_id(session_id)
        self.ensure_session(session_id)

        with self._connect() as connection:
            # 取 tool_calls 表里本会话最大的 sequence，加 1。
            # COALESCE(MAX(sequence), 0)：如果表里还没有记录，MAX 返回 NULL，
            # COALESCE 把它变成 0，再 +1 得到 1。
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1
                FROM tool_calls
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        # 返回形如 "call-1"、"call-2" 的字符串。
        return f"call-{int(row[0])}"

    def _initialize_schema(self) -> None:
        """创建本阶段需要的四张表；CREATE IF NOT EXISTS 可重复执行。"""

        with self._connect() as connection:
            # executescript 可以一次执行多条 SQL 语句。
            connection.executescript(
                """
                -- 会话表：每个会话一行。
                -- session_id 是主键，created_at 记录创建时间（UTC ISO 字符串）。
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                -- 消息表：每个会话的多条消息。
                -- id 是全局自增主键（仅用于数据库内部）。
                -- sequence 是「会话内」序号，从 1 开始，用于排序。
                -- tool_call_json 可空：普通消息为 NULL，带工具调用的消息存 JSON。
                -- UNIQUE (session_id, sequence)：同一会话内序号不能重复。
                -- FOREIGN KEY：session_id 必须存在于 sessions 表。
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_call_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE (session_id, sequence)
                );

                -- 事件表：每个会话的事件轨迹。
                -- kind 是事件类型（如 tool_requested / tool_result）。
                -- payload_json 是事件载荷的 JSON 字符串。
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE (session_id, sequence)
                );

                -- 工具调用表：把 tool_requested 和 tool_result 拼成一条记录。
                -- call_id 是业务层使用的 ID（如 "call-1"）。
                -- status 记录状态：requested / completed。
                -- result 可空：工具还没返回时为 NULL。
                -- UNIQUE (session_id, call_id)：同一会话内 call_id 不能重复。
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    call_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE (session_id, call_id)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """创建并可靠关闭启用外键约束的连接。

        ``sqlite3.Connection`` 自己的上下文管理器只负责提交或回滚，
        不会自动关闭连接。Windows 会锁定打开的数据库文件，因此这里
        必须显式在 finally 中 close，测试临时目录才能安全清理。
        """

        # 打开一个 SQLite 连接。
        connection = sqlite3.connect(self.database_path)

        # SQLite 默认不启用外键约束，必须每个连接手动开启。
        # 否则 messages/events/tool_calls 的外键只是装饰，不会真正生效。
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            # 把连接交给 with 块里的代码使用。
            yield connection
        except Exception:
            # 如果 with 块里抛出异常，回滚这次方法的全部写入，
            # 避免留下「写了一半」的脏数据。
            connection.rollback()
            raise   # 重新抛出，让调用方知道出错了
        else:
            # 如果 with 块正常结束，提交这次方法的全部写入。
            connection.commit()
        finally:
            # 无论成功、失败、还是异常，都一定要关闭连接。
            # 不关闭会占用文件句柄，Windows 上还会导致临时目录删不掉。
            connection.close()

    @staticmethod
    def _next_sequence(
        connection: sqlite3.Connection,
        table: str,
        session_id: str,
    ) -> int:
        """读取会话内下一个稳定顺序号。"""

        # 表名来自本类内部固定调用，不接受用户输入，不能用参数占位符。
        # 参数占位符 ? 只能用于「值」，不能用于「表名/列名」。
        # 这里 table 只会是 "messages" 或 "events"，是安全的。
        row = connection.execute(
            f"SELECT COALESCE(MAX(sequence), 0) + 1 FROM {table} WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        # row[0] 是查询结果的第一列；int() 确保返回 Python 整数。
        return int(row[0])

    @staticmethod
    def _sync_tool_call_evidence(
        connection: sqlite3.Connection,
        session_id: str,
        event: AgentEvent,
    ) -> None:
        """将两个工具事件拼成一条可查询的工具调用记录。"""

        if event.kind == "tool_requested":
            # 工具调用请求事件：
            # 在 tool_calls 表插入一条新记录，status='requested'。
            call_id = str(event.payload["call_id"])
            connection.execute(
                """
                INSERT INTO tool_calls
                    (session_id, sequence, call_id, name, arguments_json, status)
                VALUES (?, COALESCE(
                    (SELECT MAX(sequence) + 1 FROM tool_calls WHERE session_id = ?),
                    1
                ), ?, ?, ?, 'requested')
                """,
                (
                    session_id,
                    # 子查询：同一会话已有记录时取 MAX(sequence)+1；
                    # 没有记录时 MAX 返回 NULL，COALESCE 用 1 兜底。
                    session_id,
                    call_id,
                    str(event.payload["name"]),
                    # arguments 是 dict，需要序列化成 JSON 字符串存储。
                    json.dumps(event.payload["arguments"], ensure_ascii=False),
                ),
            )
        elif event.kind == "tool_result":
            # 工具返回结果事件：
            # 找到之前那条 call_id 对应的记录，更新为 completed，并写入结果。
            call_id = str(event.payload["call_id"])
            connection.execute(
                """
                UPDATE tool_calls
                SET status = 'completed', result = ?
                WHERE session_id = ? AND call_id = ?
                """,
                # result 直接存字符串，不存 JSON。
                (str(event.payload["content"]), session_id, call_id),
            )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        # strip() 去掉首尾空白；如果去掉后为空，说明是空字符串或纯空格。
        if not session_id.strip():
            raise ValueError("session_id must not be empty")


def _tool_call_from_json(raw_json: str) -> ToolCall:
    """从消息表中的 JSON 恢复不可变的 ToolCall 对象。"""

    # 把 JSON 字符串解析回 dict。
    payload = json.loads(raw_json)

    # 重新构造 ToolCall 对象。
    # reasoning_content 用 .get()：旧数据可能没有这个字段，缺失时返回 None。
    return ToolCall(
        name=payload["name"],
        arguments=payload["arguments"],
        reasoning_content=payload.get("reasoning_content"),
    )


def _message_from_row(row: tuple[object, ...]) -> Message:
    """兼容恢复单工具和批量工具 assistant 消息。"""

    raw_json = row[2]
    if raw_json is None:
        return Message(role=str(row[0]), content=str(row[1]))

    payload = json.loads(str(raw_json))
    if "calls" in payload:
        return Message(
            role=str(row[0]),
            content=str(row[1]),
            tool_calls=tuple(
                ToolCall(
                    name=call["name"],
                    arguments=call["arguments"],
                    reasoning_content=call.get("reasoning_content"),
                )
                for call in payload["calls"]
            ),
        )
    return Message(
        role=str(row[0]),
        content=str(row[1]),
        tool_call=_tool_call_from_json(str(raw_json)),
    )


def _utc_now() -> str:
    """返回带时区的 UTC 时间，避免不同机器的本地时间混在一起。"""

    # timezone.utc 保证时间带时区信息，isoformat() 转成 ISO 8601 字符串，
    # 例如 "2026-10-15T08:30:00+00:00"。
    return datetime.now(timezone.utc).isoformat()
