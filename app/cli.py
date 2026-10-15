"""旅行规划 Agent 的命令行入口。

CLI（Command-Line Interface）在本项目中只是最外层适配器：

    命令行文本 -> TravelAgent.run -> 事件轨迹和最终答案 -> 终端

它不负责实现 Agent 规则。这样以后把入口从 CLI 换成 FastAPI 时，核心
Agent 仍然可以复用，不需要把运行逻辑复制到 HTTP 路由里。
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

from app.agent import TravelAgent
from app.config import ConfigurationError, load_model_config
from app.model import FakeModel, ModelProviderError, OpenAICompatibleModel
from app.storage import SQLiteSessionStore


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    单独提取这个函数，是为了让参数定义和实际运行分开：
    解析器负责“用户如何输入”，Agent 负责“系统如何处理”。
    """

    parser = argparse.ArgumentParser(description="运行旅行规划 Agent。")
    parser.add_argument(
        "user_text",
        help="用户的旅行规划需求，例如：我喜欢历史景点",
    )
    parser.add_argument(
        "--provider",
        choices=["fake", "deepseek"],
        default="fake",
        help="模型提供方；默认 fake，不需要 API key",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="SQLite 数据库路径；必须和 --session-id 一起使用",
    )
    parser.add_argument(
        "--session-id",
        help="要恢复或继续的会话 ID；必须和 --db 一起使用",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """运行一次命令行 Agent，并打印可观察轨迹。

    参数：
        argv：可选的参数序列。测试传入列表；真实命令行调用时传入 None，
        argparse 会从 sys.argv 读取参数。

    返回：
        进程退出码。阶段 0 的正常运行返回 0。

    这里把固定回复放在 CLI，而不是 TravelAgent 内部，是为了保持
    Agent 与模型实现的边界：CLI 决定本次演示使用哪个模型，Agent 只负责
    组织一次运行。
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    # 只给其中一个参数会让持久化配置不完整，因此在启动模型前直接拒绝。
    # 这样用户不会误以为消息已经保存，却实际上没有稳定的会话键。
    if (args.db is None) != (args.session_id is None):
        parser.error("--db and --session-id must be provided together")

    session_store = (
        SQLiteSessionStore(args.db)
        if args.db is not None and args.session_id is not None
        else None
    )

    # 运行模型前先读一次历史。这个数字让 CLI 变成可观察的学习入口：
    # 第一次运行通常是 0，第二次使用同一个数据库和 session ID 就会变大。
    history_before = (
        session_store.load_messages(args.session_id)
        if session_store is not None and args.session_id is not None
        else []
    )
    if session_store is not None and args.session_id is not None:
        print(f"storage_path={session_store.database_path}")
        print(f"session_id={args.session_id}")
        print(f"history_before_count={len(history_before)}")
        print(f"history_before_roles={[message.role for message in history_before]!r}")
        for index, message in enumerate(history_before, start=1):
            # 直接显示数据库恢复出的消息内容，便于对照下面 model_request 的
            # 角色顺序，理解上一轮内容如何成为本轮模型上下文的一部分。
            description = (
                f"history_before[{index}] role={message.role} "
                f"content={message.content!r}"
            )
            if message.tool_call is not None:
                description += f" tool_call={message.tool_call!r}"
            print(description)

    try:
        config = load_model_config(args.provider, env=env)
    except ConfigurationError as exc:
        # 配置错误属于用户可修复的启动错误；不打印 API key，只打印变量名
        # 和修复方向，并用非零退出码通知命令行调用者。
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if config.provider == "fake":
        # FakeModel 让 CLI 不需要 API Key 或网络，也保证演示结果可重复。
        model = FakeModel(["这是一个确定性的旅行规划示例。"])
    else:
        # API key 已由配置层从环境变量读取；这里不把它打印到终端。
        model = OpenAICompatibleModel(
            base_url=config.base_url or "",
            api_key=config.api_key or "",
            model=config.model or "",
        )

    agent = TravelAgent(model, session_store=session_store)
    try:
        result = agent.run(args.user_text, session_id=args.session_id)
    except ModelProviderError as exc:
        # 供应商错误不应打印 Python 堆栈或请求中的密钥；保留简短错误并
        # 返回非零退出码，方便用户判断是配置/网络/供应商响应问题。
        print(f"model provider error: {exc}", file=sys.stderr)
        return 1

    # 按事件发生顺序输出轨迹。
    # repr(payload) 能保留字典结构，方便学习者观察数据形状；它不是生产日志格式。
    for event in result.events:
        print(f"event={event.kind} payload={event.payload!r}")

    # 最终答案单独输出，方便用户区分“过程证据”和“最终结果”。
    print(f"answer={result.answer}")

    # 运行结束后重新从数据库读取，而不是直接用 Agent 内存中的列表。
    # 这样输出验证的是“已经写入磁盘并能被 Store 重新读出”的结果。
    if session_store is not None and args.session_id is not None:
        persisted_messages = session_store.load_messages(args.session_id)
        persisted_events = session_store.load_events(args.session_id)
        persisted_tool_calls = session_store.load_tool_calls(args.session_id)
        print(f"persisted_message_count={len(persisted_messages)}")
        print(f"persisted_event_count={len(persisted_events)}")
        print(f"persisted_tool_call_count={len(persisted_tool_calls)}")
        print(
            "persisted_roles="
            f"{[message.role for message in persisted_messages]!r}"
        )
        print(
            "persisted_tool_call_ids="
            f"{[call['call_id'] for call in persisted_tool_calls]!r}"
        )
    return 0
