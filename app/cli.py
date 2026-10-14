"""旅行规划 Agent 的命令行入口。

CLI（Command-Line Interface）在本项目中只是最外层适配器：

    命令行文本 -> TravelAgent.run -> 事件轨迹和最终答案 -> 终端

它不负责实现 Agent 规则。这样以后把入口从 CLI 换成 FastAPI 时，核心
Agent 仍然可以复用，不需要把运行逻辑复制到 HTTP 路由里。
"""

import argparse
from collections.abc import Sequence

from app.agent import TravelAgent
from app.model import FakeModel


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    单独提取这个函数，是为了让参数定义和实际运行分开：
    解析器负责“用户如何输入”，Agent 负责“系统如何处理”。
    """

    parser = argparse.ArgumentParser(
        description="运行一个确定性的旅行规划 Agent 示例。"
    )
    parser.add_argument(
        "user_text",
        help="用户的旅行规划需求，例如：我喜欢历史景点",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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

    # FakeModel 让 CLI 不需要 API Key 或网络，也保证演示结果可重复。
    model = FakeModel(["这是一个确定性的旅行规划示例。"])
    agent = TravelAgent(model)
    result = agent.run(args.user_text)

    # 按事件发生顺序输出轨迹。
    # repr(payload) 能保留字典结构，方便学习者观察数据形状；它不是生产日志格式。
    for event in result.events:
        print(f"event={event.kind} payload={event.payload!r}")

    # 最终答案单独输出，方便用户区分“过程证据”和“最终结果”。
    print(f"answer={result.answer}")
    return 0
