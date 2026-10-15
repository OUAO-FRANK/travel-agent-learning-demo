"""旅行规划 Agent 的命令行入口。

CLI（Command-Line Interface）在本项目中只是最外层适配器：

    命令行文本 -> TravelAgent.run -> 事件轨迹和最终答案 -> 终端

它不负责实现 Agent 规则。这样以后把入口从 CLI 换成 FastAPI 时，核心
Agent 仍然可以复用，不需要把运行逻辑复制到 HTTP 路由里。
"""

import argparse
from collections.abc import Mapping, Sequence
import sys

from app.agent import TravelAgent
from app.config import ConfigurationError, load_model_config
from app.model import FakeModel, ModelProviderError, OpenAICompatibleModel


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

    agent = TravelAgent(model)
    try:
        result = agent.run(args.user_text)
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
    return 0
