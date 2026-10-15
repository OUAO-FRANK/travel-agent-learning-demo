"""旅行 Agent 的运行配置。

配置模块的职责是把“程序从哪里读取配置”集中起来，避免 CLI、Agent 和模型
适配器各自读取环境变量。API key 只从进程环境读取，不写入源码和测试文件。
"""

from dataclasses import dataclass
import os
from collections.abc import Mapping


class ConfigurationError(ValueError):
    """用户选择的运行模式缺少必要配置时抛出。"""


@dataclass(frozen=True)
class ModelConfig:
    """一次模型运行所需的最小配置。"""

    provider: str
    api_key: str | None
    base_url: str | None
    model: str | None


def load_model_config(
    provider: str,
    env: Mapping[str, str] | None = None,
) -> ModelConfig:
    """根据 provider 名称和环境变量生成模型配置。

    ``env`` 主要用于测试注入一份隔离字典；真实运行时省略它，函数会读取
    当前进程环境。这样测试不会修改开发者机器上的环境变量。
    """

    environment = os.environ if env is None else env

    if provider == "fake":
        return ModelConfig(
            provider="fake",
            api_key=None,
            base_url=None,
            model=None,
        )

    if provider != "deepseek":
        raise ConfigurationError(f"unsupported provider: {provider}")

    api_key = environment.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "DEEPSEEK_API_KEY is required when --provider deepseek is used"
        )

    return ModelConfig(
        provider="deepseek",
        api_key=api_key,
        base_url=(
            environment.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            .strip()
            .rstrip("/")
        ),
        # DeepSeek 当前官方模型列表使用 deepseek-flash；如果用户明确要
        # 测试兼容旧名称，可通过 DEEPSEEK_MODEL 覆盖，而不改代码。
        model=environment.get("DEEPSEEK_MODEL", "deepseek-flash").strip(),
    )
