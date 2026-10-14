"""旅行 Agent 的工具契约、注册表和分发器。

工具系统解决的是一个边界问题：模型只能提出结构化的 ``ToolCall``，
真正是否允许执行、调用哪个 Python handler、参数是否合法，都由 Runtime
侧的注册表决定。

本模块分成三层：

``ToolDefinition``
    描述一个工具的名称、说明、参数类型和实际 handler。

``ToolRegistry``
    保存当前 Agent 被允许使用的工具，并负责查找、校验和分发。

``build_default_registry``
    构造本项目当前默认启用的工具集合。
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.domain import ToolCall


ToolHandler = Callable[[dict[str, object]], str]
"""工具 handler 的统一签名：接收参数字典，返回文本观察结果。"""


@dataclass(frozen=True)
class ToolDefinition:
    """一个可被 Agent 使用的工具定义。

    ``schema`` 使用 ``{参数名: Python 类型}`` 的最小形式。例如：

    ``{"city": str}``

    这还不是完整的 JSON Schema，但足以让我们先看清“工具调用必须经过
    契约校验”这一核心机制。后续接入真实模型时，再映射为供应商需要的
    JSON Schema 格式。
    """

    # name 是模型请求中使用的稳定工具名。
    name: str

    # description 给模型或调试界面说明工具用途。
    description: str

    # schema 声明必需参数和每个参数的 Python 类型。
    # 阶段 2 将所有 schema 字段视为必需字段。
    schema: dict[str, type]

    # handler 是真正执行能力的函数；模型不能直接获得这个函数对象。
    handler: ToolHandler

    def model_schema(self) -> dict[str, object]:
        """返回适合展示给模型的简化 schema。

        这里故意返回类型名称而不是 Python ``type`` 对象，因为模型接口
        最终需要可序列化的数据。完整 JSON Schema 规范暂不在本阶段展开。
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                argument_name: argument_type.__name__
                for argument_name, argument_type in self.schema.items()
            },
        }


class ToolRegistry:
    """保存允许使用的工具，并负责安全分发调用。

    注册表是 Runtime 和具体工具 handler 之间的边界：

    * Runtime 只提交 ``ToolCall``；
    * Registry 决定工具是否存在；
    * Registry 校验参数；
    * 通过定义中的 handler 执行能力。

    这样可以避免在 Agent 里不断增加：

    ``if call.name == "get_weather": ...``

    也避免根据模型返回的字符串动态执行任意函数。
    """

    def __init__(self) -> None:
        # 字典的 key 是模型可见的工具名，value 是完整工具定义。
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """注册一个工具，拒绝重复名称覆盖。

        重复注册通常意味着配置错误。静默覆盖会让同一个 ToolCall 在
        不同启动顺序下执行不同 handler，难以调试，因此这里直接失败。
        """

        if definition.name in self._definitions:
            raise ValueError(f"tool already registered: {definition.name}")

        self._definitions[definition.name] = definition

    def dispatch(self, call: ToolCall) -> str:
        """校验并执行一个工具调用请求。

        失败顺序固定为：工具不存在、缺少参数、额外参数、类型错误。
        只有所有校验通过后才调用 handler。
        """

        definition = self._definitions.get(call.name)
        if definition is None:
            raise ValueError(f"unknown tool: {call.name}")

        expected_arguments = set(definition.schema)
        provided_arguments = set(call.arguments)

        missing_arguments = expected_arguments - provided_arguments
        if missing_arguments:
            missing = sorted(missing_arguments)[0]
            raise ValueError(f"missing required argument: {missing}")

        unexpected_arguments = provided_arguments - expected_arguments
        if unexpected_arguments:
            unexpected = sorted(unexpected_arguments)[0]
            raise ValueError(f"unexpected argument: {unexpected}")

        for argument_name, expected_type in definition.schema.items():
            value = call.arguments[argument_name]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"argument '{argument_name}' must be {expected_type.__name__}"
                )

        # 传入副本，防止 handler 通过修改字典反向污染 ToolCall。
        return definition.handler(dict(call.arguments))

    def model_schemas(self) -> list[dict[str, object]]:
        """返回当前注册工具的可序列化描述，保持注册顺序。"""

        return [definition.model_schema() for definition in self._definitions.values()]


def get_weather(arguments: dict[str, object]) -> str:
    """返回固定天气数据，作为教学用外部能力替身。"""

    city = arguments["city"]
    weather_by_city = {
        "上海": "上海：晴，25°C",
        "北京": "北京：多云，20°C",
    }

    try:
        return weather_by_city[city]  # type: ignore[index]
    except KeyError as exc:
        # 没有数据时明确失败，不根据城市名称猜测天气。
        raise ValueError(f"weather unavailable for city: {city}") from exc


def build_default_registry() -> ToolRegistry:
    """构造当前 Agent 默认允许使用的工具集合。"""

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_weather",
            description="查询固定城市的天气示例数据",
            schema={"city": str},
            handler=get_weather,
        )
    )
    return registry


def execute_tool_call(call: ToolCall) -> str:
    """兼容旧调用方：使用默认注册表执行一次工具调用。"""

    return build_default_registry().dispatch(call)
