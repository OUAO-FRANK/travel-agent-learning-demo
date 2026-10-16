"""模型适配层的教学用实现。

真实 Agent 通常不会直接在 Runtime 里写死某家模型供应商的 HTTP 调用，
而是把“给模型一组消息，得到一个回复”抽象成一个小接口。本模块同时提供
确定性的 FakeModel 和一个可注入传输函数的 OpenAI-compatible 适配器。

这里先定义一个很小的 ``ModelProvider`` 协议，再提供 FakeModel 实现。
协议的意义是：Agent 只依赖“能够接收消息和工具说明并返回模型结果”的能力，
不依赖某个具体供应商或 SDK。

FakeModel 的价值不是模拟真实模型的智能程度，而是提供：

* 确定性：同样的预设回复会产生同样的结果；
* 可观察性：可以检查 Runtime 实际发给模型的消息；
* 可测试性：不需要网络、API Key 或模型服务。
"""

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.domain import Message, ToolCall, ToolCallBatch


ModelOutput = str | ToolCall | ToolCallBatch
"""模型一次请求可以返回文本、单个工具调用或多个调用组成的批次。"""


class ModelProviderError(RuntimeError):
    """模型供应商请求或响应不符合 Provider 契约时抛出的错误。"""


ProviderTransport = Callable[
    [str, dict[str, str], dict[str, object]],
    dict[str, object],
]
"""可注入的 HTTP 传输函数，方便测试而不访问真实网络。"""


class ModelProvider(Protocol):
    """Agent 所需要的最小模型接口。

    这是 Python 的结构化接口：实现类不必继承本类，只要拥有相同签名，
    就可以被 Agent 使用。真实模型适配器和 FakeModel 都遵守这个契约。
    """

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ModelOutput:
        """根据消息和可用工具说明返回一次模型结果。"""

        ...


class OpenAICompatibleModel:
    """最小 OpenAI-compatible Chat Completions 适配器。

    这个类只负责供应商边界：

    * 把内部 ``Message`` 转成供应商请求字典；
    * 把简化工具 schema 转成 JSON Schema；
    * 把供应商响应解析成文本或 ``ToolCall``。

    Agent 的循环、工具权限和停止条件仍然由 ``TravelAgent`` 负责。
    ``transport`` 默认使用 Python 标准库发 HTTP；测试时可以注入一个
    本地函数，从而验证请求和响应解析而不需要 API Key 或网络。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        transport: ProviderTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.transport = transport or _urllib_transport

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ModelOutput:
        """发送一次请求，并把供应商响应转换为 ``ModelOutput``。"""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": self._message_payloads(messages),
        }

        # tools=None 表示调用方没有提供工具目录；不主动发送空字段，
        # 让适配器同时兼容只接受纯文本对话的 OpenAI-compatible 服务。
        if tools is not None:
            payload["tools"] = [self._tool_payload(tool) for tool in tools]

        try:
            response = self.transport(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                payload,
            )
        except ModelProviderError:
            raise
        except Exception as exc:
            raise ModelProviderError("provider request failed") from exc

        return self._parse_response(response)

    @staticmethod
    def _message_payloads(messages: list[Message]) -> list[dict[str, object]]:
        """把内部消息转换成兼容 Chat Completions 的消息列表。"""

        result: list[dict[str, object]] = []
        next_tool_call_id = 0
        pending_tool_call_ids: list[str] = []

        for message in messages:
            requested_tool_calls = []
            if message.tool_call is not None:
                requested_tool_calls = [message.tool_call]
            elif message.tool_calls is not None:
                requested_tool_calls = list(message.tool_calls)

            if requested_tool_calls:
                serialized_tool_calls = []
                for requested_tool_call in requested_tool_calls:
                    tool_call_id = f"call-{next_tool_call_id}"
                    next_tool_call_id += 1
                    pending_tool_call_ids.append(tool_call_id)
                    serialized_tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": requested_tool_call.name,
                                "arguments": json.dumps(
                                    requested_tool_call.arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                result.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": serialized_tool_calls,
                    }
                )
                reasoning_content = requested_tool_calls[0].reasoning_content
                if reasoning_content is not None:
                    result[-1]["reasoning_content"] = reasoning_content
            elif message.role == "tool":
                tool_call_id = (
                    pending_tool_call_ids.pop(0)
                    if pending_tool_call_ids
                    else "call-0"
                )
                result.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": tool_call_id,
                    }
                )
            else:
                result.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        return result

    @staticmethod
    def _tool_payload(tool: dict[str, object]) -> dict[str, object]:
        """把内部简化工具描述转换成供应商要求的 function 包装。"""

        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not isinstance(description, str):
            raise ModelProviderError("tool schema requires name and description")

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": _parameters_to_json_schema(parameters),
            },
        }

    @staticmethod
    def _parse_response(response: dict[str, object]) -> ModelOutput:
        """解析供应商响应，拒绝无法安全解释的结构。"""

        if not isinstance(response, dict):
            raise ModelProviderError("provider response must be an object")

        if "error" in response:
            raise ModelProviderError(f"provider returned error: {response['error']}")

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelProviderError("provider response missing choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ModelProviderError("provider choice must be an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ModelProviderError("provider choice missing message")

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first_tool_call = tool_calls[0]
            if not isinstance(first_tool_call, dict):
                raise ModelProviderError("provider tool call must be an object")

            function = first_tool_call.get("function")
            if not isinstance(function, dict):
                raise ModelProviderError("provider tool call missing function")

            name = function.get("name")
            arguments_text = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments_text, str):
                raise ModelProviderError("provider function requires name and arguments")

            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError as exc:
                raise ModelProviderError("provider tool arguments are not JSON") from exc

            if not isinstance(arguments, dict):
                raise ModelProviderError("provider tool arguments must be an object")

            reasoning_content = message.get("reasoning_content")
            if reasoning_content is not None and not isinstance(reasoning_content, str):
                raise ModelProviderError("provider reasoning_content must be text")

            parsed_calls = [
                ToolCall(
                    name=name,
                    arguments=arguments,
                    reasoning_content=reasoning_content,
                )
            ]
            for additional_tool_call in tool_calls[1:]:
                if not isinstance(additional_tool_call, dict):
                    raise ModelProviderError("provider tool call must be an object")
                additional_function = additional_tool_call.get("function")
                if not isinstance(additional_function, dict):
                    raise ModelProviderError("provider tool call missing function")
                additional_name = additional_function.get("name")
                additional_arguments_text = additional_function.get("arguments")
                if (
                    not isinstance(additional_name, str)
                    or not isinstance(additional_arguments_text, str)
                ):
                    raise ModelProviderError(
                        "provider function requires name and arguments"
                    )
                try:
                    additional_arguments = json.loads(additional_arguments_text)
                except json.JSONDecodeError as exc:
                    raise ModelProviderError(
                        "provider tool arguments are not JSON"
                    ) from exc
                if not isinstance(additional_arguments, dict):
                    raise ModelProviderError(
                        "provider tool arguments must be an object"
                    )
                parsed_calls.append(
                    ToolCall(
                        name=additional_name,
                        arguments=additional_arguments,
                    )
                )

            if len(parsed_calls) == 1:
                return parsed_calls[0]
            return ToolCallBatch(calls=tuple(parsed_calls))

        content = message.get("content")
        if isinstance(content, str):
            return content

        raise ModelProviderError("provider message has neither text nor tool call")


def _parameters_to_json_schema(parameters: object) -> dict[str, object]:
    """将项目内部的 ``{"city": "str"}`` 转成 JSON Schema。"""

    if not isinstance(parameters, dict):
        raise ModelProviderError("tool parameters must be an object")

    # 如果未来 ToolRegistry 直接提供完整 JSON Schema，这里允许它原样通过。
    if parameters.get("type") == "object":
        return parameters

    type_names = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    properties: dict[str, object] = {}
    for argument_name, argument_type in parameters.items():
        if not isinstance(argument_name, str) or not isinstance(argument_type, str):
            raise ModelProviderError("tool parameter types must be strings")
        json_type = type_names.get(argument_type)
        if json_type is None:
            raise ModelProviderError(f"unsupported tool parameter type: {argument_type}")
        properties[argument_name] = {"type": json_type}

    return {
        "type": "object",
        "properties": properties,
        "required": list(parameters),
    }


def _urllib_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> dict[str, object]:
    """使用标准库发送 JSON POST 请求，作为默认传输实现。"""

    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ModelProviderError(f"provider returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise ModelProviderError("provider connection failed") from exc
    except json.JSONDecodeError as exc:
        raise ModelProviderError("provider returned invalid JSON") from exc

    if not isinstance(decoded, dict):
        raise ModelProviderError("provider response must be an object")
    return decoded


class FakeModel:
    """按照预设顺序返回文本的模型替身。

    参数：
        replies：本次 FakeModel 运行允许返回的文本列表。

    例如：

    ``FakeModel(["第一轮回答", "第二轮回答"])``

    第一次调用 ``complete`` 返回第一条文本，第二次调用返回第二条文本。
    这正好可以模拟后续 Agent Loop 中的多轮模型交互。

    重要边界：
        回复列表耗尽后会抛出 ``RuntimeError``，而不是重复最后一条回复。
        如果自动重复结果，测试可能看起来通过，但实际上 Agent 可能已经
        多调用了一轮模型；明确失败更容易暴露控制流错误。
    """

    def __init__(self, replies: list[ModelOutput]) -> None:
        # list(replies) 创建一份独立列表。
        # 这样调用方后来修改自己的 replies 列表，不会改变 FakeModel 的脚本。
        self._replies = list(replies)

        # 这个游标表示下一次 complete 应该消费哪一条预设回复。
        # 它是 FakeModel 的运行状态，而不是模型消息的一部分。
        self._next_reply_index = 0

        # requests 用于观测 Runtime 发给模型的每一次消息列表。
        # 外部测试可以据此检查 system/user 消息是否正确组装。
        self.requests: list[list[Message]] = []

        # tool_requests 与 requests 一一对应，保存每次模型请求看到的工具
        # 目录。这样可以观察 Agent 是否把 ToolRegistry 的能力正确传给模型。
        self.tool_requests: list[list[dict[str, object]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ModelOutput:
        """记录一次模型请求，并返回下一条预设回复。

        输入：
            messages：本次发送给模型的消息列表。

            tools：当前 Runtime 允许模型使用的工具说明。直接调用 FakeModel
                时可以省略，省略会按空工具列表记录。

        输出：
            一条预设的文本回复或一个工具调用请求。

        注意：这里保存的是 ``list(messages)``，不是原始 ``messages`` 引用。
        如果直接保存原始引用，调用方之后对消息列表的修改会回头改变历史，
        导致调试轨迹和实际请求不一致。
        """

        # 只复制列表容器，不复制 Message 对象。
        # 当前 Message 使用 frozen=True，字段不能被重新绑定，因此浅拷贝
        # 已经足以保护这层“请求列表结构”。后续若消息内部包含可变内容块，
        # 再根据需要引入深拷贝或不可变内容结构。
        self.requests.append(list(messages))

        # 工具 schema 内部还有 parameters 字典，不能只复制最外层列表；
        # 深拷贝可以避免调用方后来修改 schema 影响历史请求记录。
        self.tool_requests.append(deepcopy(tools) if tools is not None else [])

        if self._next_reply_index >= len(self._replies):
            # 不允许静默重用最后一条回复，因为那会隐藏多调用模型的 bug。
            raise RuntimeError("FakeModel has no remaining replies")

        reply = self._replies[self._next_reply_index]
        self._next_reply_index += 1
        return reply
