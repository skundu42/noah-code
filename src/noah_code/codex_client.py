"""NOOA model interface backed by the official account-authenticated Codex CLI.

Each call uses an ephemeral, environment-free thread. Only Noah's dynamic tools
are advertised. A dynamic tool request is returned to NOOA, and Codex is closed
before Noah applies its existing tool validation and permission checks.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from noah_code.codex_rpc import CodexAppServer, CodexError
from noah_code.model_streaming import external_model_stream


def _content_parts(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Codex message content must be text or content parts.")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Invalid Codex message content part.")
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text"} and isinstance(part.get("text"), str):
            parts.extend(_content_parts(part["text"], role))
        elif kind in {"image_url", "input_image"} and role == "user":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str):
                raise ValueError("Codex image input requires an image URL.")
            parts.append({"type": "input_image", "image_url": url})
        else:
            raise ValueError(f"Codex does not support message part {kind!r}.")
    return parts


def _prepare_messages(messages: list[dict]) -> tuple[str, list[dict], list[dict]]:
    instructions: list[str] = []
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            parts = _content_parts(message.get("content"), role)
            instructions.extend(part["text"] for part in parts)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Codex tool results require a tool_call_id.")
            content = message.get("content", "")
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
        elif role in {"assistant", "user", "developer"}:
            parts = _content_parts(message.get("content"), role)
            if parts:
                history.append({"type": "message", "role": role, "content": parts})
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                if role != "assistant" or not all(
                    isinstance(value, str) and value
                    for value in (call.get("id"), function.get("name"), function.get("arguments"))
                ):
                    raise ValueError("Invalid Codex function-call history.")
                history.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
        elif message.get("type") in {"function_call", "function_call_output", "reasoning"}:
            history.append(dict(message))
        else:
            raise ValueError(f"Codex does not support message role {role!r}.")

    # App-server starts turns from user input. When the previous item was a
    # Noah tool result, preserve it in history and explicitly ask to continue.
    user_input = [
        {"type": "text", "text": "Continue from the preceding conversation and tool results."}
    ]
    if history and history[-1].get("role") == "user":
        final = history.pop()
        user_input = [
            {"type": "text", "text": part["text"]}
            if part["type"] == "input_text"
            else {"type": "image", "url": part["image_url"]}
            for part in final["content"]
        ]
    return "\n\n".join(instructions), history, user_input


def _tool_specs(tools: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.get_parameter_schema(),
        }
        for tool in tools or []
    ]


def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    values = value.get("total")
    if not isinstance(values, dict):
        return None
    mapping = {
        "inputTokens": "prompt_tokens",
        "outputTokens": "completion_tokens",
        "totalTokens": "total_tokens",
        "cachedInputTokens": "cached_tokens",
        "reasoningOutputTokens": "reasoning_tokens",
    }
    return {
        target: values[source]
        for source, target in mapping.items()
        if isinstance(values.get(source), int)
    } or None


def _validate_options(options: dict[str, Any]) -> None:
    # NOOA supplies prompt_cache_key to every call as a cache-sharding hint;
    # Codex manages caching internally and has no equivalent protocol field.
    accepted = {"reasoning_effort", "timeout", "stream", "prompt_cache_key"}
    # App-server's native dynamic-tool selection is automatic, matching NOOA's
    # CodeAct default. Forced or disabled tool selection has no RPC equivalent.
    if options.get("tool_choice") == "auto":
        accepted.add("tool_choice")
    unsupported = {
        key for key, value in options.items() if value is not None and key not in accepted
    }
    if unsupported:
        raise ValueError(
            f"Codex account transport does not support: {', '.join(sorted(unsupported))}."
        )


class CodexClient:
    def __init__(self, model: str, **config: Any) -> None:
        if not model.startswith("codex/") or not model.removeprefix("codex/").strip():
            raise ValueError("Codex models use codex/<model-id>.")
        _validate_options(config)
        self.model = model
        self.config = config
        self.context_window: int | None = None

    def count_tokens(self, text: str) -> int:
        import litellm

        return int(
            litellm.token_counter(model="openai/" + self.model.removeprefix("codex/"), text=text)
        )

    def get_model_info(self) -> None:
        # Account usage is subscription usage, not billable OpenAI API usage.
        return None

    def call(
        self, messages: list[dict], tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall(messages, tools=tools, output_model=output_model, **kwargs)
            )
        raise RuntimeError("Use CodexClient.acall() inside an active event loop.")

    async def acall(
        self, messages: list[dict], tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        from nooa.unifiedllm import LLMResponse, ToolCall

        _validate_options(kwargs)
        instructions, history, user_input = _prepare_messages(messages)
        specs = _tool_specs(tools)
        tool_names = {spec["name"] for spec in specs}
        timeout = float(kwargs.get("timeout") or self.config.get("timeout") or 180)
        model = self.model.removeprefix("codex/")
        usage = None
        text_items: dict[str, str] = {}
        with external_model_stream(
            self.model, enabled=kwargs.get("stream", self.config.get("stream")) is not False
        ) as emit:
            async with CodexAppServer() as server:
                account = await server.request("account/read", {"refreshToken": False})
                if (
                    not isinstance(account.get("account"), dict)
                    or account["account"].get("type") != "chatgpt"
                ):
                    raise CodexError("Connect your Codex account in Noah's provider setup first.")
                started = await server.request(
                    "thread/start",
                    {
                        "model": model,
                        "modelProvider": "openai",
                        "allowProviderModelFallback": False,
                        "ephemeral": True,
                        "environments": [],
                        "runtimeWorkspaceRoots": [],
                        "cwd": server.cwd,
                        "baseInstructions": instructions,
                        "developerInstructions": "",
                        "dynamicTools": specs,
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "personality": "none",
                    },
                )
                if started.get("instructionSources"):
                    raise CodexError(
                        "Codex loaded unexpected local instructions; account request stopped."
                    )
                if started.get("model") != model:
                    raise CodexError("Codex selected a different model; account request stopped.")
                thread_id = started["thread"]["id"]
                if history:
                    await server.request(
                        "thread/inject_items", {"threadId": thread_id, "items": history}
                    )
                params: dict[str, Any] = {
                    "threadId": thread_id,
                    "input": user_input,
                    "environments": [],
                    "runtimeWorkspaceRoots": [],
                    "approvalPolicy": "never",
                }
                effort = kwargs.get("reasoning_effort", self.config.get("reasoning_effort"))
                if effort and effort != "default":
                    params["effort"] = effort
                if output_model is not None:
                    params["outputSchema"] = output_model.model_json_schema()
                turn = await server.request("turn/start", params)
                turn_id = turn["turn"]["id"]
                async with asyncio.timeout(timeout):
                    while True:
                        event = await server.next_notification(timeout=timeout)
                        event_params = event.get("params", {})
                        if event_params.get("threadId") != thread_id:
                            continue
                        method = event.get("method")
                        if method == "thread/tokenUsage/updated":
                            usage = _usage(event_params.get("tokenUsage"))
                            self.context_window = event_params.get("tokenUsage", {}).get(
                                "modelContextWindow"
                            )
                        elif method == "item/agentMessage/delta":
                            emit("text", event_params.get("delta", ""))
                        elif method in {
                            "item/reasoning/summaryTextDelta",
                            "item/reasoning/textDelta",
                        }:
                            emit("reasoning", event_params.get("delta", ""))
                        elif method == "item/tool/call":
                            name = event_params.get("tool")
                            if (
                                name not in tool_names
                                or event_params.get("namespace")
                                or event_params.get("turnId") != turn_id
                            ):
                                raise CodexError("Codex requested an unregistered Noah tool.")
                            call_id = event_params.get("callId")
                            if not isinstance(call_id, str) or not call_id:
                                raise CodexError("Codex returned an invalid tool call.")
                            arguments = json.dumps(event_params["arguments"])
                            tool_call = ToolCall(id=call_id, name=name, arguments=arguments)
                            message = {
                                "role": "assistant",
                                "content": "\n".join(text_items.values()),
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                ],
                            }
                            return LLMResponse(
                                raw_response={"model": self.model},
                                content=message["content"],
                                tool_calls=[tool_call],
                                finish_reason="tool_calls",
                                assistant_message=message,
                                usage=usage,
                            )
                        elif method in {"item/started", "item/completed"}:
                            item = event_params.get("item", {})
                            if item.get("type") == "agentMessage" and method == "item/completed":
                                text_items[item["id"]] = item["text"]
                            elif item.get("type") not in {
                                "userMessage",
                                "agentMessage",
                                "reasoning",
                                "dynamicToolCall",
                            }:
                                raise CodexError(
                                    "Codex attempted a built-in tool; account request stopped."
                                )
                        elif method == "turn/completed":
                            completed = event_params["turn"]
                            if completed.get("id") != turn_id:
                                continue
                            if completed.get("status") != "completed":
                                raise CodexError(
                                    "Codex could not complete this request; check your account limits or reconnect."
                                )
                            for item in completed.get("items", []):
                                if item.get("type") == "agentMessage":
                                    text_items[item["id"]] = item["text"]
                            text = "\n".join(text_items.values())
                            content = (
                                output_model.model_validate_json(text) if output_model else text
                            )
                            return LLMResponse(
                                raw_response={"model": self.model, "turn": completed},
                                content=content,
                                tool_calls=[],
                                finish_reason="stop",
                                assistant_message={"role": "assistant", "content": text},
                                usage=usage,
                            )
