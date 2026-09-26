"""Normalize CodeAct prose replies and recover bounded output truncation."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from nooa.unifiedllm import LLMResponse, ToolCall


def _codeact_session(tools: Any) -> bool:
    return any(getattr(tool, "name", "") == "execute_python" for tool in tools or [])


def _reply_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if content is None or hasattr(content, "model_dump"):
        return ""
    return str(content).strip()


def recover_empty_truncated_response(response: Any) -> Any:
    """Let CodeAct's existing bounded continuation path handle empty truncation.

    NOOA otherwise aborts immediately when reasoning consumes the output limit.
    Keep ``length`` so this notice cannot become DONE: its synthetic-comment
    route counts an iteration and stops after repeated incomplete responses.
    There is no extra provider call here; the session and budget wrappers retain
    control of every continuation. Preserve the provider payload and accounting,
    and label the replacement content so persisted traces cannot mistake it for
    model output.
    """
    content = getattr(response, "content", None)
    if (
        getattr(response, "finish_reason", None) != "length"
        or getattr(response, "tool_calls", None)
        or (content is not None and (not isinstance(content, str) or content.strip()))
    ):
        return response
    notice = (
        "[Noah host recovery notice; not model output]\n"
        "The previous response reached its output limit without any answer or tool call. "
        "Continue the same task with shorter reasoning and one concrete execute_python "
        "action within the unchanged output limit. The task is not complete.\n"
        f"Original response: finish_reason=length; content={json.dumps(content)}; tool_calls=[]"
    )
    return LLMResponse(
        raw_response=getattr(response, "raw_response", None),
        content=notice,
        tool_calls=getattr(response, "tool_calls", []),
        finish_reason="length",
        assistant_message=getattr(response, "assistant_message", {}),
        reasoning=getattr(response, "reasoning", None),
        usage=getattr(response, "usage", None),
    )


def coerce_text_only_response(response: Any) -> Any:
    """Rewrite a bare assistant message as ``self.message`` + ``return_result``."""

    # A token-limited reply is incomplete. Let CodeAct request continuation
    # instead of silently reporting a partial answer as successfully finished.
    if getattr(response, "tool_calls", None) or getattr(response, "finish_reason", None) != "stop":
        return response
    text = _reply_text(response)
    if not text:
        return response
    explanation = " ".join(text.split())[:80] or "answered"
    code = (
        f"self.message({text!r})\n"
        f"return_result(RespondReason.DONE, explanation={explanation!r})"
    )
    return LLMResponse(
        raw_response=getattr(response, "raw_response", None),
        content="",
        tool_calls=[
            ToolCall(
                id=f"reply-{uuid4().hex[:8]}",
                name="execute_python",
                arguments=json.dumps({"code": code}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
        reasoning=getattr(response, "reasoning", None),
        usage=getattr(response, "usage", None),
    )


class ConversationalReplyLLM:
    """Wrap a UnifiedLLM so text-only CodeAct turns still answer the user."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def _coerce(self, response: Any, tools: Any) -> Any:
        if not _codeact_session(tools):
            return response
        response = recover_empty_truncated_response(response)
        return coerce_text_only_response(response)

    async def acall(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        response = await self._inner.acall(
            messages, tools=tools, output_model=output_model, **kwargs
        )
        return self._coerce(response, tools)

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        response = self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)
        return self._coerce(response, tools)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def get_model_info(self) -> Any:
        return self._inner.get_model_info()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_conversational_replies(client: Any) -> Any:
    """Identity when already wrapped; otherwise add the CodeAct reply shim."""

    if client is None:
        return client
    seen: set[int] = set()
    current = client
    while current is not None and id(current) not in seen:
        if isinstance(current, ConversationalReplyLLM):
            return client
        seen.add(id(current))
        current = getattr(current, "_inner", None)
    return ConversationalReplyLLM(client)
