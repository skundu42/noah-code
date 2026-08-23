"""Turn CodeAct prose replies into a visible answer plus DONE."""

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


def coerce_text_only_response(response: Any) -> Any:
    """Rewrite a bare assistant message as ``self.message`` + ``return_result``."""

    if getattr(response, "tool_calls", None):
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
