"""Normalize CodeAct prose replies and recover bounded output truncation."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, ToolCall
from nooa.unifiedllm.unifiedllm import UnifiedLLM

from noah_code import nooa_compat

_RECOVERY_HEADER = "[Noah host recovery notice; not model output]"


def _codeact_session(tools: Any) -> bool:
    return any(getattr(tool, "name", "") == "execute_python" for tool in tools or [])


def _reply_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if content is None or hasattr(content, "model_dump"):
        return ""
    return str(content).strip()


def _reasoning_parts(response: Any) -> tuple[AssistantReasoning, ...]:
    parts = getattr(response, "parts", None)
    if isinstance(parts, tuple | list):
        kept = tuple(part for part in parts if isinstance(part, AssistantReasoning))
        if kept:
            return kept
    reasoning = getattr(response, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return (AssistantReasoning(text=reasoning),)
    return ()


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
        f"{_RECOVERY_HEADER}\n"
        "The previous response reached its output limit without any answer or tool call. "
        "Continue the same task with shorter reasoning and one concrete execute_python "
        "action within the unchanged output limit. The task is not complete.\n"
        f"Original response: finish_reason=length; content={json.dumps(content)}; tool_calls=[]"
    )
    reasoning_parts = _reasoning_parts(response)
    return LLMResponse(
        raw_response=getattr(response, "raw_response", None),
        parts=(*reasoning_parts, AssistantText(text=notice)),
        replay_scope=getattr(response, "replay_scope", None),
        finish_reason="length",
        usage=getattr(response, "usage", None),
        model_name=getattr(response, "model_name", "") or "",
        generation_id=getattr(response, "generation_id", "") or "",
    )


def recover_truncated_prose_response(response: Any) -> Any:
    """Preserve partial truncated prose and append an explicit max-tokens notice."""

    content = getattr(response, "content", None)
    if (
        getattr(response, "finish_reason", None) != "length"
        or getattr(response, "tool_calls", None)
        or not isinstance(content, str)
        or not content.strip()
        or _RECOVERY_HEADER in content
    ):
        return response
    notice = (
        f"{content.rstrip()}\n\n"
        f"{_RECOVERY_HEADER}\n"
        "The previous response hit its output token limit (finish_reason=length) "
        "without completing a tool call. "
        "Do not repeat the partial text above; continue concisely with one concrete "
        "execute_python action (or self.message(...) and return_result(...) if ready "
        "to finish) within the unchanged output limit. The task is not complete."
    )
    reasoning_parts = _reasoning_parts(response)
    return LLMResponse(
        raw_response=getattr(response, "raw_response", None),
        parts=(*reasoning_parts, AssistantText(text=notice)),
        replay_scope=getattr(response, "replay_scope", None),
        finish_reason="length",
        usage=getattr(response, "usage", None),
        model_name=getattr(response, "model_name", "") or "",
        generation_id=getattr(response, "generation_id", "") or "",
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
        f"return_result(Done(explanation={explanation!r}))"
    )
    reasoning_parts = _reasoning_parts(response)
    return LLMResponse(
        raw_response=getattr(response, "raw_response", None),
        parts=(
            *reasoning_parts,
            ToolCall(
                id=f"reply-{uuid4().hex[:8]}",
                name="execute_python",
                arguments=json.dumps({"code": code}),
            ),
        ),
        replay_scope=getattr(response, "replay_scope", None),
        finish_reason="tool_calls",
        usage=getattr(response, "usage", None),
        model_name=getattr(response, "model_name", "") or "",
        generation_id=getattr(response, "generation_id", "") or "",
    )


class ConversationalReplyLLM(UnifiedLLM):
    """Wrap a UnifiedLLM so text-only CodeAct turns still answer the user."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        nooa_compat.init_llm_wrapper(self, inner)

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def context_window(self) -> Any:
        return getattr(self._inner, "context_window", None)

    def get_context_limits(
        self, overrides: dict[str, Any] | None = None, *, fallback_reserve: int = 0
    ) -> Any:
        get_limits = getattr(self._inner, "get_context_limits", None)
        if callable(get_limits):
            return get_limits(overrides, fallback_reserve=fallback_reserve)
        return super().get_context_limits(overrides, fallback_reserve=fallback_reserve)

    def _coerce(self, response: Any, tools: Any) -> Any:
        if not _codeact_session(tools) or nooa_compat.in_summary_fork():
            return response
        truncated_without_tools = (
            getattr(response, "finish_reason", None) == "length"
            and not getattr(response, "tool_calls", None)
        )
        response = recover_empty_truncated_response(response)
        response = recover_truncated_prose_response(response)
        if truncated_without_tools:
            return (
                response.model_copy(update={"finish_reason": "stop"})
                if hasattr(response, "model_copy")
                else response
            )
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

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()
        else:
            await super().aclose()

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
        else:
            super().close()

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
