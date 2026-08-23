"""Ask the user structured questions mid-turn."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nooa import Skill

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine


@dataclass(frozen=True)
class QuestionPrompt:
    header: str
    prompt: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class QuestionAnswer:
    selections: list[str]
    custom: str = ""


QuestionHandler = Callable[[list[QuestionPrompt]], Awaitable[QuestionAnswer]]


class QuestionTools(Skill):
    """Pause the turn and collect a structured choice from the user."""

    def __init__(
        self,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        handler: QuestionHandler | None = None,
        runtime: Any = None,
        timeout_seconds: float = 86_400.0,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._approvals = approvals
        self._handler = handler
        self._runtime = runtime
        self._timeout_seconds = timeout_seconds
        self._ui_lock = asyncio.Lock()

    def set_handler(self, handler: QuestionHandler | None) -> None:
        self._handler = handler

    async def question(self, header: str, prompt: str, options: list[str]) -> str:
        """Ask one multiple-choice question and return the user's answer."""

        cleaned = [item.strip() for item in options if str(item).strip()]
        if not cleaned:
            raise ValueError("question requires at least one option")
        await self._approvals.require(
            self._engine.decide(
                PermissionCategory.QUESTION,
                header.strip() or prompt.strip(),
                tool="question",
            )
        )
        item = QuestionPrompt(
            header=header.strip() or "Question",
            prompt=prompt.strip(),
            options=tuple(cleaned),
        )
        if self._handler is None:
            raise PermissionError("question tool has no UI handler")
        runtime = self._runtime
        interaction_id = ""
        if runtime is not None:
            interaction_id = runtime.begin_interaction(
                "question",
                {
                    "header": item.header,
                    "prompt": item.prompt,
                    "options": list(item.options),
                },
            )
        try:
            async with self._ui_lock:
                answer = await asyncio.wait_for(
                    self._handler([item]),
                    timeout=self._timeout_seconds,
                )
        except asyncio.CancelledError:
            if interaction_id:
                assert runtime is not None
                runtime.resolve_interaction(interaction_id, "cancelled", state="cancelled")
            raise
        except TimeoutError as exc:
            if interaction_id:
                assert runtime is not None
                runtime.resolve_interaction(interaction_id, "timeout", state="timed_out")
            raise TimeoutError("user question timed out") from exc
        except Exception as exc:
            if interaction_id:
                assert runtime is not None
                runtime.resolve_interaction(interaction_id, str(exc), state="error")
            raise
        chosen = answer.selections or []
        custom = answer.custom.strip()
        parts = [
            f"Q: {item.header}",
            item.prompt,
            "A: " + "; ".join([*chosen, *([custom] if custom else [])]),
        ]
        if interaction_id:
            assert runtime is not None
            runtime.resolve_interaction(
                interaction_id,
                {"selections": chosen, "custom": custom},
            )
        return "\n".join(part for part in parts if part).strip()


async def console_question_handler(
    prompts: list[QuestionPrompt],
    *,
    printer: Callable[[str], None] | None = None,
    reader: Callable[[str], Awaitable[str]] | None = None,
) -> QuestionAnswer:
    """Numbered-option fallback for the line-oriented console."""

    emit = printer or print

    async def _read(label: str) -> str:
        if reader is not None:
            return await reader(label)
        return await asyncio.to_thread(input, label)

    selections: list[str] = []
    custom_bits: list[str] = []
    for item in prompts:
        emit(f"{item.header}: {item.prompt}")
        for index, option in enumerate(item.options, start=1):
            emit(f"  [{index}] {option}")
        emit("  [0] other")
        raw = (await _read("answer> ")).strip()
        if raw in {"0", "other", "o"}:
            custom_bits.append((await _read("custom> ")).strip())
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(item.options):
            selections.append(item.options[int(raw) - 1])
            continue
        if raw:
            custom_bits.append(raw)
    return QuestionAnswer(
        selections=selections, custom=" ".join(part for part in custom_bits if part)
    )
