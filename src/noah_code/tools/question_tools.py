"""Ask the user structured questions mid-turn."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

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
    ) -> None:
        super().__init__()
        self._engine = engine
        self._approvals = approvals
        self._handler = handler

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
        answer = await self._handler([item])
        chosen = answer.selections or []
        custom = answer.custom.strip()
        parts = [
            f"Q: {item.header}",
            item.prompt,
            "A: " + "; ".join([*chosen, *([custom] if custom else [])]),
        ]
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
