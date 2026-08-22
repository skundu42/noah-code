"""Console UI rendering and approval parsing tests."""

from __future__ import annotations

import io
import uuid
from typing import Any

import pytest
from rich.console import Console

from noah_code.events import HostEvent, HostEventKind
from noah_code.permissions import PermissionCategory, PermissionDecision
from noah_code.themes import THEME_NAMES
from noah_code.ui.console import ConsoleUI
from noah_code.ui.protocol import HostUI


def _make_ui(markdown: bool = True) -> tuple[ConsoleUI, io.StringIO]:
    buffer = io.StringIO()
    ui = ConsoleUI(markdown=markdown, file=buffer)
    ui.console = Console(file=buffer, width=100, color_system=None)
    return ui, buffer


def _decision(action: str = "ask") -> PermissionDecision:
    return PermissionDecision(
        category=PermissionCategory.BASH,
        target="pytest -q",
        action=action,  # type: ignore[arg-type]
        matching_rule=None,
        reason="shell requires approval",
        remember_pattern="pytest -q",
    )


def _request() -> Any:  # noqa: ANN401 - test helper keeps typing local
    from noah_code.approvals import ApprovalRequest

    return ApprovalRequest(
        id=str(uuid.uuid4()),
        decision=_decision(),
        created_at=0.0,
        future=None,  # type: ignore[arg-type]
    )



@pytest.mark.parametrize(
    ("kind", "needle"),
    [
        (HostEventKind.MESSAGE, "model answer"),
        (HostEventKind.REASONING, "weighing options"),
        (HostEventKind.TOOL_START, "Reading src/a.py"),
        (HostEventKind.TOOL_FINISH, "complete"),
        (HostEventKind.SHELL_CHUNK, "1 passed"),
        (HostEventKind.ERROR, "boom"),
        (HostEventKind.STATUS, "mode set to plan"),
        (HostEventKind.STOP, "Completed"),
        (HostEventKind.DIFF_REVIEW, "Changes · 1"),
    ],
)
def test_render_handles_every_host_event_kind(kind: HostEventKind, needle: str) -> None:
    ui, buffer = _make_ui()
    meta = {"stream": "stdout"} if kind == HostEventKind.SHELL_CHUNK else {}
    ui.render(HostEvent(kind, needle, meta=meta))
    assert needle in buffer.getvalue()


def test_render_plain_format_skips_markdown() -> None:
    ui, buffer = _make_ui(markdown=True)
    ui.render(
        HostEvent(
            HostEventKind.MESSAGE,
            "**bold** claim",
            meta={"format": "plain", "source": "command"},
        )
    )
    assert "**bold** claim" in buffer.getvalue()


def test_shell_chunk_stderr_is_styled_not_dropped() -> None:
    ui, buffer = _make_ui()
    ui.render(HostEvent(HostEventKind.SHELL_CHUNK, "traceback here\n", meta={"stream": "stderr"}))
    assert "traceback here" in buffer.getvalue()


@pytest.mark.asyncio
async def test_ask_approval_parses_once_session_reject_and_invalid(monkeypatch) -> None:
    ui, buffer = _make_ui()

    answers = iter(["bogus", "y"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    choice = await ui.ask_approval(_request())
    assert choice.value == "once"
    assert "Enter 1/2/3" in buffer.getvalue()

    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "s")
    assert (await ui.ask_approval(_request())).value == "session"

    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "3")
    assert (await ui.ask_approval(_request())).value == "reject"


@pytest.mark.asyncio
async def test_ask_approval_rejects_on_eof(monkeypatch) -> None:
    ui, _buffer = _make_ui()

    def _raise_eof(*_a, **_k):  # noqa: ANN002, ANN003
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert (await ui.ask_approval(_request())).value == "reject"


@pytest.mark.asyncio
async def test_prompt_returns_none_on_eof(monkeypatch) -> None:
    ui, _buffer = _make_ui()

    def _raise_eof(*_a, **_k):  # noqa: ANN002, ANN003
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert await ui.prompt("noah> ") is None


@pytest.mark.asyncio
async def test_prompt_returns_line(monkeypatch) -> None:
    ui, _buffer = _make_ui()
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "fix the test")
    assert await ui.prompt("noah> ") == "fix the test"


def test_console_ui_satisfies_hostui_protocol() -> None:
    ui, _buffer = _make_ui()
    assert isinstance(ui, HostUI)


def test_theme_names_are_non_empty() -> None:
    assert len(THEME_NAMES) >= 4
