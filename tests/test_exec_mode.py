"""Exec driver, JSON UI, and permission-rule parsing tests."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from noah_code.budget import BudgetGuard
from noah_code.config import BudgetConfig
from noah_code.events import HostEvent, HostEventKind
from noah_code.exec_mode import (
    EXIT_AGENT,
    EXIT_BUDGET,
    EXIT_CONFIG,
    EXIT_DENIED,
    EXIT_OK,
    ExecDriver,
    JsonUI,
    event_payload,
    parse_rule_spec,
    read_followup_prompts,
)


def _event(kind: HostEventKind, text: str = "", **meta) -> HostEvent:
    return HostEvent(kind, text, meta=meta)


def test_event_payload_strips_review_and_keeps_meta() -> None:
    payload = event_payload(
        _event(HostEventKind.TOOL_START, "Reading x", activity_id="a1", tool="ws", review=object())
    )
    assert payload == {"type": "tool_start", "text": "Reading x", "activity_id": "a1", "tool": "ws"}


def test_json_ui_streams_ndjson_and_records() -> None:
    stream = io.StringIO()
    ui = JsonUI(stream)
    ui.render(_event(HostEventKind.MESSAGE, "hi"))
    ui.set_status("ready")
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [item["type"] for item in lines] == ["message", "status_line"]
    assert len(ui.events) == 2


def test_mirror_text_mode_prints_without_json() -> None:
    stream = io.StringIO()
    ui = JsonUI(stream, mirror_text=True)
    ui.render(_event(HostEventKind.MESSAGE, "answer body"))
    assert "answer body" in stream.getvalue()
    assert "{" not in stream.getvalue()


@pytest.mark.asyncio
async def test_ask_approval_rejects_and_questions_return_empty() -> None:
    from noah_code.permissions import PermissionCategory, PermissionDecision

    decision = PermissionDecision(
        category=PermissionCategory.BASH,
        target="rm -rf /",
        action="ask",
        matching_rule=None,
        reason="risky",
        remember_pattern="rm -rf /",
    )
    request = SimpleNamespace(id="req-1", decision=decision)
    stream = io.StringIO()
    ui = JsonUI(stream)
    assert (await ui.ask_approval(request)).value == "reject"  # type: ignore[arg-type]
    assert (await ui.ask_questions([])).selections == []
    assert "approval_request" in stream.getvalue()


def test_followup_prompt_reading_skips_blank_lines() -> None:
    stream = io.StringIO("first\n\n  second  \n")
    assert read_followup_prompts(stream) == ["first", "second"]


class StubHost:
    """Minimal AgentHost stand-in for driver tests."""

    def __init__(self, *, guard: BudgetGuard | None = None) -> None:
        self.meta = SimpleNamespace(session_id="abc123", model="fake-model", mode="build")
        self._guard = guard
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def usage_snapshot(self):
        from noah_code.usage import UsageSnapshot

        return UsageSnapshot(
            calls=2,
            failed_calls=0,
            prompt_tokens=10,
            cached_tokens=0,
            completion_tokens=5,
            reasoning_tokens=0,
            cost_usd=0.01,
            llm_seconds=0.5,
            tool_output_chars=0,
        )

    @property
    def _budget_guard(self):
        return self._guard


def _make_driver(host: StubHost) -> ExecDriver:
    ui = JsonUI(io.StringIO())
    return ExecDriver(host, ui, output_format="json"), ui  # type: ignore[return-value]


def test_parse_rule_spec_defaults_and_validates() -> None:
    assert parse_rule_spec("edit:*", "allow") == ("edit", "*", "allow")
    assert parse_rule_spec("*:git status*", "deny") == ("*", "git status*", "deny")
    assert parse_rule_spec("bash:git push*", "deny") == ("bash", "git push*", "deny")
    with pytest.raises(ValueError, match="unknown permission category"):
        parse_rule_spec("teleport:x", "allow")
    with pytest.raises(ValueError, match="needs a pattern"):
        parse_rule_spec("bash:", "deny")


@pytest.mark.asyncio
async def test_driver_success_flow_reports_turns_and_summary() -> None:
    class EmittingHost(StubHost):
        def __init__(self) -> None:
            super().__init__()
            self.ui: JsonUI | None = None

        async def handle_line(self, line: str) -> str:
            assert self.ui is not None
            if line == "/nope":
                self.ui.render(_event(HostEventKind.ERROR, "provider exploded"))
            else:
                self.ui.render(_event(HostEventKind.MESSAGE, f"reply to {line}"))
                self.ui.render(_event(HostEventKind.STOP, "Completed · done"))
            return "continue"

    host = EmittingHost()
    ui = JsonUI(io.StringIO())
    host.ui = ui
    driver = ExecDriver(host, ui, output_format="json")  # type: ignore[arg-type]

    code = await driver.run(["explain", "/nope"])

    assert code == EXIT_AGENT
    assert host.started and host.closed
    assert driver.turn_results[0]["exit_code"] == EXIT_OK
    assert driver.turn_results[0]["response"] == "reply to explain"
    assert driver.turn_results[1]["exit_code"] == EXIT_AGENT


@pytest.mark.asyncio
async def test_driver_handle_line_exception_is_structured_failure() -> None:
    class ExplodingHost(StubHost):
        async def handle_line(self, line: str) -> str:
            raise RuntimeError("provider exploded")

    host = ExplodingHost()
    stream = io.StringIO()
    stderr = io.StringIO()
    ui = JsonUI(stream)
    driver = ExecDriver(  # type: ignore[arg-type]
        host,
        ui,
        output_format="stream-json",
        stderr=stderr,
    )

    code = await driver.run(["explain"])

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert code == EXIT_AGENT
    assert host.started and host.closed
    assert driver.turn_results[0]["exit_code"] == EXIT_AGENT
    assert driver.turn_results[0]["error"] == "provider exploded"
    assert records[-2]["type"] == "turn_result"
    assert records[-2]["error"] == "provider exploded"
    assert records[-1]["type"] == "result"
    assert records[-1]["exit_code"] == EXIT_AGENT
    assert records[-1]["error"] == "provider exploded"
    assert "exec turn 1 failed: provider exploded" in stderr.getvalue()


@pytest.mark.asyncio
async def test_driver_redacts_and_bounds_exception_messages() -> None:
    secret = "sk-1234567890abcdefghijklmnop"

    class ExplodingHost(StubHost):
        async def handle_line(self, line: str) -> str:
            raise RuntimeError(
                f"Authorization: Bearer {secret} apiKey=short-secret "
                f"password='two words secret' "
                f"https://operator:password@example.test/ {'x' * 900}"
            )

    host = ExplodingHost()
    stream = io.StringIO()
    stderr = io.StringIO()
    driver = ExecDriver(  # type: ignore[arg-type]
        host,
        JsonUI(stream),
        output_format="stream-json",
        stderr=stderr,
    )

    assert await driver.run(["explain"]) == EXIT_AGENT
    output = stream.getvalue() + stderr.getvalue()
    assert secret not in output
    assert "short-secret" not in output
    assert "two words secret" not in output
    assert "words secret" not in output
    assert "operator:password" not in output
    assert "***" in output
    assert len(driver.turn_results[0]["error"]) <= 700


@pytest.mark.asyncio
async def test_driver_redacts_host_emitted_error_events() -> None:
    class EmittingHost(StubHost):
        def __init__(self) -> None:
            super().__init__()
            self.ui: JsonUI | None = None

        async def handle_line(self, line: str) -> str:
            assert self.ui is not None
            self.ui.render(_event(HostEventKind.ERROR, "apiKey=HOST-LEAK-123456"))
            return "handled"

    host = EmittingHost()
    stream = io.StringIO()
    ui = JsonUI(stream)
    host.ui = ui
    driver = ExecDriver(host, ui, output_format="stream-json")  # type: ignore[arg-type]

    assert await driver.run(["explain"]) == EXIT_AGENT
    assert "HOST-LEAK-123456" not in stream.getvalue()
    assert "apiKey=***" in stream.getvalue()


@pytest.mark.asyncio
async def test_driver_startup_exception_closes_partial_host_and_reports_failure() -> None:
    class PartialStartupHost(StubHost):
        async def start(self) -> None:
            self.started = True
            raise RuntimeError("startup exploded")

    host = PartialStartupHost()
    stream = io.StringIO()
    ui = JsonUI(stream)
    driver = ExecDriver(  # type: ignore[arg-type]
        host,
        ui,
        output_format="stream-json",
        stderr=io.StringIO(),
    )

    code = await driver.run(["explain"])

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert code == EXIT_CONFIG
    assert host.started and host.closed
    assert records[-1]["type"] == "result"
    assert records[-1]["exit_code"] == EXIT_CONFIG
    assert records[-1]["error"] == "startup exploded"


@pytest.mark.asyncio
async def test_driver_maps_denials_to_exit_code() -> None:
    class DenyingHost(StubHost):
        def __init__(self) -> None:
            super().__init__()
            self.ui: JsonUI | None = None

        async def handle_line(self, line: str) -> str:
            assert self.ui is not None
            self.ui.render(_event(HostEventKind.ERROR, "denied [bash] git push: destructive"))
            return "handled"

    host = DenyingHost()
    ui = JsonUI(io.StringIO())
    host.ui = ui
    driver = ExecDriver(host, ui, output_format="json")  # type: ignore[arg-type]
    code = await driver.run(["try push"])
    assert code == EXIT_DENIED


@pytest.mark.asyncio
async def test_driver_budget_exceeded_exit_code() -> None:
    guard = BudgetGuard(BudgetConfig(max_tokens=1))
    guard.add_usage(prompt_tokens=100)

    class BudgetHost(StubHost):
        pass

    host = BudgetHost(guard=guard)
    ui = JsonUI(io.StringIO())
    driver = ExecDriver(host, ui, output_format="stream-json")  # type: ignore[arg-type]

    async def handle_line(line: str) -> str:
        return "continue"

    host.handle_line = handle_line  # type: ignore[method-assign]
    code = await driver.run(["one"])
    assert code == EXIT_BUDGET
    assert driver.turn_results[0]["budget_exceeded"]
