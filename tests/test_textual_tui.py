"""Headless tests for the adaptive Textual cockpit."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.config import NoahCodeConfig
from noah_code.events import HostEvent, HostEventKind
from noah_code.permissions import PermissionDecision
from noah_code.sessions import SessionEventRecord
from noah_code.ui.textual_app import (
    MAX_TRANSCRIPT_LINES,
    ActivityHistoryScreen,
    ApprovalModal,
    ConversationHistoryScreen,
    FilteredPicker,
    NoahCodeApp,
    TextualUI,
)


def _fake_host(tmp_path: Path):
    host = MagicMock()
    host.config = NoahCodeConfig(model="fake-model")
    host.meta = MagicMock(session_id="abcd1234efgh", model="fake-model", title="t")
    host.workspace.root = tmp_path
    host._custom_commands = {}
    host._agent = MagicMock(mode="build")
    host.agent.mode = "build"
    host.agent.todos.list_todos.return_value = []
    host.handle_line = AsyncMock(return_value="continue")
    host.cancel_active_turn = MagicMock()
    host.load_history_page = AsyncMock(return_value=[])
    host.list_session_metas.return_value = []
    host.list_skill_infos.return_value = []
    host.list_mcp_infos.return_value = []
    host.list_provider_infos.return_value = []
    host.set_provider_api_key = AsyncMock()
    host.configure_provider = AsyncMock(return_value="provider configured")
    host._mcp_attached = set()
    return host


def _rendered_text(renderable) -> str:  # noqa: ANN001
    output = StringIO()
    console = Console(file=output, width=160, color_system=None)
    console.print(renderable)
    return output.getvalue()


def _log_text(log) -> str:  # noqa: ANN001
    return "\n".join(strip.text for strip in log.lines)


@pytest.mark.asyncio
async def test_tui_renders_host_events_and_header(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(HostEvent(HostEventKind.MESSAGE, "hello from agent"))
        await pilot.pause()

        assert "hello from agent" in _log_text(app.query_one("#conversation"))
        assert "fake-model" in _rendered_text(app.query_one("#header").content)
        assert app.query_one("#conversation").max_lines == MAX_TRANSCRIPT_LINES


@pytest.mark.asyncio
async def test_command_output_preserves_lines_spacing_and_literal_brackets(tmp_path: Path) -> None:
    ui = TextualUI()
    app = NoahCodeApp(_fake_host(tmp_path), ui)
    output = (
        "Available Skills (activate with self.skills.activate(['name'])):\n"
        "  nemo.context    Dict-like context API\n"
        "  nemo.events     Query past events"
    )
    async with app.run_test(size=(120, 30)) as pilot:
        ui.render(
            HostEvent(
                HostEventKind.MESSAGE,
                output,
                meta={"format": "plain", "source": "command"},
            )
        )
        await pilot.pause()

        rendered = _log_text(app.query_one("#conversation"))
        assert "Command output" in rendered
        assert "self.skills.activate(['name'])" in rendered
        assert "nemo.context    Dict-like context API" in rendered
        assert "nemo.events     Query past events" in rendered


@pytest.mark.asyncio
async def test_tui_submit_calls_host(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "Explain the repo"
        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited()
        assert "Explain the repo" in host.handle_line.await_args.args[0]


@pytest.mark.asyncio
async def test_tui_paints_before_host_start_and_queues_first_prompt(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host._agent = None
    host.meta = None
    start_gate = asyncio.Event()

    async def _start():  # noqa: ANN202
        await start_gate.wait()
        host._agent = MagicMock(mode="build")
        host.agent.mode = "build"
        host.meta = MagicMock(session_id="abcd1234efgh", model="fake-model", title="t")
        return host.meta

    host.start = AsyncMock(side_effect=_start)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        await pilot.pause()
        assert "Starting agent" in _rendered_text(app.query_one("#welcome").content)
        host.start.assert_awaited_once()

        composer = app.query_one("#composer")
        composer.text = "Run after startup"
        await pilot.press("enter")
        assert app._pending_submit == "Run after startup"
        host.handle_line.assert_not_awaited()

        start_gate.set()
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("Run after startup")


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_without_submitting(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "first line"
        composer.cursor_location = (0, len("first line"))
        await pilot.press("shift+enter")
        await pilot.pause()
        assert composer.text == "first line\n"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_tab_toggles_between_build_and_plan_modes(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)

    async def _handle(line: str) -> str:
        host.agent.mode = line.removeprefix("/mode ")
        return "handled"

    host.handle_line = AsyncMock(side_effect=_handle)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "Keep this draft"

        await pilot.press("tab")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/mode plan")
        assert host.agent.mode == "plan"
        assert composer.text == "Keep this draft"

        host.handle_line.reset_mock()
        await pilot.press("tab")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/mode build")
        assert host.agent.mode == "build"


@pytest.mark.asyncio
async def test_slash_suggestions_filter_navigate_and_complete(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        suggestions = app.query_one("#command-suggestions")

        composer.text = "/mo"
        await pilot.pause()
        rendered = _rendered_text(suggestions.content)
        assert suggestions.styles.display == "block"
        assert "/mode" in rendered
        assert "/model" in rendered
        assert "/help" not in rendered

        await pilot.press("down", "tab")
        assert composer.text == "/model "
        assert suggestions.styles.display == "none"

        composer.text = "/mode "
        await pilot.pause()
        rendered = _rendered_text(suggestions.content)
        assert "/mode build" in rendered
        assert "/mode plan" in rendered

        await pilot.press("enter")
        assert composer.text == "/mode build"
        assert suggestions.styles.display == "none"
        host.handle_line.assert_not_awaited()

        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/mode build")

        composer.text = "/config ui."
        await pilot.pause()
        rendered = _rendered_text(suggestions.content)
        assert "/config ui.theme" in rendered
        assert "atom-one-dark" in rendered

        await pilot.press("escape")
        assert suggestions.styles.display == "none"

        composer.text = "/"
        await pilot.pause()
        await pilot.press("up")
        rendered = _rendered_text(suggestions.content)
        assert "/model --global MODEL" in rendered
        assert app._suggestion_index == len(app._suggestion_matches) - 1


@pytest.mark.asyncio
async def test_skills_have_dedicated_search_and_insert_selected_skill(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_skill_infos.return_value = [
        SimpleNamespace(
            registry_name="cmd.portable-review",
            name="portable-review",
            description="Review changes with a shared rubric",
            source=str(tmp_path / ".agents" / "skills" / "portable-review"),
            active=False,
            document_skill=True,
        ),
        SimpleNamespace(
            registry_name="cmd.release-notes",
            name="release-notes",
            description="Prepare release notes",
            source=str(tmp_path / ".claude" / "skills" / "release-notes"),
            active=False,
            document_skill=True,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        picker = app.screen
        search = picker.query_one("#picker-filter")
        search.value = "review"
        await pilot.pause()

        assert [row[1] for row in picker._filtered] == ["$portable-review"]
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one("#composer").text == "$portable-review "


@pytest.mark.asyncio
async def test_exact_skills_command_opens_picker_instead_of_printing_chat(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/skills"
        await pilot.pause()
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_has_dedicated_searchable_connection_picker(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_mcp_infos.return_value = [
        SimpleNamespace(
            name="github",
            transport="streamable-http",
            target="https://example.com/mcp",
            source=str(tmp_path / ".mcp.json"),
        )
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        app.action_mcp()
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        picker = app.screen
        search = picker.query_one("#picker-filter")
        search.value = "github"
        await pilot.pause()

        assert [row[1] for row in picker._filtered] == ["github"]


@pytest.mark.asyncio
async def test_providers_have_searchable_secret_free_setup(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_provider_infos.return_value = [
        SimpleNamespace(
            key="openai",
            label="OpenAI",
            description="OpenAI API models",
            model_hint="openai/MODEL_NAME",
            credential_hint="OPENAI_API_KEY",
            configured=True,
            active=False,
        ),
        SimpleNamespace(
            key="anthropic",
            label="Anthropic Claude",
            description="Claude API models",
            model_hint="anthropic/MODEL_NAME",
            credential_hint="ANTHROPIC_API_KEY",
            configured=False,
            active=False,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        app.action_providers()
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        picker = app.screen
        picker.query_one("#picker-filter").value = "openai"
        await pilot.pause()
        assert [row[1] for row in picker._filtered][:2] == [
            "OpenAI",
            "+ Custom OpenAI-compatible",
        ]

        await pilot.press("enter")
        await pilot.pause()
        model_input = app.screen.query_one("#prompt-input")
        model_input.value = "example-model"
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.configure_provider.assert_awaited_once_with("openai", "example-model")


@pytest.mark.asyncio
async def test_exact_model_command_runs_masked_provider_key_model_setup(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_provider_infos.return_value = [
        SimpleNamespace(
            key="openai",
            label="OpenAI",
            description="OpenAI API models",
            model_hint="openai/MODEL_NAME",
            credential_hint="OPENAI_API_KEY",
            configured=False,
            active=False,
        )
    ]
    host.set_provider_api_key.return_value = SimpleNamespace(
        message="OPENAI_API_KEY saved in the OS credential store"
    )
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/model"
        await pilot.pause()
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        assert "1 OF 3" in app.screen.query_one("#picker-title").render().plain
        await pilot.press("enter")
        await pilot.pause()

        key_input = app.screen.query_one("#prompt-input")
        assert key_input.password is True
        key_input.value = "never-render-this-key"
        await pilot.press("enter")
        for _ in range(20):
            if host.set_provider_api_key.await_count:
                break
            await pilot.pause()

        model_input = app.screen.query_one("#prompt-input")
        assert model_input.password is False
        model_input.value = "example-model"
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.set_provider_api_key.assert_awaited_once_with("openai", "never-render-this-key")
        host.configure_provider.assert_awaited_once_with("openai", "example-model")
        assert "never-render-this-key" not in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_model_setup_recovers_a_missing_credential_startup_failure(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host._agent = None
    host.list_provider_infos.return_value = [
        SimpleNamespace(
            key="openai",
            label="OpenAI",
            description="OpenAI API models",
            model_hint="openai/MODEL_NAME",
            credential_hint="OPENAI_API_KEY",
            configured=False,
            active=False,
        )
    ]
    host.set_provider_api_key.return_value = SimpleNamespace(message="credential saved")
    attempts = 0

    async def _start():  # noqa: ANN202
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("OPENAI_API_KEY is missing")
        host._agent = MagicMock(mode="build")
        host.agent.mode = "build"
        return host.meta

    host.start = AsyncMock(side_effect=_start)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        for _ in range(20):
            if app._phase == "startup failed":
                break
            await pilot.pause()
        assert "Type /model" in _log_text(app.query_one("#conversation"))

        composer = app.query_one("#composer")
        composer.text = "/model"
        await pilot.pause()
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app.screen.query_one("#prompt-input").value = "replacement-key"
        await pilot.press("enter")
        for _ in range(20):
            if host.set_provider_api_key.await_count:
                break
            await pilot.pause()
        app.screen.query_one("#prompt-input").value = "example-model"
        await pilot.press("enter")

        for _ in range(40):
            if app._agent_ready:
                break
            await pilot.pause()
        assert app._agent_ready is True
        assert host.start.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "wide", "compact"),
    [
        ((80, 24), False, True),
        ((109, 30), False, False),
        ((110, 30), True, False),
        ((140, 40), True, False),
    ],
)
async def test_adaptive_layout_breakpoints(
    tmp_path: Path,
    size: tuple[int, int],
    wide: bool,
    compact: bool,
) -> None:
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())
    async with app.run_test(size=size):
        assert app.screen.has_class("wide") is wide
        assert app.screen.has_class("compact") is compact
        expected = "block" if wide else "none"
        assert app.query_one("#context-rail").styles.display == expected


@pytest.mark.asyncio
async def test_activity_streams_live_then_compacts(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "execute_python: run tests",
                meta={"activity_id": "tool-1", "tool": "execute_python"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.SHELL_CHUNK,
                "one\ntwo\n",
                meta={"activity_id": "tool-1", "stream": "stdout"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "code cell complete",
                meta={"activity_id": "tool-1", "result_status": "complete"},
            )
        )
        await pilot.pause(0.08)

        assert app.query_one("#live-activity").styles.display == "none"
        assert len(app._activity_history) == 1
        assert app._activity_history[0].output == "one\ntwo\n"
        transcript = _log_text(app.query_one("#conversation"))
        assert "execute_python: run tests" in transcript
        assert "2 lines" in transcript

        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ActivityHistoryScreen)
        assert "one" in _log_text(app.screen.query_one("#activity-detail"))


@pytest.mark.asyncio
async def test_event_burst_uses_one_drain_and_bounded_writes(tmp_path: Path, monkeypatch) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        output = app.query_one("#activity-output")
        wrapped_write = MagicMock(wraps=output.write)
        monkeypatch.setattr(output, "write", wrapped_write)

        for index in range(2_000):
            ui.render(
                HostEvent(
                    HostEventKind.SHELL_CHUNK,
                    f"chunk {index}\n",
                    meta={"stream": "stdout"},
                )
            )
        assert ui._event_pending is True
        assert len(ui._events) == 2_000

        await pilot.pause(0.1)

        assert ui._event_pending is False
        assert not ui._events
        assert wrapped_write.call_count <= 2


@pytest.mark.asyncio
async def test_transcript_treats_rich_markup_as_text(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "[bold red]not markup[/]"
        await pilot.press("enter")
        await pilot.pause()
        assert "[bold red]not markup[/]" in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_scrolled_transcript_counts_new_output_until_end(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(80, 20)) as pilot:
        for index in range(60):
            ui.render(HostEvent(HostEventKind.MESSAGE, f"message {index}"))
        await pilot.pause()
        log = app.query_one("#conversation")
        assert app._unread_count == 0

        log.scroll_home(animate=False)
        await pilot.pause()
        ui.render(HostEvent(HostEventKind.MESSAGE, "new while reading"))
        await pilot.pause()

        assert app._unread_count == 1
        assert log.is_vertical_scroll_end is False
        await pilot.press("end")
        await pilot.pause()
        assert app._unread_count == 0
        assert log.is_vertical_scroll_end is True


@pytest.mark.asyncio
async def test_conversation_history_loads_persisted_events(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.load_history_page.return_value = [
        SessionEventRecord(1, "event-1", "Task", {"prompt": "prior question"}),
        SessionEventRecord(2, "event-2", "Message", {"content": "prior answer"}),
    ]
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = _log_text(app.query_one("#conversation"))
        assert "prior question" in transcript
        assert "prior answer" in transcript

        await pilot.press("f3")
        await pilot.pause()
        assert isinstance(app.screen, ConversationHistoryScreen)


@pytest.mark.asyncio
async def test_approval_modal_reject_is_safe_default(tmp_path: Path) -> None:
    decision = PermissionDecision(
        category="edit",
        target="[bold]a.py[/bold]",
        action="ask",
        matching_rule=None,
        reason="needs approval",
        remember_pattern="*.py",
    )
    request = ApprovalRequest(id="req-1", decision=decision, created_at=0.0, future=MagicMock())
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    result_box: list[ApprovalChoice] = []

    async with app.run_test() as pilot:

        async def _ask() -> None:
            result_box.append(await app.push_screen_wait(ApprovalModal(request)))

        app.run_worker(_ask)
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one("#reject")
        body = _rendered_text(app.screen.query_one("#approval-body").content)
        assert "[bold]a.py[/bold]" in body
        await pilot.press("1")
        for _ in range(40):
            if result_box:
                break
            await pilot.pause()
        assert result_box == [ApprovalChoice.ONCE]
