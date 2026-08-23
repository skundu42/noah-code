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
from noah_code.steer import SteerQueue
from noah_code.ui.textual_app import (
    MAX_TRANSCRIPT_LINES,
    ActivityHistoryScreen,
    ApprovalModal,
    ConversationHistoryScreen,
    DiffReviewScreen,
    FilteredPicker,
    NoahCodeApp,
    TextualUI,
    _coalesce_activity_text,
    _completed_activity_label,
    _normalize_markdown,
    _record_to_entries,
)
from noah_code.updates import UpdateStatus


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
    host.steer_queue = SteerQueue()
    host._pending_attach_paths = []

    def take_pending_attaches():
        paths, host._pending_attach_paths = host._pending_attach_paths, []
        return paths

    def enqueue_steer(text, attach_paths=None):
        paths = list(attach_paths or []) + take_pending_attaches()
        return host.steer_queue.push(text, attach_paths=paths or None)

    host.take_pending_attaches = take_pending_attaches
    host.enqueue_steer = enqueue_steer
    return host


@pytest.fixture(autouse=True)
def _disable_live_update_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        "noah_code.ui.textual_app.maybe_check_for_update",
        lambda **_kwargs: None,
    )


def _rendered_text(renderable) -> str:
    output = StringIO()
    console = Console(file=output, width=160, color_system=None)
    console.print(renderable)
    return output.getvalue()


def _log_text(log) -> str:
    return "\n".join(strip.text for strip in log.lines)


def test_markdown_normalization_repairs_indented_model_output() -> None:
    raw = (
        "# Noah Code\n\n"
        "        **Noah Code** is a coding agent.\n\n"
        "        ### Capabilities\n\n"
        "        - **Inspect** repositories"
    )

    assert _normalize_markdown(raw) == (
        "# Noah Code\n\n"
        "**Noah Code** is a coding agent.\n\n"
        "### Capabilities\n\n"
        "- **Inspect** repositories"
    )


def test_markdown_normalization_unwraps_redundant_markdown_fence() -> None:
    assert _normalize_markdown("```markdown\n## Result\n\n**Passed**\n```") == (
        "## Result\n\n**Passed**"
    )


def test_persisted_python_activity_uses_human_label() -> None:
    record = SessionEventRecord(
        1,
        "event-1",
        "ToolCallEvent",
        {
            "name": "execute_python",
            "arguments": {"code": 'await self.ws.run("pytest -q")'},
            "result": {"result_status": "complete"},
        },
    )

    entries = _record_to_entries(record)

    assert [entry.text for entry in entries] == ["✓ Bash pytest -q"]
    assert "execute_python" not in entries[0].text


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
async def test_diff_event_opens_change_ledger_with_validation_and_patch(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.agent.lsp.document_symbols = AsyncMock(return_value="a.py:1  function changed")
    review = SimpleNamespace(
        files=[
            SimpleNamespace(
                key="unstaged:a.py",
                path="a.py",
                scope="unstaged",
                status="modified",
                additions=1,
                deletions=1,
                diagnostics="clean",
                patch="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
            )
        ],
        additions=1,
        deletions=1,
    )
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(120, 40)) as pilot:
        ui.render(HostEvent(HostEventKind.DIFF_REVIEW, "changes", meta={"review": review}))
        await pilot.pause()

        assert isinstance(app.screen, DiffReviewScreen)
        assert "a.py" in _rendered_text(app.screen.query_one("#diff-file-header").content)
        assert "+new" in _log_text(app.screen.query_one("#diff-patch"))
        assert "clean" in _rendered_text(app.screen.query_one("#diff-validation").content)
        assert "changed" in _rendered_text(app.screen.query_one("#diff-validation").content)


@pytest.mark.asyncio
async def test_active_context_rail_shows_semantic_tool_state_not_code(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(120, 30)) as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Bash pytest -q",
                meta={"activity_id": "tool-1", "tool": "execute_python"},
            )
        )
        await pilot.pause()

        rail = _rendered_text(app.query_one("#context-rail").content)
        assert "Running\nBash pytest -q" in rail
        assert "result = await" not in rail

        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "code cell success",
                meta={"activity_id": "tool-1", "result_status": "success"},
            )
        )
        ui.render(HostEvent(HostEventKind.STOP, "Completed · tests passed"))
        await pilot.pause()
        assert "ready" in _rendered_text(app.query_one("#header").content)
        transcript = _log_text(app.query_one("#conversation"))
        assert "Completed · tests passed" in transcript
        assert "\n  ·\n" not in transcript


@pytest.mark.asyncio
async def test_busy_banner_is_obvious_and_internal_cells_do_not_clutter_chat(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(120, 30)) as pilot:
        ui.set_busy(True)
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Inspecting repository",
                meta={"activity_id": "tool-1", "tool": "execute_python"},
            )
        )
        await pilot.pause()

        banner = app.query_one("#working-banner")
        assert banner.styles.display == "block"
        banner_text = _rendered_text(banner.content)
        assert "WORKING" not in banner_text
        assert "Inspecting repository" in banner_text
        assert any(frame in banner_text for frame in "◐◓◑◒")
        assert banner.styles.height.value == 1
        live = app.query_one("#live-activity")
        assert live.styles.display == "block"
        title = _rendered_text(app.query_one("#activity-title").content)
        assert "Inspecting repository" in title

        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "complete",
                meta={"activity_id": "tool-1", "result_status": "complete"},
            )
        )
        await pilot.pause()
        transcript = _log_text(app.query_one("#conversation"))
        assert "✓ Inspect" in transcript
        assert "execute_python" not in transcript
        assert "Activity" not in transcript

        ui.set_busy(False)
        await pilot.pause()
        assert banner.styles.display == "none"


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

    async def _start():
        await start_gate.wait()
        host._agent = MagicMock(mode="build")
        host.agent.mode = "build"
        host.meta = MagicMock(session_id="abcd1234efgh", model="fake-model", title="t")
        return host.meta

    host.start = AsyncMock(side_effect=_start)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        await pilot.pause()
        welcome = _rendered_text(app.query_one("#welcome").content)
        assert "NOAH" in welcome
        assert "Starting agent" not in welcome
        assert "Starting" in _rendered_text(app.query_one("#context-rail").content)
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
async def test_first_run_opens_model_setup_before_starting_agent(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host._agent = None
    host.meta = None
    host.set_provider_api_key.return_value = SimpleNamespace(message="credential saved")

    async def _start():
        host._agent = MagicMock(mode="build")
        host.agent.mode = "build"
        host.meta = MagicMock(
            session_id="abcd1234efgh",
            model="openai/example-model",
            title="untitled",
        )
        return host.meta

    host.start = AsyncMock(side_effect=_start)
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
    app = NoahCodeApp(host, TextualUI(), onboarding_required=True)

    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(20):
            if isinstance(app.screen, FilteredPicker):
                break
            await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        assert "MODEL SETUP" in app.screen.query_one("#picker-title").render().plain
        host.start.assert_not_awaited()
        assert app.query_one("#welcome").styles.display == "block"
        rail = _rendered_text(app.query_one("#context-rail").content)
        assert "Choose a model" in rail

        await pilot.press("enter")
        await pilot.pause()
        app.screen.query_one("#prompt-input").value = "first-run-key"
        await pilot.press("enter")
        for _ in range(20):
            if host.set_provider_api_key.await_count:
                break
            await pilot.pause()
        app.screen.query_one("#prompt-input").value = "example-model"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(40):
            if app._agent_ready:
                break
            await pilot.pause()
        assert app._agent_ready is True
        host.start.assert_awaited_once()
        assert app.query_one("#welcome").styles.display == "block"


@pytest.mark.asyncio
async def test_available_update_uses_temporary_banner_and_persistent_rail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "noah_code.ui.textual_app.maybe_check_for_update",
        lambda **_kwargs: UpdateStatus(current="0.2.1", latest="0.3.0"),
    )
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())

    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(20):
            if app._available_update is not None:
                break
            await pilot.pause()

        banner = app.query_one("#notice-banner")
        assert banner.styles.display == "block"
        assert "0.2.1 → 0.3.0" in _rendered_text(banner.content)
        assert "noah update" in _rendered_text(banner.content)
        rail = _rendered_text(app.query_one("#context-rail").content)
        assert "UPDATE" in rail
        assert "0.2.1 → 0.3.0" in rail


@pytest.mark.asyncio
async def test_theme_picker_applies_and_persists_theme(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("noah_code.config._user_config_path", lambda: config_path)
    app = NoahCodeApp(_fake_host(tmp_path), TextualUI())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/theme"
        await pilot.pause()
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)

        await pilot.press("down", "enter")
        for _ in range(20):
            if app._theme_name == "noah-ocean":
                break
            await pilot.pause()

        assert app._theme_name == "noah-ocean"
        assert composer.theme == "noah-ocean"
        assert app.get_css_variables()["nc-canvas"] == "#07151d"
        assert 'theme = "noah-ocean"' in config_path.read_text()


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
async def test_at_mention_suggestions_complete_in_place(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.py").write_text("x = 1\n")
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "Fix @src/par"
        await pilot.pause()
        rendered = _rendered_text(app.query_one("#command-suggestions").content)
        assert "@src/parser.py" in rendered
        await pilot.press("enter")
        assert "Fix @src/parser.py" in composer.text
        host.handle_line.assert_not_awaited()


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
async def test_slash_suggestion_selection_remains_visible_after_first_page(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(160, 50)) as pilot:
        composer = app.query_one("#composer")
        suggestions = app.query_one("#command-suggestions")
        composer.text = "/"
        await pilot.pause()

        rendered = _rendered_text(suggestions.content)
        assert "1–5 of 27" in rendered
        assert "› /help" in rendered

        await pilot.press("down", "down", "down", "down", "down")
        rendered = _rendered_text(suggestions.content)

        assert "2–6 of 27" in rendered
        assert "› /reasoning" in rendered
        assert "/help" not in rendered


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
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.configure_provider.assert_awaited_once_with(
            "openai", "example-model", reasoning_effort="default"
        )


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
        message="OPENAI_API_KEY saved in ~/.local/share/noah-code/auth.json"
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
        assert "1 OF 4" in app.screen.query_one("#picker-title").render().plain
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
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        assert "REASONING" in app.screen.query_one("#picker-title").render().plain
        for _ in range(5):
            await pilot.press("down")
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.set_provider_api_key.assert_awaited_once_with("openai", "never-render-this-key")
        host.configure_provider.assert_awaited_once_with(
            "openai", "example-model", reasoning_effort="high"
        )
        assert "never-render-this-key" not in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_exact_reasoning_command_opens_picker_and_switches_effort(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/reasoning"
        await pilot.pause()
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        for _ in range(3):
            await pilot.press("down")
        await pilot.press("enter")
        for _ in range(20):
            if host.handle_line.await_count:
                break
            await pilot.pause()

        host.handle_line.assert_awaited_once_with("/reasoning low")


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

    async def _start():
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
        banner = app.query_one("#notice-banner")
        assert banner.styles.display == "block"
        assert "open /model" in _rendered_text(banner.content).lower()
        assert app.query_one("#welcome").styles.display == "block"

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
        await pilot.pause()
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


def test_completed_activity_label_keeps_file_paths() -> None:
    assert _completed_activity_label("Read src/parser.py", failed=False) == "✓ Read src/parser.py"
    assert _completed_activity_label("Reading src/parser.py", failed=False) == "✓ Read src/parser.py"
    assert _completed_activity_label("Write src/parser.py", failed=False) == "✓ Write src/parser.py"
    assert (
        _completed_activity_label("Reading src/a.py · Writing src/b.py", failed=False)
        == "✓ Read src/a.py · Write src/b.py"
    )
    assert _completed_activity_label("Inspecting repository", failed=False) == "✓ Inspect"
    assert _completed_activity_label("Bash pytest -q", failed=False) == "✓ Bash pytest -q"
    assert _completed_activity_label("Think", failed=False) is None
    assert _completed_activity_label("Preparing", failed=False) is None


def test_consecutive_file_activity_compacts_into_one_line() -> None:
    assert _coalesce_activity_text("✓ Read src/a.py", "✓ Read src/b.py") == "✓ Read src/a.py, src/b.py"
    assert (
        _coalesce_activity_text("✓ Read src/a.py, src/b.py", "✓ Read src/c.py")
        == "✓ Read src/a.py, src/b.py +1"
    )
    assert _coalesce_activity_text("✓ Read src/a.py", "✓ Write src/b.py") is None
    assert _coalesce_activity_text("✓ Wrote src/a.py", "✓ Write src/b.py") == "✓ Write src/a.py, src/b.py"


@pytest.mark.asyncio
async def test_activity_streams_live_then_compacts(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Bash pytest -q",
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
        assert "✓ Bash pytest -q" in transcript
        assert "execute_python" not in transcript
        assert "2 lines" not in transcript

        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ActivityHistoryScreen)
        assert "one" in _log_text(app.screen.query_one("#activity-detail"))


@pytest.mark.asyncio
async def test_file_activity_is_visible_live_then_compacts_together(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Reading src/a.py",
                meta={"activity_id": "read-1", "tool": "execute_python"},
            )
        )
        await pilot.pause()
        live = app.query_one("#live-activity")
        assert live.styles.display == "block"
        assert "src/a.py" in _rendered_text(app.query_one("#activity-title").content)

        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "complete",
                meta={"activity_id": "read-1", "result_status": "complete"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Reading src/b.py",
                meta={"activity_id": "read-2", "tool": "execute_python"},
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "complete",
                meta={"activity_id": "read-2", "result_status": "complete"},
            )
        )
        await pilot.pause()

        transcript = _log_text(app.query_one("#conversation"))
        assert "✓ Read src/a.py, src/b.py" in transcript
        assert transcript.count("✓ Read") == 1
        assert live.styles.display == "none"


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


@pytest.mark.asyncio
async def test_reasoning_attaches_to_activity_and_banner_shows_thought(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.config.ui.show_reasoning = False
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Reading src/a.py",
                meta={"activity_id": "read-1", "tool": "execute_python"},
            )
        )
        await pilot.pause()
        ui.render(HostEvent(HostEventKind.REASONING, "Check the export path first"))
        await pilot.pause()

        record = app._activities["read-1"]
        assert "export path" in record.thought
        assert app._last_thought == "Check the export path first"
        banner_text = _rendered_text(app.query_one("#working-banner").content)
        assert "✎" in banner_text
        # Reasoning stays out of the transcript when show_reasoning is off.
        transcript = _log_text(app.query_one("#conversation"))
        assert "Thinking:" not in transcript


@pytest.mark.asyncio
async def test_activity_inspector_expands_thought_and_action_sections(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.render(
            HostEvent(
                HostEventKind.TOOL_START,
                "Reading src/a.py",
                meta={
                    "activity_id": "read-1",
                    "tool": "execute_python",
                    "detail": "text = await self.ws.read('src/a.py')",
                },
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.REASONING,
                "Confirm the module exports before editing",
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.TOOL_FINISH,
                "complete",
                meta={"activity_id": "read-1", "result_status": "complete"},
            )
        )
        await pilot.pause(0.08)

        await pilot.press("f2")
        await pilot.pause()
        inspector = app.screen
        detail_text = _log_text(inspector.query_one("#activity-detail"))
        # Thought and action start collapsed with previews; output starts open.
        assert "▶ THOUGHT" in detail_text
        assert "▶ ACTION" in detail_text
        assert "▼ OUTPUT" in detail_text

        await pilot.press("t")
        await pilot.pause()
        expanded = _log_text(inspector.query_one("#activity-detail"))
        assert "▼ THOUGHT" in expanded
        assert "Confirm the module exports" in expanded

        await pilot.press("a")
        await pilot.pause()
        expanded = _log_text(inspector.query_one("#activity-detail"))
        assert "▼ ACTION" in expanded
        assert "ws.read('src/a.py')" in expanded

        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_session_actions_are_refused_while_turn_is_busy(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    host = _fake_host(tmp_path)
    host.start_new_session = AsyncMock(side_effect=AssertionError("must not switch"))
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        await pilot.pause()
        await pilot.press("ctrl+n")
        await pilot.pause()
        host.start_new_session.assert_not_called()
        assert "cancel it" in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_enter_on_exact_command_submits_directly(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")

        composer.text = "/help"
        await pilot.pause()
        assert app.suggestions_open
        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/help")
        assert not app.suggestions_open
        assert composer.text == ""


@pytest.mark.asyncio
async def test_enter_on_fully_typed_option_submits_without_second_press(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/mode plan"
        await pilot.pause()
        assert app._suggestion_matches[app._suggestion_index].invocation == "/mode plan"

        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/mode plan")


@pytest.mark.asyncio
async def test_enter_with_args_beyond_placeholder_submits(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "/config model"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/config model")


@pytest.mark.asyncio
async def test_busy_enter_queues_follow_up_without_starting_a_turn(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        composer = app.query_one("#composer")
        composer.text = "also run pytest"
        await pilot.press("enter")
        await pilot.pause()

        host.handle_line.assert_not_awaited()
        assert host.steer_queue.snapshot()["count"] == 1
        assert "queued · 1" in _rendered_text(app.query_one("#header").content)
        item = host.steer_queue.pop()
        assert item is not None and item.text == "also run pytest"
        assert "also run pytest" in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_chrome_shows_queued_count(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test(size=(120, 30)) as pilot:
        host.steer_queue.push("one")
        host.steer_queue.push("two")
        app.update_chrome(force=True)
        await pilot.pause()
        header = _rendered_text(app.query_one("#header").content)
        rail = _rendered_text(app.query_one("#context-rail").content)
        assert "queued · 2" in header
        assert "queued · 2" in rail


@pytest.mark.asyncio
async def test_approval_modal_does_not_queue_composer_text(tmp_path: Path) -> None:
    request = ApprovalRequest(
        id="req-1",
        decision=PermissionDecision(
            category="edit",
            target="a.py",
            action="ask",
            matching_rule=None,
            reason="needs approval",
            remember_pattern="a.py",
        ),
        created_at=0.0,
        future=MagicMock(),
    )
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        app.run_worker(app.push_screen_wait(ApprovalModal(request)))
        await pilot.pause()
        composer = app.query_one("#composer")
        composer.text = "do not queue this"
        app.action_submit()
        await pilot.pause()
        assert len(host.steer_queue) == 0
        assert composer.text == "do not queue this"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_tokens_slash_runs_while_busy(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        composer = app.query_one("#composer")
        composer.text = "/tokens"
        await pilot.press("enter")
        for _ in range(40):
            if host.handle_line.await_count:
                break
            await pilot.pause()
        host.handle_line.assert_awaited_once_with("/tokens")
        assert len(host.steer_queue) == 0


@pytest.mark.asyncio
async def test_undo_slash_is_blocked_while_busy(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    async with app.run_test() as pilot:
        ui.set_busy(True)
        composer = app.query_one("#composer")
        composer.text = "/undo"
        await pilot.press("enter")
        await pilot.pause()
        host.handle_line.assert_not_awaited()
        assert "blocked" in _log_text(app.query_one("#conversation"))
        assert len(host.steer_queue) == 0
