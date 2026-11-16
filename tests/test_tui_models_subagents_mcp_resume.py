"""Headless Textual tests for quick coding model picker, multi-subagent strip, 1-click MCP/Claude skills, and /resume."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.text import Text
from textual.widgets import Input, OptionList, RichLog, Static

from noah_code.events import HostEvent, HostEventKind
from noah_code.provider_discovery import ModelDiscoveryResult, ModelInfo
from noah_code.ui.modals import (
    FilteredPicker,
    ResumeSessionPicker,
    TextPromptModal,
    WorkLedgerScreen,
)
from noah_code.ui.textual_app import (
    AgentDisplayState,
    ComposerTextArea,
    NoahCodeApp,
    TextualUI,
)

_tui_test_helpers: Any = importlib.import_module("test_textual_tui")
_fake_host: Any = _tui_test_helpers._fake_host
_log_text: Any = _tui_test_helpers._log_text
_rendered_text: Any = _tui_test_helpers._rendered_text


@pytest.fixture(autouse=True)
def _disable_live_update_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "noah_code.ui.textual_app.maybe_check_for_update",
        lambda **_kwargs: None,
    )

# ---------------------------------------------------------------------------
# 1. Model Picker Upgrade (Ctrl+L / /model / Ctrl+P quick coding models)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_coding_models_picker_selects_openrouter_model_without_typing_id(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    host.list_provider_infos.return_value = [
        SimpleNamespace(
            key="openai",
            label="OpenAI",
            description="OpenAI API models",
            model_hint="openai/MODEL_NAME",
            credential_hint="OPENAI_API_KEY",
            configured=True,
            active=True,
        ),
        SimpleNamespace(
            key="openrouter",
            label="OpenRouter",
            description="One API for models from many providers",
            model_hint="openrouter/PROVIDER/MODEL",
            credential_hint="OPENROUTER_API_KEY",
            configured=True,
            active=False,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)

        # Compatibility contract: ordered_infos remain at indices 0..len-1, quick pick sits right above __advanced__.
        top_ids = [row[0] for row in app.screen._rows]
        assert top_ids == [
            "provider:openai",
            "provider:openrouter",
            "__coding_models__",
            "__advanced__",
        ]

        app.screen.query_one("#picker-filter", Input).value = "Popular coding"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        assert "POPULAR CODING MODELS" in str(
            app.screen.query_one("#picker-title", Static).render()
        )

        # Filter to an OpenRouter coding flagship and select with Enter (zero model ID typing).
        app.screen.query_one("#picker-filter", Input).value = "openrouter claude-sonnet-5.5"
        await pilot.pause()
        assert app.screen._filtered
        assert app.screen._filtered[0][0] == "quick:openrouter:anthropic/claude-sonnet-5.5"
        await pilot.press("enter")
        await pilot.pause()

        # Straight to reasoning effort since OpenRouter is configured.
        assert isinstance(app.screen, FilteredPicker)
        assert "REASONING" in str(app.screen.query_one("#picker-title", Static).render())
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.set_provider_api_key.assert_not_awaited()
        host.configure_provider.assert_awaited_once_with(
            "openrouter",
            "anthropic/claude-sonnet-5.5",
            reasoning_effort="default",
        )


@pytest.mark.asyncio
async def test_quick_coding_models_prompts_masked_key_when_unconfigured_and_openrouter_badges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = AsyncMock(
        return_value=ModelDiscoveryResult(
            (
                ModelInfo("openrouter/auto-code", "★ Coding router"),
                ModelInfo("anthropic/claude-sonnet-5.5", "★ Coding flagship"),
                ModelInfo("openai/gpt-5.3-codex", "★ Agentic coding"),
            ),
            "catalog",
            "Bundled catalog",
        )
    )
    monkeypatch.setattr("noah_code.provider_discovery.discover_models", discovery)

    host = _fake_host(tmp_path)
    host.meta.model = "openrouter/anthropic/claude-sonnet-5.5"
    host.config.model = "openrouter/anthropic/claude-sonnet-5.5"
    host.list_provider_infos.return_value = [
        SimpleNamespace(
            key="openrouter",
            label="OpenRouter",
            description="One API for models from many providers",
            model_hint="openrouter/PROVIDER/MODEL",
            credential_hint="OPENROUTER_API_KEY",
            configured=False,
            active=True,
        ),
    ]
    host.set_provider_api_key.return_value = SimpleNamespace(
        message="OPENROUTER_API_KEY saved"
    )
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 34)) as pilot:
        # 1. Verify Ctrl+P action_coding_models + masked API key flow without typing model ID.
        await pilot.press("ctrl+p")
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "Popular coding models"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        app.screen.query_one("#picker-filter", Input).value = "openrouter gpt-5.3-codex"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TextPromptModal)
        key_input = app.screen.query_one("#prompt-input", Input)
        assert key_input.password is True
        key_input.value = "sk-or-secret-token"
        await pilot.press("enter")
        for _ in range(20):
            if host.set_provider_api_key.await_count:
                break
            await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        await pilot.press("enter")
        for _ in range(20):
            if host.configure_provider.await_count:
                break
            await pilot.pause()

        host.set_provider_api_key.assert_awaited_once_with("openrouter", "sk-or-secret-token")
        host.configure_provider.assert_awaited_once_with(
            "openrouter",
            "openai/gpt-5.3-codex",
            reasoning_effort="default",
        )

        # 2. Verify _pick_provider_model keeps curated order, badges active model, and shows OpenRouter filter hint.
        host.list_provider_infos.return_value[0].configured = True
        await pilot.press("ctrl+l")
        await pilot.pause()
        await pilot.press("enter")  # Row 0 = OpenRouter
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        model_rows = app.screen._rows
        assert [r[0] for r in model_rows[:3]] == [
            "model:openrouter/auto-code",
            "model:anthropic/claude-sonnet-5.5",
            "model:openai/gpt-5.3-codex",
        ]
        assert model_rows[1][2].startswith("active · ")
        hint_plain = _rendered_text(app.screen.query_one("#picker-hint", Static).content)
        assert "openai/" in hint_plain and "anthropic/" in hint_plain and "coder" in hint_plain
        await pilot.press("escape")


# ---------------------------------------------------------------------------
# 2. Live Multi-Subagent Coordination Display (#subagent-strip, rail, F4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_subagent_strip_parallel_roster_sidebar_work_ledger_and_turn_reset(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(140, 38)) as pilot:
        strip = app.query_one("#subagent-strip", Static)
        assert strip.display is False

        # Launch 3 subagents in parallel: explore, planner, and isolated general.
        ui.set_busy(True)
        ui.render(
            HostEvent(
                HostEventKind.STATUS,
                "agent explore · queued · Map auth routes",
                meta={
                    "kind": "subagent",
                    "id": "sub-explore",
                    "agent": "explore",
                    "state": "queued",
                    "prompt": "Map auth routes",
                    "workflow": "build",
                    "phase": "explore",
                    "step_count": 0,
                },
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.STATUS,
                "agent planner · running",
                meta={
                    "kind": "subagent",
                    "id": "sub-plan",
                    "agent": "planner",
                    "state": "running",
                    "prompt": "Design migration steps",
                    "workflow": "build",
                    "phase": "plan",
                    "current_action": "Read schema.py",
                    "step_count": 2,
                },
            )
        )
        ui.render(
            HostEvent(
                HostEventKind.STATUS,
                "agent general · running",
                meta={
                    "kind": "subagent",
                    "id": "sub-impl",
                    "agent": "general",
                    "state": "running",
                    "prompt": "Patch router endpoints",
                    "workflow": "build",
                    "phase": "implement",
                    "isolated": True,
                    "current_action": "Edit router.py",
                    "step_count": 4,
                },
            )
        )
        # Emit additional running progress updates — must NOT spam transcript ACTIVITY entries!
        ui.render(
            HostEvent(
                HostEventKind.STATUS,
                "agent general · running · step 5",
                meta={
                    "kind": "subagent",
                    "id": "sub-impl",
                    "agent": "general",
                    "state": "running",
                    "isolated": True,
                    "current_action": "Run pytest router",
                    "step_count": 5,
                },
            )
        )
        await pilot.pause()

        assert strip.display is True
        assert app._agent_state == AgentDisplayState.RUNNING
        assert app._state_detail == "3 subagents active"

        # Only the single initial queued event was logged; all running step updates were suppressed!
        activity_entries = [e for e in app._transcript_entries if e.role == "ACTIVITY"]
        assert len(activity_entries) == 1

        strip_text = cast(Text, strip.content)
        assert "⬡ SUBAGENTS · 3 running · 0 done · build" in strip_text.plain
        assert "F4 / Alt+I inspect" in strip_text.plain
        assert "explore" in strip_text.plain and "(explore)" in strip_text.plain
        assert "planner" in strip_text.plain and "Read schema.py" in strip_text.plain and "step 2" in strip_text.plain
        assert "general [worktree] (implement)" in strip_text.plain
        assert "Run pytest router" in strip_text.plain and "step 5" in strip_text.plain

        # Complete explore wave while planner & general stay running -> completed wave stays visible!
        ui.render(
            HostEvent(
                HostEventKind.STATUS,
                "agent explore · completed · 1.8s · Map auth routes",
                meta={
                    "kind": "subagent",
                    "id": "sub-explore",
                    "agent": "explore",
                    "state": "completed",
                    "duration": 1.8,
                    "step_count": 3,
                },
            )
        )
        await pilot.pause()

        assert app._state_detail == "2 subagents active"
        updated_strip = cast(Text, strip.content)
        assert "⬡ SUBAGENTS · 2 running · 1 done · build" in updated_strip.plain
        assert "✓ explore" in updated_strip.plain and "Completed" in updated_strip.plain

        # Sidebar context rail shows Agents section with counts, [worktree], and live action.
        rail_plain = app._build_rail_text().plain
        assert "Agents" in rail_plain
        assert "2 active · 1 done" in rail_plain
        assert "[worktree]" in rail_plain
        assert "Run pytest router · s5" in rail_plain

        # Theme switch updates #subagent-strip cleanly without losing live state.
        app.apply_theme("daylight")
        await pilot.pause()
        assert strip.display is True
        assert "⬡ SUBAGENTS" in cast(Text, strip.content).plain

        # Click "F4 / Alt+I inspect" on #subagent-strip -> opens WorkLedgerScreen with live step & action!
        inspect_spans = [
            span
            for span in cast(Text, strip.content).spans
            if isinstance(span.style, object)
            and getattr(span.style, "meta", None)
            and span.style.meta.get("@click") == "app.work_ledger"
        ]
        assert inspect_spans
        first_line = cast(Text, strip.content).plain.splitlines()[0]
        col = first_line.index("F4 / Alt+I inspect") + 2
        offset = strip.content_region.offset - strip.region.offset
        await pilot.click(strip, offset=(offset.x + col, offset.y))
        await pilot.pause()

        assert isinstance(app.screen, WorkLedgerScreen)
        work_list = app.screen.query_one("#work-list", OptionList)
        prompts_plain = "\n".join(
            str(work_list.get_option_at_index(i).prompt)
            for i in range(work_list.option_count)
        )
        assert "Live step 5 · Run pytest router" in prompts_plain

        # Select general in WorkLedgerScreen and verify #work-detail shows Live step 5.
        for idx in range(work_list.option_count):
            opt = work_list.get_option_at_index(idx)
            if opt.id == "agent:sub-impl":
                work_list.highlighted = idx
                break
        await pilot.pause()
        detail_text = _log_text(app.screen.query_one("#work-detail", RichLog))
        assert "Live step 5 · Run pytest router" in detail_text
        await pilot.press("escape")
        await pilot.pause()

        # Turn reset clears roster and hides #subagent-strip cleanly.
        app._reset_transient_turn_widgets()
        await pilot.pause()
        assert len(app._subagent_roster) == 0
        assert strip.display is False


# ---------------------------------------------------------------------------
# 3. 1-Click MCP Presets, Claude/Cursor Import & Claude Skill Badges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_one_click_presets_secret_prompt_external_import_and_claude_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    host = _fake_host(tmp_path)
    host.connect_mcp_preset = AsyncMock(return_value="Connected MCP server context7 as self.context7")
    host.import_external_mcp_servers = AsyncMock(
        return_value="Imported 2 MCP servers from Claude / Cursor"
    )
    host.list_importable_mcp_servers = MagicMock(
        return_value=[
            SimpleNamespace(name="linear", source="Claude Code"),
            SimpleNamespace(name="sentry", source="Cursor"),
        ]
    )
    host.list_skill_infos.return_value = [
        SimpleNamespace(
            registry_name="cmd.audit-security",
            name="audit-security",
            description="Run OWASP code review",
            source=str(tmp_path / ".claude" / "skills" / "audit-security"),
            active=False,
            document_skill=True,
        ),
        SimpleNamespace(
            registry_name="cmd.local-helper",
            name="local-helper",
            description="Workspace helper",
            source=str(tmp_path / ".agents" / "skills" / "local-helper"),
            active=False,
            document_skill=True,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 34)) as pilot:
        # 1. /mcp -> ★ Quick-connect popular MCP server -> 1-click Context7 (no secret prompt!)
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "/mcp"
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, FilteredPicker)
        row_ids = [r[0] for r in app.screen._rows]
        assert "__mcp_presets__" in row_ids
        assert "__mcp_import__" in row_ids
        assert "__add_smart__" in row_ids
        assert "__add_stdio__" in row_ids
        assert "__add_http__" in row_ids

        await pilot.press("enter")  # Row 0 = __mcp_presets__
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        assert "POPULAR MCP SERVERS" in str(
            app.screen.query_one("#picker-title", Static).render()
        )
        preset_keys = {r[0] for r in app.screen._rows}
        for expected in (
            "preset:context7",
            "preset:playwright",
            "preset:filesystem",
            "preset:fetch",
            "preset:memory",
            "preset:sequential-thinking",
            "preset:github",
            "preset:postgres",
            "preset:brave-search",
            "preset:linear",
            "preset:sentry",
            "preset:cloudflare-docs",
        ):
            assert expected in preset_keys

        app.screen.query_one("#picker-filter", Input).value = "context7"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            if host.connect_mcp_preset.await_count:
                break
            await pilot.pause()
        host.connect_mcp_preset.assert_awaited_once_with("context7", env_value=None)

        # 2. Secret-requiring preset (GitHub) prompts masked token before connecting.
        host.connect_mcp_preset.reset_mock()
        host.connect_mcp_preset.return_value = "Connected MCP server github as self.github"
        app.action_mcp()
        await pilot.pause()
        await pilot.press("enter")  # __mcp_presets__
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "GitHub"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TextPromptModal)
        token_input = app.screen.query_one("#prompt-input", Input)
        assert token_input.password is True
        token_input.value = "ghp_secret_pat_value"
        await pilot.press("enter")
        for _ in range(20):
            if host.connect_mcp_preset.await_count:
                break
            await pilot.pause()
        host.connect_mcp_preset.assert_awaited_once_with(
            "github", env_value="ghp_secret_pat_value"
        )

        # 3. 1-click Import MCPs from Claude / Cursor.
        app.action_mcp()
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "Import MCPs"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            if host.import_external_mcp_servers.await_count:
                break
            await pilot.pause()
        host.import_external_mcp_servers.assert_awaited_once()
        assert "Imported 2 MCP servers" in _log_text(app.query_one("#conversation", RichLog))

        # 4. Claude skill badge in /skills (Ctrl+G) searchable by 'Claude skill' / skill name.
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(app.screen, FilteredPicker)
        app.screen.query_one("#picker-filter", Input).value = "Claude skill audit"
        await pilot.pause()
        assert [row[1] for row in app.screen._filtered] == ["$audit-security"]
        assert "Claude skill" in app.screen._filtered[0][2]
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "$audit-security "


# ---------------------------------------------------------------------------
# 4. /resume Last-3-Chats Picker & --resume Startup Auto-Open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_picker_last_three_previews_quick_digit_and_direct_arg(
    tmp_path: Path,
) -> None:
    now = time.time()
    host = _fake_host(tmp_path)
    host.recent_repo_sessions = MagicMock(
        return_value=[
            SimpleNamespace(
                session_id="111111111111",
                title="Refactor OAuth middleware",
                mode="build",
                model="openrouter/anthropic/claude-sonnet-5.5",
                worktree_name="",
                updated_at=now - 120,
                preview="Added PKCE verifier and unit tests for callback handler",
            ),
            SimpleNamespace(
                session_id="222222222222",
                title="Fix SQLite WAL contention",
                mode="build",
                model="openai/gpt-5.3-codex",
                worktree_name="wal-fix",
                updated_at=now - 3600,
                preview="Configured busy_timeout and readonly URI connections",
            ),
            SimpleNamespace(
                session_id="333333333333",
                title="Audit API rate limits",
                mode="plan",
                model="anthropic/claude-opus-5-5",
                worktree_name="",
                updated_at=now - 7200,
                preview="Reviewed token bucket thresholds across edge routes",
            ),
            SimpleNamespace(
                session_id="444444444444",
                title="Fourth older session",
                mode="build",
                model="openai/gpt-5.4",
                worktree_name="",
                updated_at=now - 86400,
                preview="Should be capped at 3",
            ),
        ]
    )
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(120, 34)) as pilot:
        # Verify /resume is registered in _base_commands without duplicates.
        resume_cmds = [
            cmd for cmd in app._base_commands if cmd.invocation.split()[0] == "/resume"
        ]
        assert len(resume_cmds) == 1

        # Busy guard refuses cleanly before switching sessions.
        ui.set_busy(True)
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "/resume"
        app.close_suggestions()
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ResumeSessionPicker)
        assert "Cancel the running turn before resuming another chat" in _rendered_text(
            app.query_one("#notice-banner", Static).content
        )
        ui.set_busy(False)
        await pilot.pause()

        # Bare /resume opens ResumeSessionPicker with exactly 3 chats and message previews.
        composer.text = "/resume"
        app.close_suggestions()
        await pilot.press("enter")
        for _ in range(20):
            if isinstance(app.screen, ResumeSessionPicker):
                break
            await pilot.pause()

        assert isinstance(app.screen, ResumeSessionPicker)
        assert "RESUME CHAT · LAST 3 IN THIS REPO" in str(
            app.screen.query_one("#picker-title", Static).render()
        )
        assert len(app.screen._rows) == 3
        assert app.screen._rows[0][1] == "1. Refactor OAuth middleware"
        assert '"Added PKCE verifier' in app.screen._rows[0][2]
        assert app.screen._rows[1][1] == "2. Fix SQLite WAL contention"
        assert "wal-fix" in app.screen._rows[1][2]
        assert app.screen._rows[2][1] == "3. Audit API rate limits"

        # Pressing '2' when filter input is empty immediately resumes chat #2!
        await pilot.press("2")
        for _ in range(20):
            if host.switch_session.await_count:
                break
            await pilot.pause()

        host.switch_session.assert_awaited_once_with("222222222222")
        assert "Resumed · Fix SQLite WAL contention" in _rendered_text(
            app.query_one("#notice-banner", Static).content
        )
        assert app.focused is composer

        # Direct '/resume 3' switches straight to chat #3 without opening a modal.
        host.switch_session.reset_mock()
        composer.text = "/resume 3"
        app.close_suggestions()
        await pilot.press("enter")
        for _ in range(20):
            if host.switch_session.await_count:
                break
            await pilot.pause()
        host.switch_session.assert_awaited_once_with("333333333333")


@pytest.mark.asyncio
async def test_dash_dash_resume_flag_auto_opens_picker_on_startup(tmp_path: Path) -> None:
    now = time.time()
    host = _fake_host(tmp_path)
    host._agent = None  # Exercise cold _start_host() path as in `noah --resume` CLI launch!
    host.start = AsyncMock()
    host._open_resume_on_start = True
    host.list_session_metas.return_value = [
        SimpleNamespace(
            session_id="aaaaaaaa1111",
            title="Ship OpenRouter catalog",
            mode="build",
            model="openrouter/auto-code",
            worktree_name="",
            updated_at=now - 90,
        ),
        SimpleNamespace(
            session_id="bbbbbbbb2222",
            title="Parallel subagent UI",
            mode="build",
            model="openai/gpt-5.3-codex",
            worktree_name="",
            updated_at=now - 600,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(110, 32)) as pilot:
        for _ in range(30):
            if isinstance(app.screen, ResumeSessionPicker):
                break
            await pilot.pause()

        assert isinstance(app.screen, ResumeSessionPicker)
        assert host._open_resume_on_start is False
        assert len(app.screen._rows) == 2

        await pilot.press("enter")
        for _ in range(20):
            if host.switch_session.await_count:
                break
            await pilot.pause()

        host.switch_session.assert_awaited_once_with("aaaaaaaa1111")


@pytest.mark.asyncio
async def test_mcp_smart_add_fallback_subagent_overflow_and_resume_filter_typing(
    tmp_path: Path,
) -> None:
    now = time.time()
    host = _fake_host(tmp_path)
    host.add_mcp_server = AsyncMock(return_value="Saved MCP server linear")
    host.list_session_metas.return_value = [
        SimpleNamespace(
            session_id="aaaa1111bbbb",
            title="Chat Alpha",
            mode="build",
            model="openai/gpt-5.4",
            worktree_name="",
            updated_at=now - 60,
        ),
        SimpleNamespace(
            session_id="cccc2222dddd",
            title="Chat Beta 2",
            mode="build",
            model="openai/gpt-5.3-codex",
            worktree_name="",
            updated_at=now - 120,
        ),
    ]
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(120, 34)) as pilot:
        # 1. >5 subagents shows '+2 more in F4' overflow link.
        for idx in range(7):
            ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"agent explore · running · shard {idx}",
                    meta={
                        "kind": "subagent",
                        "id": f"shard-{idx}",
                        "agent": "explore",
                        "state": "running",
                        "current_action": f"Scan module {idx}",
                        "step_count": 1,
                    },
                )
            )
        await pilot.pause()
        strip_plain = cast(Text, app.query_one("#subagent-strip", Static).content).plain
        assert "+2 more in F4" in strip_plain
        app._reset_transient_turn_widgets()

        # 2. Preset fallback to host.add_mcp_server when host.connect_mcp_preset is absent (MagicMock).
        app.action_mcp()
        await pilot.pause()
        await pilot.press("enter")  # __mcp_presets__
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "Linear"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            if host.add_mcp_server.await_count:
                break
            await pilot.pause()
        host.add_mcp_server.assert_awaited_once_with(
            "http", "linear", "https://mcp.linear.app/mcp"
        )

        # 3. Smart single-step MCP add auto-detects https:// endpoint vs stdio command.
        host.add_mcp_server.reset_mock()
        app.action_mcp()
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "command or URL"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TextPromptModal)
        app.screen.query_one("#prompt-input", Input).value = "custom-http"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TextPromptModal)
        app.screen.query_one("#prompt-input", Input).value = "https://mcp.example.dev/sse"
        await pilot.press("enter")
        for _ in range(20):
            if host.add_mcp_server.await_count:
                break
            await pilot.pause()
        host.add_mcp_server.assert_awaited_once_with(
            "http", "custom-http", "https://mcp.example.dev/sse"
        )

        # 4. Typing non-empty filter text in ResumeSessionPicker allows typing digits ('2') without premature jump.
        await pilot.press("ctrl+p")
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "Resume recent chat"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            if isinstance(app.screen, ResumeSessionPicker):
                break
            await pilot.pause()
        assert isinstance(app.screen, ResumeSessionPicker)
        await pilot.press("b", "e", "t", "a", " ", "2")
        await pilot.pause()
        assert isinstance(app.screen, ResumeSessionPicker)
        assert app.screen.query_one("#picker-filter", Input).value == "beta 2"
        await pilot.press("enter")
        for _ in range(20):
            if host.switch_session.await_count:
                break
            await pilot.pause()
        host.switch_session.assert_awaited_once_with("cccc2222dddd")

