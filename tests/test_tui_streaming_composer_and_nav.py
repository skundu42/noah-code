"""Headless tests for inline throttled streaming, O(1) transcript updates, composer ergonomics, and navigation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.text import Text
from textual.app import ComposeResult
from textual.geometry import Offset
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.widgets import Input, Static

from noah_code.events import HostEvent, HostEventKind
from noah_code.project_notes import PlanStore
from noah_code.tools.git_tools import DiffFile, DiffReview
from noah_code.ui import modals as modals_module
from noah_code.ui import textual_app as tui
from noah_code.ui.command_menu import command_menu_renderable
from noah_code.ui.modals import SelectableRichLog
from noah_code.ui.textual_app import (
    MODEL_STREAM_FLUSH_SECONDS,
    TRANSCRIPT_MAX_ENTRIES,
    TRANSCRIPT_TRIM_THRESHOLD,
    ActivityHistoryScreen,
    ComposerTextArea,
    ConfirmationModal,
    DiffReviewScreen,
    NoahCodeApp,
    RepositorySnapshot,
    TextualUI,
    TranscriptEntry,
    WorkLedgerScreen,
    _active_mention,
    _active_skill,
    _close_unbalanced_markdown_fence,
    _cursor_index,
)
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import (
    _fake_host,
    _log_text,
    _rendered_text,
)


def test_close_unbalanced_markdown_fence_handles_open_and_sliced_fences() -> None:
    raw = "Here is Python:\n```python\ndef add(a: int, b: int) -> int:\n    return a + b"
    closed = _close_unbalanced_markdown_fence(raw)
    assert closed.endswith("\n```")
    assert raw.endswith("return a + b")  # Original string untouched.

    balanced = "```python\nx = 1\n```\nTrailing prose"
    assert _close_unbalanced_markdown_fence(balanced) == balanced

    # Long streaming code block larger than tail window preserves opening fence header.
    long_code = "```python\n" + "".join(f"x_{i} = {i}\n" for i in range(500))
    sliced = _close_unbalanced_markdown_fence(long_code, max_tail_chars=400)
    assert sliced.startswith("```python\n")
    assert sliced.endswith("\n```")


@pytest.mark.asyncio
async def test_inline_throttled_model_stream_leading_and_trailing_edge_and_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(120, 36)) as pilot:
        primary_children = [child.id for child in app.query_one("#primary-pane").children]
        assert primary_children.index("transcript-find-bar") < primary_children.index("conversation")
        assert primary_children.index("conversation") < primary_children.index("model-stream-preview")
        assert primary_children.index("model-stream-preview") < primary_children.index("working-banner")
        assert primary_children.index("model-stream-preview") < primary_children.index("live-activity")

        render_calls = 0
        real_render = app._render_model_stream_preview

        def counting_render() -> None:
            nonlocal render_calls
            render_calls += 1
            real_render()

        monkeypatch.setattr(app, "_render_model_stream_preview", counting_render)

        ui.render(HostEvent(HostEventKind.MODEL_STREAM, "", meta={"phase": "start", "call_id": "c1"}))
        ui.render(
            HostEvent(
                HostEventKind.MODEL_STREAM,
                "```python\ndef compute():\n",
                meta={"phase": "text", "call_id": "c1"},
            )
        )
        # Burst 15 additional SSE deltas in the same drain window.
        for idx in range(15):
            ui.render(
                HostEvent(
                    HostEventKind.MODEL_STREAM,
                    f"    val_{idx} = {idx}\n",
                    meta={"phase": "text", "call_id": "c1"},
                )
            )
        app._drain_host_events()

        preview = app.query_one("#model-stream-preview", Static)
        assert preview.display is True
        # Leading edge rendered immediately once; burst chunks coalesced into dirty flag!
        assert render_calls == 1
        assert app._model_preview_dirty is True
        assert app._model_preview_timer is not None

        # Wait for trailing-edge flush window to fire.
        await pilot.pause(MODEL_STREAM_FLUSH_SECONDS + 0.05)
        assert render_calls == 2
        assert app._model_preview_dirty is False
        rendered_preview = _rendered_text(preview.content)
        assert "Noah · streaming" in rendered_preview
        assert "val_14 = 14" in rendered_preview
        assert "```" not in rendered_preview  # Auto-closed fence rendered as Markdown code block!

        # MESSAGE event immediately hides preview and cancels timer even without explicit finish.
        ui.render(HostEvent(HostEventKind.MESSAGE, "Completed reply"))
        await pilot.pause()
        assert preview.display is False
        assert app._model_preview == ""
        assert app._model_preview_timer is None
        assert "Completed reply" in _log_text(app.query_one("#conversation"))


@pytest.mark.asyncio
async def test_transcript_hysteresis_o1_tail_updates_and_stat_guarded_plan_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(130, 32)) as pilot:
        await pilot.pause()
        log = app.query_one("#conversation", SelectableRichLog)

        rerenders = 0
        real_rerender = app._rerender_transcript

        def counting_rerender() -> None:
            nonlocal rerenders
            rerenders += 1
            real_rerender()

        monkeypatch.setattr(app, "_rerender_transcript", counting_rerender)
        clear_spy = MagicMock(wraps=log.clear)
        monkeypatch.setattr(log, "clear", clear_spy)

        # 1. O(1) ACTIVITY coalescing without log.clear() or _rerender_transcript().
        app._append_entry(TranscriptEntry("ACTIVITY", "✓ Read src/one.py"))
        await pilot.pause()
        app._append_entry(TranscriptEntry("ACTIVITY", "✓ Read src/two.py"))
        await pilot.pause()
        assert app._transcript_entries[-1].text == "✓ Read src/one.py, src/two.py"
        assert "✓ Read src/one.py, src/two.py" in _log_text(log)
        assert rerenders == 0
        assert clear_spy.call_count == 0

        # 2. O(1) checkpoint receipt update (also exercises log.replace_tail_lines when present).
        app._append_entry(TranscriptEntry("RECEIPT", "Turn complete · 1.2s · checks: passed"))
        await pilot.pause()
        ui.render(HostEvent(HostEventKind.STATUS, "◆", meta={"kind": "checkpoint"}))
        await pilot.pause()
        assert app._transcript_entries[-1].text.startswith("◆ Turn complete")
        assert "◆ Turn complete" in _log_text(log)
        assert rerenders == 0
        assert clear_spy.call_count == 0

        # Also verify delegation to SelectableRichLog.replace_tail_lines contract if present.
        custom_calls: list[int] = []

        def fake_replace_tail(old_rows: int, _renderable: object, scroll_end: bool = False) -> int:
            _ = scroll_end
            custom_calls.append(old_rows)
            return old_rows

        monkeypatch.setattr(
            log,
            "replace_tail_lines",
            fake_replace_tail,
            raising=False,
        )
        app._append_entry(TranscriptEntry("ACTIVITY", "✓ Edit src/a.py"))
        app._append_entry(TranscriptEntry("ACTIVITY", "✓ Edit src/b.py"))
        assert len(custom_calls) == 1 and custom_calls[0] > 0
        monkeypatch.delattr(log, "replace_tail_lines", raising=False)

        # 3. Hysteresis cap: entries up to 549 do 0 full rerenders; 550th trims back to 500 once.
        while len(app._transcript_entries) < TRANSCRIPT_TRIM_THRESHOLD - 1:
            app._append_entry(TranscriptEntry("STATUS", f"line {len(app._transcript_entries)}"))
        assert len(app._transcript_entries) == TRANSCRIPT_TRIM_THRESHOLD - 1
        assert rerenders == 0

        app._append_entry(TranscriptEntry("STATUS", "trigger trim"))
        assert len(app._transcript_entries) == TRANSCRIPT_MAX_ENTRIES
        assert rerenders == 1

        # Next 10 entries again append in O(1) without rerendering.
        for idx in range(10):
            app._append_entry(TranscriptEntry("STATUS", f"after trim {idx}"))
        assert len(app._transcript_entries) == TRANSCRIPT_MAX_ENTRIES + 10
        assert rerenders == 1

        # 4. Stat-guarded PlanStore caching: 0 reads when absent or unchanged, instant update on write.
        read_calls = 0
        real_plan_read = PlanStore.read

        def counting_plan_read(self_store: PlanStore) -> str:
            nonlocal read_calls
            read_calls += 1
            return real_plan_read(self_store)

        monkeypatch.setattr(PlanStore, "read", counting_plan_read)
        app._build_rail_text()
        app._build_rail_text()
        assert read_calls == 0

        PlanStore(tmp_path).write("# Ship streaming TUI\n- Step 1\n")
        rail_text = app._build_rail_text().plain
        assert "Ship streaming TUI" in rail_text
        assert read_calls == 1

        # Repeated rail builds with unchanged plan.md perform 0 file reads.
        app._build_rail_text()
        app._build_rail_text()
        assert read_calls == 1


@pytest.mark.asyncio
async def test_sub_entry_fenced_code_selection_and_code_file_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src").mkdir()
    target_file = tmp_path / "src" / "helper.py"
    target_file.write_text("def helper():\n    return 1\n")

    host = _fake_host(tmp_path)
    host.diff_review = AsyncMock(
        return_value=DiffReview(
            [DiffFile("src/helper.py", "unstaged", "modified", patch="@@ -1 +1 @@\n-old\n+new\n", loaded=True)]
        )
    )
    host.agent.lsp.document_symbols = AsyncMock(return_value="")
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    reply = (
        "Introductory explanation paragraph.\n\n"
        "```python\n"
        "def compute_total(items: list[int]) -> int:\n"
        "    return sum(x * 2 for x in items)\n"
        "```\n\n"
        "Closing remarks."
    )

    async with app.run_test(size=(100, 36)) as pilot:
        ui.render(HostEvent(HostEventKind.MESSAGE, reply))
        await pilot.pause()

        transcript = app.query_one("#conversation", SelectableRichLog)
        code_rows = [
            idx
            for idx, strip in enumerate(transcript.lines)
            if "def compute_total" in strip.text or "return sum" in strip.text
        ]
        assert len(code_rows) == 2

        # Dragging strictly inside the fenced code block copies ONLY pristine source lines!
        app.screen.selections = {
            transcript: Selection(Offset(0, code_rows[0]), Offset(30, code_rows[-1]))
        }
        await pilot.press("ctrl+shift+c")
        assert app.clipboard == (
            "def compute_total(items: list[int]) -> int:\n"
            "    return sum(x * 2 for x in items)"
        )

        # Selecting just the indented body line still preserves exact 4-space source indentation.
        app.screen.selections = {
            transcript: Selection(Offset(0, code_rows[1]), Offset(30, code_rows[1]))
        }
        await pilot.press("ctrl+shift+c")
        assert app.clipboard == "    return sum(x * 2 for x in items)"
        app.screen.clear_selection()

        # Alt+C copies the latest fenced code block directly.
        app._clipboard = ""
        await pilot.press("alt+c")
        assert app.clipboard == (
            "def compute_total(items: list[int]) -> int:\n"
            "    return sum(x * 2 for x in items)"
        )
        assert "Copied latest code block" in _rendered_text(
            app.query_one("#notice-banner", Static).content
        )

        # action_copy_code_id looks up registered code blocks gracefully.
        monkeypatch.setattr(
            modals_module,
            "get_registered_code_block",
            lambda bid: "print('registered')" if bid == "blk-1" else None,
            raising=False,
        )
        app.action_copy_code_id("blk-1")
        assert app.clipboard == "print('registered')"

        # action_open_file_id opens editor at +line when clean, or DiffReviewScreen when modified.
        editor_cmds: list[list[str]] = []
        monkeypatch.setenv("VISUAL", "")
        monkeypatch.setenv("EDITOR", "nvim")
        def record_editor_cmd(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
            if cmd and cmd[0] == "git":
                return subprocess.CompletedProcess(cmd, 1, stdout=b"")
            editor_cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(tui.subprocess, "run", record_editor_cmd)
        monkeypatch.setattr(
            modals_module,
            "get_registered_file_ref",
            lambda rid: ("src/helper.py", 12) if rid == "ref-1" else None,
            raising=False,
        )
        app._repository_snapshot = RepositorySnapshot("main", paths=())
        app.action_open_file_id("ref-1")
        await pilot.pause()
        assert editor_cmds == [["nvim", "+12", str(target_file.resolve())]]

        app._repository_snapshot = RepositorySnapshot("main", modified=1, paths=("src/helper.py",))
        app.action_open_file_id("ref-1")
        await pilot.pause()
        assert isinstance(app.screen, DiffReviewScreen)
        app.screen.dismiss(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_cursor_zero_mention_fix_and_inline_skill_autocomplete(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("a = 1\n")

    # Direct unit verification of the (0, 0) cursor fix.
    assert _cursor_index("explain @src/al", (0, 0)) == 0
    assert _active_mention("explain @src/al", (0, 0)) is None
    assert _active_skill("explain $rev", (0, 0)) is None
    assert _active_skill("line one\n  $rev", (1, 6)) == "$rev"

    host = _fake_host(tmp_path)
    host.list_skill_infos.return_value = [
        SimpleNamespace(
            registry_name="cmd.review-pr",
            name="review-pr",
            description="Review pull request diffs",
            source="local",
            active=False,
            document_skill=True,
        ),
        SimpleNamespace(
            registry_name="cmd.release",
            name="release",
            description="Draft release changelog",
            source="local",
            active=False,
            document_skill=True,
        ),
        SimpleNamespace(
            registry_name="tool.internal",
            name="internal-tool",
            description="Non-document skill",
            source="local",
            active=True,
            document_skill=False,
        ),
    ]
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 34)) as pilot:
        composer = app.query_one("#composer", ComposerTextArea)

        # 1. Typing @mention at cursor (0, 0) before existing text completes at start of prompt!
        composer.text = "summarize this"
        composer.cursor_location = (0, 0)
        await pilot.press("@", "s", "r", "c", "/", "a", "l", " ")
        # Backspace space to reopen @src/al before "summarize this"
        composer.text = "summarize this"
        composer.cursor_location = (0, 0)
        composer.insert("@src/al ")
        composer.cursor_location = (0, 7)
        app._update_command_suggestions(composer.text, composer.cursor_location)
        assert app.suggestions_open
        await pilot.press("tab")
        assert composer.text == "@src/alpha.py  summarize this"

        # Moving cursor to (0, 0) when @mention is at the end closes stale suggestions.
        composer.text = "check @src/al"
        await pilot.pause()
        assert app.suggestions_open
        composer.cursor_location = (0, 0)
        await pilot.pause()
        assert not app.suggestions_open

        # 2. Single-line inline $skill autocomplete with "Skills" menu header; Enter does not submit!
        composer.text = "Please $rev"
        await pilot.pause()
        assert app.suggestions_open
        menu_rows = [
            cast(Text, row)
            for row in command_menu_renderable(
                app._suggestion_matches,
                0,
                width=70,
                window_size=5,
                palette=app.theme_palette,
            ).renderables
        ]
        assert menu_rows[0].plain.startswith("Skills")
        assert "$review-pr" in menu_rows[1].plain
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "Please $review-pr "
        host.handle_line.assert_not_awaited()

        # 3. Multiline inline $skill autocomplete replaces token on line 2 in-place.
        composer.text = "Context line\nRun $rel"
        composer.cursor_location = (1, 8)
        app._update_command_suggestions(composer.text, composer.cursor_location)
        assert app.suggestions_open
        assert [m.invocation for m in app._suggestion_matches] == ["$release "]
        await pilot.press("tab")
        assert composer.text == "Context line\nRun $release "
        assert composer.cursor_location == (1, len("Run $release "))


@pytest.mark.asyncio
async def test_up_down_composer_prompt_history_preserves_unsent_draft(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.list_prompt_history = MagicMock(
        return_value=["recent in-session", "older cross-session prompt"]
    )
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(100, 30)) as pilot:
        app._append_entry(TranscriptEntry("YOU", "first in-session"))
        app._append_entry(TranscriptEntry("YOU", "recent in-session"))
        composer = app.query_one("#composer", ComposerTextArea)
        composer.focus()

        # Multiline draft: pressing up on line 1 moves to line 0 first without cycling history.
        composer.text = "draft line 1\ndraft line 2"
        composer.cursor_location = (1, 5)
        await pilot.press("up")
        assert composer.text == "draft line 1\ndraft line 2"
        assert composer.cursor_location[0] == 0

        # Pressing up on row 0 stashes the unsent multiline draft and cycles backward.
        await pilot.press("up")
        assert composer.text == "recent in-session"
        await pilot.press("up")
        assert composer.text == "first in-session"
        await pilot.press("up")
        assert composer.text == "older cross-session prompt"
        # Extra up at oldest stays on oldest.
        await pilot.press("up")
        assert composer.text == "older cross-session prompt"

        # Pressing down steps forward all the way back to the stashed unsent draft.
        await pilot.press("down")
        assert composer.text == "first in-session"
        await pilot.press("down")
        assert composer.text == "recent in-session"
        await pilot.press("down")
        assert composer.text == "draft line 1\ndraft line 2"
        assert app._history_index is None


@pytest.mark.asyncio
async def test_ctrl_e_external_editor_updates_composer_and_cleans_up_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    seen_paths: list[Path] = []

    def fake_editor_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if cmd and cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 1, stdout=b"")
        assert cmd[0] == "custom-ed"
        temp_path = Path(cmd[-1])
        seen_paths.append(temp_path)
        assert temp_path.exists()
        assert (temp_path.stat().st_mode & 0o777) == 0o600
        assert temp_path.read_text(encoding="utf-8") == "initial draft"
        temp_path.write_text("edited prompt\nsecond line\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("VISUAL", "custom-ed")
    monkeypatch.setattr(tui.subprocess, "run", fake_editor_run)

    async with app.run_test(size=(100, 30)) as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.focus()
        composer.text = "initial draft"

        await pilot.press("ctrl+e")
        await pilot.pause()

        assert composer.text == "edited prompt\nsecond line"
        assert composer.cursor_location == composer.document.end
        assert len(seen_paths) == 1
        assert not seen_paths[0].exists()

        # Non-zero exit keeps current draft untouched.
        monkeypatch.setattr(
            tui.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 1),
        )
        await pilot.press("ctrl+e")
        await pilot.pause()
        assert composer.text == "edited prompt\nsecond line"


@pytest.mark.asyncio
async def test_clickable_attachment_chips_remove_pending_files(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    file_a = tmp_path / "diagram.png"
    file_b = tmp_path / "notes.md"
    host._pending_attach_paths.extend([file_a, file_b])
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 30)) as pilot:
        app.update_chrome(force=True)
        await pilot.pause()

        input_ctx = app.query_one("#input-context", Static)
        assert input_ctx.styles.display == "block"
        rendered = cast(Text, input_ctx.content)
        assert "[× diagram.png]" in rendered.plain
        assert "[× notes.md]" in rendered.plain
        assert any(
            (span.style.meta if isinstance(span.style, tui.Style) and span.style.meta else {}).get(
                "@click"
            )
            == "app.remove_attachment(0)"
            for span in rendered.spans
        )

        # Click the first chip directly in the UI!
        col = rendered.plain.index("[× diagram.png]") + 2
        offset = input_ctx.content_region.offset - input_ctx.region.offset
        await pilot.click(input_ctx, offset=(offset.x + col, offset.y))
        await pilot.pause()

        assert list(host.pending_attach_paths()) == [file_b]
        updated_rendered = cast(Text, input_ctx.content)
        assert "[× diagram.png]" not in updated_rendered.plain
        assert "[× notes.md]" in updated_rendered.plain

        app.action_remove_attachment(0)
        await pilot.pause()
        assert list(host.pending_attach_paths()) == []
        assert input_ctx.styles.display == "none"


@pytest.mark.asyncio
async def test_unified_palette_executes_ui_action_without_clobbering_draft(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(120, 34)) as pilot:
        composer = app.query_one("#composer", ComposerTextArea)
        composer.text = "keep this exact draft"

        await pilot.press("ctrl+p")
        await pilot.pause()
        picker = app.screen
        picker.query_one("#picker-filter", Input).value = "Work ledger"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, WorkLedgerScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert composer.text == "keep this exact draft"
        host.handle_line.assert_not_awaited()


@pytest.mark.asyncio
async def test_ctrl_f_in_transcript_find_bar_jumps_matches_and_respects_modals(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)

    async with app.run_test(size=(90, 22)) as pilot:
        for idx in range(45):
            marker = f"NEEDLE-{idx}" if idx in {2, 20, 42} else f"filler message {idx}"
            ui.render(HostEvent(HostEventKind.MESSAGE, marker))
        await pilot.pause()

        await pilot.press("ctrl+f")
        await pilot.pause()
        bar = app.query_one("#transcript-find-bar")
        find_input = app.query_one("#transcript-find-input", Input)
        count_label = app.query_one("#transcript-find-count", Static)
        log = app.query_one("#conversation")

        assert bar.display is True
        assert app.focused is find_input

        find_input.value = "needle-"
        await pilot.pause()
        assert _rendered_text(count_label.content).strip() == "1 of 3"
        first_scroll = log.scroll_y

        await pilot.press("enter")
        await pilot.pause()
        assert _rendered_text(count_label.content).strip() == "2 of 3"
        second_scroll = log.scroll_y
        assert second_scroll > first_scroll

        await pilot.press("shift+enter")
        await pilot.pause()
        assert _rendered_text(count_label.content).strip() == "1 of 3"
        assert log.scroll_y == first_scroll

        find_input.value = "no-such-term"
        await pilot.pause()
        assert "No matches" in _rendered_text(count_label.content)

        await pilot.press("escape")
        await pilot.pause()
        assert bar.display is False
        assert app.focused is app.query_one("#composer")

        # Inside ActivityHistoryScreen modal, Ctrl+F focuses modal search instead of transcript find.
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.screen, ActivityHistoryScreen)
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one("#activity-filter")
        assert bar.display is False


@pytest.mark.asyncio
async def test_live_context_window_meter_checkpoint_undo_and_session_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())

    async with app.run_test(size=(130, 30)) as pilot:
        palette = app.theme_palette

        for last_in, expected_text, expected_color in [
            (45_200, "ctx 45.2k/200k (23%)", palette.muted),
            (150_000, "ctx 150.0k/200k (75%)", palette.warning),
            (184_000, "ctx 184.0k/200k (92%)", palette.error),
        ]:
            host.usage_snapshot.return_value = SimpleNamespace(
                prompt_tokens=120_000,
                cached_tokens=60_000,
                completion_tokens=4_000,
                cost_usd=0.25,
                cache_hit_ratio=0.5,
                last_input_tokens=last_in,
                context_window=200_000,
            )
            app.update_chrome(force=True)
            await pilot.pause()
            hint_content = cast(Text, app.query_one("#context-hint", Static).content)
            assert expected_text in hint_content.plain
            ctx_spans = [
                span
                for span in hint_content.spans
                if isinstance(span.style, tui.Style)
                and (span.style.meta or {}).get("@click") == "app.context_visibility"
            ]
            assert ctx_spans
            assert isinstance(ctx_spans[0].style, tui.Style)
            assert expected_color.lstrip("#").lower() in str(ctx_spans[0].style.color).lower()

        # Checkpoint undo fallback when shell_may_bypass=True and Git checkpoint exists.
        mutation = SimpleNamespace(path=str(tmp_path / "main.py"))
        host.agent.journal.latest_turn.return_value = SimpleNamespace(
            turn_id="abcdef123456",
            mutations=[mutation],
            shell_may_bypass=True,
        )
        host.has_git_checkpoint = MagicMock(return_value=True)
        host.checkpoint_undo_preview = MagicMock(return_value="Restore checkpoint turn-0003")
        host.undo_last_turn_async = AsyncMock(return_value="restored checkpoint turn-0003")

        app.action_confirm_undo()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        assert "RESTORE GIT CHECKPOINT?" in str(app.screen.query_one("#confirmation-title").render())
        assert "Restore checkpoint turn-0003" in str(
            app.screen.query_one("#confirmation-body").render()
        )
        await pilot.press("y")
        await pilot.pause()
        host.undo_last_turn_async.assert_awaited_once_with(use_checkpoint=True)

        # SessionPickerScreen integration when available on modals_module.
        opened_hosts: list[object] = []

        class FakeSessionPicker(ModalScreen[str | None]):
            def __init__(self, passed_host: object) -> None:
                super().__init__()
                opened_hosts.append(passed_host)

            def compose(self) -> ComposeResult:
                yield Static("fake session picker")

            def on_mount(self) -> None:
                self.dismiss("target-session-id")

        monkeypatch.setattr(modals_module, "SessionPickerScreen", FakeSessionPicker, raising=False)
        await pilot.press("ctrl+o")
        for _ in range(10):
            if host.switch_session.await_count:
                break
            await pilot.pause()
        assert opened_hosts == [host]
        host.switch_session.assert_awaited_once_with("target-session-id")
