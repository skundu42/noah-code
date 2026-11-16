"""Tests for Workstream 2: themes, rich-log tail/selection, sessions, work ledger, diff review, and questions."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from nooa.events import Message, Task
from nooa.runtime import EventManager
from nooa.tools.shell_tools import ShellTools
from nooa.unifiedllm import FakeLLMClient
from pygments.styles import get_style_by_name  # type: ignore[import-untyped]
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.style import Style
from rich.text import Text
from textual.geometry import Offset
from textual.selection import SELECT_ALL, Selection
from textual.strip import Strip
from textual.widgets import Input, OptionList, RichLog, Static

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.checkpoints import CheckpointManager
from noah_code.config import DEFAULT_PERMISSION_RULES, load_config
from noah_code.host import AgentHost
from noah_code.permissions import PermissionEngine
from noah_code.runtime_state import RuntimeStateStore
from noah_code.sessions import SessionStore
from noah_code.snapshots import SnapshotJournal
from noah_code.themes import THEME_NAMES, THEMES, ThemePalette
from noah_code.tools.git_tools import DiffFile, DiffReview, GitTools
from noah_code.tools.process_tools import ProcessTools
from noah_code.tools.question_tools import QuestionAnswer, QuestionPrompt, QuestionTools
from noah_code.tools.task_tools import TaskTools
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.ui import modals
from noah_code.ui.diff_screen import DiffReviewScreen, _diff_renderable, extract_patch_hunks
from noah_code.ui.modals import (
    ActivityHistoryScreen,
    ActivityRecord,
    ConfirmationModal,
    ContextVisibilityScreen,
    QuestionModal,
    SelectableRichLog,
    SessionPickerScreen,
    TextPromptModal,
    TranscriptEntry,
    WorkLedgerScreen,
    _role_renderable,
    extract_fenced_code_blocks,
    get_registered_code_block,
    get_registered_file_ref,
    register_code_block,
    register_file_ref,
)
from noah_code.ui.textual_app import NoahCodeApp, TextualUI
from noah_code.workspace import Workspace
from noah_code.worktree import WorktreeManager

_tui_test_helpers: Any = importlib.import_module("test_textual_tui")
_fake_host: Any = _tui_test_helpers._fake_host
_log_text: Any = _tui_test_helpers._log_text


@pytest.fixture(autouse=True)
def _disable_live_update_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "noah_code.ui.textual_app.maybe_check_for_update",
        lambda **_kwargs: None,
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "mod.py").write_text(
        "\n".join(f"def func_{i}(value: int) -> int:\n    return value + {i}\n" for i in range(1, 12))
    )
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "initial")
    return path


async def _approve_once(_req: object) -> ApprovalChoice:
    return ApprovalChoice.ONCE


# ---------------------------------------------------------------------------
# 1. Theme-Aware Code Syntax Highlighting
# ---------------------------------------------------------------------------


def test_every_theme_palette_maps_to_verified_pygments_style() -> None:
    assert ThemePalette(
        name="atom-one-dark",
        label="x",
        description="x",
        canvas="#000000",
        surface="#111111",
        raised="#222222",
        border="#333333",
        text="#ffffff",
        muted="#888888",
        accent="#aaaaaa",
        success="#00ff00",
        warning="#ffff00",
        error="#ff0000",
    ).code_theme == "monokai"

    for name in THEME_NAMES:
        palette = THEMES[name]
        style_cls = get_style_by_name(palette.code_theme)
        assert style_cls is not None

    assert THEMES["atom-one-dark"].code_theme == "one-dark"
    assert THEMES["atom-one-dark"].is_light is False
    assert THEMES["daylight"].code_theme == "friendly"
    assert THEMES["daylight"].is_light is True
    assert THEMES["high-contrast"].code_theme in {"native", "github-dark"}
    assert THEMES["high-contrast"].is_light is False


def test_role_renderable_passes_palette_code_theme_to_markdown() -> None:
    entry = TranscriptEntry("NOAH", "```python\ndef hi() -> int:\n    return 1\n```", markdown=True)
    for theme_name in ("atom-one-dark", "daylight", "high-contrast"):
        palette = THEMES[theme_name]
        group = _role_renderable(entry, palette)
        md_padding = group.renderables[1]
        md = getattr(md_padding, "renderable", None)
        assert isinstance(md, Markdown)
        assert md.code_theme == palette.code_theme


# ---------------------------------------------------------------------------
# 2. SelectableRichLog Slicing, replace_tail_lines & Safe @click Registries
# ---------------------------------------------------------------------------


def test_selectable_rich_log_get_selection_slices_only_selected_rows() -> None:
    accesses = 0

    class CountingStrip(Strip):
        @property
        def text(self) -> str:
            nonlocal accesses
            accesses += 1
            return super().text

    log = SelectableRichLog()
    log.lines = [CountingStrip.blank(20) for _ in range(2_000)]
    from rich.segment import Segment

    log.lines[500] = CountingStrip([Segment("alpha line 500   ")], 17)
    log.lines[501] = CountingStrip([Segment("bravo line 501   ")], 17)
    log.lines[502] = CountingStrip([Segment("charlie line 502 ")], 17)

    accesses = 0
    sel = Selection(Offset(6, 500), Offset(14, 502))
    extracted, sep = log.get_selection(sel)
    assert sep == "\n"
    assert extracted == "line 500\nbravo line 501\ncharlie line 5"
    assert accesses == 3  # Only the 3 selected rows were read, never all 2,000!

    # Single-line selection
    accesses = 0
    single, _ = log.get_selection(Selection(Offset(0, 501), Offset(5, 501)))
    assert single == "bravo"
    assert accesses == 1

    # Open-start and open-end selections
    head, _ = log.get_selection(Selection(None, Offset(5, 1)))
    assert head == "\n"
    log.lines[1999] = CountingStrip([Segment("omega tail")], 10)
    accesses = 0
    tail, _ = log.get_selection(Selection(Offset(6, 1999), None))
    assert tail == "tail"
    assert accesses == 1


@pytest.mark.asyncio
async def test_selectable_rich_log_replace_tail_lines_updates_in_place(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(100, 30)) as pilot:
        log = app.query_one("#conversation", SelectableRichLog)
        log.styles.display = "block"
        await pilot.pause()

        assert log.replace_tail_lines(0, Text("ignored")) == -1
        assert log.replace_tail_lines(5, Text("too many")) == -1

        log.write(Text("Header row\nSecond row"), scroll_end=False)
        before_tail = len(log.lines)
        log.write(Text("✓ Read a.py"), scroll_end=False)
        old_tail_rows = len(log.lines) - before_tail
        assert old_tail_rows == 1
        await pilot.pause()

        # Warm _line_cache by rendering the tail line before replacing it.
        _ = log.render_line(len(log.lines) - 1)

        new_rows = log.replace_tail_lines(
            old_tail_rows,
            Text("✓ Read a.py, b.py\n✓ Read c.py"),
            scroll_end=False,
        )
        await pilot.pause()
        assert new_rows == 2
        assert len(log.lines) == before_tail + 2
        assert log.virtual_size.height == len(log.lines)
        assert "✓ Read a.py, b.py" in log.render_line(before_tail).text
        assert "✓ Read c.py" in log.render_line(before_tail + 1).text


def test_code_block_and_file_ref_registries_and_role_renderable_links() -> None:
    markdown = (
        "Here are two snippets:\n\n"
        "```python\n"
        "def f(x=\"a'b)\"):\n"
        "    return x\n"
        "```\n\n"
        "Inline ```not a block``` stays inline.\n\n"
        "~~~bash\n"
        "pytest -q\n"
        "~~~\n"
    )
    blocks = extract_fenced_code_blocks(markdown)
    assert blocks == ['def f(x="a\'b)"):\n    return x', "pytest -q"]

    cid = register_code_block(blocks[0])
    assert len(cid) == 12 and all(ch in "0123456789abcdef" for ch in cid)
    assert get_registered_code_block(cid) == blocks[0]

    fid = register_file_ref("src/noah_code/host.py", 142)
    assert len(fid) == 12
    assert get_registered_file_ref(fid) == ("src/noah_code/host.py", 142)

    # Verify bounded LRU eviction at 512 items
    first_id = register_code_block("unique-block-0")
    for idx in range(1, 520):
        register_code_block(f"unique-block-{idx}")
    assert len(modals._CODE_BLOCK_REGISTRY) == 512
    assert get_registered_code_block(first_id) is None

    # Verify NOAH copy action bar (single and multi-block)
    single_entry = TranscriptEntry("NOAH", "```py\nx = 1\n```", markdown=True)
    single_group = _role_renderable(single_entry, THEMES["atom-one-dark"])
    assert len(single_group.renderables) == 3
    single_pad = single_group.renderables[2]
    assert isinstance(single_pad, Padding) and isinstance(single_pad.renderable, Text)
    single_bar = single_pad.renderable
    assert "⎘ Copy code" in single_bar.plain
    assert any(
        "app.copy_code_id(" in str((span.style.meta or {}).get("@click", ""))
        for span in single_bar.spans
        if isinstance(span.style, Style)
    )

    multi_entry = TranscriptEntry("NOAH", markdown, markdown=True)
    multi_group = _role_renderable(multi_entry, THEMES["atom-one-dark"])
    multi_pad = multi_group.renderables[2]
    assert isinstance(multi_pad, Padding) and isinstance(multi_pad.renderable, Text)
    assert "⎘ Copy block 1 · ⎘ Copy block 2" in multi_pad.renderable.plain

    # Verify path:line click links in ACTIVITY/ERROR while preserving activity_history
    act_entry = TranscriptEntry("ACTIVITY", "✗ Check src/noah_code/host.py:88 failed")
    act_group = _role_renderable(act_entry, THEMES["atom-one-dark"])
    console = Console(width=120, force_terminal=True)
    act_pad = act_group.renderables[0]
    assert isinstance(act_pad, Padding) and isinstance(act_pad.renderable, Text)
    segments = list(act_pad.renderable.render(console))
    clicks = {
        seg.text.strip(): str(seg.style.meta.get("@click") or "")
        for seg in segments
        if seg.style and seg.style.meta and seg.text.strip()
    }
    assert any(val == "app.activity_history" for val in clicks.values())
    file_click = clicks.get("src/noah_code/host.py:88", "")
    assert file_click.startswith("app.open_file_id('")
    extracted_ref_id = file_click.split("'")[1]
    assert get_registered_file_ref(extracted_ref_id) == ("src/noah_code/host.py", 88)


@pytest.mark.asyncio
async def test_select_all_copy_remains_byte_identical_with_copy_code_bar(tmp_path: Path) -> None:
    from noah_code.events import HostEvent, HostEventKind

    host = _fake_host(tmp_path)
    ui = TextualUI()
    app = NoahCodeApp(host, ui)
    reply = "Result:\n\n```python\nprint('exact copy')\n```"
    async with app.run_test(size=(90, 30)) as pilot:
        ui.render(HostEvent(HostEventKind.MESSAGE, reply))
        await pilot.pause()
        conv = app.query_one("#conversation", SelectableRichLog)
        assert "⎘ Copy code" in _log_text(conv)
        app.screen.selections = {conv: SELECT_ALL}
        await pilot.press("ctrl+shift+c")
        assert app.clipboard == reply


# ---------------------------------------------------------------------------
# 3. Prompt History, Session Rename/Fork/Delete, Usage & Context Fallbacks
# ---------------------------------------------------------------------------


def test_session_store_prompt_history_atomic_dedup_and_family_sharing(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    store = SessionStore(tmp_path / "data" / "sessions")
    ws_main = Workspace(root=repo)
    wt = WorktreeManager(repo, tmp_path / "data" / "worktree").create("feat")
    ws_wt = Workspace(root=wt.directory)

    store.record_prompt_history(ws_main, "   ")
    store.record_prompt_history(ws_main, "/help")
    store.record_prompt_history(ws_main, "  first prompt  ")
    store.record_prompt_history(ws_wt, "second prompt")
    store.record_prompt_history(ws_main, "first prompt")  # moves back to top

    history = store.list_prompt_history(ws_wt)
    assert history == ["first prompt", "second prompt"]

    hist_files = list((store.session_dir / ".prompt_history").glob("*.json"))
    assert len(hist_files) == 1
    assert (hist_files[0].stat().st_mode & 0o777) == 0o600

    # Cap at 100 items and 8000 chars per item
    store.record_prompt_history(ws_main, "x" * 9000)
    assert len(store.list_prompt_history(ws_main)[0]) == 8000
    for idx in range(110):
        store.record_prompt_history(ws_main, f"prompt-{idx}")
    capped = store.list_prompt_history(ws_main, limit=200)
    assert len(capped) == 100
    assert capped[0] == "prompt-109"


@pytest.mark.asyncio
async def test_session_store_and_host_rename_fork_delete_and_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    session_dir = tmp_path / "sessions"
    ws = Workspace(root=repo)
    config = load_config(repo, cli_overrides={"session_dir": str(session_dir), "auto_approve": True})
    store = SessionStore(session_dir)

    host = AgentHost(ws, config, llm=FakeLLMClient(), store=store)
    meta = await host.start()
    try:
        # Record an event + journal blob in the active session before forking
        assert host._storage is not None
        EventManager(host._storage.event_backend).add(Task(prompt="original prompt"))
        EventManager(host._storage.event_backend).add(Message(content="original answer"))
        target_file = repo / "mod.py"
        host.agent.journal.begin_turn()
        mut = host.agent.journal.record_preimage(target_file)
        target_file.write_text("# edited\n")
        host.agent.journal.record_postimage(mut, target_file)
        host.agent.journal.end_turn()
        await host._persist_async()

        # Rename active session
        renamed = await host.rename_session(meta.session_id, "Auth Refactor")
        assert renamed.title == "Auth Refactor"
        assert host.meta is not None and host.meta.title == "Auth Refactor"

        # Active session cannot be deleted while active
        with pytest.raises(ValueError, match="Switch to another session"):
            await host.delete_session_by_id(meta.session_id)

        # Fork active session -> switches host onto forked copy with history & journal preserved
        forked = await host.fork_session(meta.session_id)
        assert forked.session_id != meta.session_id
        assert forked.title == "Auth Refactor (fork)"
        assert host.meta.session_id == forked.session_id
        events = store.load_event_page(forked.session_id)
        assert any(e.payload.get("prompt") == "original prompt" for e in events)
        assert (store.session_dir / forked.session_id / "journal.json").is_file()

        # Now original session is non-active and can be deleted cleanly
        await host.delete_session_by_id(meta.session_id)
        assert not (store.session_dir / meta.session_id).exists()

        # Checkpoint undo preview helpers
        assert host.has_git_checkpoint() is False
        assert host.checkpoint_undo_preview() is None
        cp = CheckpointManager(repo, forked.session_id)
        cp.capture("before shell")
        host._checkpoints = cp
        assert host.has_git_checkpoint() is True
        preview = host.checkpoint_undo_preview()
        assert preview is not None and "Git checkpoint" in preview

        # Context snapshot dynamic instruction discovery hook
        fake_instr = types.ModuleType("noah_code.instructions")

        @dataclass
        class FakeInstr:
            label: str
            path: Path
            scope: str

        fake_instr.discover_instruction_files = lambda _root: [  # type: ignore[attr-defined]
            FakeInstr("AGENTS.md", repo / "AGENTS.md", "repo"),
            FakeInstr("~/.noah-code/instructions.md", tmp_path / "user.md", "user"),
        ]
        monkeypatch.setitem(sys.modules, "noah_code.instructions", fake_instr)
        ctx = host.context_snapshot()
        instr_rows = [r for r in ctx if r["kind"] == "instruction"]
        assert instr_rows == [
            {"kind": "instruction", "label": "AGENTS.md", "detail": f"Repository instructions · {repo / 'AGENTS.md'}"},
            {"kind": "instruction", "label": "~/.noah-code/instructions.md", "detail": f"User instructions · {tmp_path / 'user.md'}"},
        ]

        # Live context-window fallback on host.usage_snapshot()
        snap = host.usage_snapshot()
        if hasattr(snap, "context_window"):
            assert getattr(snap, "context_window", 0) > 0
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_session_picker_screen_rename_delete_fork_and_active_guard(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    host.meta.session_id = "111111111111"
    metas = [
        SimpleNamespace(
            session_id="111111111111",
            title="Current work",
            mode="build",
            model="gpt-test",
            worktree_name="",
        ),
        SimpleNamespace(
            session_id="222222222222",
            title="Old spike",
            mode="plan",
            model="gpt-test",
            worktree_name="calm-river",
        ),
    ]
    host.list_session_metas.side_effect = lambda: list(metas)

    async def fake_rename(sid: str, title: str):
        for m in metas:
            if m.session_id == sid:
                m.title = title
                return m
        raise AssertionError("missing session")

    async def fake_delete(sid: str):
        metas[:] = [m for m in metas if m.session_id != sid]

    host.rename_session = AsyncMock(side_effect=fake_rename)
    host.delete_session_by_id = AsyncMock(side_effect=fake_delete)
    host.fork_session = AsyncMock()
    host.create_worktree_session = AsyncMock()

    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(110, 32)) as pilot:
        picker = SessionPickerScreen(host)
        await app.push_screen(picker)
        await pilot.pause()

        # Guard deleting active session (row 0)
        picker.action_delete_session()
        await pilot.pause()
        host.delete_session_by_id.assert_not_awaited()
        assert "Switch to another session" in str(picker.query_one("#picker-hint", Static).render())

        # Rename row 0 via Ctrl+R -> TextPromptModal
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert isinstance(app.screen, TextPromptModal)
        app.screen.query_one("#prompt-input", Input).value = "Renamed Current"
        await pilot.press("enter")
        for _ in range(20):
            if host.rename_session.await_count:
                break
            await pilot.pause()
        host.rename_session.assert_awaited_once_with("111111111111", "Renamed Current")
        opt0_prompt = picker.query_one("#picker-list", OptionList).get_option_at_index(0).prompt
        assert "Renamed Current" in str(opt0_prompt)

        # Move to non-active session (row 1) and delete via Delete key + ConfirmationModal
        await pilot.press("down")
        await pilot.press("delete")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        await pilot.press("y")
        for _ in range(20):
            if host.delete_session_by_id.await_count:
                break
            await pilot.pause()
        host.delete_session_by_id.assert_awaited_once_with("222222222222")
        assert picker.query_one("#picker-list", OptionList).option_count == 1

        # Fork via Ctrl+F dismisses picker after forking
        await pilot.press("ctrl+f")
        for _ in range(20):
            if host.fork_session.await_count:
                break
            await pilot.pause()
        host.fork_session.assert_awaited_once_with("111111111111")
        assert not isinstance(app.screen, SessionPickerScreen)


# ---------------------------------------------------------------------------
# 4. Actionable WorkLedgerScreen, ProcessTools.peek_logs & TaskTools.snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_tools_peek_logs_and_task_tools_durable_snapshot(tmp_path: Path) -> None:
    ws = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    approvals = ApprovalBroker(engine, handler=_approve_once)
    journal = SnapshotJournal()
    ws_tools = WorkspaceTools(ws, ShellTools(cwd=str(tmp_path)), engine, approvals, journal)
    runtime = RuntimeStateStore(tmp_path / "session")
    proc_tools = ProcessTools(ws_tools, runtime=runtime)
    try:
        started = await proc_tools.start(
            f"{sys.executable} -c \"import sys; sys.stdout.write('out-line\\n'); sys.stderr.write('err-line\\n')\"",
            name="check",
        )
        job_id = started.split()[1]
        for _ in range(60):
            if proc_tools.snapshot()[0]["state"] != "running":
                break
            await asyncio.sleep(0.05)

        cursor_before = proc_tools.snapshot()[0]["cursor"]
        peeked = proc_tools.peek_logs(job_id)
        assert "out-line" in peeked
        assert "! err-line" in peeked
        assert proc_tools.snapshot()[0]["cursor"] == cursor_before
        # Model logs(cursor=0) still receives full output untouched!
        full_logs = await proc_tools.logs(job_id, cursor=0)
        assert "out-line" in full_logs and "! err-line" in full_logs

        # Evict in-memory job and verify durable JSONL log tailing fallback in peek_logs
        proc_tools._jobs.pop(job_id, None)
        durable_peeked = proc_tools.peek_logs(job_id)
        assert "out-line" in durable_peeked
        assert "! err-line" in durable_peeked

        # Durable TaskTools.snapshot includes isolated records loaded from RuntimeStateStore
        runtime.set_state(
            "child_sessions",
            {
                "abcdef123456": {
                    "id": "abcdef123456",
                    "agent": "general",
                    "prompt": "Implement isolated feature",
                    "directory": str(tmp_path / "wt"),
                    "isolated": True,
                    "applied": True,
                    "worktree_cleaned": False,
                    "state": "completed",
                    "result": "Implemented cleanly",
                }
            },
        )
        parent_stub = SimpleNamespace(_runtime=runtime, _config=load_config(tmp_path))
        tasks = TaskTools(ws, engine, approvals, parent=parent_stub)
        snap = tasks.snapshot()
        assert len(snap) == 1
        assert snap[0]["id"] == "abcdef123456"
        assert snap[0]["isolated"] is True
        assert snap[0]["applied"] is True
        assert snap[0]["result_preview"] == "Implemented cleanly"
    finally:
        await proc_tools.close()
        await ws_tools.close()
        runtime.close()


@pytest.mark.asyncio
async def test_work_ledger_live_logs_badges_actions_search_guard_and_tabs(tmp_path: Path) -> None:
    host = _fake_host(tmp_path)
    state_box = {
        "agents": [
            {
                "id": "aa11bb22cc33",
                "agent": "general",
                "prompt": "Build auth patch",
                "mode": "build",
                "readonly": False,
                "state": "completed",
                "result_preview": "Ready to apply",
                "duration": 3.2,
                "isolated": True,
                "applied": False,
                "worktree_cleaned": False,
            }
        ],
        "jobs": [
            {
                "id": "job12345",
                "name": "pytest-watch",
                "kind": "process",
                "state": "running",
                "command": "pytest -q",
                "elapsed": 1.5,
                "returncode": None,
                "cursor": 2,
            }
        ],
    }
    host.work_snapshot.side_effect = lambda: state_box
    host.work_job_logs.return_value = "collected 12 items\n! 1 warning"
    host.stop_work_item = AsyncMock(return_value="stopped")
    host.follow_up_work_agent = AsyncMock(return_value="queued")
    host.work_agent_diff = AsyncMock(return_value="@@ -1 +1 @@\n-old_auth\n+new_auth")
    host.apply_work_agent = AsyncMock(return_value="Applied 1 file")
    host.cleanup_work_agent = AsyncMock(return_value="Cleaned up worktree")

    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(130, 40)) as pilot:
        await pilot.press("f4")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, WorkLedgerScreen)

        # Job is running (sorted before completed), shows live stdout/stderr logs in #work-detail
        detail_log = screen.query_one("#work-detail", RichLog)
        detail_text = _log_text(detail_log)
        assert "OUTPUT" in detail_text
        assert "collected 12 items" in detail_text
        assert "! 1 warning" in detail_text

        # Typing action letters k/d/a/f/x in #work-filter types into search Input without firing actions!
        await pilot.press("slash")
        search = screen.query_one("#work-filter", Input)
        assert screen.focused is search
        await pilot.press("k", "d", "a", "f", "x")
        await pilot.pause()
        assert search.value == "kdafx"
        host.stop_work_item.assert_not_awaited()
        search.value = ""
        await pilot.press("enter")
        await pilot.pause()

        # Stop running job via 'k' + ConfirmationModal
        await pilot.press("k")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        await pilot.press("y")
        for _ in range(20):
            if host.stop_work_item.await_count:
                break
            await pilot.pause()
        host.stop_work_item.assert_awaited_once_with("job", "job12345")

        # Select isolated child agent -> check [isolated] badge, preview diff ('d'), apply ('a'), follow-up ('f'), cleanup ('x')
        work_list = screen.query_one("#work-list", OptionList)
        assert "[isolated]" in str(work_list.get_option_at_index(1).prompt)
        work_list.highlighted = 1
        await pilot.pause()

        await pilot.press("d")
        for _ in range(20):
            if host.work_agent_diff.await_count:
                break
            await pilot.pause()
        assert "+new_auth" in _log_text(detail_log)

        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        state_box["agents"][0]["applied"] = True
        await pilot.press("y")
        for _ in range(20):
            if host.apply_work_agent.await_count:
                break
            await pilot.pause()
        host.apply_work_agent.assert_awaited_once_with("aa11bb22cc33")
        assert "[applied]" in str(work_list.get_option_at_index(1).prompt)

        await pilot.press("f")
        await pilot.pause()
        assert isinstance(app.screen, TextPromptModal)
        app.screen.query_one("#prompt-input", Input).value = "Add regression test"
        await pilot.press("enter")
        for _ in range(20):
            if host.follow_up_work_agent.await_count:
                break
            await pilot.pause()
        host.follow_up_work_agent.assert_awaited_once_with("aa11bb22cc33", "Add regression test")

        await pilot.press("x")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        await pilot.press("y")
        for _ in range(20):
            if host.cleanup_work_agent.await_count:
                break
            await pilot.pause()
        host.cleanup_work_agent.assert_awaited_once_with("aa11bb22cc33")

        # Unified inspector tabs: Alt+3 -> ContextVisibilityScreen -> Alt+1 -> ActivityHistoryScreen -> Alt+2 -> WorkLedgerScreen
        app._timeline_history.append(ActivityRecord("act-1", "Read a.py", state="complete"))
        await pilot.press("alt+3")
        await pilot.pause()
        assert isinstance(app.screen, ContextVisibilityScreen)

        await pilot.press("alt+1")
        await pilot.pause()
        assert isinstance(app.screen, ActivityHistoryScreen)

        await pilot.press("alt+2")
        await pilot.pause()
        assert isinstance(app.screen, WorkLedgerScreen)


# ---------------------------------------------------------------------------
# 5. Upgraded DiffReviewScreen & GitTools Hunk Stage / Hunk Revert
# ---------------------------------------------------------------------------


def test_diff_renderable_syntax_highlighting_and_intraline_word_diff() -> None:
    patch = (
        "--- a/sample.py\n"
        "+++ b/sample.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def compute(old_arg: int) -> int:\n"
        "+def compute(new_arg: int) -> int:\n"
        "     return 42\n"
    )
    for theme_name in ("atom-one-dark", "daylight"):
        palette = THEMES[theme_name]
        group = _diff_renderable(patch, palette, path="sample.py")
        del_line = group.renderables[3]
        add_line = group.renderables[4]
        assert isinstance(del_line, Text) and isinstance(add_line, Text)

        console = Console(width=120, color_system="truecolor")
        del_segs = [seg for seg in del_line.render(console) if seg.text]
        add_segs = [seg for seg in add_line.render(console) if seg.text]

        # Changed span ('old' vs 'new') has bold + emphasized background distinct from unchanged 'compute('
        del_unchanged_bg = next(seg.style.bgcolor for seg in del_segs if "compute" in seg.text and seg.style)
        del_changed = next(seg for seg in del_segs if "old" in seg.text and seg.style)
        assert del_changed.style is not None
        assert del_changed.style.bold is True
        assert del_changed.style.bgcolor != del_unchanged_bg

        unchanged_bg = next(seg.style.bgcolor for seg in add_segs if "compute" in seg.text and seg.style)
        changed_seg = next(seg for seg in add_segs if "new" in seg.text and seg.style)
        assert changed_seg.style is not None
        assert changed_seg.style.bold is True
        assert changed_seg.style.bgcolor != unchanged_bg


@pytest.mark.asyncio
async def test_git_tools_stage_hunk_unstage_hunk_and_journaled_revert_hunk(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    ws = Workspace(repo.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    journal = SnapshotJournal()
    ws_tools = WorkspaceTools(
        ws,
        ShellTools(cwd=str(repo)),
        engine,
        ApprovalBroker(engine, handler=_approve_once),
        journal,
    )
    git = GitTools(ws_tools)
    try:
        mod = repo / "mod.py"
        original_text = mod.read_text()
        lines = original_text.splitlines()
        lines[1] = "    return value + 100"
        lines[-1] = "    return value + 999"
        mod.write_text("\n".join(lines) + "\n")

        review = await git.review()
        assert len(review.files) == 1
        item = review.files[0]
        hunks = extract_patch_hunks(item.patch, item.path)
        assert len(hunks) == 2

        # Stage only hunk 1; hunk 2 remains unstaged!
        msg = await git.stage_hunk(item.path, "unstaged", hunks[0], expected_revision=item.revision)
        assert "staged hunk" in msg
        review2 = await git.review()
        assert {(f.path, f.scope) for f in review2.files} == {
            ("mod.py", "staged"),
            ("mod.py", "unstaged"),
        }

        # Unstage the staged hunk
        staged_item = next(f for f in review2.files if f.scope == "staged")
        staged_hunks = extract_patch_hunks(staged_item.patch, staged_item.path)
        unstage_msg = await git.stage_hunk(
            staged_item.path, "staged", staged_hunks[0], expected_revision=staged_item.revision
        )
        assert "unstaged hunk" in unstage_msg

        # Revert only hunk 2 (unstaged) inside a journal turn, then verify /undo restores hunk 2!
        review3 = await git.review()
        unstaged_item = review3.files[0]
        hunks3 = extract_patch_hunks(unstaged_item.patch, unstaged_item.path)
        assert len(hunks3) == 2
        journal.begin_turn()
        rev_msg = await git.revert_hunk(
            unstaged_item.path, "unstaged", hunks3[1], expected_revision=unstaged_item.revision
        )
        journal.end_turn()
        assert "reverted unstaged hunk" in rev_msg
        after_revert = mod.read_text()
        assert "value + 100" in after_revert
        assert "value + 999" not in after_revert

        # Journal undo restores hunk 2 cleanly!
        journal.undo()
        assert "value + 999" in mod.read_text()
    finally:
        await ws_tools.close()


@pytest.mark.asyncio
async def test_diff_review_screen_wrap_split_hunk_actions_confirm_revert_and_checkpoint_undo(
    tmp_path: Path,
) -> None:
    host = _fake_host(tmp_path)
    host._turn_running.return_value = False
    host.agent.lsp.document_symbols = AsyncMock(return_value="func_1")
    host.diff_diagnostics = AsyncMock()
    host.agent.git._review_signature = AsyncMock(return_value="rev-1")

    patch_text = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def func_1(x: int) -> int:\n"
        "-    return x + 1\n"
        "+    return x + 100\n"
        "@@ -20,3 +20,3 @@\n"
        " def func_9(x: int) -> int:\n"
        "-    return x + 9\n"
        "+    return x + 900\n"
    )
    item = DiffFile(
        "mod.py",
        "unstaged",
        "modified",
        additions=2,
        deletions=2,
        patch=patch_text,
        loaded=True,
        revision="rev-1",
    )
    host.diff_review = AsyncMock(return_value=DiffReview([item]))
    host.stage_diff_hunk = AsyncMock(return_value="staged hunk in mod.py")
    host.revert_diff_hunk = AsyncMock(return_value="reverted unstaged hunk in mod.py")
    host.revert_diff_file = AsyncMock(return_value="reverted unstaged changes in mod.py")

    # Configure shell-bypassed journal turn + Git checkpoint fallback for 'u'
    journal = SnapshotJournal()
    p = tmp_path / "mod.py"
    p.write_text("before\n")
    journal.begin_turn()
    mut = journal.record_preimage(p)
    p.write_text("after\n")
    journal.record_postimage(mut, p)
    journal.mark_shell_bypass()
    journal.end_turn()
    host.agent.journal = journal
    host.has_git_checkpoint.return_value = True
    host.checkpoint_undo_preview.return_value = (
        "Journal undo is unavailable because shell commands ran, "
        "but Git checkpoint refs/noah-code/checkpoints/s/00000001 (deadbeefcafe) can restore the worktree."
    )
    host.undo_last_turn_async = AsyncMock(return_value="restored Git checkpoint")

    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(120, 40)) as pilot:
        screen = DiffReviewScreen(host, DiffReview([item]))
        await app.push_screen(screen)
        await pilot.pause()

        patch_log = screen.query_one("#diff-patch", RichLog)
        assert patch_log.wrap is False

        # 'w' toggles soft-wrap
        await pilot.press("w")
        await pilot.pause()
        assert patch_log.wrap is True
        assert "Soft wrap on" in str(screen.query_one("#diff-status", Static).render())

        # 's' toggles side-by-side split view
        await pilot.press("s")
        await pilot.pause()
        assert screen._split_view is True
        assert "│" in _log_text(patch_log)
        await pilot.press("s")
        await pilot.pause()
        assert screen._split_view is False

        # Navigate to hunk 2 ('n', 'n') and stage hunk ('g')
        await pilot.press("n", "n")
        await pilot.pause()
        assert screen._hunk_index == 1
        await pilot.press("g")
        for _ in range(20):
            if host.stage_diff_hunk.await_count:
                break
            await pilot.pause()
        staged_arg = host.stage_diff_hunk.await_args.args[2]
        assert "@@ -20,3 +20,3 @@" in staged_arg
        assert "@@ -1,3 +1,3 @@" not in staged_arg

        # Revert hunk ('h') opens ConfirmationModal
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        await pilot.press("y")
        for _ in range(20):
            if host.revert_diff_hunk.await_count:
                break
            await pilot.pause()
        host.revert_diff_hunk.assert_awaited_once()

        # File revert ('r') uses ConfirmationModal (no typing REVERT!)
        await pilot.press("r")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        await pilot.press("y")
        for _ in range(20):
            if host.revert_diff_file.await_count:
                break
            await pilot.pause()
        host.revert_diff_file.assert_awaited_once_with(
            "mod.py", "unstaged", expected_revision="rev-1"
        )

        # Undo ('u') offers Git checkpoint ConfirmationModal when journal turn used shell mutations
        await pilot.press("u")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationModal)
        assert "Git checkpoint" in str(app.screen.query_one("#confirmation-body", Static).render())
        await pilot.press("y")
        for _ in range(20):
            if host.undo_last_turn_async.await_count:
                break
            await pilot.pause()
        host.undo_last_turn_async.assert_awaited_once_with(use_checkpoint=True)


# ---------------------------------------------------------------------------
# 6. Multi-Select & Rich Option Descriptions in QuestionModal & QuestionTools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_question_tools_multi_select_and_modal_space_toggle_and_descriptions(
    tmp_path: Path,
) -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    approvals = ApprovalBroker(engine, handler=_approve_once)

    captured_prompts: list[QuestionPrompt] = []

    async def handler(prompts: list[QuestionPrompt]) -> QuestionAnswer:
        captured_prompts.extend(prompts)
        return QuestionAnswer(selections=[prompts[0].options[0], prompts[0].options[2]])

    qtools = QuestionTools(engine, approvals, handler=handler)
    res = await qtools.question(
        "Checks",
        "Which checks should run?",
        ["pytest — Unit test suite", "ruff - Fast linter", "mypy — Static types"],
        multi_select=True,
    )
    assert captured_prompts[0].multi_select is True
    assert "pytest — Unit test suite; mypy — Static types" in res

    host = _fake_host(tmp_path)
    app = NoahCodeApp(host, TextualUI())
    async with app.run_test(size=(90, 28)) as pilot:
        # Case A: multi_select=True starts in checkbox mode; 1-9 toggles; descriptions styled muted
        prompt_multi = QuestionPrompt(
            "Checks",
            "Pick checks",
            ("pytest — Unit tests", "ruff - Linter", "mypy — Typecheck"),
            multi_select=True,
        )
        out_multi: list[QuestionAnswer | None] = []
        app.run_worker(_push_and_collect(app, QuestionModal(prompt_multi), out_multi))
        await pilot.pause()
        qlist = app.screen.query_one("#question-list", OptionList)
        first_opt = qlist.get_option_at_index(0).prompt
        assert isinstance(first_opt, Text)
        assert "[ ]" in first_opt.plain
        assert any(span.style == app.theme_palette.muted for span in first_opt.spans)

        await pilot.press("1", "3", "enter")
        for _ in range(20):
            if out_multi:
                break
            await pilot.pause()
        assert out_multi == [
            QuestionAnswer(selections=["pytest — Unit tests", "mypy — Typecheck"], custom="")
        ]

        # Case B: multi_select=False supports Space opt-in multi-select while keeping exact option strings
        prompt_opt_in = QuestionPrompt(
            "Targets",
            "Choose targets",
            ("linux — x86_64", "macos - arm64", "windows — x64"),
            multi_select=False,
        )
        out_opt_in: list[QuestionAnswer | None] = []
        app.run_worker(_push_and_collect(app, QuestionModal(prompt_opt_in), out_opt_in))
        await pilot.pause()
        await pilot.press("space", "down", "space", "enter")
        for _ in range(20):
            if out_opt_in:
                break
            await pilot.pause()
        assert out_opt_in == [
            QuestionAnswer(selections=["linux — x86_64", "macos - arm64"], custom="")
        ]


async def _push_and_collect(app: NoahCodeApp, modal: QuestionModal, sink: list[QuestionAnswer | None]) -> None:
    sink.append(await app.push_screen_wait(modal))
