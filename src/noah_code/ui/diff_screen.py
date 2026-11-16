"""Read-only diff review screen with accurate hunk offset navigation and explicit mutations."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.measure import measure_renderables
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from noah_code.themes import THEMES, ThemePalette
from noah_code.ui.modals import (
    ConfirmationModal,
    SelectableRichLog,
    TextPromptModal,
    _undo_preview,
)

if TYPE_CHECKING:
    from noah_code.host import AgentHost


def _split_patch_lines(patch: str) -> list[str]:
    text = patch or "(no textual diff)"
    raw_lines = text.split("\n")
    if text.endswith("\n"):
        raw_lines.pop()
    return [line.rstrip("\r") for line in raw_lines]


def _style_diff_line(raw: str, palette: ThemePalette) -> Text:
    sanitized = raw.replace("\v", " ").replace("\f", " ").replace("\r", "")
    if sanitized.startswith("@@"):
        closing = sanitized.find("@@", 2)
        if closing != -1:
            header = sanitized[: closing + 2]
            context = sanitized[closing + 2 :]
            line = Text(no_wrap=False)
            line.append(
                header,
                style=Style(color=palette.accent, bgcolor=palette.raised, bold=True),
            )
            if context:
                line.append(context, style=palette.muted)
            return line
        style: str | Style = f"bold {palette.accent}"
    elif sanitized.startswith(("diff ", "--- ", "+++ ")):
        style = f"bold {palette.text}"
    elif sanitized.startswith(("index ", "new file", "deleted file", "similarity ", "rename ")):
        style = palette.muted
    elif sanitized.startswith("+"):
        style = palette.success
    elif sanitized.startswith("-"):
        style = palette.error
    else:
        style = palette.text
    return Text(sanitized, style=style, no_wrap=False)


def _diff_renderable(patch: str, palette: ThemePalette | None = None) -> Group:
    """Render a readable unified diff without interpreting arbitrary markup."""

    resolved = palette or THEMES["atom-one-dark"]
    return Group(*(_style_diff_line(raw, resolved) for raw in _split_patch_lines(patch)))


def _recorded_checks_text(results: list[dict[str, Any]]) -> str:
    """Summarize the latest attempt of each shared check against current code."""

    latest = {
        (record["source"], record.get("cwd"), record["command"]): record for record in results
    }
    if not latest:
        return "checks: no results recorded"
    if len(latest) > 3:
        counts = Counter(record["state"] for record in latest.values())
        return "recorded check commands: " + ", ".join(
            f"{count} {state}" for state, count in counts.items()
        )
    checks = []
    for record in latest.values():
        status = record["state"]
        if status == "failed":
            status += f" (exit {record['returncode']})"
        elif status == "stale":
            status += f" (last exit {record['returncode']})"
        elif status == "unknown":
            status = "unverified (workspace state unavailable)"
        checks.append(f"{record['label']} {status}")
    return "recorded check commands: " + ", ".join(checks)


class DiffReviewScreen(ModalScreen[None]):
    """Read-only, progressively loaded review with explicit mutation controls."""

    BINDINGS = [
        Binding("j,down", "next_file", "Next file", show=True),
        Binding("k,up", "previous_file", "Previous file", show=True),
        Binding("n,]", "next_hunk", "Next hunk", show=True),
        Binding("p,[", "previous_hunk", "Previous hunk", show=True),
        Binding("slash", "filter", "Find file", show=True),
        Binding("ctrl+r", "refresh_review", "Refresh", show=True),
        Binding("c", "copy_patch", "Copy patch", show=True),
        Binding("o", "open_editor", "Open editor", show=True),
        Binding("v", "toggle_diagnostics", "Diagnostics", show=True),
        Binding("r", "revert", "Revert file", show=True),
        Binding("u", "undo", "Undo turn", show=True),
        Binding("escape,q", "close", "Close", show=True),
    ]

    def __init__(self, host: AgentHost, review: Any) -> None:
        super().__init__()
        self.host = host
        self.review = review
        self._by_option: dict[str, Any] = {}
        self._symbols: dict[str, str] = {}
        self._selected_key: str | None = None
        self._preamble_lines: list[Text] = []
        self._hunk_blocks: list[list[Text]] = []
        self._hunks: list[int] = []
        self._hunk_layout_key: tuple[int, bool, int] | None = None
        self._hunk_index = -1

    def compose(self) -> ComposeResult:
        with Vertical(id="diff-dialog"):
            yield Static("CHANGE REVIEW", id="diff-title")
            yield Static("", id="diff-summary")
            yield Input(placeholder="Filter files… (/ to focus)", id="diff-filter")
            with Horizontal(id="diff-body"):
                yield OptionList(id="diff-files", compact=True)
                with Vertical(id="diff-inspector"):
                    yield Static("", id="diff-file-header")
                    yield SelectableRichLog(
                        id="diff-patch",
                        markup=False,
                        highlight=False,
                        wrap=False,
                        min_width=0,
                        max_lines=None,
                    )
                    yield Static("", id="diff-validation")
            yield Static("", id="diff-status", markup=False)
            yield Static(
                "J/K file · N/P hunk · / find · C copy · O editor · V diagnostics · Ctrl+R refresh · R revert · U undo · Esc close",
                id="diff-hint",
            )

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 100, "narrow")
        self._render_review()
        self.query_one("#diff-files", OptionList).focus()

    def on_resize(self) -> None:
        self.set_class(self.app.size.width < 100, "narrow")
        self._hunk_layout_key = None
        self.call_after_refresh(self._recompute_hunk_offsets)

    def _render_review(self) -> None:
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        query = self.query_one("#diff-filter", Input).value.casefold()
        files = [item for item in self.review.files if query in item.path.casefold()]
        captured = getattr(self.review, "captured_at", time.time())
        stamp = time.strftime("%H:%M:%S", time.localtime(captured))
        self.query_one("#diff-summary", Static).update(
            Text(
                f"{len(files)}/{len(self.review.files)} file views · list captured {stamp} · "
                "details load on selection; Ctrl+R refreshes",
                style=palette.muted,
            ),
            layout=False,
        )
        option_list = self.query_one("#diff-files", OptionList)
        option_list.clear_options()
        self._by_option.clear()
        selected = 0
        for index, item in enumerate(files):
            option_id = f"change-{index}"
            self._by_option[option_id] = item
            if item.key == self._selected_key:
                selected = index
            stage = "S" if item.scope == "staged" else "U"
            prompt = Text(f"{stage} ", style=f"bold {palette.accent}")
            prompt.append(f"{item.path}\n", style=palette.text)
            prompt.append(f"   {item.status}", style=palette.muted)
            if getattr(item, "loaded", bool(item.patch)):
                prompt.append(f"  +{item.additions}", style=palette.success)
                prompt.append(f" -{item.deletions}", style=palette.error)
            option_list.add_option(Option(prompt, id=option_id))
        if files:
            option_list.highlighted = selected
            self._show_item(files[selected])
        else:
            option_list.add_option(
                Option("No matching files" if query else "Working tree clean", disabled=True)
            )
            self._selected_key = None
            self._preamble_lines = []
            self._hunk_blocks = []
            self._hunks = []
            self._hunk_layout_key = None
            self.query_one("#diff-file-header", Static).update("")
            self.query_one("#diff-patch", RichLog).clear()
            self.query_one("#diff-validation", Static).update("")

    @on(Input.Changed, "#diff-filter")
    def _filter_changed(self) -> None:
        self._render_review()

    @on(Input.Submitted, "#diff-filter")
    def _filter_submitted(self) -> None:
        self.query_one("#diff-files", OptionList).focus()

    def action_filter(self) -> None:
        self.query_one("#diff-filter", Input).focus()

    @on(OptionList.OptionHighlighted, "#diff-files")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id and event.option.id in self._by_option:
            item = self._by_option[event.option.id]
            if item.key != self._selected_key:
                self._show_item(item)

    def _prepare_hunk_blocks(self, raw_patch: str, palette: ThemePalette) -> Group:
        preamble: list[Text] = []
        blocks: list[list[Text]] = []
        logical_offsets: list[int] = []
        styled_all: list[Text] = []
        raw_lines = _split_patch_lines(raw_patch) if raw_patch else []
        for index, raw in enumerate(raw_lines):
            styled = _style_diff_line(raw, palette)
            styled_all.append(styled)
            if raw.startswith("@@"):
                logical_offsets.append(index)
                blocks.append([styled])
            elif blocks:
                blocks[-1].append(styled)
            else:
                preamble.append(styled)
        self._preamble_lines = preamble
        self._hunk_blocks = blocks
        self._hunks = logical_offsets
        self._hunk_layout_key = None
        if not styled_all:
            return _diff_renderable("Loading patch…", palette)
        return Group(*styled_all)

    def _block_visual_rows(
        self,
        lines: list[Text],
        *,
        width: int,
        min_width: int,
        wrap: bool,
    ) -> int:
        if not lines:
            return 0
        if not wrap or width <= 0:
            return len(lines)
        console = self.app.console
        base_options = console.options
        group = Group(*lines)
        measured = measure_renderables(console, base_options, [group]).maximum
        render_width = max(min(measured, width), min_width)
        segments = console.render(group, base_options.update_width(render_width))
        return sum(1 for _ in Segment.split_lines(segments))

    def _recompute_hunk_offsets(self) -> None:
        if not self._hunk_blocks:
            self._hunks = []
            return
        try:
            patch = self.query_one("#diff-patch", RichLog)
        except Exception:  # noqa: BLE001
            return
        width = patch.scrollable_content_region.width
        wrap = bool(patch.wrap)
        min_width = patch.min_width
        layout_key = (width, wrap, min_width)
        if layout_key == self._hunk_layout_key:
            return
        offset = self._block_visual_rows(
            self._preamble_lines,
            width=width,
            min_width=min_width,
            wrap=wrap,
        )
        offsets: list[int] = []
        for index, block in enumerate(self._hunk_blocks):
            offsets.append(offset)
            if index + 1 < len(self._hunk_blocks):
                offset += self._block_visual_rows(
                    block,
                    width=width,
                    min_width=min_width,
                    wrap=wrap,
                )
        self._hunks = offsets
        if width > 0 or not wrap:
            self._hunk_layout_key = layout_key

    def _show_item(self, item: Any) -> None:
        self._selected_key = item.key
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        captured = getattr(item, "captured_at", 0)
        stamp = (
            time.strftime("%H:%M:%S", time.localtime(captured))
            if captured
            else "not recorded"
            if item.patch
            else "pending"
        )
        self.query_one("#diff-file-header", Static).update(
            Text.assemble(
                (item.path, f"bold {palette.text}"),
                (f"   {item.scope} · patch {stamp}", palette.muted),
            ),
            layout=False,
        )
        patch = self.query_one("#diff-patch", RichLog)
        patch.clear()
        renderable = self._prepare_hunk_blocks(item.patch, palette)
        patch.write(renderable, scroll_end=False)
        patch.scroll_home(animate=False)
        self._recompute_hunk_offsets()
        self.call_after_refresh(self._recompute_hunk_offsets)
        self._hunk_index = -1
        self.query_one("#diff-validation", Static).update(
            self._validation_text(item, self._symbols.get(item.path, "Loading symbols…")),
            layout=False,
        )
        if not getattr(item, "loaded", bool(item.patch)):
            self._load_item(item)
        elif item.diagnostics == "pending" or item.path not in self._symbols:
            self._load_diagnostics(item)

    @work(exclusive=True, group="diff-details")
    async def _load_item(self, item: Any) -> None:
        try:
            await self.host.diff_review_file(item)
        except Exception as exc:  # noqa: BLE001
            item.patch = f"Patch unavailable: {exc}"
            item.loaded = True
        if any(current is item for current in self.review.files) and self._selected_key == item.key:
            self._show_item(item)

    @work(exclusive=True, group="diff-diagnostics")
    async def _load_diagnostics(self, item: Any) -> None:
        try:
            async with asyncio.timeout(5):
                if item.diagnostics == "pending":
                    await self.host.diff_diagnostics(item)
                symbols = await self.host.agent.lsp.document_symbols(item.path)
            compact = " · ".join(
                line.split("  ", 1)[-1] for line in symbols.splitlines()[:4] if line.strip()
            )
            if len(symbols.splitlines()) > 4:
                compact += " · …"
        except Exception as exc:  # noqa: BLE001
            if item.diagnostics == "pending":
                item.diagnostics = "unavailable"
            compact = f"unavailable — {exc}" if str(exc) else "unavailable — timed out"
        if not any(current is item for current in self.review.files):
            return
        self._symbols[item.path] = compact or "no declarations"
        if self._selected_key == item.key:
            self.query_one("#diff-validation", Static).update(
                self._validation_text(item, self._symbols[item.path]),
                layout=False,
            )

    def _validation_text(self, item: Any, symbols: str) -> Text:
        palette: ThemePalette = self.app.theme_palette  # type: ignore[attr-defined]
        text = Text("EDITOR DIAGNOSTICS (current worktree)  ", style=f"bold {palette.accent}")
        text.append(
            item.diagnostics,
            style=palette.success if item.diagnostics == "clean" else palette.warning,
        )
        text.append("\nSYMBOLS  ", style=f"bold {palette.accent}")
        text.append(symbols, style=palette.text)
        text.append("\nExecuted checks are reported in the turn receipt.", style=palette.muted)
        return text

    def action_toggle_diagnostics(self) -> None:
        diagnostics = self.query_one("#diff-validation", Static)
        diagnostics.display = not diagnostics.display

    def _move(self, delta: int) -> None:
        option_list = self.query_one("#diff-files", OptionList)
        if self._by_option:
            option_list.highlighted = ((option_list.highlighted or 0) + delta) % len(
                self._by_option
            )

    def action_next_file(self) -> None:
        self._move(1)

    def action_previous_file(self) -> None:
        self._move(-1)

    def _move_hunk(self, delta: int) -> None:
        self._recompute_hunk_offsets()
        if self._hunks:
            if self._hunk_index < 0:
                self._hunk_index = 0 if delta > 0 else len(self._hunks) - 1
            else:
                self._hunk_index = (self._hunk_index + delta) % len(self._hunks)
            patch = self.query_one("#diff-patch", RichLog)
            target_y = max(self._hunks[self._hunk_index] - getattr(patch, "_start_line", 0), 0)
            patch.scroll_to(y=target_y, animate=False, force=True)
            self.query_one("#diff-status", Static).update(
                f"Hunk {self._hunk_index + 1}/{len(self._hunks)}"
            )

    def action_next_hunk(self) -> None:
        self._move_hunk(1)

    def action_previous_hunk(self) -> None:
        self._move_hunk(-1)

    @work(exclusive=True, group="diff-refresh")
    async def action_refresh_review(self) -> None:
        try:
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Refresh failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(
            "Refreshed; select a file to capture its current patch"
        )

    def action_copy_patch(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        if not getattr(item, "loaded", bool(item.patch)) or not item.patch:
            self.query_one("#diff-status", Static).update("No patch text loaded to copy")
            return
        self.app.copy_to_clipboard(item.patch)
        self.query_one("#diff-status", Static).update(f"Copied patch for {item.path}")

    @work(exclusive=True, group="diff-editor")
    async def action_open_editor(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        try:
            blocked = self.host.agent.git._review_path_error(item.path)
            if blocked:
                raise ValueError(blocked)
            path = self.host.workspace.resolve(item.path)
            if not path.is_file():
                raise ValueError("The selected file no longer exists in the worktree")
            raw_editor = (
                (os.environ.get("VISUAL") or "").strip()
                or (os.environ.get("EDITOR") or "").strip()
                or "vi"
            )
            editor = shlex.split(raw_editor) or ["vi"]
            with self.app.suspend():
                result = subprocess.run([*editor, str(path)], check=False)
            self.query_one("#diff-status", Static).update(
                f"Editor exited {result.returncode}; Ctrl+R refreshes the captured review"
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Editor unavailable: {exc}")

    def _selected_item(self) -> Any:
        return next((item for item in self.review.files if item.key == self._selected_key), None)

    def _mutation_blocked(self) -> bool:
        if self.host._turn_running() is True:
            self.query_one("#diff-status", Static).update(
                "Read-only review while working. Stop the active turn before reverting or undoing."
            )
            return True
        return False

    @work(exclusive=True, group="diff-mutation")
    async def action_revert(self) -> None:
        if self._mutation_blocked():
            return
        item = self._selected_item()
        if item is None:
            return
        revision = getattr(item, "revision", None)
        if revision is None or await self.host.agent.git._review_signature(item.path) != revision:
            self.query_one("#diff-status", Static).update(
                "File or index changed, or patch not loaded. Ctrl+R refreshes before reverting."
            )
            return
        if self._mutation_blocked():
            return
        confirmation = await self.app.push_screen_wait(
            TextPromptModal(
                f"Revert {item.path}?",
                "Type REVERT",
                "This discards the selected file changes. Staged reverts also change the Git index.",
            )
        )
        if confirmation != "REVERT" or self._mutation_blocked():
            return
        try:
            status = await self.host.revert_diff_file(
                item.path, item.scope, expected_revision=revision
            )
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Revert failed: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    @work(exclusive=True, group="diff-mutation")
    async def action_undo(self) -> None:
        if self._mutation_blocked():
            return
        preview = _undo_preview(self.host)
        if preview.startswith("There is no reversible") or "full undo is unavailable" in preview:
            self.query_one("#diff-status", Static).update(preview.splitlines()[-1])
            return
        confirmed = await self.app.push_screen_wait(
            ConfirmationModal("Undo last turn?", preview, "Undo turn")
        )
        if not confirmed or self._mutation_blocked():
            return
        try:
            status = await self.host.undo_last_turn_async()
            self.review = await self.host.diff_review()
        except Exception as exc:  # noqa: BLE001
            self.query_one("#diff-status", Static).update(f"Undo unavailable: {exc}")
            return
        self._symbols.clear()
        self._render_review()
        self.query_one("#diff-status", Static).update(status)

    def action_close(self) -> None:
        if self.query_one("#diff-filter", Input).has_focus:
            self.query_one("#diff-files", OptionList).focus()
        else:
            self.dismiss(None)
