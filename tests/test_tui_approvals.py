"""Compact permission cards keep scope, controls, and long content accessible."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.config import PermissionRule
from noah_code.permissions import PermissionDecision
from noah_code.ui import textual_app as tui
from test_textual_tui import (
    _disable_live_update_checks as _disable_live_update_checks,
)
from test_textual_tui import _fake_host


def _request(*, target: str = "python -m pytest tests/test_auth.py", elevated: bool = False):
    return ApprovalRequest(
        "compact-approval",
        PermissionDecision(
            category="bash", target=target, action="ask",
            matching_rule=PermissionRule(category="bash", pattern="python *", action="ask"),
            reason="A configured rule requires approval before running this command.",
            remember_pattern="python *", tool="bash", elevated_floor=elevated,
        ),
        0.0, MagicMock(),
    )


@pytest.mark.parametrize("size", [(120, 40), (80, 24), (52, 18)])
async def test_short_permission_card_fits_content_and_keeps_choices_visible(tmp_path: Path, size):
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=size) as pilot:
        app.push_screen(tui.ApprovalModal(_request()))
        await pilot.pause()
        dialog = app.screen.query_one("#approval-dialog")
        assert dialog.size.height <= 13
        assert dialog.size.width <= 70
        assert not app.screen.query_one("#approval-details-scroll").display
        scope = app.screen.query_one("#approval-scope", Static).content.plain
        assert "python *" in scope and "Session pattern" in scope
        for name in ("once", "session", "reject"):
            button = app.screen.query_one(f"#{name}", Button)
            assert button.region.bottom < size[1]
            assert button.region.right < size[0]
            assert button.region.height == 3
        assert app.screen.focused is app.screen.query_one("#reject")


async def test_long_target_can_scroll_and_expand_without_hiding_choices(tmp_path: Path):
    target = "\n".join(f"[literal] command-{number}" for number in range(60))
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(tui.ApprovalModal(_request(target=target, elevated=True)))
        await pilot.pause()
        scroll = app.screen.query_one("#approval-scroll", VerticalScroll)
        assert scroll.size.height <= 3
        assert app.screen.query_one("#approval-body", Static).content.plain == target
        scroll.scroll_end(animate=False)
        await pilot.pause()
        assert scroll.scroll_y == scroll.max_scroll_y > 0
        assert app.screen.query_one("#approval-risk").display
        await pilot.press("d")
        await pilot.pause()
        assert app.screen.query_one("#approval-details-scroll").display
        assert scroll.size.height > 3
        details = app.screen.query_one("#approval-details", Static).content.plain
        assert "configured rule" in details and "python *" in details
        assert app.screen.focused is app.screen.query_one("#reject")
        await pilot.resize_terminal(52, 18)
        await pilot.pause()
        assert app.screen.query_one("#reject").region.bottom < 18
        assert app.screen.query_one("#approval-scroll").size.height >= 1
        await pilot.press("d")
        assert not app.screen.query_one("#approval-details-scroll").display


@pytest.mark.parametrize("key,choice", [
    ("1", ApprovalChoice.ONCE), ("2", ApprovalChoice.SESSION),
    ("3", ApprovalChoice.REJECT), ("escape", ApprovalChoice.REJECT),
    ("enter", ApprovalChoice.REJECT),
])
async def test_permission_keys_preserve_decisions(tmp_path: Path, key, choice):
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    answers = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(tui.ApprovalModal(_request()), answers.append)
        await pilot.pause()
        await pilot.press("d", "d", key)
        await pilot.pause()
        assert answers == [choice]


async def test_permission_details_can_be_opened_with_mouse(tmp_path: Path):
    app = tui.NoahCodeApp(_fake_host(tmp_path), tui.TextualUI())
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(tui.ApprovalModal(_request()))
        await pilot.pause()
        await pilot.click("#approval-toggle")
        await pilot.pause(0.25)
        assert app.screen.query_one("#approval-details-scroll").display
        await pilot.click("#approval-toggle")
        await pilot.pause()
        assert not app.screen.query_one("#approval-details-scroll").display
