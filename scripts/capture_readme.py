"""Capture the real TUI with deterministic sample data, without provider calls.

Run from a development checkout: ``uv run python scripts/capture_readme.py``.
Screenshots illustrate the interface; they are not records of an actual agent run.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from noah_code.approvals import ApprovalRequest
from noah_code.permissions import PermissionDecision
from noah_code.ui import textual_app as tui

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
REPOSITORY = tui.RepositorySnapshot(
    branch="pagination",
    modified=2,
    untracked=1,
    paths=("src/api/search.py", "src/api/cursors.py", "tests/test_pagination.py"),
)


def sample_host() -> MagicMock:
    # Reuse the isolated host fixture; only the production Textual UI is rendered.
    sys.path.insert(0, str(ROOT / "tests"))
    from test_textual_tui import _fake_host

    host = _fake_host(Path("/workspace/search-api"))
    host.config.model = "configured model"
    host.meta.model = "configured model"
    host.meta.session_id = "sample-session"
    host.meta.title = "Sample: cursor pagination"
    host.config.ui.animations = False
    return host


def save(app: tui.NoahCodeApp, filename: str, title: str) -> None:
    svg = app.export_screenshot(title=f"Noah Code · {title} · Sample session")
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    (OUTPUT / filename).write_text(svg, encoding="utf-8")
    print(f"Saved {OUTPUT / filename}")


async def capture_team() -> None:
    host = sample_host()
    host.agent.todos.list_todos.return_value = [
        SimpleNamespace(status="done", title="Trace search ordering"),
        SimpleNamespace(status="done", title="Implement cursor pagination"),
        SimpleNamespace(status="open", title="Review edge cases"),
    ]
    host.work_snapshot.return_value = {
        "agents": [
            {
                "id": "sample-analyze",
                "agent": "explore",
                "prompt": "Trace query ordering and API conventions",
                "state": "completed",
                "mode": "plan",
                "readonly": True,
                "duration": 18,
                "workflow": "build",
                "team_id": "sample-build",
                "phase": "analyze",
                "result_preview": "Use a stable timestamp and ID ordering.",
            },
            {
                "id": "sample-implement",
                "agent": "general",
                "prompt": "Implement cursors and pagination coverage",
                "state": "completed",
                "mode": "build",
                "readonly": False,
                "duration": 42,
                "workflow": "build",
                "team_id": "sample-build",
                "phase": "implement",
                "result_preview": "Cursor support and edge-case tests are ready for review.",
            },
            {
                "id": "sample-review",
                "agent": "reviewer",
                "prompt": "Review pagination edge cases",
                "state": "running",
                "mode": "plan",
                "readonly": True,
                "duration": 8,
                "workflow": "build",
                "team_id": "sample-build",
                "phase": "review",
                "result_preview": "",
            },
        ],
        "jobs": [],
    }
    app = tui.NoahCodeApp(host, tui.TextualUI())
    async with app.run_test(size=(132, 32)) as pilot:
        await pilot.pause()
        for entry in [
            tui.TranscriptEntry(
                "YOU", "/team build Add cursor pagination to search and cover its edge cases."
            ),
            tui.TranscriptEntry(
                "NOAH",
                "I'll trace the query, implement pagination, then hand the changes to a "
                "read-only reviewer.",
            ),
            tui.TranscriptEntry(
                "ACTIVITY", "✓ explore · analyze · Traced src/api/search.py and query ordering"
            ),
            tui.TranscriptEntry(
                "NOAH",
                "The query needs a stable cursor across pages:\n\n"
                "- Order by timestamp and ID to avoid duplicate results.\n"
                "- Validate cursors and bound the requested page size.\n"
                "- Cover empty pages, invalid cursors, and tied timestamps.",
                markdown=True,
            ),
            tui.TranscriptEntry(
                "ACTIVITY", "✓ general · implement · Updated search, cursor helpers, and tests"
            ),
            tui.TranscriptEntry(
                "NOAH",
                "The reviewer is checking boundary cases and compatibility. "
                "Open F4 to inspect each handoff, or Ctrl+D to review the changes.",
            ),
        ]:
            app._append_entry(entry)
        app._set_agent_state(tui.AgentDisplayState.RUNNING, "Reviewing pagination edge cases")
        app.update_chrome(force=True)
        await pilot.pause()
        save(app, "noah-in-action.svg", "Build with an agent team")


async def capture_commands() -> None:
    app = tui.NoahCodeApp(sample_host(), tui.TextualUI())
    async with app.run_test(size=(104, 24)) as pilot:
        app._append_entry(tui.TranscriptEntry("YOU", "Help me find the right workflow."))
        app._append_entry(tui.TranscriptEntry(
            "NOAH", "Type / to explore commands. Use F9 to launch a build, review, "
            "or investigation team."
        ))
        app.query_one("#composer").text = "/"
        await pilot.pause()
        save(app, "noah-command-menu.svg", "Find a command")


async def capture_permissions() -> None:
    app = tui.NoahCodeApp(sample_host(), tui.TextualUI())
    command = "python -m pytest tests/test_pagination.py -q"
    request = ApprovalRequest(
        "sample-approval",
        PermissionDecision(
            "bash", command, "ask", None,
            "This command needs approval under your workspace rules.",
            command, tool="bash",
        ),
        0,
        MagicMock(),
    )
    async with app.run_test(size=(104, 24)) as pilot:
        app._append_entry(tui.TranscriptEntry("YOU", "Run the focused pagination tests."))
        app._append_entry(tui.TranscriptEntry(
            "NOAH", "I will run the pagination tests to check the cursor behavior."
        ))
        app.push_screen(tui.ApprovalModal(request))
        await pilot.pause()
        save(app, "noah-permissions.svg", "Approve with context")


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("NO_COLOR", None)
    environment.update(TERM="xterm-256color", COLORTERM="truecolor", FORCE_COLOR="1")
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(tui, "maybe_check_for_update", return_value=None),
        patch.object(tui, "_read_repository_snapshot", return_value=REPOSITORY),
    ):
        await capture_team()
        await capture_commands()
        await capture_permissions()


if __name__ == "__main__":
    asyncio.run(main())
