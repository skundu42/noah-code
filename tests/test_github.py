"""Host-owned GitHub pull-request manager and tools."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.github import GithubError, GithubManager
from noah_code.permissions import PermissionEngine
from noah_code.tools.github_tools import GithubTools


class _FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess[str]]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, cwd: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        _ = cwd
        key = tuple(argv)
        if key in self.responses:
            return self.responses[key]
        if argv[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        if argv[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            return subprocess.CompletedProcess(argv, 0, "true\n", "")
        return subprocess.CompletedProcess(argv, 1, "", f"unexpected: {argv}")


def _ok(argv: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout, "")


def test_list_parses_open_pull_requests(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {
                "number": 12,
                "title": "Add worktrees",
                "url": "https://github.com/acme/repo/pull/12",
                "headRefName": "feat",
                "baseRefName": "main",
                "state": "OPEN",
            }
        ]
    )
    runner = _FakeRunner(
        {
            ("gh", "pr", "list", "--limit", "20", "--json", "number,title,url,headRefName,baseRefName,state"): _ok(
                ["gh"], payload
            )
        }
    )
    rows = GithubManager(tmp_path, runner=runner).list()
    assert rows[0].number == 12
    assert "Add worktrees" in rows[0].format_row()


def test_create_pushes_then_opens_pr(tmp_path: Path) -> None:
    runner = _FakeRunner(
        {
            ("git", "log", "-1", "--pretty=%s"): _ok(["git"], "Add worktrees\n"),
            ("git", "push", "-u", "origin", "HEAD"): _ok(["git"], "pushed\n"),
            ("gh", "pr", "create", "--title", "Add worktrees", "--body", ""): _ok(
                ["gh"], "https://github.com/acme/repo/pull/12\n"
            ),
            ("gh", "pr", "view", "--json", "number,title,url,headRefName,baseRefName,state"): _ok(
                ["gh"],
                json.dumps(
                    {
                        "number": 12,
                        "title": "Add worktrees",
                        "url": "https://github.com/acme/repo/pull/12",
                        "headRefName": "feat",
                        "baseRefName": "main",
                        "state": "OPEN",
                    }
                ),
            ),
        }
    )
    info = GithubManager(tmp_path, runner=runner).create()
    assert info.number == 12
    assert ["git", "push", "-u", "origin", "HEAD"] in runner.calls
    assert ["gh", "pr", "create", "--title", "Add worktrees", "--body", ""] in runner.calls


def test_checkout_uses_opencode_branch_name(tmp_path: Path) -> None:
    runner = _FakeRunner(
        {
            ("gh", "pr", "checkout", "12", "--branch", "pr/12", "--force"): _ok(["gh"]),
        }
    )
    assert GithubManager(tmp_path, runner=runner).checkout(12) == "pr/12"


def test_missing_gh_explains_install(tmp_path: Path) -> None:
    runner = _FakeRunner({("gh", "auth", "status"): subprocess.CompletedProcess(["gh"], 127, "", "not found")})
    with pytest.raises(GithubError, match="gh CLI is required"):
        GithubManager(tmp_path, runner=runner).list()


def test_unauthenticated_gh_explains_login(tmp_path: Path) -> None:
    runner = _FakeRunner({("gh", "auth", "status"): subprocess.CompletedProcess(["gh"], 1, "", "not logged in")})
    with pytest.raises(GithubError, match="gh auth login"):
        GithubManager(tmp_path, runner=runner).list()


@pytest.mark.asyncio
async def test_github_tools_create_asks_then_runs(tmp_path: Path) -> None:
    runner = _FakeRunner(
        {
            ("git", "push", "-u", "origin", "HEAD"): _ok(["git"]),
            ("gh", "pr", "create", "--title", "Ship it", "--body", "body"): _ok(
                ["gh"], "https://github.com/acme/repo/pull/3\n"
            ),
            ("gh", "pr", "view", "--json", "number,title,url,headRefName,baseRefName,state"): _ok(
                ["gh"],
                json.dumps(
                    {
                        "number": 3,
                        "title": "Ship it",
                        "url": "https://github.com/acme/repo/pull/3",
                    }
                ),
            ),
        }
    )
    asked: list[str] = []

    async def once(request):
        asked.append(request.decision.target)
        return ApprovalChoice.ONCE

    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    tools = GithubTools(
        tmp_path,
        engine,
        ApprovalBroker(engine, handler=once),
        manager=GithubManager(tmp_path, runner=runner),
    )
    text = await tools.create("Ship it", "body")
    assert asked == ["create"]
    assert "#3" in text


@pytest.mark.asyncio
async def test_github_tools_list_is_allowed(tmp_path: Path) -> None:
    runner = _FakeRunner(
        {
            (
                "gh",
                "pr",
                "list",
                "--limit",
                "20",
                "--json",
                "number,title,url,headRefName,baseRefName,state",
            ): _ok(["gh"], "[]")
        }
    )
    asked = {"n": 0}

    async def once(_request):
        asked["n"] += 1
        return ApprovalChoice.ONCE

    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    tools = GithubTools(
        tmp_path,
        engine,
        ApprovalBroker(engine, handler=once),
        manager=GithubManager(tmp_path, runner=runner),
    )
    assert await tools.list() == "(none)"
    assert asked["n"] == 0
