"""Host-owned GitHub pull-request operations (OpenCode-shaped)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[[Path, list[str]], subprocess.CompletedProcess[str]]


class GithubError(RuntimeError):
    """GitHub / pull-request failure."""


@dataclass(frozen=True)
class PullRequestInfo:
    number: int
    title: str
    url: str
    head: str = ""
    base: str = ""
    state: str = ""

    def format_row(self) -> str:
        return f"#{self.number}  {self.title}  {self.url}"


def _default_run(cwd: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        program = argv[0] if argv else "command"
        return subprocess.CompletedProcess(argv, 127, "", f"{program} not found")


def _message(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "github command failed").strip()


class GithubManager:
    """Create, list, view, push, checkout, and comment via trusted ``gh`` / ``git``."""

    def __init__(self, checkout: Path, *, runner: Runner | None = None) -> None:
        self.root = checkout.resolve()
        self._run = runner or _default_run

    def check(self) -> None:
        """Raise if ``gh`` or git is not ready."""

        self._require_ready()

    def list(self, limit: int = 20) -> list[PullRequestInfo]:
        self._require_ready()
        listed = self._run(
            self.root,
            [
                "gh",
                "pr",
                "list",
                "--limit",
                str(limit),
                "--json",
                "number,title,url,headRefName,baseRefName,state",
            ],
        )
        if listed.returncode != 0:
            raise GithubError(_message(listed))
        return [_parse_pr(item) for item in _parse_json_list(listed.stdout)]

    def view(self, number: int | None = None) -> str:
        self._require_ready()
        argv = ["gh", "pr", "view"]
        if number is not None:
            argv.append(str(number))
        viewed = self._run(self.root, argv)
        if viewed.returncode != 0:
            raise GithubError(_message(viewed))
        return (viewed.stdout or viewed.stderr).strip()

    def create(
        self,
        title: str | None = None,
        body: str = "",
        base: str | None = None,
    ) -> PullRequestInfo:
        self._require_ready()
        resolved_title = (title or self._latest_subject()).strip()
        if not resolved_title:
            raise GithubError("pull request title is required")
        self.push()
        argv = ["gh", "pr", "create", "--title", resolved_title, "--body", body]
        if base:
            argv.extend(["--base", base])
        created = self._run(self.root, argv)
        if created.returncode != 0:
            raise GithubError(_message(created))
        url = (created.stdout or "").strip().splitlines()[-1] if created.stdout.strip() else ""
        viewed = self._run(
            self.root,
            [
                "gh",
                "pr",
                "view",
                "--json",
                "number,title,url,headRefName,baseRefName,state",
            ],
        )
        if viewed.returncode == 0:
            return _parse_pr(_parse_json_object(viewed.stdout))
        if url.startswith("http"):
            number = _number_from_url(url)
            return PullRequestInfo(number=number, title=resolved_title, url=url)
        raise GithubError("opened a pull request but could not read it back")

    def push(self) -> str:
        self._require_ready()
        pushed = self._run(self.root, ["git", "push", "-u", "origin", "HEAD"])
        if pushed.returncode != 0:
            raise GithubError(_message(pushed))
        return (pushed.stdout or pushed.stderr or "pushed HEAD to origin").strip()

    def checkout(self, number: int) -> str:
        self._require_ready()
        branch = f"pr/{int(number)}"
        result = self._run(
            self.root,
            ["gh", "pr", "checkout", str(int(number)), "--branch", branch, "--force"],
        )
        if result.returncode != 0:
            raise GithubError(_message(result))
        return branch

    def comment(self, number: int, body: str) -> str:
        self._require_ready()
        text = body.strip()
        if not text:
            raise GithubError("comment text is required")
        result = self._run(
            self.root,
            ["gh", "pr", "comment", str(int(number)), "--body", text],
        )
        if result.returncode != 0:
            raise GithubError(_message(result))
        return (result.stdout or f"commented on #{int(number)}").strip()

    def _latest_subject(self) -> str:
        result = self._run(self.root, ["git", "log", "-1", "--pretty=%s"])
        return result.stdout.strip() if result.returncode == 0 else ""

    def _require_ready(self) -> None:
        auth = self._run(self.root, ["gh", "auth", "status"])
        if auth.returncode == 127:
            raise GithubError("gh CLI is required; install https://cli.github.com/")
        if auth.returncode != 0:
            raise GithubError("gh is not authenticated; run gh auth login")
        git = self._run(self.root, ["git", "rev-parse", "--is-inside-work-tree"])
        if git.returncode != 0 or git.stdout.strip() != "true":
            raise GithubError("pull requests need a git repo")


def _parse_json_list(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise GithubError("unexpected gh list payload")
    return [item for item in data if isinstance(item, dict)]


def _parse_json_object(raw: str) -> dict:
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        raise GithubError("unexpected gh view payload")
    return data


def _parse_pr(item: dict) -> PullRequestInfo:
    return PullRequestInfo(
        number=int(item.get("number") or 0),
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        head=str(item.get("headRefName") or ""),
        base=str(item.get("baseRefName") or ""),
        state=str(item.get("state") or ""),
    )


def _number_from_url(url: str) -> int:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else 0
