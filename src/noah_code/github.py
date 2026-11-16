"""Host-owned GitHub pull-request operations (OpenCode-shaped)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[[Path, list[str]], subprocess.CompletedProcess[str]]
_PR_URL_RE = re.compile(r"^https://[^/\s]+/[^/\s]+/[^/\s]+/pull/[1-9]\d*(?:[/?#].*)?$")
_REMOTE_OWNER_RE = re.compile(r"(?:github\.com[:/]|://[^/]+/)([^/]+)/[^/]+?(?:\.git)?/?$")


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


def _default_run(
    cwd: Path, argv: list[str], *, timeout: float = 60.0
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("GH_PROMPT_DISABLED", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        program = argv[0] if argv else "command"
        return subprocess.CompletedProcess(
            argv, 124, "", f"{program} timed out after {timeout:.0f}s"
        )
    except FileNotFoundError:
        program = argv[0] if argv else "command"
        return subprocess.CompletedProcess(argv, 127, "", f"{program} not found")


def _validate_pr_number(number: int) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("pull request number must be an integer >= 1")
    return number


def _validate_pr_target(target: int | str | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, bool):
        raise ValueError("pull request target must be a positive integer or PR URL")
    if isinstance(target, int):
        return str(_validate_pr_number(target))
    if isinstance(target, str):
        cleaned = target.strip()
        if cleaned.isdigit() and int(cleaned) >= 1:
            return str(int(cleaned))
        if _PR_URL_RE.match(cleaned):
            return cleaned
        raise ValueError(
            "pull request target must be a positive integer or GitHub pull request URL"
        )
    raise ValueError("pull request target must be a positive integer or PR URL")


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

    def view(self, number: int | str | None = None) -> str:
        target = _validate_pr_target(number)
        self._require_ready()
        argv = ["gh", "pr", "view"]
        if target is not None:
            argv.append(target)
        viewed = self._run(self.root, argv)
        if viewed.returncode == 0:
            return (viewed.stdout or viewed.stderr).strip()
        if target is None:
            fork_ref = self._fork_head_ref()
            if fork_ref:
                retry = self._run(self.root, ["gh", "pr", "view", fork_ref])
                if retry.returncode == 0:
                    return (retry.stdout or retry.stderr).strip()
        raise GithubError(_message(viewed))

    def create(
        self,
        title: str | None = None,
        body: str = "",
        base: str | None = None,
        *,
        recover: bool = False,
    ) -> PullRequestInfo:
        self._require_ready()
        if recover:
            existing = self._run(
                self.root,
                [
                    "gh",
                    "pr",
                    "view",
                    "--json",
                    "number,title,url,headRefName,baseRefName,state",
                ],
            )
            if existing.returncode == 0:
                return _parse_pr(_parse_json_object(existing.stdout))
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
        remote = self._push_remote()
        pushed = self._run(self.root, ["git", "push", "-u", remote, "HEAD"])
        if pushed.returncode != 0:
            raise GithubError(_message(pushed))
        return (pushed.stdout or pushed.stderr or f"pushed HEAD to {remote}").strip()

    def checkout(self, number: int) -> str:
        pr_num = _validate_pr_number(number)
        self._require_ready()
        branch = f"pr/{pr_num}"
        result = self._run(
            self.root,
            ["gh", "pr", "checkout", str(pr_num), "--branch", branch, "--force"],
        )
        if result.returncode != 0:
            raise GithubError(_message(result))
        return branch

    def comment(self, number: int, body: str, *, recover: bool = False) -> str:
        pr_num = _validate_pr_number(number)
        self._require_ready()
        text = body.strip()
        if not text:
            raise GithubError("comment text is required")
        if recover:
            viewed = self._run(
                self.root,
                ["gh", "pr", "view", str(pr_num), "--json", "comments"],
            )
            if viewed.returncode == 0:
                payload = _parse_json_object(viewed.stdout)
                comments = payload.get("comments") or []
                if any(
                    isinstance(item, dict) and str(item.get("body") or "").strip() == text
                    for item in comments
                ):
                    return f"comment already present on #{pr_num}"
        result = self._run(
            self.root,
            ["gh", "pr", "comment", str(pr_num), "--body", text],
        )
        if result.returncode != 0:
            raise GithubError(_message(result))
        return (result.stdout or f"commented on #{pr_num}").strip()

    def _push_remote(self) -> str:
        upstream = self._run(
            self.root,
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        )
        if upstream.returncode == 0:
            ref = upstream.stdout.strip()
            if "/" in ref:
                return ref.split("/", 1)[0]
        push_default = self._run(self.root, ["git", "config", "remote.pushDefault"])
        if push_default.returncode == 0 and push_default.stdout.strip():
            return push_default.stdout.strip()
        remotes = self._run(self.root, ["git", "remote"])
        if remotes.returncode == 0:
            names = [line.strip() for line in remotes.stdout.splitlines() if line.strip()]
            if len(names) == 1:
                return names[0]
            if "origin" in names:
                return "origin"
        return "origin"

    def _fork_head_ref(self) -> str | None:
        branch_res = self._run(self.root, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if branch_res.returncode != 0:
            return None
        branch = branch_res.stdout.strip()
        if not branch or branch == "HEAD":
            return None
        remote = self._push_remote()
        url_res = self._run(self.root, ["git", "remote", "get-url", remote])
        if url_res.returncode != 0:
            return None
        match = _REMOTE_OWNER_RE.search(url_res.stdout.strip())
        if not match:
            return None
        return f"{match.group(1)}:{branch}"

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
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GithubError(f"invalid JSON from gh: {exc}") from exc
    if not isinstance(data, list):
        raise GithubError("unexpected gh list payload")
    return [item for item in data if isinstance(item, dict)]


def _parse_json_object(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise GithubError(f"invalid JSON from gh: {exc}") from exc
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
