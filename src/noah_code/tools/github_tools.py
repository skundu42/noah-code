"""Permission-gated GitHub pull-request tools."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from nooa import Skill, spec

from noah_code.approvals import ApprovalBroker
from noah_code.github import (
    GithubError,
    GithubManager,
    PullRequestInfo,
    _validate_pr_number,
    _validate_pr_target,
)
from noah_code.permissions import PermissionCategory, PermissionEngine

_AUTH_MARKERS = (
    "http 401",
    "http 403",
    "bad credentials",
    "unauthorized",
    "not authenticated",
    "authentication",
)

_NETWORK_MARKERS = (
    "dial tcp",
    "connection refused",
    "connection reset",
    "could not resolve host",
    "no such host",
    "network is unreachable",
    "i/o timeout",
    "tls handshake timeout",
    "temporary failure in name resolution",
)


def _user_facing(raw: str) -> str | None:
    """Translate gh-reported auth or connectivity failures into guidance."""

    text = raw.strip()
    lowered = text.lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return f"github auth failed ({text}); run `gh auth login` and retry"
    if any(marker in lowered for marker in _NETWORK_MARKERS):
        return f"github is unreachable ({text}); check network connectivity and retry"
    return None


class GithubTools(Skill):
    """List, view, create, push, checkout, and comment on GitHub pull requests."""

    def __init__(
        self,
        checkout: Path,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        manager: GithubManager | None = None,
        runtime: Any = None,
        coordinator: Any = None,
        workspace_tools: Any = None,
        checkpoints: Any = None,
    ) -> None:
        super().__init__()
        self._checkout = checkout.resolve()
        self._manager = manager or GithubManager(checkout)
        self._engine = engine
        self._approvals = approvals
        self._runtime = runtime
        self._coordinator = coordinator
        self._ws = workspace_tools
        self._checkpoints = checkpoints
        self._checkout_lock = asyncio.Lock()

    async def list(self) -> str:
        """List open pull requests for this repository."""

        await self._approve("list")
        rows = await self._call(self._manager.list)
        return "\n".join(item.format_row() for item in rows) or "(none)"

    async def view(
        self,
        number: Annotated[
            int | str | None,
            spec(description="PR number or PR URL; omit for the current branch"),
        ] = None,
    ) -> str:
        """Show one pull request."""

        target = _validate_pr_target(number)
        await self._approve("view")
        return await self._call(lambda: self._manager.view(target))

    async def create(
        self,
        title: Annotated[str | None, spec(description="PR title; defaults to the latest commit")] = None,
        body: Annotated[str, spec(description="PR body")] = "",
        base: Annotated[str | None, spec(description="Base branch")] = None,
    ) -> str:
        """Push the current branch and open a pull request."""

        await self._approve("create")
        head_meta = await asyncio.to_thread(self._git_head_identity)
        return await self._durable_effect(
            "create",
            "pull-request",
            {"title": title, "body": body, "base": base, **head_meta},
            lambda recovering: _format_created(
                self._manager.create(title, body, base, recover=recovering)
            ),
        )

    async def push(self) -> str:
        """Push the current branch to origin (updates an existing PR)."""

        await self._approve("push")
        head_meta = await asyncio.to_thread(self._git_head_identity)
        return await self._durable_effect(
            "push", "HEAD", dict(head_meta), lambda _recovering: self._manager.push()
        )

    async def checkout(
        self,
        number: Annotated[int, spec(description="PR number to check out")],
    ) -> str:
        """Fetch and check out a pull request as ``pr/<number>``."""

        pr_num = _validate_pr_number(number)
        await self._approve("checkout")
        lock = self._coordinator.lock if self._coordinator is not None else self._checkout_lock
        async with lock:
            if self._checkpoints is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        self._checkpoints.capture, f"before github checkout #{pr_num}"
                    )
            elif self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.checkpoint_before_shell(f"github checkout #{pr_num}")
            try:
                return await self._durable_effect(
                    "checkout",
                    str(pr_num),
                    {"number": pr_num},
                    lambda _recovering: (
                        f"checked out #{pr_num} as {self._manager.checkout(pr_num)}"
                    ),
                )
            finally:
                if self._ws is not None:
                    self._ws._read_fingerprints.clear()

    async def comment(
        self,
        number: Annotated[int, spec(description="PR number")],
        body: Annotated[str, spec(description="Comment text")],
    ) -> str:
        """Comment on a pull request."""

        pr_num = _validate_pr_number(number)
        await self._approve("comment")
        return await self._durable_effect(
            "comment",
            str(pr_num),
            {"number": pr_num, "body": body},
            lambda recovering: self._manager.comment(pr_num, body, recover=recovering),
        )

    def _git_head_identity(self) -> dict[str, str]:
        def _git(*args: str) -> str:
            try:
                proc = subprocess.run(
                    ["git", *args],
                    cwd=self._checkout,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return proc.stdout.strip() if proc.returncode == 0 else ""

        return {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head_sha": _git("rev-parse", "HEAD"),
        }

    async def _durable_effect(
        self,
        kind: str,
        target: str,
        request: dict[str, Any],
        operation: Callable[[bool], str],
    ) -> str:
        if self._runtime is None:
            return await self._call(lambda: operation(False))
        effect_key, cached, result, recovering = self._runtime.begin_effect(
            kind, target, request
        )
        if cached:
            return str(result or "completed")
        try:
            value = await self._call(lambda: operation(recovering))
        except Exception as exc:
            self._runtime.fail_effect(effect_key, str(exc))
            raise
        self._runtime.complete_effect(effect_key, value)
        return value

    async def _call(self, func: Callable[[], Any]) -> Any:
        try:
            return await asyncio.to_thread(func)
        except GithubError as exc:
            friendly = _user_facing(str(exc))
            if friendly is None:
                raise
            raise GithubError(friendly) from exc
        except ConnectionError as exc:
            raise GithubError(
                f"github is unreachable ({exc}); check network connectivity and retry"
            ) from exc

    async def _approve(self, operation: str) -> None:
        await self._approvals.require(
            self._engine.decide(PermissionCategory.GITHUB, operation, tool=f"github_{operation}")
        )


def _format_created(info: PullRequestInfo) -> str:
    return f"created #{info.number}  {info.title}  {info.url}".strip()
