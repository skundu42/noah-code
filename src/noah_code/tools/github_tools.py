"""Permission-gated GitHub pull-request tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from nooa import Skill, spec

from noah_code.approvals import ApprovalBroker
from noah_code.github import GithubManager, PullRequestInfo
from noah_code.permissions import PermissionCategory, PermissionEngine


class GithubTools(Skill):
    """List, view, create, push, checkout, and comment on GitHub pull requests."""

    def __init__(
        self,
        checkout: Path,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        manager: GithubManager | None = None,
    ) -> None:
        super().__init__()
        self._manager = manager or GithubManager(checkout)
        self._engine = engine
        self._approvals = approvals

    async def list(self) -> str:
        """List open pull requests for this repository."""

        await self._approve("list")
        rows = await asyncio.to_thread(self._manager.list)
        return "\n".join(item.format_row() for item in rows) or "(none)"

    async def view(
        self,
        number: Annotated[int | None, spec(description="PR number; omit for the current branch")] = None,
    ) -> str:
        """Show one pull request."""

        await self._approve("view")
        return await asyncio.to_thread(self._manager.view, number)

    async def create(
        self,
        title: Annotated[str | None, spec(description="PR title; defaults to the latest commit")] = None,
        body: Annotated[str, spec(description="PR body")] = "",
        base: Annotated[str | None, spec(description="Base branch")] = None,
    ) -> str:
        """Push the current branch and open a pull request."""

        await self._approve("create")
        info = await asyncio.to_thread(self._manager.create, title, body, base)
        return _format_created(info)

    async def push(self) -> str:
        """Push the current branch to origin (updates an existing PR)."""

        await self._approve("push")
        return await asyncio.to_thread(self._manager.push)

    async def checkout(
        self,
        number: Annotated[int, spec(description="PR number to check out")],
    ) -> str:
        """Fetch and check out a pull request as ``pr/<number>``."""

        await self._approve("checkout")
        branch = await asyncio.to_thread(self._manager.checkout, number)
        return f"checked out #{int(number)} as {branch}"

    async def comment(
        self,
        number: Annotated[int, spec(description="PR number")],
        body: Annotated[str, spec(description="Comment text")],
    ) -> str:
        """Comment on a pull request."""

        await self._approve("comment")
        return await asyncio.to_thread(self._manager.comment, number, body)

    async def _approve(self, operation: str) -> None:
        await self._approvals.require(
            self._engine.decide(PermissionCategory.GITHUB, operation, tool=f"github_{operation}")
        )


def _format_created(info: PullRequestInfo) -> str:
    return f"created #{info.number}  {info.title}  {info.url}".strip()
