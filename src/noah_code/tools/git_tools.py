"""Narrow git helpers - status/diff/log only by default."""

from __future__ import annotations

from typing import Annotated

from nooa import Skill, spec

from noah_code.tools.workspace_tools import WorkspaceTools


class GitTools(Skill):
    """Read-oriented git helpers. Mutating git still goes through workspace.run with policy."""

    def __init__(self, workspace_tools: WorkspaceTools) -> None:
        super().__init__()
        self._ws = workspace_tools

    async def status(self) -> str:
        """Return ``git status --short --branch`` output."""
        result = await self._ws.run_trusted_readonly("git status --short --branch")
        return result.stdout or result.stderr

    async def diff(
        self,
        path: Annotated[str | None, spec(description="Optional path limit")] = None,
    ) -> str:
        """Return ``git diff`` (unstaged + staged summary via --stat if no path)."""
        if path:
            import shlex

            cmd = f"git diff -- {shlex.quote(path)}"
        else:
            cmd = "git diff"
        result = await self._ws.run_trusted_readonly(cmd)
        return result.stdout or "(no diff)"

    async def log(
        self,
        n: Annotated[int, spec(description="Number of commits")] = 5,
    ) -> str:
        """Return recent commit subjects."""
        result = await self._ws.run_trusted_readonly(f"git log -n {int(n)} --oneline")
        return result.stdout or "(no commits)"
