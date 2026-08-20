"""Narrow git helpers - status/diff/log only by default."""

from __future__ import annotations

import asyncio
import difflib
import subprocess
from dataclasses import dataclass, field
from typing import Annotated

from nooa import Skill, spec

from noah_code.permissions import is_secret_path
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import WorkspaceError


@dataclass
class DiffFile:
    path: str
    scope: str
    status: str
    additions: int = 0
    deletions: int = 0
    diagnostics: str = "pending"
    patch: str = ""

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.path}"


@dataclass
class DiffReview:
    files: list[DiffFile] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(item.additions for item in self.files)

    @property
    def deletions(self) -> int:
        return sum(item.deletions for item in self.files)


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
        import shlex

        if path:
            blocked = self._review_path_error(path)
            if blocked is not None:
                return blocked
            cmd = f"git diff -- {shlex.quote(path)}"
            result = await self._ws.run_trusted_readonly(cmd)
            return result.stdout or "(no diff)"
        names = await self._ws.run_trusted_readonly("git diff --name-only")
        chunks: list[str] = []
        for file_path in names.stdout.splitlines():
            file_path = file_path.strip()
            if not file_path:
                continue
            blocked = self._review_path_error(file_path)
            if blocked is not None:
                chunks.append(blocked)
                continue
            piece = await self._ws.run_trusted_readonly(f"git diff -- {shlex.quote(file_path)}")
            if piece.stdout:
                chunks.append(piece.stdout)
        return "\n".join(chunks) or "(no diff)"

    async def log(
        self,
        n: Annotated[int, spec(description="Number of commits")] = 5,
    ) -> str:
        """Return recent commit subjects."""
        result = await self._ws.run_trusted_readonly(f"git log -n {int(n)} --oneline")
        return result.stdout or "(no commits)"

    async def review(self) -> DiffReview:
        """Return staged and unstaged files with bounded per-file patches."""
        status = await self._git("status", "--porcelain=v1", "-z")
        if status.returncode != 0:
            raise RuntimeError(status.stderr.strip() or "git status failed")
        entries = [entry for entry in status.stdout.split("\0") if entry]
        changed: list[tuple[str, str, str]] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            code = entry[:2]
            path = entry[3:] if len(entry) >= 4 else ""
            if ("R" in code or "C" in code) and index + 1 < len(entries):
                index += 1
                # -z rename records place the destination in the status entry.
            x, y = code[0], code[1]
            if x not in {" ", "?", "!"}:
                changed.append((path, "staged", x))
            if y not in {" ", "!"} or code == "??":
                changed.append((path, "unstaged", "?" if code == "??" else y))
            index += 1

        files: list[DiffFile] = []
        for path, scope, status_code in changed:
            patch = await self._patch(path, scope, status_code)
            additions, deletions = await self._counts(path, scope, status_code)
            files.append(
                DiffFile(
                    path=path,
                    scope=scope,
                    status=self._status_name(status_code),
                    additions=additions,
                    deletions=deletions,
                    patch=patch,
                )
            )
        files.sort(key=lambda item: (item.path, item.scope != "staged"))
        return DiffReview(files)

    async def revert(self, path: str, scope: str) -> str:
        """Revert one explicitly selected file after host/UI confirmation."""
        resolved = await self._ws._authorize_path(path, "edit")
        if scope == "unstaged":
            current = resolved.read_text(errors="strict") if resolved.exists() else None
            index_content = await self._git("show", f":{path}")
            if index_content.returncode == 0:
                if current is None:
                    changes = [{"path": path, "old": None, "new": index_content.stdout}]
                else:
                    changes = [{"path": path, "old": current, "new": index_content.stdout}]
            elif current is not None:
                changes = [{"path": path, "old": current, "new": None}]
            else:
                return f"{path} is already absent"
            await self._ws.apply_patch(changes)
            return f"reverted unstaged changes in {path}"
        if scope != "staged":
            raise ValueError("scope must be staged or unstaged")
        # Staging metadata is outside the file journal. Capture the worktree
        # preimage, perform the explicit Git restore, and mark this turn as not
        # fully reversible so /undo never overpromises.
        mutation = self._ws._journal.record_preimage(resolved)
        result = await self._ws.run(
            f"git restore --source=HEAD --staged --worktree -- {__import__('shlex').quote(path)}"
        )
        if result.returncode != 0:
            self._ws._journal.discard_mutation(mutation)
            raise RuntimeError(result.stderr or "git restore failed")
        self._ws._journal.record_postimage(mutation, resolved)
        return f"reverted staged and worktree changes in {path}"

    async def _patch(self, path: str, scope: str, status: str) -> str:
        blocked = self._review_path_error(path)
        if blocked is not None:
            return blocked
        if status == "?":
            target = self._ws._workspace.resolve(path)
            try:
                text = target.read_text(errors="replace")
            except OSError as exc:
                return f"diff unavailable: {exc}"
            lines = text.splitlines(keepends=True)
            return "".join(
                difflib.unified_diff([], lines, fromfile="/dev/null", tofile=f"b/{path}", n=3)
            )[:80_000]
        args = ["diff", "--no-ext-diff", "--unified=3"]
        if scope == "staged":
            args.append("--cached")
        args.extend(["--", path])
        result = await self._git(*args)
        return (result.stdout or result.stderr or "(no textual diff)")[:80_000]

    async def _counts(self, path: str, scope: str, status: str) -> tuple[int, int]:
        if self._review_path_error(path) is not None:
            return 0, 0
        if status == "?":
            target = self._ws._workspace.resolve(path)
            try:
                return len(target.read_text(errors="replace").splitlines()), 0
            except (OSError, WorkspaceError):
                return 0, 0
        args = ["diff", "--numstat"]
        if scope == "staged":
            args.append("--cached")
        args.extend(["--", path])
        result = await self._git(*args)
        first = result.stdout.splitlines()[0].split("\t") if result.stdout.strip() else []
        if len(first) < 2:
            return 0, 0
        try:
            return int(first[0]), int(first[1])
        except ValueError:  # binary files report '-'
            return 0, 0

    async def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args],
                cwd=self._ws._workspace.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        return await asyncio.to_thread(run)

    def _review_path_error(self, path: str) -> str | None:
        if is_secret_path(path):
            return f"diff unavailable: secret path denied: {path}"
        try:
            self._ws._workspace.resolve(path)
        except WorkspaceError as exc:
            return f"diff unavailable: {exc}"
        return None

    @staticmethod
    def _status_name(code: str) -> str:
        return {
            "?": "untracked",
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
            "U": "conflict",
        }.get(code, "changed")
