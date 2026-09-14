"""Narrow git helpers - status/diff/log only by default."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Annotated

from nooa import Skill, hidden, spec

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
    loaded: bool = False
    captured_at: float = 0.0
    truncated: bool = False
    revision: str | None = None

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.path}"


@dataclass
class DiffReview:
    files: list[DiffFile] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)

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

    async def review(self, *, eager: bool = True) -> DiffReview:
        """List changes immediately; optionally load patches for noninteractive callers."""
        status = await self._git("status", "--porcelain=v1", "-z", "--untracked-files=all")
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

        files = [
            DiffFile(path=path, scope=scope, status=self._status_name(status_code))
            for path, scope, status_code in changed
        ]
        files.sort(key=lambda item: (item.path, item.scope != "staged"))
        review = DiffReview(files)
        if eager:
            for item in files:
                await self.review_file(item)
        return review

    @hidden
    async def review_file(self, item: DiffFile) -> None:
        """Capture one file's patch; keep it fixed until the next explicit refresh."""
        before = await self._review_signature(item.path)
        status = "?" if item.status == "untracked" else item.status
        patch = await self._patch(item.path, item.scope, status)
        item.additions, item.deletions = await self._counts(item.path, item.scope, status)
        item.truncated = len(patch) > 80_000
        item.patch = patch[:80_000]
        if item.truncated:
            item.patch += "\n… Patch truncated at 80,000 characters. Press O to open the file in your editor."
        after = await self._review_signature(item.path)
        item.revision = after if before == after else None
        if item.revision is None:
            item.patch += "\nFile changed while loading. Refresh before reverting."
        item.captured_at = time.time()
        item.loaded = True

    @hidden
    async def change_fingerprints(self) -> dict[str, tuple[object, ...]]:
        """Observe Git-visible changes without reading or hashing file contents."""
        review = await self.review(eager=False)
        index = await self._git("ls-files", "--stage", "-z")
        index_entries = {}
        for entry in index.stdout.split("\0"):
            metadata, separator, path = entry.partition("\t")
            if separator:
                index_entries[path] = metadata
        fingerprints: dict[str, tuple[object, ...]] = {}
        for item in review.files:
            fingerprints[item.key] = (item.status, index_entries.get(item.path), *self._stat_signature(item.path))
        return fingerprints

    def _stat_signature(self, path: str) -> tuple[int, ...]:
        # lstat observes links without following them.
        try:
            stat = (self._ws._workspace.root / path).lstat()
            return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_mode)
        except OSError:
            return ()

    async def _review_signature(self, path: str) -> str:
        index = await self._git("ls-files", "--stage", "--", path)
        head = await self._git("rev-parse", "--verify", "HEAD")
        raw = repr((self._stat_signature(path), index.returncode, index.stdout, head.stdout))
        return hashlib.sha256(raw.encode()).hexdigest()

    async def revert(self, path: str, scope: str, *, expected_revision: str | None = None) -> str:
        """Revert one explicitly selected file after host/UI confirmation."""
        resolved = await self._ws._authorize_path(path, "edit", tool="git_revert")
        if expected_revision is not None and await self._review_signature(path) != expected_revision:
            raise RuntimeError("File or index changed since this review; refresh before reverting")
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
        command = f"git --literal-pathspecs restore --source=HEAD --staged --worktree -- {shlex.quote(path)}"
        await self._ws._approvals.require(self._ws._shell_decision(command))
        mutation = None

        async def preflight() -> None:
            nonlocal mutation
            if expected_revision is not None and await self._review_signature(path) != expected_revision:
                raise RuntimeError("File or index changed since this review; refresh before reverting")
            mutation = self._ws._journal.record_preimage(resolved)

        result = await self._ws._run_authorized(command, before_run=preflight)
        assert mutation is not None
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
                def read_preview() -> str:
                    with target.open(errors="replace") as stream:
                        return stream.read(80_001)
                text = await asyncio.to_thread(read_preview)
            except OSError as exc:
                return f"diff unavailable: {exc}"
            lines = text.splitlines(keepends=True)
            return "".join(
                difflib.unified_diff([], lines, fromfile="/dev/null", tofile=f"b/{path}", n=3)
            )
        args = ["diff", "--no-ext-diff", "--unified=3"]
        if scope == "staged":
            args.append("--cached")
        args.extend(["--", path])
        result = await self._git(*args)
        return result.stdout or result.stderr or "(no textual diff)"

    async def _counts(self, path: str, scope: str, status: str) -> tuple[int, int]:
        if self._review_path_error(path) is not None:
            return 0, 0
        if status == "?":
            target = self._ws._workspace.resolve(path)
            try:
                def count_lines() -> int:
                    count = 0
                    last = ""
                    with target.open(errors="replace") as stream:
                        while chunk := stream.read(1_000_000):
                            count += chunk.count("\n")
                            last = chunk[-1]
                    return count + int(bool(last) and last != "\n")
                return await asyncio.to_thread(count_lines), 0
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
            try:
                return subprocess.run(
                    ["git", "--literal-pathspecs", *args],
                    cwd=self._ws._workspace.root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                # Callers already branch on returncode/stderr; a synthetic
                # result keeps every path (review/revert/patch) uniform.
                return subprocess.CompletedProcess(
                    args,
                    returncode=124,
                    stdout="",
                    stderr=f"git {args[0]!r} timed out after 10s",
                )
            except FileNotFoundError:
                return subprocess.CompletedProcess(
                    args,
                    returncode=127,
                    stdout="",
                    stderr="git is not installed or not on PATH",
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
