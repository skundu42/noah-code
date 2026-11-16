"""Narrow git helpers - status/diff/log only by default."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Annotated

from nooa import Skill, hidden, spec

from noah_code.permissions import PermissionCategory, is_secret_path
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import WorkspaceError, parse_porcelain_z


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
        await self._ws._approvals.require(
            self._ws._engine.decide(PermissionCategory.READ, ".", tool="git_status")
        )
        result = await self._ws.run_trusted_readonly("git status --short --branch")
        return result.stdout or result.stderr

    async def diff(
        self,
        path: Annotated[str | None, spec(description="Optional path limit")] = None,
    ) -> str:
        """Return ``git diff`` (unstaged + staged summary via --stat if no path)."""
        if path:
            blocked = self._review_path_error(path)
            if blocked is not None:
                return blocked
            resolved = self._ws._workspace.resolve(path)
            rel = self._ws._workspace.relpath(resolved)
            target_rel = rel if rel not in {"", "."} else "."
            await self._ws._approvals.require(
                self._ws._engine.decide(PermissionCategory.READ, target_rel, tool="git_diff")
            )
            if not resolved.is_dir():
                cmd = f"git diff --relative -- {shlex.quote(rel)}"
                result = await self._ws.run_trusted_readonly(cmd)
                return result.stdout or "(no diff)"
            scope_args: tuple[str, ...] = ("--", target_rel)
        else:
            await self._ws._approvals.require(
                self._ws._engine.decide(PermissionCategory.READ, ".", tool="git_diff")
            )
            scope_args = ()

        names = await self._git("diff", "--relative", "--name-only", "-z", *scope_args)
        chunks: list[str] = []
        for file_path in names.stdout.split("\0"):
            if not file_path:
                continue
            blocked = self._review_path_error(file_path)
            if blocked is not None:
                chunks.append(blocked)
                continue
            await self._ws._approvals.require(
                self._ws._engine.decide(PermissionCategory.READ, file_path, tool="git_diff")
            )
            piece = await self._ws.run_trusted_readonly(
                f"git diff --relative -- {shlex.quote(file_path)}"
            )
            if piece.stdout:
                chunks.append(piece.stdout)
        return "\n".join(chunks) or "(no diff)"

    async def log(
        self,
        n: Annotated[int, spec(description="Number of commits")] = 5,
    ) -> str:
        """Return recent commit subjects."""
        await self._ws._approvals.require(
            self._ws._engine.decide(PermissionCategory.READ, ".", tool="git_log")
        )
        if int(n) <= 0:
            return "(no commits)"
        result = await self._ws.run_trusted_readonly(f"git log -n {int(n)} --oneline")
        return result.stdout or "(no commits)"

    async def _repo_prefix(self) -> str:
        res = await self._git("rev-parse", "--show-prefix")
        return res.stdout.strip() if res.returncode == 0 else ""

    async def review(self, *, eager: bool = True) -> DiffReview:
        """List changes immediately; optionally load patches for noninteractive callers."""
        prefix = await self._repo_prefix()
        status = await self._git(
            "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."
        )
        if status.returncode != 0:
            raise RuntimeError(status.stderr.strip() or "git status failed")
        _, raw_entries = parse_porcelain_z(status.stdout)
        changed: list[tuple[str, str, str]] = []
        for entry in raw_entries:
            path = entry.path
            if prefix:
                if not path.startswith(prefix):
                    continue
                path = path[len(prefix) :]
            if not path:
                continue
            x, y = entry.index_status, entry.worktree_status
            code = f"{x}{y}"
            if x not in {" ", "?", "!"}:
                changed.append((path, "staged", x))
            if y not in {" ", "!"} or code == "??":
                changed.append((path, "unstaged", "?" if code == "??" else y))

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
        rel = self._ws._workspace.relpath(resolved)
        if scope == "unstaged":
            ls_res = await self._git("ls-files", "--error-unmatch", "--", rel)
            if ls_res.returncode not in {0, 1}:
                raise RuntimeError(ls_res.stderr or "git ls-files failed")
            async with self._ws._coordinator.lock:
                if (
                    expected_revision is not None
                    and await self._review_signature(rel) != expected_revision
                ):
                    raise RuntimeError(
                        "File or index changed since this review; refresh before reverting"
                    )
                if ls_res.returncode == 0:
                    index_bytes = await self._git_bytes("show", f":./{rel}")
                    if index_bytes.returncode != 0:
                        err = (index_bytes.stderr or b"").decode(
                            "utf-8", errors="replace"
                        ).strip()
                        raise RuntimeError(err or f"git show :{rel} failed")
                    mode = resolved.lstat().st_mode & 0o7777 if os.path.lexists(resolved) else None
                    mut = self._ws._journal.record_preimage(resolved)
                    try:
                        SnapshotJournal._write_state(resolved, index_bytes.stdout, mode)
                        self._ws._journal.record_postimage(mut, resolved)
                    except Exception:
                        self._ws._journal.discard_mutation(mut)
                        raise
                    self._ws._refresh_read_fingerprint(resolved)
                elif os.path.lexists(resolved):
                    mut = self._ws._journal.record_preimage(resolved)
                    try:
                        SnapshotJournal._write_state(resolved, None, None)
                        self._ws._journal.record_postimage(mut, resolved)
                    except Exception:
                        self._ws._journal.discard_mutation(mut)
                        raise
                    self._ws._read_fingerprints.pop(str(resolved), None)
                else:
                    return f"{path} is already absent"
            return f"reverted unstaged changes in {path}"
        if scope != "staged":
            raise ValueError("scope must be staged or unstaged")
        head_has = await self._git("cat-file", "-e", f"HEAD:./{rel}")
        # Staging metadata is outside the file journal. Capture the worktree
        # preimage, perform the explicit Git restore/rm, and mark this turn as not
        # fully reversible so /undo never overpromises.
        if head_has.returncode == 0:
            command = f"git --literal-pathspecs restore --source=HEAD --staged --worktree -- {shlex.quote(rel)}"
        else:
            command = f"git --literal-pathspecs rm -f -- {shlex.quote(rel)}"
        await self._ws._approvals.require(self._ws._shell_decision(command))
        mutation = None

        async def preflight() -> None:
            nonlocal mutation
            if (
                expected_revision is not None
                and await self._review_signature(rel) != expected_revision
            ):
                raise RuntimeError(
                    "File or index changed since this review; refresh before reverting"
                )
            mutation = self._ws._journal.record_preimage(resolved)

        result = await self._ws._run_authorized(command, before_run=preflight)
        assert mutation is not None
        if result.returncode != 0:
            self._ws._journal.discard_mutation(mutation)
            raise RuntimeError(result.stderr or "git restore failed")
        self._ws._journal.record_postimage(mutation, resolved)
        self._ws._refresh_read_fingerprint(resolved)
        return f"reverted staged and worktree changes in {path}"

    @hidden
    async def stage_hunk(
        self,
        path: str,
        scope: str,
        hunk_patch: str,
        *,
        expected_revision: str | None = None,
    ) -> str:
        """Stage (when unstaged) or unstage (when staged) one unified diff hunk."""

        if scope not in {"unstaged", "staged"}:
            raise ValueError("scope must be staged or unstaged")
        resolved = await self._ws._authorize_path(path, "edit", tool="git_stage_hunk")
        rel = self._ws._workspace.relpath(resolved)
        payload = hunk_patch if hunk_patch.endswith("\n") else f"{hunk_patch}\n"
        async with self._ws._coordinator.lock:
            if (
                expected_revision is not None
                and await self._review_signature(rel) != expected_revision
            ):
                raise RuntimeError(
                    "File or index changed since this review; refresh before staging"
                )
            prefix = await self._repo_prefix()
            args = ["apply", "--cached", "--unidiff-zero"]
            if scope == "staged":
                args.append("--reverse")
            if prefix:
                args.append(f"--directory={prefix.rstrip('/')}")
            result = await self._git_input(*args, input_text=payload)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "git apply --cached failed")
        action = "unstaged" if scope == "staged" else "staged"
        return f"{action} hunk in {path}"

    @hidden
    async def revert_hunk(
        self,
        path: str,
        scope: str,
        hunk_patch: str,
        *,
        expected_revision: str | None = None,
    ) -> str:
        """Revert one unified diff hunk; journaled in SnapshotJournal for /undo."""

        if scope not in {"unstaged", "staged"}:
            raise ValueError("scope must be staged or unstaged")
        resolved = await self._ws._authorize_path(path, "edit", tool="git_revert_hunk")
        rel = self._ws._workspace.relpath(resolved)
        payload = hunk_patch if hunk_patch.endswith("\n") else f"{hunk_patch}\n"
        async with self._ws._coordinator.lock:
            if (
                expected_revision is not None
                and await self._review_signature(rel) != expected_revision
            ):
                raise RuntimeError(
                    "File or index changed since this review; refresh before reverting"
                )
            prefix = await self._repo_prefix()
            args = ["apply", "--reverse", "--unidiff-zero"]
            if scope == "staged":
                args.append("--index")
            if prefix:
                args.append(f"--directory={prefix.rstrip('/')}")
            mut = self._ws._journal.record_preimage(resolved)
            try:
                result = await self._git_input(*args, input_text=payload)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "git apply --reverse failed")
                if scope == "staged":
                    self._ws._journal.mark_shell_bypass()
                self._ws._journal.record_postimage(mut, resolved)
            except Exception:
                self._ws._journal.discard_mutation(mut)
                raise
            self._ws._refresh_read_fingerprint(resolved)
        return f"reverted {scope} hunk in {path}"

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
        args = ["diff", "--relative", "--no-ext-diff", "--unified=3"]
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
        args = ["diff", "--relative", "--numstat"]
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

    async def _git_bytes(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        def run() -> subprocess.CompletedProcess[bytes]:
            try:
                return subprocess.run(
                    ["git", "--literal-pathspecs", *args],
                    cwd=self._ws._workspace.root,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return subprocess.CompletedProcess(
                    args,
                    returncode=124,
                    stdout=b"",
                    stderr=f"git {args[0]!r} timed out after 10s".encode(),
                )
            except FileNotFoundError:
                return subprocess.CompletedProcess(
                    args,
                    returncode=127,
                    stdout=b"",
                    stderr=b"git is not installed or not on PATH",
                )

        return await asyncio.to_thread(run)

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

    async def _git_input(
        self,
        *args: str,
        input_text: str,
    ) -> subprocess.CompletedProcess[str]:
        def run() -> subprocess.CompletedProcess[str]:
            try:
                return subprocess.run(
                    ["git", "--literal-pathspecs", *args],
                    cwd=self._ws._workspace.root,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
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
