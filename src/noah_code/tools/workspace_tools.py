"""Permission-gated workspace tools wrapping ShellTools."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import gc
import hashlib
import os
import re
import shlex
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from nooa import Skill, hidden, spec
from nooa.tools.shell_tools import (
    FileWrite,
    Match,
    ShellResult,
    ShellTools,
    StreamDone,
    StreamEvent,
)

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import (
    PermissionCategory,
    PermissionDecision,
    PermissionEngine,
    is_secret_path,
)
from noah_code.snapshots import SnapshotJournal
from noah_code.tool_output import ToolOutputStore
from noah_code.workspace import Workspace, WorkspaceError

if TYPE_CHECKING:
    from noah_code.runtime_state import RuntimeStateStore

# Module-level aliases: inside the class body the tool methods named ``list``
# would shadow builtins.list in parameter annotations.
InspectTargets = list[str]
PatchChanges = list[dict[str, str | None]]

_IGNORED_LIST_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cursor",
    }
)


class WorkspaceMutationCoordinator:
    """One mutation lane shared by a parent agent and all nested agents."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


def _pattern_keeps_dir(pattern: str, name: str) -> bool:
    return name in pattern.replace("\\", "/").split("/")


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob with ``**`` into a regex that works on Python 3.12+."""
    parts: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        if pattern.startswith("**", index):
            rest = pattern[index + 2 :]
            if rest.startswith("/"):
                parts.append("(?:.*/)?")
                index += 3
            elif rest == "":
                parts.append(".*")
                index += 2
            else:
                parts.append("[^/]*[^/]*")
                index += 2
            continue
        char = pattern[index]
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(char))
        index += 1
    return re.compile("^" + "".join(parts) + "$")


def _matches_glob(relative: str, pattern: str) -> bool:
    posix = relative.replace("\\", "/")
    normalized = pattern.replace("\\", "/").removeprefix("./")
    try:
        return _glob_to_regex(normalized).fullmatch(posix) is not None
    except re.error:
        return False


class WorkspaceTools(Skill):
    """Read, search, edit, and run commands inside the active workspace.

    All mutating operations go through the permission engine. Prefer
    Match-based ``replace`` over rewriting whole files. Familiar ``list``,
    ``edit``, and ``write`` aliases are available for model compatibility.
    Paths are canonicalized and must remain inside the workspace unless
    ``external_directory`` is approved.
    """

    def __init__(
        self,
        workspace: Workspace,
        shell: ShellTools,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        journal: SnapshotJournal,
        *,
        max_output_chars: int = 16_000,
        max_output_lines: int = 250,
        max_search_results: int = 100,
        max_file_results: int = 500,
        output_retention_hours: int = 24,
        default_timeout: float = 60.0,
        lsp: Any = None,
        runtime: RuntimeStateStore | None = None,
        coordinator: WorkspaceMutationCoordinator | None = None,
        output_store_root: Path | None = None,
        output_store_max_bytes: int = 2_000_000_000,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._shell = shell
        self._engine = engine
        self._approvals = approvals
        self._journal = journal
        self._max_output = max_output_chars
        self._max_output_lines = max_output_lines
        self._max_search_results = max_search_results
        self._max_file_results = max_file_results
        self._output_store = ToolOutputStore(
            root=output_store_root,
            retention_hours=None if output_store_root is not None else output_retention_hours,
            max_total_bytes=output_store_max_bytes,
        )
        self._default_timeout = default_timeout
        self._on_shell_chunk: Any = None
        self._lsp = lsp
        self._runtime = runtime
        self._coordinator = coordinator or WorkspaceMutationCoordinator()
        self._mutation_checkpoint_handler: Any = None
        # NOOA 0.0.9 starts BashSession lazily without guarding concurrent
        # callers. Batched inspections can otherwise launch multiple shells
        # and orphan every process except the last one assigned to the session.
        self._shell_start_lock = asyncio.Lock()
        # Every operation on the persistent shell shares one transaction lock.
        # Pinned operations hold it across pin, command, and cwd restoration;
        # streamed commands hold it until their iterator finishes.
        self._file_op_lock = asyncio.Lock()
        # sha256 of raw bytes at read() time, keyed by absolute path. Anchored
        # (Match) edits verify this fingerprint before splicing so a stale
        # anchor fails loudly instead of corrupting a concurrently changed file.
        self._read_fingerprints: dict[str, str] = {}

    def set_lsp(self, lsp: Any) -> None:
        """Attach diagnostics after both services have been constructed."""
        self._lsp = lsp

    def set_efficiency_profile(self, profile: str) -> None:
        """Adjust model-facing output limits for the current session."""

        if profile == "fast":
            self._max_output = 16_000
            self._max_output_lines = 250
        elif profile == "balanced":
            self._max_output = 24_000
            self._max_output_lines = 400
        elif profile == "deep":
            self._max_output = 80_000
            self._max_output_lines = 2_000
        else:
            raise ValueError("profile must be fast, balanced, or deep")

    def set_shell_chunk_handler(self, handler: Any) -> None:
        """Optional callback(stream: str, text: str) for UI streaming."""
        self._on_shell_chunk = handler

    def set_mutation_checkpoint_handler(self, handler: Any) -> None:
        """Install an async host callback used before untracked shell effects."""

        self._mutation_checkpoint_handler = handler

    async def checkpoint_before_shell(self, command: str) -> None:
        if self._mutation_checkpoint_handler is not None:
            await self._mutation_checkpoint_handler(command)

    @hidden
    @property
    def raw_shell(self) -> ShellTools:
        """Raw shell - host only; not for the model."""
        return self._shell

    async def read(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        lines: Annotated[
            tuple[int, int] | None,
            spec(description="Optional (start, end) 1-indexed inclusive range"),
        ] = None,
    ) -> Match | str:
        """Read a file range; oversized reads return a managed preview, not an edit anchor."""
        resolved = await self._authorize_path(path, PermissionCategory.READ, tool="ws_read")
        async with self._pinned_shell_cwd():
            result = await self._shell.read(str(resolved), lines=lines)
        self._record_read_fingerprint(resolved)
        bounded = self._bound(result.text)
        if bounded != result.text:
            # A head/tail preview is not contiguous file content and therefore
            # must never masquerade as an editable Match anchor.
            return f"{result.path}:{result.start}-{result.end}\n{bounded}"
        return result

    async def search(
        self,
        pattern: Annotated[str, spec(description="Regex or fixed pattern for ripgrep")],
        path: Annotated[str, spec(description="Subdirectory or file to search")] = ".",
    ) -> ShellResult:
        """Return bounded ripgrep text; call read() on a result to get an edit anchor."""
        resolved = await self._authorize_path(path, PermissionCategory.READ, tool="ws_search")
        target = str(resolved) if resolved != self._workspace.root.resolve() else "."
        cmd = " ".join(
            shlex.quote(a) for a in ["rg", "-n", "--no-heading", "-S", "--", pattern, target]
        )
        async with self._pinned_shell_cwd():
            result = await self._shell.run(cmd, timeout=self._default_timeout)
        return self._cap_shell_result(self._redact_secret_search(result))

    async def list_files(
        self,
        pattern: Annotated[str, spec(description="Glob pattern")] = "**/*",
        path: Annotated[str, spec(description="Subdirectory")] = ".",
    ) -> list[str]:
        """List files under path matching a glob (deterministic, no shell)."""
        self._assert_internal_glob(pattern)
        root = await self._authorize_path(path, PermissionCategory.READ, tool="ws_list")
        workspace_root = self._workspace.root.resolve()
        matches: list[str] = []
        truncated = False

        for dirpath, dirnames, filenames in os.walk(
            root,
            onerror=lambda _error: None,
            followlinks=False,
        ):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if _pattern_keeps_dir(pattern, name) or name not in _IGNORED_LIST_DIRS
            )
            directory = Path(dirpath)
            for name in sorted(filenames):
                candidate = directory / name
                try:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    resolved = candidate.resolve()
                    glob_rel = candidate.relative_to(root).as_posix()
                    workspace_rel = resolved.relative_to(workspace_root).as_posix()
                except (OSError, ValueError):
                    continue
                if is_secret_path(workspace_rel):
                    continue
                if not _matches_glob(glob_rel, pattern):
                    continue
                matches.append(workspace_rel)
                if len(matches) >= self._max_file_results:
                    truncated = True
                    dirnames.clear()
                    break
            if truncated:
                break

        matches.sort()
        if truncated:
            matches.append(f"...[truncated at {self._max_file_results}]")
        return matches

    async def list(
        self,
        pattern: Annotated[str, spec(description="Glob pattern")] = "**/*",
        path: Annotated[str, spec(description="Subdirectory")] = ".",
    ) -> list[str]:
        """Compatibility alias for list_files()."""

        return await self.list_files(pattern=pattern, path=path)

    async def read_output(
        self,
        output_id: Annotated[str, spec(description="Managed output id from a truncated result")],
        lines: Annotated[
            tuple[int, int],
            spec(description="1-indexed inclusive line range to retrieve"),
        ],
    ) -> str:
        """Read a focused slice of a previously truncated full tool result."""

        text = self._output_store.read(output_id, lines)
        return self._bound(text)

    async def inspect(
        self,
        searches: Annotated[
            InspectTargets | None,
            spec(description="Regex/fixed searches to run concurrently"),
        ] = None,
        files: Annotated[
            InspectTargets | None,
            spec(description="Workspace files to read concurrently"),
        ] = None,
        symbols: Annotated[
            bool,
            spec(description="Also return class/function/type declarations"),
        ] = False,
    ) -> str:
        """Batch focused repository searches and reads into one compact result."""

        queries: list[str] = list(dict.fromkeys(searches or []))
        paths: list[str] = list(dict.fromkeys(files or []))
        if len(queries) > 8 or len(paths) > 8:
            raise ValueError("inspect accepts at most 8 searches and 8 files")
        if not queries and not paths and not symbols:
            raise ValueError("inspect requires searches, files, or symbols=True")

        import asyncio

        labels: list[tuple[str, str]] = []
        operations = []
        for query in queries:
            labels.append(("search", query))
            operations.append(self.search(query))
        for path in paths:
            labels.append(("file", path))
            operations.append(self.read(path, lines=(1, 300)))
        if symbols:
            labels.append(("symbols", "definitions"))
            operations.append(
                self.search(r"^(?:class|def|async def|interface|type|enum|struct)\s+", ".")
            )

        results = await asyncio.gather(*operations, return_exceptions=True)
        sections: list[str] = []
        for (kind, label), result in zip(labels, results, strict=True):
            sections.append(f"## {kind}: {label}")
            if isinstance(result, Exception):
                sections.append(f"error: {type(result).__name__}: {result}")
            elif isinstance(result, ShellResult):
                sections.append(result.stdout or result.stderr or "(no matches)")
            elif isinstance(result, Match):
                sections.append(f"{result.path}:{result.start}-{result.end}\n{result.text}")
            else:
                sections.append(str(result))
        return self._bound("\n\n".join(sections))

    async def replace(
        self,
        match: Annotated[Any, spec(description="Match from read()/search() or path string")],
        new_text: Annotated[str, spec(description="Replacement text for Match form")] = "",
        new: Annotated[str | None, spec(description="Path-form replacement")] = None,
    ) -> Any:
        """Edit via Match anchor (preferred) or unique string replacement."""
        async with self._coordinator.lock:
            return await self._replace_locked(match, new_text, new)

    async def _replace_locked(self, match: Any, new_text: str, new: str | None) -> Any:
        if isinstance(match, Match):
            resolved = await self._authorize_path(
                match.path, PermissionCategory.EDIT, tool="ws_edit"
            )
            expected = self._read_fingerprints.get(str(resolved))
            if expected is not None:
                actual = self._hash_path(resolved)
                if actual != expected:
                    raise ValueError(
                        f"stale edit anchor: {self._workspace.relpath(resolved)} changed "
                        "since read(); call read() again to refresh the Match"
                    )
            durable = self._begin_durable_file_operation(resolved)
            mut = self._journal.record_preimage(resolved)
            try:
                result = self._native_replace_match(resolved, match, new_text)
                self._journal.record_postimage(mut, resolved)
                self._complete_durable_file_operation(durable, resolved)
            except Exception:
                self._journal.discard_mutation(mut)
                self._rollback_durable_file_operation(durable)
                raise
            return result
        if isinstance(match, str):
            resolved = await self._authorize_path(match, PermissionCategory.EDIT, tool="ws_edit")
            durable = self._begin_durable_file_operation(resolved)
            mut = self._journal.record_preimage(resolved)
            try:
                async with self._pinned_shell_cwd():
                    result = await self._shell.replace(str(resolved), new_text, new)
                self._journal.record_postimage(mut, resolved)
                self._complete_durable_file_operation(durable, resolved)
            except Exception:
                self._journal.discard_mutation(mut)
                self._rollback_durable_file_operation(durable)
                raise
            return result
        raise TypeError("replace expects a Match or path string")

    async def edit(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        old: Annotated[str, spec(description="Unique text to replace")],
        new: Annotated[str, spec(description="Replacement text")],
    ) -> Any:
        """Replace one unique string in a file; compatibility alias for replace()."""

        return await self.replace(path, old, new)

    async def write_file(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> Any:
        """Create or overwrite a file with content."""
        async with self._coordinator.lock:
            return await self._write_file_locked(path, content)

    async def _write_file_locked(self, path: str, content: str) -> Any:
        resolved = await self._authorize_path(path, PermissionCategory.EDIT, tool="write_file")
        durable = self._begin_durable_file_operation(resolved)
        mut = self._journal.record_preimage(resolved)
        try:
            result = self._atomic_write_bytes(resolved, content.encode("utf-8"), content)
            self._journal.record_postimage(mut, resolved)
            self._complete_durable_file_operation(durable, resolved)
        except Exception:
            self._journal.discard_mutation(mut)
            self._rollback_durable_file_operation(durable)
            raise
        return result

    async def write(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> Any:
        """Create or overwrite a file; compatibility alias for write_file()."""

        return await self.write_file(path, content)

    async def apply_patch(
        self,
        changes: Annotated[
            PatchChanges,
            spec(
                description=(
                    "Atomic file changes: path plus exact old text and new text; "
                    "old=null creates, new=null deletes"
                )
            ),
        ],
    ) -> str:
        """Apply one exact, transactional multi-file patch and report diagnostics.

        Each update replaces one unique ``old`` preimage. A create uses
        ``old=None``; a delete uses ``new=None`` and requires ``old`` to equal
        the entire current file. Every target is authorized and preflighted
        before any file changes. A failed commit rolls the whole batch back.
        """
        async with self._coordinator.lock:
            return await self._apply_patch_locked(changes)

    async def _apply_patch_locked(self, changes: PatchChanges) -> str:
        if not changes:
            raise ValueError("patch requires at least one change")
        if len(changes) > 50:
            raise ValueError("patch accepts at most 50 files")

        prepared: list[dict[str, Any]] = []
        seen: set[Path] = set()
        # Resolve and authorize the entire batch before reading preimages.
        for raw in changes:
            path = str(raw.get("path") or "").strip()
            if not path:
                raise ValueError("every patch change requires path")
            resolved = await self._authorize_path(
                path, PermissionCategory.EDIT, tool="apply_patch"
            )
            if resolved in seen:
                raise ValueError(f"patch target repeated: {path}")
            seen.add(resolved)
            prepared.append(
                {
                    "path": path,
                    "resolved": resolved,
                    "old": raw.get("old"),
                    "new": raw.get("new"),
                }
            )

        for item in prepared:
            target = cast(Path, item["resolved"])
            old = item["old"]
            new = item["new"]
            exists = target.is_file()
            if target.exists() and not exists:
                raise ValueError(f"patch target is not a file: {item['path']}")
            before_bytes = target.read_bytes() if exists else None
            if before_bytes is not None and len(before_bytes) > self._journal.blob_limit:
                raise ValueError(
                    f"patch target exceeds atomic rollback limit ({self._journal.blob_limit} bytes): "
                    f"{item['path']}"
                )
            if before_bytes is not None and b"\0" in before_bytes:
                raise ValueError(f"binary patch targets are not supported: {item['path']}")
            before = before_bytes.decode("utf-8") if before_bytes is not None else None
            if old is None:
                if exists:
                    raise ValueError(f"create preimage failed; file already exists: {item['path']}")
                if new is None:
                    raise ValueError(f"create requires new content: {item['path']}")
                after = new
                operation = "add"
            elif not exists:
                raise ValueError(f"update preimage failed; file does not exist: {item['path']}")
            elif new is None:
                if old != before:
                    raise ValueError(
                        f"delete preimage mismatch; old must equal the full file: {item['path']}"
                    )
                after = None
                operation = "delete"
            else:
                current_text = before or ""
                occurrences = current_text.count(str(old))
                if occurrences != 1:
                    raise ValueError(
                        f"update preimage must match exactly once in {item['path']} "
                        f"(found {occurrences})"
                    )
                after = current_text.replace(str(old), str(new), 1)
                operation = "update"
            item.update(
                before=before,
                before_bytes=before_bytes,
                after=after,
                after_bytes=after.encode() if after is not None else None,
                operation=operation,
                mode=target.stat().st_mode if exists else None,
            )

        temporary: dict[Path, Path] = {}
        mutations = []
        durable_operations: list[tuple[str, Path]] = []
        committed: list[dict[str, Any]] = []
        created_dirs: list[Path] = []
        try:
            # Stage every write on the target filesystem before committing.
            for item in prepared:
                target = cast(Path, item["resolved"])
                after_bytes: bytes | None = item["after_bytes"]
                if after_bytes is None:
                    continue
                missing: list[Path] = []
                parent = target.parent
                probe = parent
                while not probe.exists() and probe.is_relative_to(self._workspace.root):
                    missing.append(probe)
                    probe = probe.parent
                parent.mkdir(parents=True, exist_ok=True)
                created_dirs.extend(reversed(missing))
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{target.name}.noah-", dir=parent
                )
                temp_path = Path(temp_name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(after_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                if item["mode"] is not None:
                    temp_path.chmod(item["mode"])
                temporary[target] = temp_path

            # Close the TOCTOU window: every file must still match its preflight bytes.
            for item in prepared:
                target = cast(Path, item["resolved"])
                current = target.read_bytes() if target.exists() else None
                if current != item["before_bytes"]:
                    raise RuntimeError(f"concurrent modification detected: {item['path']}")

            operation_group = uuid.uuid4().hex
            for item in prepared:
                target = cast(Path, item["resolved"])
                durable = self._begin_durable_file_operation(
                    target, operation_group=operation_group
                )
                if durable:
                    durable_operations.append((durable, target))
            for item in prepared:
                mutations.append(self._journal.record_preimage(item["resolved"]))
            for item in prepared:
                target = cast(Path, item["resolved"])
                if item["after_bytes"] is None:
                    target.unlink()
                else:
                    os.replace(temporary.pop(target), target)
                self._fsync_directory(target.parent)
                committed.append(item)
            for mutation, item in zip(mutations, prepared, strict=True):
                self._journal.record_postimage(mutation, item["resolved"])
            if durable_operations and self._runtime is not None:
                self._runtime.complete_file_operations(durable_operations)
        except Exception as exc:
            rollback_error: Exception | None = None
            try:
                for item in reversed(committed):
                    SnapshotJournal._write_state(
                        item["resolved"], item["before_bytes"], item["mode"]
                    )
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_error = rollback_exc
            for mutation in mutations:
                self._journal.discard_mutation(mutation)
            for durable, _target in reversed(durable_operations):
                self._rollback_durable_file_operation(durable)
            for directory in reversed(created_dirs):
                with contextlib.suppress(OSError):
                    directory.rmdir()
            if rollback_error is not None:
                raise RuntimeError(
                    f"atomic patch failed and rollback also failed: {exc}; {rollback_error}"
                ) from exc
            raise RuntimeError(f"atomic patch failed; all changes rolled back: {exc}") from exc
        finally:
            for temp_path in temporary.values():
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()

        rows = ["Applied atomic patch:"]
        changed_paths: list[str] = []
        for item in prepared:
            before_lines = (item["before"] or "").splitlines()
            after_lines = (item["after"] or "").splitlines()
            delta = list(difflib.ndiff(before_lines, after_lines))
            additions = sum(line.startswith("+ ") for line in delta)
            deletions = sum(line.startswith("- ") for line in delta)
            marker = {"add": "A", "update": "M", "delete": "D"}[item["operation"]]
            rows.append(f"  {marker} {item['path']}  +{additions} -{deletions}")
            if item["after"] is not None:
                changed_paths.append(item["path"])
        if self._lsp is not None and changed_paths:
            diagnostics = await self._lsp.diagnostics_for_paths(changed_paths)
            rows.append("Diagnostics:")
            for path, result in diagnostics.items():
                first = result.splitlines()[0] if result else "unavailable"
                rows.append(f"  {path}: {first}")
        return "\n".join(rows)

    async def apply_unified_diff(
        self,
        diff_text: Annotated[
            str,
            spec(description="Unified diff text (git-style ---/+++/@@ hunks) to apply atomically"),
        ],
    ) -> str:
        """Apply a unified diff as one transactional, journaled patch.

        Context lines are verified against current file content; a mismatch
        aborts before any file changes. Creates use '--- /dev/null' and
        deletes use '+++ /dev/null'.
        """
        from noah_code.tools.diff_tools import (
            materialize_change,
            parse_unified_diff,
        )

        if not diff_text or not diff_text.strip():
            raise ValueError("diff text is required")
        parsed = parse_unified_diff(diff_text)
        changes: list[dict[str, str | None]] = []
        for file_diff in parsed:
            resolved = await self._authorize_path(
                file_diff.path, PermissionCategory.EDIT, tool="apply_patch"
            )
            current: str | None = None
            if not file_diff.is_create:
                try:
                    data = resolved.read_bytes()
                except OSError as exc:
                    raise ValueError(f"cannot read patch target {file_diff.path}: {exc}") from exc
                if b"\0" in data or len(data) > self._journal.blob_limit:
                    raise ValueError(
                        f"binary or oversized patch targets are not supported: {file_diff.path}"
                    )
                try:
                    current = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"non-UTF-8 patch targets are not supported: {file_diff.path}"
                    ) from exc
            materialized = materialize_change(file_diff, current)
            changes.append(
                {
                    "path": str(resolved),
                    "old": materialized.old,
                    "new": materialized.new,
                }
            )
        report = await self.apply_patch(changes)
        return f"Applied unified diff ({len(parsed)} file(s)):\n{report}"

    async def run(
        self,
        command: Annotated[str, spec(description="Shell command")],
        stdin: Annotated[str | None, spec(description="Optional stdin payload")] = None,
        timeout: Annotated[float | None, spec(description="Timeout seconds")] = None,
    ) -> ShellResult:
        """Run a command in the workspace shell session."""
        decision = self._shell_decision(command)
        await self._approvals.require(decision)
        if not self._engine.is_readonly_command(command):
            await self.checkpoint_before_shell(command)
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
        mutating = not self._engine.is_readonly_command(command)
        async with self._mutation_guard(mutating), self._file_op_lock:
            await self._ensure_shell_started()
            result = await self._shell.run(
                command,
                stdin=stdin,
                timeout=timeout or self._default_timeout,
            )
        if self._on_shell_chunk is not None:
            if result.stdout:
                self._on_shell_chunk("stdout", result.stdout)
            if result.stderr:
                self._on_shell_chunk("stderr", result.stderr)
            self._on_shell_chunk("status", f"[exit {result.returncode}]\n")
        return self._cap_shell_result(result)

    async def run_stream(
        self,
        command: Annotated[str, spec(description="Shell command")],
        timeout: Annotated[float | None, spec(description="Timeout seconds")] = None,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Stream command output; same permission rules as run()."""
        decision = self._shell_decision(command)
        await self._approvals.require(decision)
        if not self._engine.is_readonly_command(command):
            await self.checkpoint_before_shell(command)
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
        mutating = not self._engine.is_readonly_command(command)
        async with self._mutation_guard(mutating), self._file_op_lock:
            await self._ensure_shell_started()
            try:
                async for event in self._shell.run_stream(
                    command, timeout=timeout or self._default_timeout
                ):
                    if self._on_shell_chunk is not None and hasattr(event, "kind"):
                        self._on_shell_chunk(
                            getattr(event, "kind", "stdout"), getattr(event, "text", "")
                        )
                    yield event
            finally:
                # NOOA updates BashSession.cwd after a streamed command but
                # currently leaves ShellTools.cwd stale.
                self._shell.cwd = self._shell.session.cwd.resolve()

    def _shell_decision(self, command: str, *, tool: str = "ws_run") -> PermissionDecision:
        decision = self._engine.decide(PermissionCategory.BASH, command, tool=tool)
        if decision.denied or not self._engine.is_uncertain_shell(command):
            return decision
        # Shell syntax is too ambiguous to auto-approve safely. Interactive
        # sessions may still approve the exact command once.
        action: Literal["allow", "ask", "deny"] = (
            "deny" if self._engine.auto_approve else "ask"
        )
        reason = (
            "compound/uncertain shell commands cannot be auto-approved"
            if action == "deny"
            else "compound/uncertain shell command requires approval"
        )
        return replace(
            decision,
            action=action,
            reason=reason,
        )

    @contextlib.asynccontextmanager
    async def _mutation_guard(self, enabled: bool):  # noqa: ANN202
        if not enabled:
            yield
            return
        async with self._coordinator.lock:
            yield

    def _begin_durable_file_operation(
        self, path: Path, *, operation_group: str = ""
    ) -> str:
        if self._runtime is None:
            return ""
        return self._runtime.begin_file_operation(path, operation_group=operation_group)

    def _complete_durable_file_operation(self, operation_id: str, path: Path) -> None:
        if operation_id and self._runtime is not None:
            self._runtime.complete_file_operation(operation_id, path)

    def _rollback_durable_file_operation(self, operation_id: str) -> None:
        if operation_id and self._runtime is not None:
            self._runtime.rollback_file_operation(operation_id)

    @hidden
    async def run_trusted_readonly(self, command: str) -> ShellResult:
        """Run a host-constructed, strictly read-only command without a model approval."""
        if not self._engine.is_readonly_command(command):
            raise PermissionError(f"trusted command is not read-only: {command}")
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise PermissionError(f"trusted command is not read-only: {command}") from exc
        if any(is_secret_path(token) or is_secret_path(Path(token).name) for token in tokens):
            raise PermissionError(f"trusted command targets a secret path: {command}")
        async with self._pinned_shell_cwd():
            result = await self._shell.run(command, timeout=self._default_timeout)
        return self._cap_shell_result(result)

    async def _ensure_shell_started(self) -> None:
        """Single-flight NOOA's lazy persistent-shell startup."""
        async with self._shell_start_lock:
            await self._shell.session.start()

    @contextlib.asynccontextmanager
    async def _pinned_shell_cwd(self):
        """Temporarily pin delegated operations to the canonical workspace root.

        NOOA keeps both a Python-side cwd and a live persistent-shell cwd.
        Pinning both keeps a model-driven ``cd`` from redirecting authorized
        file operations, trusted Git commands, or repository searches. Restore
        the prior cwd afterward so user shell state retains its documented
        persistence.
        """

        async with self._file_op_lock:
            await self._ensure_shell_started()
            original = self._shell.session.cwd.resolve()
            self._shell.cwd = original
            workspace_root = self._workspace.root
            if original != workspace_root:
                pin = await self._shell.run(
                    f"cd -- {shlex.quote(str(workspace_root))}",
                    timeout=min(self._default_timeout, 5.0),
                )
                if pin.returncode != 0:
                    raise RuntimeError(f"cannot pin shell cwd to workspace: {pin.stderr.strip()}")
            primary_error: BaseException | None = None
            try:
                yield
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                if original != workspace_root:
                    restore_error: BaseException | None = None
                    message = ""
                    try:
                        restored = await self._shell.run(
                            f"cd -- {shlex.quote(str(original))}",
                            timeout=min(self._default_timeout, 5.0),
                        )
                        if restored.returncode != 0:
                            detail = restored.stderr.strip() or restored.stdout.strip()
                            message = f"cannot restore shell cwd to {original}"
                            if detail:
                                message += f": {detail}"
                    except BaseException as exc:
                        if primary_error is None and not isinstance(exc, Exception):
                            raise
                        restore_error = exc
                        message = f"cannot restore shell cwd to {original}: {exc}"
                    if message:
                        if primary_error is not None:
                            primary_error.add_note(message)
                        else:
                            raise RuntimeError(message) from restore_error

    def _record_read_fingerprint(self, resolved: Path) -> None:
        digest = self._hash_path(resolved)
        if digest is None:
            return
        if len(self._read_fingerprints) > 512:
            # Conservative bound: forget everything and require fresh reads.
            self._read_fingerprints.clear()
        self._read_fingerprints[str(resolved)] = digest

    @staticmethod
    def _hash_path(path: Path) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _native_replace_match(
        self, resolved: Path, match: Match, new_text: str
    ) -> FileWrite:
        """Byte-preserving anchored splice with an atomic commit.

        Unlike the upstream shell splice this never re-encodes or newline-
        normalizes untouched regions: bytes are decoded losslessly, only the
        anchor lines change, and the result lands via temp+fsync+rename.
        """

        data = resolved.read_bytes()
        if b"\0" in data:
            raise ValueError("binary files are not editable via replace()")
        codec = "utf-8"
        try:
            text = data.decode(codec)
        except UnicodeDecodeError:
            codec = "latin-1"  # byte-lossless fallback for legacy encodings
            text = data.decode(codec)
        all_lines = text.splitlines(keepends=True)
        total = len(all_lines)
        start = max(1, int(match.start))
        end = min(total, int(match.end))
        if start > total:
            raise ValueError(
                f"stale edit anchor: {self._workspace.relpath(resolved)} has {total} lines; "
                "call read() again to refresh the Match"
            )
        removed = all_lines[start - 1 : end]
        replacement = new_text
        if replacement and not replacement.endswith("\n") and end < total:
            eol = "\r\n" if removed and removed[-1].endswith("\r\n") else "\n"
            replacement += eol
        new_content = "".join(all_lines[: start - 1]) + replacement + "".join(all_lines[end:])
        mode = resolved.stat().st_mode & 0o7777
        self._atomic_write_bytes(resolved, new_content.encode(codec), new_content, mode=mode)
        diff = f"--- a/{match.path}\n+++ b/{match.path}\n"
        diff += f"@@ -{start},{end - start + 1} @@\n"
        return FileWrite(
            path=match.path,
            message=f"Edited {match.path} (replaced lines {start}-{end})",
            diff=diff,
            new_text=replacement,
        )

    def _atomic_write_bytes(
        self,
        resolved: Path,
        data: bytes,
        display_source: str,
        *,
        mode: int | None = None,
    ) -> FileWrite:
        """Atomically create/overwrite a file without newline translation."""

        resolved.parent.mkdir(parents=True, exist_ok=True)
        if mode is None and resolved.exists():
            mode = resolved.stat().st_mode & 0o7777
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{resolved.name}.noah-", dir=resolved.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if mode is not None:
                os.chmod(temp_name, mode)
            os.replace(temp_name, resolved)
            self._fsync_directory(resolved.parent)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        line_count = (
            display_source.count("\n") + (1 if display_source and not display_source.endswith("\n") else 0)
        )
        display = self._workspace.relpath(resolved)
        return FileWrite(
            path=str(resolved),
            message=f"Created {display} ({line_count} lines)",
            new_text=display_source,
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Make a completed rename durable across sudden power loss."""

        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _cap_shell_result(self, result: ShellResult) -> ShellResult:
        stdout = self._bound(result.stdout)
        stderr = self._bound(result.stderr)
        matches = result.matches
        if matches and len(matches) > self._max_search_results:
            matches = matches[: self._max_search_results]
        if stdout == result.stdout and stderr == result.stderr and matches is result.matches:
            return result
        return ShellResult(
            stdout=stdout,
            stderr=stderr,
            returncode=result.returncode,
            matches=matches,
            timed_out=result.timed_out,
        )

    def _bound(self, text: str) -> str:
        return self._output_store.bound(
            text,
            max_chars=self._max_output,
            max_lines=self._max_output_lines,
        ).text

    @staticmethod
    def _assert_internal_glob(pattern: str) -> None:
        candidate = Path(pattern)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("glob pattern must stay inside the workspace")

    def _redact_secret_search(self, result: ShellResult) -> ShellResult:
        stdout = "".join(
            line
            for line in (result.stdout or "").splitlines(keepends=True)
            if not is_secret_path(_rg_hit_path(line) or "")
        )
        matches = result.matches
        if matches:
            matches = [
                match
                for match in matches
                if not is_secret_path(str(getattr(match, "path", "") or ""))
            ]
        if stdout == (result.stdout or "") and matches is result.matches:
            return result
        return ShellResult(
            stdout=stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            matches=matches,
            timed_out=result.timed_out,
        )

    async def _authorize_path(self, path: str, category: str, *, tool: str = "") -> Path:
        try:
            resolved = self._workspace.resolve(path)
            rel = str(resolved.relative_to(self._workspace.root))
            decision = self._engine.decide(category, rel, tool=tool)
        except WorkspaceError as exc:
            abs_path = Path(path).expanduser()
            if not abs_path.is_absolute():
                abs_path = (self._workspace.root / path).resolve()
            else:
                abs_path = abs_path.resolve()
            decision = self._engine.decide(
                PermissionCategory.EXTERNAL_DIRECTORY, str(abs_path), tool=tool
            )
            await self._approvals.require(decision)
            raise WorkspaceError(
                f"path outside workspace requires dedicated external handling: {path}"
            ) from exc
        await self._approvals.require(decision)
        return resolved

    @hidden
    async def close(self) -> None:
        # NOOA 0.0.9 waits for its persistent shell but can leave asyncio's
        # subprocess transport open on Linux until after the test/app loop exits.
        session = self._shell.session
        process = getattr(session, "_process", None)
        try:
            await self._shell.close()
        finally:
            transport = getattr(process, "_transport", None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.close()
                # Closing a subprocess transport schedules each pipe's
                # connection_lost callback, followed by the parent transport's
                # finalizer. Give that callback chain time to complete while
                # its event loop is still alive.
                for _ in range(3):
                    await asyncio.sleep(0)
                    if getattr(transport, "_finished", True):
                        break
            # NOOA's timeout recovery creates a short-lived helper subprocess
            # that can remain in a transport/protocol cycle on CPython 3.12
            # Linux. Collect it before pytest or the app closes this loop.
            if sys.platform.startswith("linux") and sys.version_info < (3, 13):
                gc.collect()
            await asyncio.sleep(0)


def _rg_hit_path(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(":", 2)
    if len(parts) < 2:
        return None
    return parts[0]
