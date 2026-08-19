"""Permission-gated workspace tools wrapping ShellTools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from nooa import Skill, hidden, spec
from nooa.tools.shell_tools import Match, ShellResult, ShellTools, StreamDone, StreamEvent

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionDecision, PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tool_output import ToolOutputStore
from noah_code.workspace import Workspace, WorkspaceError


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
        self._output_store = ToolOutputStore(retention_hours=output_retention_hours)
        self._default_timeout = default_timeout
        self._on_shell_chunk: Any = None

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
        resolved = await self._authorize_path(path, PermissionCategory.READ)
        rel = self._workspace.relpath(resolved)
        result = await self._shell.read(rel, lines=lines)
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
        resolved = await self._authorize_path(path, PermissionCategory.READ)
        import shlex

        target = self._workspace.relpath(resolved) or "."
        cmd = " ".join(
            shlex.quote(a) for a in ["rg", "-n", "--no-heading", "-S", "--", pattern, target]
        )
        result = await self._shell.run(cmd, timeout=self._default_timeout)
        return self._cap_shell_result(result)

    async def list_files(
        self,
        pattern: Annotated[str, spec(description="Glob pattern")] = "**/*",
        path: Annotated[str, spec(description="Subdirectory")] = ".",
    ) -> list[str]:
        """List files under path matching a glob (deterministic, no shell)."""
        root = await self._authorize_path(path, PermissionCategory.READ)
        matches = sorted(
            str(p.relative_to(self._workspace.root)) for p in root.glob(pattern) if p.is_file()
        )
        if len(matches) > self._max_file_results:
            return matches[: self._max_file_results] + [
                f"...[{len(matches) - self._max_file_results} more]"
            ]
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
            list[str] | None,
            spec(description="Regex/fixed searches to run concurrently"),
        ] = None,
        files: Annotated[
            list[str] | None,
            spec(description="Workspace files to read concurrently"),
        ] = None,
        symbols: Annotated[
            bool,
            spec(description="Also return class/function/type declarations"),
        ] = False,
    ) -> str:
        """Batch focused repository searches and reads into one compact result."""

        queries = list(dict.fromkeys(searches or []))
        paths = list(dict.fromkeys(files or []))
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
                sections.append(
                    f"{result.path}:{result.start}-{result.end}\n{result.text}"
                )
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
        if isinstance(match, Match):
            resolved = await self._authorize_path(match.path, PermissionCategory.EDIT)
            mut = self._journal.record_preimage(resolved)
            try:
                result = await self._shell.replace(match, new_text)
            except Exception:
                self._journal.discard_mutation(mut)
                raise
            self._journal.record_postimage(mut, resolved)
            return result
        if isinstance(match, str):
            resolved = await self._authorize_path(match, PermissionCategory.EDIT)
            mut = self._journal.record_preimage(resolved)
            try:
                result = await self._shell.replace(match, new_text, new)
            except Exception:
                self._journal.discard_mutation(mut)
                raise
            self._journal.record_postimage(mut, resolved)
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
        resolved = await self._authorize_path(path, PermissionCategory.EDIT)
        mut = self._journal.record_preimage(resolved)
        try:
            result = await self._shell.write_file(path, content)
        except Exception:
            self._journal.discard_mutation(mut)
            raise
        self._journal.record_postimage(mut, resolved)
        return result

    async def write(
        self,
        path: Annotated[str, spec(description="File path relative to workspace")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> Any:
        """Create or overwrite a file; compatibility alias for write_file()."""

        return await self.write_file(path, content)

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
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
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
            self._journal.mark_shell_bypass()
        if self._on_shell_chunk is not None:
            self._on_shell_chunk("status", f"$ {command}\n")
        async for event in self._shell.run_stream(
            command, timeout=timeout or self._default_timeout
        ):
            if self._on_shell_chunk is not None and hasattr(event, "kind"):
                self._on_shell_chunk(getattr(event, "kind", "stdout"), getattr(event, "text", ""))
            yield event

    def _shell_decision(self, command: str) -> PermissionDecision:
        decision = self._engine.decide(PermissionCategory.BASH, command)
        if decision.denied or not self._engine.is_uncertain_shell(command):
            return decision
        # Shell syntax is too ambiguous to auto-approve safely. Interactive
        # sessions may still approve the exact command once.
        action = "deny" if self._engine.auto_approve else "ask"
        reason = (
            "compound/uncertain shell commands cannot be auto-approved"
            if action == "deny"
            else "compound/uncertain shell command requires approval"
        )
        return PermissionDecision(
            category=PermissionCategory.BASH,
            target=command,
            action=action,
            matching_rule=decision.matching_rule,
            reason=reason,
            remember_pattern=decision.remember_pattern,
        )

    @hidden
    async def run_trusted_readonly(self, command: str) -> ShellResult:
        """Run a host-constructed, strictly read-only command without a model approval."""
        if not self._engine.is_readonly_command(command):
            raise PermissionError(f"trusted command is not read-only: {command}")
        result = await self._shell.run(command, timeout=self._default_timeout)
        return self._cap_shell_result(result)

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

    async def _authorize_path(self, path: str, category: str) -> Path:
        try:
            resolved = self._workspace.resolve(path)
            rel = str(resolved.relative_to(self._workspace.root))
            decision = self._engine.decide(category, rel)
        except WorkspaceError as exc:
            abs_path = Path(path).expanduser()
            if not abs_path.is_absolute():
                abs_path = (self._workspace.root / path).resolve()
            else:
                abs_path = abs_path.resolve()
            decision = self._engine.decide(PermissionCategory.EXTERNAL_DIRECTORY, str(abs_path))
            await self._approvals.require(decision)
            raise WorkspaceError(
                f"path outside workspace requires dedicated external handling: {path}"
            ) from exc
        await self._approvals.require(decision)
        return resolved

    @hidden
    async def close(self) -> None:
        await self._shell.close()
