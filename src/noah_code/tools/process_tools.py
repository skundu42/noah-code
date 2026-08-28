"""Bounded, session-owned background process tools."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from nooa import Skill, hidden, spec

from noah_code.tools.workspace_tools import WorkspaceTools

if TYPE_CHECKING:
    from noah_code.runtime_state import RuntimeStateStore


_TERMINAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")
_POSIX_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


@dataclass(frozen=True)
class ProcessEvent:
    sequence: int
    stream: str
    text: str
    timestamp: float


@dataclass
class BackgroundJob:
    id: str
    name: str
    command: str
    process: asyncio.subprocess.Process
    log_path: Path | None = None
    kind: str = "process"
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    returncode: int | None = None
    state: str = "running"
    events: deque[ProcessEvent] = field(default_factory=deque)
    event_chars: int = 0
    next_sequence: int = 1
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    command_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def elapsed(self) -> float:
        return max(0.0, (self.finished_at or time.monotonic()) - self.started_at)


class ProcessTools(Skill):
    """Start and control long-running commands without blocking an agent turn."""

    def __init__(
        self,
        workspace_tools: WorkspaceTools,
        *,
        max_jobs: int = 8,
        max_runtime_seconds: float = 86_400.0,
        max_buffer_chars: int = 64_000,
        max_log_bytes: int = 4_000_000,
        stop_grace_seconds: float = 2.0,
        runtime: RuntimeStateStore | None = None,
    ) -> None:
        super().__init__()
        self._ws = workspace_tools
        self._max_jobs = max_jobs
        self._max_runtime = max_runtime_seconds
        self._max_buffer = max_buffer_chars
        self._max_log_bytes = max_log_bytes
        self._stop_grace = stop_grace_seconds
        self._jobs: dict[str, BackgroundJob] = {}
        self._on_lifecycle: Any = None
        self._runtime = runtime

    async def start(
        self,
        command: Annotated[str, spec(description="Command to run in the workspace")],
        name: Annotated[str, spec(description="Short optional job label")] = "",
        timeout: Annotated[
            float | None,
            spec(description="Maximum runtime seconds; bounded by session configuration"),
        ] = None,
    ) -> str:
        """Start a bounded background job and return its id and first log cursor."""
        command = command.strip()
        if not command:
            raise ValueError("command is required")
        decision = self._ws._shell_decision(command, tool="processes_start")
        await self._ws._approvals.require(decision)
        if not self._ws._engine.is_readonly_command(command):
            await self._ws.checkpoint_before_shell(command)
            self._ws._journal.mark_shell_bypass()
        runtime = min(timeout or self._max_runtime, self._max_runtime)
        if runtime <= 0:
            raise ValueError("timeout must be positive")
        if os.name == "nt":
            argv: tuple[str, ...] = ("cmd.exe", "/d", "/s", "/c", command)
        else:
            argv = (os.environ.get("SHELL") or "/bin/sh", "-lc", command)
        job = await self._launch(
            argv,
            command=command,
            name=(name.strip() or command.split()[0])[:60],
            runtime=runtime,
        )
        return f"job {job.id} started · {job.name} · cursor=0 · timeout={runtime:g}s"

    async def open_terminal(
        self,
        name: Annotated[str, spec(description="Unique terminal session name")] = "",
        shell: Annotated[str, spec(description="Optional shell executable path or name")] = "",
        timeout: Annotated[
            float | None,
            spec(description="Maximum terminal lifetime; bounded by session configuration"),
        ] = None,
    ) -> str:
        """Open a named persistent terminal; commands remain permission-gated."""

        label = name.strip() or f"terminal-{self._terminal_count() + 1}"
        if _TERMINAL_NAME.fullmatch(label) is None:
            raise ValueError(
                "terminal name must be 1-40 letters, numbers, dots, underscores, or hyphens"
            )
        if any(
            job.kind == "terminal"
            and job.name == label
            and job.state in {"running", "stopping"}
            for job in self._jobs.values()
        ):
            raise ValueError(f"terminal name already exists: {label}")
        requested = shell.strip() or ("cmd.exe" if os.name == "nt" else os.environ.get("SHELL", ""))
        requested = requested or ("cmd.exe" if os.name == "nt" else "/bin/sh")
        executable = shutil.which(requested)
        if executable is None:
            raise ValueError(f"shell executable not found: {requested}")
        shell_name = Path(executable).name.lower()
        if os.name != "nt" and shell_name not in _POSIX_SHELLS:
            supported = ", ".join(sorted(_POSIX_SHELLS))
            raise ValueError(f"unsupported terminal shell {shell_name!r}; use one of: {supported}")
        if os.name == "nt" and shell_name not in {"cmd", "cmd.exe"}:
            raise ValueError("managed terminals currently support cmd.exe on Windows")
        await self._ws._approvals.require(
            self._ws._engine.decide("task", f"terminal:{label}", tool="terminal_open")
        )
        runtime = min(timeout or self._max_runtime, self._max_runtime)
        if runtime <= 0:
            raise ValueError("timeout must be positive")
        argv = (executable,) if os.name == "nt" else (executable, "-l")
        job = await self._launch(
            argv,
            command=f"[terminal] {executable}",
            name=label[:60],
            runtime=runtime,
            kind="terminal",
        )
        return (
            f"terminal {job.name} opened · id={job.id} · shell={executable} · "
            f"cursor=0 · timeout={runtime:g}s"
        )

    async def terminal_run(
        self,
        name: Annotated[str, spec(description="Terminal session name or id")],
        command: Annotated[str, spec(description="Command to run in the persistent terminal")],
        timeout: Annotated[float, spec(description="Maximum command wait in seconds")] = 60.0,
    ) -> str:
        """Run one permission-checked command and return output plus exit status."""

        job = self._terminal_job(name)
        command = command.strip()
        if not command:
            raise ValueError("command is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        wait_timeout = min(timeout, self._max_runtime)
        decision = self._ws._shell_decision(command, tool="terminal_run")
        await self._ws._approvals.require(decision)
        if not self._ws._engine.is_readonly_command(command):
            await self._ws.checkpoint_before_shell(command)
            self._ws._journal.mark_shell_bypass()
        async with job.command_lock:
            cursor = job.next_sequence - 1
            marker = f"__NOAH_TERMINAL_{uuid.uuid4().hex}__"
            if os.name == "nt":
                payload = f"{command}\r\necho {marker}:%errorlevel%\r\n"
            else:
                payload = f"{command}\nprintf '\\n{marker}:%s\\n' \"$?\"\n"
            await self._write_input(job, payload)
            output, returncode = await self._wait_for_terminal_marker(
                job,
                cursor=cursor,
                marker=marker,
                timeout=wait_timeout,
            )
        bounded = output.strip()
        if len(bounded) > 32_000:
            bounded = bounded[:16_000] + "\n… terminal output bounded …\n" + bounded[-16_000:]
        return f"{bounded or '(no output)'}\n\nterminal {job.name} · exit={returncode}"

    async def close_terminal(
        self,
        name: Annotated[str, spec(description="Terminal session name or id")],
    ) -> str:
        """Close a named persistent terminal session."""

        job = self._terminal_job(name)
        result = await self.stop(job.id)
        return f"terminal {job.name} closed · {result}"

    async def terminal_status(
        self,
        name: Annotated[str, spec(description="Optional terminal name or id")] = "",
    ) -> str:
        """List persistent terminals or show one terminal's state."""

        if name.strip():
            return self._status_line(self._terminal_job(name))
        terminals = [job for job in self._jobs.values() if job.kind == "terminal"]
        if not terminals:
            return "(no terminal sessions; use open_terminal to create one)"
        return "\n".join(self._status_line(job) for job in terminals)

    async def _launch(
        self,
        argv: tuple[str, ...],
        *,
        command: str,
        name: str,
        runtime: float,
        kind: str = "process",
    ) -> BackgroundJob:
        self._ensure_capacity()
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._ws._workspace.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=(
                asyncio.subprocess.STDOUT if kind == "terminal" else asyncio.subprocess.PIPE
            ),
            start_new_session=os.name != "nt",
        )
        job_id = uuid.uuid4().hex[:8]
        log_path = self._runtime.process_log_dir / f"{job_id}.jsonl" if self._runtime else None
        job = BackgroundJob(
            id=job_id,
            name=name,
            command=command,
            process=process,
            log_path=log_path,
            kind=kind,
        )
        try:
            if log_path is not None:
                log_path.touch(mode=0o600, exist_ok=False)
            self._jobs[job_id] = job
            if self._runtime is not None:
                self._runtime.register_job(
                    job_id=job_id,
                    name=job.name,
                    command=command,
                    pid=process.pid,
                    timeout_seconds=runtime,
                    log_path=log_path or self._runtime.process_log_dir / f"{job_id}.jsonl",
                )
        except Exception:
            with contextlib.suppress(Exception):
                await self._terminate(job)
            self._close_transport(process)
            self._jobs.pop(job_id, None)
            if log_path is not None:
                with contextlib.suppress(OSError):
                    log_path.unlink(missing_ok=True)
            raise
        readers = (
            asyncio.create_task(
                self._read(job, "stdout", process.stdout), name=f"noah-job-{job_id}-out"
            ),
            asyncio.create_task(
                self._read(job, "stderr", process.stderr), name=f"noah-job-{job_id}-err"
            ),
        )
        job.tasks = [
            *readers,
            asyncio.create_task(self._wait(job, readers), name=f"noah-job-{job_id}-wait"),
            asyncio.create_task(self._expire(job, runtime), name=f"noah-job-{job_id}-timeout"),
        ]
        self._emit(job, f"started pid={process.pid}")
        return job

    async def logs(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
        cursor: Annotated[
            int, spec(description="Last consumed sequence; 0 starts at retained head")
        ] = 0,
        max_chars: Annotated[int, spec(description="Maximum returned log characters")] = 12_000,
    ) -> str:
        """Return only retained log events newer than cursor, plus the next cursor."""
        selected = job_id.strip()
        if selected not in self._jobs and self._runtime is not None:
            return self._durable_logs(selected, cursor=cursor, max_chars=max_chars)
        job = self._job(selected)
        limit = min(max(max_chars, 1000), 32_000)
        events = [event for event in job.events if event.sequence > cursor]
        retained_from = job.events[0].sequence if job.events else job.next_sequence
        rows: list[str] = []
        used = 0
        next_cursor = cursor
        for event in events:
            prefix = "! " if event.stream == "stderr" else ""
            value = prefix + event.text
            if used + len(value) > limit:
                break
            rows.append(value.rstrip("\n"))
            used += len(value)
            next_cursor = event.sequence
        if cursor and cursor < retained_from - 1:
            rows.insert(0, f"… earlier output expired before cursor {retained_from} …")
        state = self._status_line(job)
        body = "\n".join(rows) or "(no new output)"
        return f"{body}\n\n{state} · next_cursor={max(next_cursor, cursor)}"

    async def status(
        self,
        job_id: Annotated[str, spec(description="Optional job id; empty lists all jobs")] = "",
    ) -> str:
        """Show one job or a compact list of all session jobs."""
        if job_id.strip():
            selected = job_id.strip()
            if selected in self._jobs:
                return self._status_line(self._job(selected))
            record = self._runtime.job(selected) if self._runtime is not None else None
            if record is None:
                raise KeyError(f"unknown background job: {selected}")
            return self._durable_status_line(record)
        if not self._jobs and self._runtime is None:
            return "(no background jobs)"
        rows = [self._status_line(job) for job in self._jobs.values()]
        if self._runtime is not None:
            active_ids = set(self._jobs)
            rows.extend(
                self._durable_status_line(record)
                for record in self._runtime.jobs()
                if str(record["job_id"]) not in active_ids
            )
        return "\n".join(rows) or "(no background jobs)"

    async def input(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
        text: Annotated[str, spec(description="Text to write to stdin")],
        newline: Annotated[bool, spec(description="Append a newline")] = True,
    ) -> str:
        """Write input to a running job."""
        job = self._job(job_id)
        if job.kind == "terminal":
            raise RuntimeError(
                f"job {job.id} is a managed terminal; use terminal_run so commands are approved"
            )
        payload = text + ("\n" if newline else "")
        await self._write_input(job, payload)
        return f"sent {len(payload.encode())} bytes to job {job.id}"

    async def _write_input(self, job: BackgroundJob, payload: str) -> None:
        if job.state != "running" or job.process.stdin is None:
            raise RuntimeError(f"job {job.id} is not accepting input ({job.state})")
        try:
            job.process.stdin.write(payload.encode())
            await job.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            # The process exited after the running-state check above; report
            # the same condition the state guard produces instead of leaking
            # a transport error.
            raise RuntimeError(f"job {job.id} is not accepting input ({job.state})") from exc

    async def stop(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
    ) -> str:
        """Stop one owned process group, escalating to kill after a grace period."""
        job = self._job(job_id)
        if job.state != "running":
            return self._status_line(job)
        job.state = "stopping"
        if self._runtime is not None:
            self._runtime.update_job(job.id, "stopping")
        await self._terminate(job)
        await asyncio.gather(
            *(task for task in job.tasks if task.get_name().endswith("-wait")),
            return_exceptions=True,
        )
        return self._status_line(job)

    def summary(self) -> str:
        """Bounded live context for the agent; never includes process output."""
        if not self._jobs:
            return "(no background jobs)"
        running = [job for job in self._jobs.values() if job.state in {"running", "stopping"}]
        recent = list(self._jobs.values())[-5:]
        lines = [f"background_jobs running={len(running)} total={len(self._jobs)}"]
        lines.extend(self._status_line(job) for job in recent)
        return "\n".join(lines)

    def snapshot(self, *, limit: int = 20) -> list[dict[str, Any]]:
        jobs = list(self._jobs.values())[-limit:] if limit > 0 else []
        return [
            {
                "id": job.id,
                "name": job.name,
                "kind": job.kind,
                "state": job.state,
                "command": job.command,
                "elapsed": job.elapsed,
                "returncode": job.returncode,
                "cursor": job.next_sequence - 1,
            }
            for job in jobs
        ]

    def has_running(self) -> bool:
        return any(job.state in {"running", "stopping"} for job in self._jobs.values())

    def set_lifecycle_handler(self, handler: Any) -> None:
        self._on_lifecycle = handler

    def _ensure_capacity(self) -> None:
        active = sum(job.state in {"running", "stopping"} for job in self._jobs.values())
        if active >= self._max_jobs:
            raise RuntimeError(f"background job limit reached ({self._max_jobs})")
        if len(self._jobs) < self._max_jobs * 4:
            return
        finished = [
            job_id
            for job_id, job in self._jobs.items()
            if job.state not in {"running", "stopping"}
        ]
        for job_id in finished[: max(1, len(self._jobs) - self._max_jobs * 3)]:
            self._jobs.pop(job_id, None)

    def _terminal_count(self) -> int:
        return sum(
            job.kind == "terminal" and job.state in {"running", "stopping"}
            for job in self._jobs.values()
        )

    def _terminal_job(self, name: str) -> BackgroundJob:
        selected = name.strip()
        matches = [
            job
            for job in self._jobs.values()
            if job.kind == "terminal" and selected in {job.id, job.name}
        ]
        if not matches:
            raise KeyError(f"unknown terminal session: {selected}")
        return next(
            (
                job
                for job in reversed(matches)
                if job.state in {"running", "stopping"}
            ),
            matches[-1],
        )

    async def _wait_for_terminal_marker(
        self,
        job: BackgroundJob,
        *,
        cursor: int,
        marker: str,
        timeout: float,
    ) -> tuple[str, int]:
        deadline = asyncio.get_running_loop().time() + timeout
        collected = ""
        next_cursor = cursor
        while True:
            events = [event for event in job.events if event.sequence > next_cursor]
            if events:
                collected += "".join(event.text for event in events)
                next_cursor = events[-1].sequence
                marker_at = collected.find(marker + ":")
                if marker_at >= 0:
                    suffix = collected[marker_at + len(marker) + 1 :]
                    code_text = suffix.splitlines()[0].strip() if suffix else ""
                    try:
                        return collected[:marker_at], int(code_text)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"terminal {job.name} returned an invalid exit marker"
                        ) from exc
            if job.state != "running":
                raise RuntimeError(f"terminal {job.name} exited before the command completed")
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"terminal command timed out after {timeout:g}s; session {job.name} is still open"
                )
            job.output_event.clear()
            try:
                await asyncio.wait_for(job.output_event.wait(), timeout=remaining)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"terminal command timed out after {timeout:g}s; session {job.name} is still open"
                ) from exc

    async def _read(
        self,
        job: BackgroundJob,
        stream_name: str,
        stream: asyncio.StreamReader | None,
    ) -> None:
        if stream is None:
            return
        # Incremental decoding keeps multibyte UTF-8 sequences that straddle
        # read-chunk boundaries from turning into U+FFFD garbage.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                tail = decoder.decode(b"", final=True)
                if tail:
                    self._append(job, stream_name, tail)
                return
            text = decoder.decode(chunk)
            if text:
                self._append(job, stream_name, text)

    def _append(self, job: BackgroundJob, stream: str, text: str) -> None:
        event = ProcessEvent(job.next_sequence, stream, text, time.monotonic())
        job.next_sequence += 1
        job.events.append(event)
        job.event_chars += len(text)
        job.output_event.set()
        while job.events and job.event_chars > self._max_buffer:
            removed = job.events.popleft()
            job.event_chars -= len(removed.text)
        if job.log_path is not None:
            payload = json.dumps(
                {
                    "sequence": event.sequence,
                    "stream": stream,
                    "text": text,
                    "timestamp": time.time(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with job.log_path.open("a", encoding="utf-8") as output:
                output.write(payload + "\n")
                output.flush()
            self._cap_durable_log(job)

    def _cap_durable_log(self, job: BackgroundJob) -> None:
        """Rotate an oversized JSONL log, keeping whole recent lines.

        The durable log is written for crash recovery, so it must stay
        bounded even when a job streams output forever. Rotation truncates
        the head down to roughly half the cap and leaves a marker line
        noting the truncation.
        """
        path = job.log_path
        if path is None:
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size <= self._max_log_bytes:
            return
        keep = max(self._max_log_bytes // 2, 1)
        # Rotation runs right after the append that crossed the cap, so the
        # file holds at most cap + one event: reading it here stays bounded.
        try:
            lines = path.read_bytes().splitlines(keepends=True)
        except OSError:
            return
        retained: list[bytes] = []
        retained_bytes = 0
        for line in reversed(lines):
            # A pipe read may deliver only the final bytes of an output line
            # as a tiny last event. When that newest record is below the
            # retention target, keep its predecessor too so rotation does not
            # preserve only a dangling tail while dropping the line prefix.
            # After that, retain older whole records only while they fit.
            minimum_records = 2 if retained_bytes < keep else 1
            if len(retained) >= minimum_records and retained_bytes + len(line) > keep:
                break
            retained.append(line)
            retained_bytes += len(line)
        retained.reverse()
        marker = json.dumps(
            {
                "sequence": 0,
                "stream": "stderr",
                "text": f"… earlier durable log truncated; kept last {retained_bytes} bytes …",
                "timestamp": time.time(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(marker.encode("utf-8") + b"\n")
                stream.writelines(retained)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    async def _wait(
        self,
        job: BackgroundJob,
        readers: tuple[asyncio.Task[Any], asyncio.Task[Any]],
    ) -> None:
        returncode = await job.process.wait()
        await self._close_stdin(job)
        await asyncio.gather(*readers, return_exceptions=True)
        if job.state == "running":
            job.state = "completed" if returncode == 0 else "failed"
        elif job.state == "stopping":
            job.state = "stopped"
        job.returncode = returncode
        job.finished_at = time.monotonic()
        if self._runtime is not None:
            try:
                self._runtime.update_job(job.id, job.state, returncode=returncode)
            except Exception as exc:
                # Durable bookkeeping must never suppress the terminal
                # lifecycle event; the host is waiting on it to wake up.
                with contextlib.suppress(Exception):
                    self._append(job, "stderr", f"[noah] runtime state update failed: {exc!r}")
        self._emit(job, f"{job.state} exit={returncode}", terminal=True)
        self._close_transport(job.process)
        await asyncio.sleep(0)

    async def _expire(self, job: BackgroundJob, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            if job.state == "running":
                job.state = "timed_out"
                await self._terminate(job)
        except asyncio.CancelledError:
            pass

    async def _terminate(self, job: BackgroundJob) -> None:
        await self._close_stdin(job)
        if job.process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            if os.name != "nt":
                try:
                    os.killpg(job.process.pid, signal.SIGTERM)
                except PermissionError:
                    # Some macOS runners deny process-group signals even for
                    # an owned child. Terminate the direct child instead.
                    job.process.terminate()
            else:
                job.process.terminate()
        try:
            await asyncio.wait_for(job.process.wait(), timeout=self._stop_grace)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                if os.name != "nt":
                    try:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    except PermissionError:
                        job.process.kill()
                else:
                    job.process.kill()
            await job.process.wait()

    @staticmethod
    async def _close_stdin(job: BackgroundJob) -> None:
        writer = job.process.stdin
        if writer is None or writer.is_closing():
            return
        writer.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await writer.wait_closed()

    @staticmethod
    def _close_transport(process: asyncio.subprocess.Process) -> None:
        # asyncio exposes no public Process.close(). Closing the private transport
        # prevents its destructor from touching an event loop that pytest or the
        # application has already shut down.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()

    def _emit(self, job: BackgroundJob, message: str, *, terminal: bool = False) -> None:
        if self._on_lifecycle is not None:
            try:
                self._on_lifecycle(job.id, job.name, message, terminal)
            except TypeError:
                # Compatibility for third-party UI handlers written against
                # the original three-argument callback.
                with contextlib.suppress(Exception):
                    self._on_lifecycle(job.id, job.name, message)
            except Exception:
                pass

    def _job(self, job_id: str) -> BackgroundJob:
        selected = job_id.strip()
        if selected not in self._jobs:
            raise KeyError(f"unknown background job: {selected}")
        return self._jobs[selected]

    @staticmethod
    def _status_line(job: BackgroundJob) -> str:
        exit_text = "" if job.returncode is None else f" exit={job.returncode}"
        kind = "terminal · " if job.kind == "terminal" else ""
        return f"{job.id} [{job.state}] {kind}{job.name} · {job.elapsed:.1f}s{exit_text}"

    @staticmethod
    def _durable_status_line(record: dict[str, Any]) -> str:
        finished = float(record.get("finished_at") or time.time())
        elapsed = max(finished - float(record.get("started_at") or finished), 0.0)
        returncode = record.get("returncode")
        exit_text = "" if returncode is None else f" exit={returncode}"
        return (
            f"{record['job_id']} [{record['state']}] {record['name']} · "
            f"{elapsed:.1f}s{exit_text}"
        )

    def _durable_logs(self, job_id: str, *, cursor: int, max_chars: int) -> str:
        assert self._runtime is not None
        record = self._runtime.job(job_id)
        if record is None:
            raise KeyError(f"unknown background job: {job_id}")
        path = Path(str(record["log_path"]))
        rows: list[str] = []
        used = 0
        next_cursor = cursor
        limit = min(max(max_chars, 1000), 32_000)
        if path.is_file():
            for raw in self._tail_lines(path, limit):
                try:
                    event = json.loads(raw)
                    sequence = int(event.get("sequence", 0))
                    if sequence <= cursor:
                        continue
                    text = str(event.get("text", ""))
                    if event.get("stream") == "stderr":
                        text = "! " + text
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if used + len(text) > limit:
                    break
                rows.append(text.rstrip("\n"))
                used += len(text)
                next_cursor = sequence
        body = "\n".join(rows) or "(no new output)"
        return (
            f"{body}\n\n{self._durable_status_line(record)} · "
            f"next_cursor={max(next_cursor, cursor)}"
        )

    @staticmethod
    def _tail_lines(path: Path, limit: int) -> list[str]:
        """Read only the newest log lines; never load the whole file.

        The window comfortably covers one page of events. Durable logs are
        capped by rotation, and cursors older than the retained window simply
        see the newest retained lines (rotation leaves a marker line).
        """
        size = path.stat().st_size
        window = min(size, max(262_144, limit * 16))
        offset = size - window
        with path.open("rb") as stream:
            if offset:
                stream.seek(offset - 1)
                starts_mid_line = stream.read(1) != b"\n"
            else:
                starts_mid_line = False
            data = stream.read(window)
        lines = data.decode("utf-8", errors="replace").splitlines()
        if starts_mid_line and lines:
            lines = lines[1:]  # drop the partially read first line
        return lines

    @hidden
    async def close(self) -> None:
        running = [job for job in self._jobs.values() if job.process.returncode is None]
        for job in running:
            if job.state == "running":
                job.state = "stopping"
                if self._runtime is not None:
                    self._runtime.update_job(job.id, "stopping")
        await asyncio.gather(*(self._terminate(job) for job in running), return_exceptions=True)
        current = asyncio.current_task()
        for job in self._jobs.values():
            for task in job.tasks:
                if task is not current and not task.done() and task.get_name().endswith("-timeout"):
                    task.cancel()
        await asyncio.gather(
            *(task for job in self._jobs.values() for task in job.tasks if task is not current),
            return_exceptions=True,
        )
