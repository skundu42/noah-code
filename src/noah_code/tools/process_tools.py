"""Bounded, session-owned background process tools."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any

from nooa import Skill, hidden, spec

from noah_code.tools.workspace_tools import WorkspaceTools


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
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    returncode: int | None = None
    state: str = "running"
    events: deque[ProcessEvent] = field(default_factory=deque)
    event_chars: int = 0
    next_sequence: int = 1
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)

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
        max_runtime_seconds: float = 3600.0,
        max_buffer_chars: int = 64_000,
        stop_grace_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self._ws = workspace_tools
        self._max_jobs = max_jobs
        self._max_runtime = max_runtime_seconds
        self._max_buffer = max_buffer_chars
        self._stop_grace = stop_grace_seconds
        self._jobs: dict[str, BackgroundJob] = {}
        self._on_lifecycle: Any = None

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
        running = sum(job.state == "running" for job in self._jobs.values())
        if running >= self._max_jobs:
            raise RuntimeError(f"background job limit reached ({self._max_jobs})")
        if len(self._jobs) >= self._max_jobs * 4:
            finished = [job_id for job_id, job in self._jobs.items() if job.state != "running"]
            for job_id in finished[: max(1, len(self._jobs) - self._max_jobs * 3)]:
                self._jobs.pop(job_id, None)
        decision = self._ws._shell_decision(command)
        await self._ws._approvals.require(decision)
        if not self._ws._engine.is_readonly_command(command):
            self._ws._journal.mark_shell_bypass()
        runtime = min(timeout or self._max_runtime, self._max_runtime)
        if runtime <= 0:
            raise ValueError("timeout must be positive")
        if os.name == "nt":
            argv: tuple[str, ...] = ("cmd.exe", "/d", "/s", "/c", command)
        else:
            argv = (os.environ.get("SHELL") or "/bin/sh", "-lc", command)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._ws._workspace.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        job_id = uuid.uuid4().hex[:8]
        job = BackgroundJob(
            id=job_id,
            name=(name.strip() or command.split()[0])[:60],
            command=command,
            process=process,
        )
        self._jobs[job_id] = job
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
        return f"job {job_id} started · {job.name} · cursor=0 · timeout={runtime:g}s"

    async def logs(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
        cursor: Annotated[
            int, spec(description="Last consumed sequence; 0 starts at retained head")
        ] = 0,
        max_chars: Annotated[int, spec(description="Maximum returned log characters")] = 12_000,
    ) -> str:
        """Return only retained log events newer than cursor, plus the next cursor."""
        job = self._job(job_id)
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
            return self._status_line(self._job(job_id))
        if not self._jobs:
            return "(no background jobs)"
        return "\n".join(self._status_line(job) for job in self._jobs.values())

    async def input(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
        text: Annotated[str, spec(description="Text to write to stdin")],
        newline: Annotated[bool, spec(description="Append a newline")] = True,
    ) -> str:
        """Write input to a running job."""
        job = self._job(job_id)
        if job.state != "running" or job.process.stdin is None:
            raise RuntimeError(f"job {job.id} is not accepting input ({job.state})")
        payload = text + ("\n" if newline else "")
        job.process.stdin.write(payload.encode())
        await job.process.stdin.drain()
        return f"sent {len(payload.encode())} bytes to job {job.id}"

    async def stop(
        self,
        job_id: Annotated[str, spec(description="Background job id")],
    ) -> str:
        """Stop one owned process group, escalating to kill after a grace period."""
        job = self._job(job_id)
        if job.state != "running":
            return self._status_line(job)
        job.state = "stopping"
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

    def has_running(self) -> bool:
        return any(job.state in {"running", "stopping"} for job in self._jobs.values())

    def set_lifecycle_handler(self, handler: Any) -> None:
        self._on_lifecycle = handler

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
        while job.events and job.event_chars > self._max_buffer:
            removed = job.events.popleft()
            job.event_chars -= len(removed.text)

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
        self._emit(job, f"{job.state} exit={returncode}")
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
                os.killpg(job.process.pid, signal.SIGTERM)
            else:
                job.process.terminate()
        try:
            await asyncio.wait_for(job.process.wait(), timeout=self._stop_grace)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                if os.name != "nt":
                    os.killpg(job.process.pid, signal.SIGKILL)
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

    def _emit(self, job: BackgroundJob, message: str) -> None:
        if self._on_lifecycle is not None:
            with contextlib.suppress(Exception):
                self._on_lifecycle(job.id, job.name, message)

    def _job(self, job_id: str) -> BackgroundJob:
        selected = job_id.strip()
        if selected not in self._jobs:
            raise KeyError(f"unknown background job: {selected}")
        return self._jobs[selected]

    @staticmethod
    def _status_line(job: BackgroundJob) -> str:
        exit_text = "" if job.returncode is None else f" exit={job.returncode}"
        return f"{job.id} [{job.state}] {job.name} · {job.elapsed:.1f}s{exit_text}"

    @hidden
    async def close(self) -> None:
        running = [job for job in self._jobs.values() if job.process.returncode is None]
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
