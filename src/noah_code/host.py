"""Pure-Python host: dispatcher loop, persistence, approvals, slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noah_code import nooa_compat
from noah_code.approvals import ApprovalChoice
from noah_code.commands import config_text, help_text, parse_slash
from noah_code.config import (
    REASONING_EFFORTS,
    NoahCodeConfig,
    save_user_default_model,
    save_user_reasoning_effort,
    save_user_theme,
    user_default_model,
)
from noah_code.custom_commands import CustomCommand, discover_custom_commands
from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEvent, HostEventKind
from noah_code.sessions import SessionEventRecord, SessionMeta, SessionStore
from noah_code.steer import SAFE_SLASH_WHILE_BUSY, SteerQueue, expansion_failed
from noah_code.themes import THEME_NAMES, get_theme
from noah_code.ui.console import ConsoleUI
from noah_code.ui.protocol import HostUI
from noah_code.usage import UsageSnapshot, UsageTracker
from noah_code.workspace import Workspace

if TYPE_CHECKING:
    from noah_code.agent import CodingAgent

logger = logging.getLogger(__name__)


def _friendly_agent_error(exc: Exception) -> str:
    """Turn framework/provider failures into bounded, actionable UI copy."""

    raw = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(exc)).strip()
    iteration = re.search(r"Generation failed after (\d+) iterations \(max_iterations=(\d+)\)", raw)
    if iteration:
        used, limit = iteration.groups()
        return (
            f"Reached the iteration limit ({used}/{limit} turns). "
            "Continue with a narrower follow-up."
        )
    retries = re.search(r"Generation failed after (\d+) errors", raw)
    if retries:
        return (
            f"The model produced invalid tool code {retries.group(1)} times. "
            "Try again; if it repeats, switch models with /model."
        )
    compact = " ".join(raw.split())
    if len(compact) > 700:
        compact = compact[:697].rstrip() + "…"
    return compact or type(exc).__name__


_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
    "requested tokens exceed",
    "input is too long",
)


def _is_context_overflow(exc: BaseException) -> bool:
    """Return True when a provider error looks like a context-window overflow."""

    text = str(exc).lower()
    return any(marker in text for marker in _OVERFLOW_MARKERS)


async def _handle_with_overflow_recovery(
    agent: Any,
    notification: dict[str, list],
    *,
    render: Any = None,
) -> Any:
    """Run handle(); on context overflow, compact once and retry the same step."""

    try:
        return await agent.handle(notification)
    except Exception as exc:
        if not _is_context_overflow(exc):
            raise
        compacted = await agent.compact_history()
        if not compacted:
            raise
        if render is not None:
            render(
                HostEvent(
                    HostEventKind.STATUS,
                    "context overflow · compacted and retrying once",
                )
            )
        return await agent.handle(notification)


def _stop_text(kind: Any, explanation: str) -> str:
    """Render the agent protocol as human language instead of enum syntax."""

    value = str(getattr(kind, "value", kind) or "").upper()
    label = {
        "DONE": "Completed",
        "NEED_INPUT": "Waiting for input",
        "GET_USER_INPUT": "Waiting for input",
        "WAIT": "Waiting for background work",
    }.get(value, value.replace("_", " ").title() or "Stopped")
    detail = " ".join((explanation or "").split())
    return f"{label} · {detail}" if detail else label


def _command_output(text: str) -> HostEvent:
    """Return exact, preformatted command output for every UI frontend."""

    return HostEvent(
        HostEventKind.MESSAGE,
        text,
        meta={"format": "plain", "source": "command"},
    )


def _format_skills_output(text: str) -> str:
    """Convert NOOA's wide fixed-column status into a narrow, readable list."""

    rendered: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("Active Skills"):
            rendered.extend(
                [
                    "Active skills",
                    "Use with self.<name>; deactivate with self.skills.deactivate(['name']).",
                ]
            )
            continue
        if line.startswith("Available Skills"):
            if rendered and rendered[-1]:
                rendered.append("")
            rendered.extend(
                [
                    "Available skills",
                    "Activate with self.skills.activate(['name']).",
                ]
            )
            continue
        match = re.fullmatch(r"\s{2}(\S+)(?:\s{2,}(.*))?", line)
        if match:
            name, description = match.groups()
            rendered.append(f"\n  {name}")
            if description:
                rendered.append(f"    {description.strip()}")
            continue
        rendered.append(line)
    return "\n".join(rendered).strip()


def _deterministic_title(text: str) -> str:
    """Create a useful title without spending an additional model call."""

    cleaned = re.sub(r"[`*_#>\[\]()]", " ", text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", cleaned)
    title = " ".join(words[:5]).strip(" ._-/")
    return title[:60] or "Coding task"


def _load_agent_runtime() -> tuple[type[CodingAgent], Any]:
    """Load the heavy NOOA/LiteLLM runtime away from the TUI event loop."""

    from noah_code.agent import CodingAgent
    from noah_code.llm import get_llm_client

    return CodingAgent, get_llm_client


def _json_safe(value: Any) -> Any:
    """Best-effort conversion for session meta JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "items") and not isinstance(value, type):
        try:
            return {str(k): _json_safe(v) for k, v in value.items()}
        except Exception:  # noqa: BLE001
            pass
    return str(value)


@dataclass
class HostResult:
    exit_code: int
    explanation: str = ""
    session_id: str | None = None


class AgentHost:
    """Owns session lifecycle and the InteractiveAgent dispatcher loop."""

    def __init__(
        self,
        workspace: Workspace,
        config: NoahCodeConfig,
        *,
        llm: Any = None,
        ui: HostUI | None = None,
        session_meta: SessionMeta | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.ui: HostUI = ui or ConsoleUI(markdown=config.ui.markdown)
        self.store = store or SessionStore(config.session_dir)
        self._llm = llm
        self._agent: CodingAgent | None = None
        self._storage: Any = None
        self.meta = session_meta
        self._exit_requested = False
        self._title_task: asyncio.Task[Any] | None = None
        self._memory_task: asyncio.Task[Any] | None = None
        self._budget_guard: Any = None
        self._llm_cache: Any = None
        self._hooks: Any = None
        self._checkpoints: Any = None
        self.last_checkpoint: dict[str, Any] | None = None
        self._post_hook_tasks: list[asyncio.Task[Any]] = []
        self._trace_info = "auto (viewer if reachable)"
        self._active_turn: asyncio.Task[Any] | None = None
        self._event_unsubs: list[Any] = []
        self._custom_commands: dict[str, CustomCommand] = {}
        self._last_turn_shell_bypass = False
        self._mcp_attached: set[str] = set()
        self._mcp_errors: dict[str, str] = {}
        self._usage = UsageTracker()
        self.steer_queue = SteerQueue()
        self._pending_attach_paths: list[Path] = []
        self._pending_worktree_name = ""
        self.on_session_changed: Any = None  # optional UI callback

    @property
    def agent(self) -> CodingAgent:
        if self._agent is None:
            raise RuntimeError("host not started")
        return self._agent

    def _setup_tracing(self, session_id: str) -> None:
        if not self.config.tracing.enabled:
            return
        from nooa.tracing import enable_tracing, exporters, set_session

        set_session(session_id)
        exps = []
        if self.config.tracing.jsonl_dir:
            path = Path(self.config.tracing.jsonl_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            exps.append(exporters.jsonl(trace_dir=str(path)))
            self._trace_info = str(path / f"{session_id}.jsonl")
        if exps:
            enable_tracing(exporters=exps)

    async def start(self) -> SessionMeta:
        from noah_code.worktree import infer_worktree_name, repo_id_for, worktree_storage_root

        if self.meta is None:
            self.meta = self.store.create(
                self.workspace,
                model=self.config.model,
                mode=self.config.mode,
                reasoning_effort=self.config.reasoning_effort,
                repo_id=repo_id_for(self.workspace.root),
                worktree_name=self._pending_worktree_name
                or infer_worktree_name(
                    self.workspace.root, worktree_storage_root(self.config.session_dir)
                ),
            )
            self._pending_worktree_name = ""
        else:
            self.store.verify_workspace(self.meta, self.workspace)
            inferred = infer_worktree_name(
                self.workspace.root, worktree_storage_root(self.config.session_dir)
            )
            repo = repo_id_for(self.workspace.root)
            dirty = False
            if repo and not self.meta.repo_id:
                self.meta.repo_id = repo
                dirty = True
            if inferred and not self.meta.worktree_name:
                self.meta.worktree_name = inferred
                dirty = True
            if dirty:
                self.store.save_meta(self.meta)

        # Python's first NOOA import initializes LiteLLM's provider registry and
        # is by far the largest cold-start cost. Do it in a worker thread so the
        # Textual shell can paint and remain responsive meanwhile.
        agent_class, get_llm_client = await asyncio.to_thread(_load_agent_runtime)

        client_kwargs: dict[str, Any] = {}
        if self._llm is None:
            from noah_code.llm import reasoning_overrides, sampling_overrides

            client_kwargs.update(sampling_overrides(self.config.sampling))
        llm = self._llm
        if llm is None:
            from noah_code.llm import reasoning_overrides

            llm = await asyncio.to_thread(
                get_llm_client,
                self.meta.model,
                **reasoning_overrides(self.meta.reasoning_effort),
                **client_kwargs,
            )
        lightweight_llm = llm
        if self.config.lightweight_model and self.config.lightweight_model != self.meta.model:
            lightweight_llm = await asyncio.to_thread(
                get_llm_client,
                self.config.lightweight_model,
                **client_kwargs,
            )

        from noah_code.budget import SharedBudgetLLM, _PrefixObserverOnly, wrap_with_budget
        from noah_code.llm_cache import resolve_cache_settings, wrap_with_cache
        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)
        lightweight_llm = wrap_conversational_replies(lightweight_llm)
        llm, self._budget_guard = wrap_with_budget(
            llm, self.config.budget, prefix_observer=self._usage
        )
        if isinstance(llm, SharedBudgetLLM):
            # Both routes draw from one guard so caps span the whole session.
            lightweight_llm = SharedBudgetLLM(
                lightweight_llm, self._budget_guard, prefix_observer=self._usage
            )
        elif isinstance(llm, _PrefixObserverOnly):
            lightweight_llm = _PrefixObserverOnly(lightweight_llm, self._usage)
        cache_mode, cache_dir = resolve_cache_settings()
        llm = wrap_with_cache(llm, cache_dir, cache_mode)
        lightweight_llm = wrap_with_cache(lightweight_llm, cache_dir, cache_mode)
        self._llm_cache = llm if hasattr(llm, "stats") else None

        self._setup_tracing(self.meta.session_id)
        self._storage = self.store.open_storage(self.meta.session_id)

        agent = agent_class(
            self.workspace,
            self.config,
            llm=llm,
            lightweight_llm=lightweight_llm,
            storage=self._storage,
        )
        # Restore snapshot if present.
        restored = self._storage.restore_latest_snapshot(agent)
        if restored:
            # Re-bind host-owned nosnapshot infrastructure after restore.
            agent._engine.mode = agent.mode
            agent._engine.load_session_rules(self.meta.permission_rules)
            journal_data = await asyncio.to_thread(self.store.load_journal, self.meta.session_id)
            agent.journal.load_dict(journal_data or self.meta.journal)
            if self.meta.todos:
                agent.todos.from_dict(self.meta.todos)
        else:
            agent.set_mode(self.meta.mode)
            agent.v.mode = self.meta.mode
            agent.v.model = self.meta.model

        agent._approvals.set_handler(self.ui.ask_approval)
        agent.ask.set_handler(self.ui.ask_questions)
        agent._render_message = self._on_agent_message
        agent.ws.set_shell_chunk_handler(self._on_shell_chunk)
        agent.processes.set_lifecycle_handler(self._on_process_lifecycle)

        from noah_code.checkpoints import CheckpointManager
        from noah_code.hooks import HookRunner

        self._hooks = HookRunner(self.config.hooks, cwd=self.workspace.root)
        if self._hooks.active:
            runner = self._hooks

            async def _pre_tool_guard(decision: Any) -> None:
                tool = str(getattr(decision, "tool", "") or "").strip() or str(decision.category)
                outcome = await runner.run_pre(
                    tool=tool[:80],
                    category=str(decision.category),
                    target=str(decision.target),
                )
                if not outcome.allowed:
                    raise PermissionError(outcome.reason)

            agent._approvals.set_guard(_pre_tool_guard)

        session_id = self.meta.session_id
        self._checkpoints = CheckpointManager(
            self.workspace.root,
            session_id,
            max_per_session=self.config.checkpoints.max_per_session,
        )
        self._teardown_event_bridge()
        self._event_unsubs = install_event_bridge(agent, self._emit_with_hooks, self._usage)

        self._custom_commands = discover_custom_commands(self.workspace.root)

        self._agent = agent

        self._mcp_attached = set()
        self._mcp_errors = {}

        # Connect configured MCP servers so their tools are on the agent at
        # the first turn. Workspace catalogs stay untrusted until `/mcp connect`.
        if not self.config.efficiency.lazy_mcp:
            try:
                from noah_code.mcp_setup import install_mcp

                mcp_result = await install_mcp(
                    agent,
                    self.workspace.root,
                    self.config,
                    engine=agent.engine,
                    approvals=agent.approvals,
                    startup=True,
                )
                self._mcp_attached = set(mcp_result.attached)
                self._mcp_errors = {
                    error.partition(":")[0]: error.partition(":")[2].strip()
                    for error in mcp_result.errors
                }
                self.ui.render(HostEvent(HostEventKind.STATUS, str(mcp_result)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("mcp setup skipped: %s", exc)

        skills_status = getattr(agent, "_skills_status", "")
        if skills_status:
            self.ui.render(HostEvent(HostEventKind.STATUS, skills_status))

        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"session={self.meta.session_id} model={self.meta.model} "
                f"mode={agent.mode} workspace={self.workspace.root}",
            )
        )
        title = self.meta.title if self.meta.title != "untitled" else ""
        if title:
            self.ui.render(HostEvent(HostEventKind.STATUS, f"title={title}"))
        self.ui.set_status(self.status_prompt())
        if self.on_session_changed:
            with contextlib.suppress(Exception):
                self.on_session_changed(self.meta)
        return self.meta

    def _emit_with_hooks(self, event: HostEvent) -> None:
        """Render an event; schedule post-tool hooks for tool completions."""

        hooks = getattr(self, "_hooks", None)
        if (
            hooks is not None
            and event.kind == HostEventKind.TOOL_FINISH
            and hooks.active
            and self.config.hooks.post_tool
        ):
            task = asyncio.create_task(
                self._run_post_hooks(
                    tool=str(event.meta.get("tool", "tool")),
                    status=str(event.meta.get("result_status", "")),
                    target=event.text,
                )
            )
            self._post_hook_tasks.append(task)
        self.ui.render(event)

    async def _run_post_hooks(self, *, tool: str, status: str, target: str) -> None:
        try:
            failures = await self._hooks.run_post(
                tool=tool, category="tool", target=target, status=status
            )
        except Exception:  # noqa: BLE001 - hooks must not break turns
            logger.debug("post-tool hook crashed", exc_info=True)
            return
        for failure in failures:
            self.ui.render(HostEvent(HostEventKind.STATUS, f"hook warning: {failure}"))

    async def _flush_post_hooks(self) -> None:
        tasks, self._post_hook_tasks = self._post_hook_tasks, []
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _sync_budget_cost(self) -> None:
        guard = getattr(self, "_budget_guard", None)
        if guard is None or not guard.active:
            return
        usage = self.usage_snapshot()
        guard.add_usage(cost_usd=usage.cost_usd - guard.status()["cost_usd"])

    async def _capture_checkpoint(self, label: str) -> None:
        manager = self._checkpoints
        if manager is None or not self.config.checkpoints.enabled:
            return
        try:
            snapshot = await asyncio.to_thread(manager.capture, label)
        except Exception as exc:  # noqa: BLE001 - checkpoints are best-effort
            logger.debug("checkpoint capture failed", exc_info=True)
            self.ui.render(HostEvent(HostEventKind.STATUS, f"checkpoint failed: {exc}"))
            return
        if snapshot is not None:
            self.last_checkpoint = snapshot
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"checkpoint saved · {snapshot['ref']} · commit {snapshot['commit'][:10]}",
                    meta={"kind": "checkpoint", "checkpoint": snapshot},
                )
            )

    def _teardown_event_bridge(self) -> None:
        for unsub in self._event_unsubs:
            with contextlib.suppress(Exception):
                unsub()
        self._event_unsubs = []

    def _on_shell_chunk(self, stream: str, text: str) -> None:
        self.ui.render(HostEvent(HostEventKind.SHELL_CHUNK, text, meta={"stream": stream}))

    def _on_process_lifecycle(self, job_id: str, name: str, message: str) -> None:
        """Push lifecycle changes to the UI without streaming logs into model context."""

        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"background job {job_id} · {name} · {message}",
                meta={"kind": "background_job", "job_id": job_id},
            )
        )

    def _on_agent_message(self, text: str, **_kwargs: Any) -> None:
        self.ui.render(HostEvent(HostEventKind.MESSAGE, text))

    def _persist_state(self) -> dict:
        """Serialize agent-owned state for persistence. Thread-safe; touches no SQLite."""

        if self._agent is None or self.meta is None:
            return {}
        self.meta.mode = self._agent.mode
        self.meta.model = getattr(self._agent.v, "model", self.meta.model) or self.meta.model
        self.meta.permission_rules = self._agent.engine.snapshot_session_rules()
        # Undo blobs live in a pruned sidecar so meta.json stays small and
        # cheap to rewrite on every turn end.
        self.meta.journal = {}
        self.meta.todos = _json_safe(self._agent.todos.to_dict())
        with contextlib.suppress(Exception):
            title = getattr(self._agent.v, "title", None)
            if title:
                self.meta.title = str(title)
        return self._agent.journal.to_dict()

    def _write_persist_files(self, journal_data: dict) -> None:
        """Write meta.json and the undo sidecar. File I/O only."""

        if self.meta is None:
            return
        with contextlib.suppress(Exception):
            self.store.save_journal(self.meta.session_id, journal_data)
        self.store.save_meta(self.meta)

    def _persist(self) -> None:
        if self._agent is None or self.meta is None or self._storage is None:
            return
        journal_data = self._persist_state()
        self._write_persist_files(journal_data)
        self._storage.save_snapshot(self._agent)

    async def _persist_async(self) -> None:
        """Serialize and write session files off the UI event loop.

        NOOA's SQLiteStorageManager is thread-affine, so the snapshot write
        itself must stay on the event-loop thread.
        """

        journal_data = await asyncio.to_thread(self._persist_state)
        await asyncio.to_thread(self._write_persist_files, journal_data)
        if self._agent is not None and self._storage is not None:
            with contextlib.suppress(Exception):
                self._storage.save_snapshot(self._agent)

    async def close(self) -> None:
        try:
            await self._persist_async()
        finally:
            self._teardown_event_bridge()
            if self._agent is not None:
                try:
                    await self._agent.close_tools()
                except Exception:  # noqa: BLE001
                    logger.debug("shell close failed", exc_info=True)
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            if self._agent is not None and self.config.tracing.enabled:
                try:
                    from nooa.tracing import flush_traces

                    flush_traces()
                except Exception:  # noqa: BLE001
                    logger.debug("trace flush failed", exc_info=True)

    def status_prompt(self) -> str:
        sid = self.meta.session_id[:8] if self.meta else "?"
        mode = self.agent.mode if self._agent else self.config.mode
        model = self.meta.model if self.meta else self.config.model
        title = ""
        if self.meta and self.meta.title and self.meta.title != "untitled":
            title = f"|{self.meta.title[:20]}"
        effort = self.meta.reasoning_effort if self.meta else self.config.reasoning_effort
        effort_label = "auto" if effort == "default" else effort
        queued = self.steer_queue.snapshot()["count"]
        suffix = f" queued · {queued}" if queued else ""
        worktree = ""
        if self.meta and self.meta.worktree_name:
            worktree = f"|wt:{self.meta.worktree_name}"
        plan = ""
        if self._agent is not None:
            from noah_code.project_notes import PlanStore

            if PlanStore(self.workspace.root).exists():
                plan = "|plan"
        return f"noah [{mode}|{model}|r:{effort_label}|{sid}{title}{worktree}{plan}]{suffix}"

    def usage_snapshot(self) -> UsageSnapshot:
        return self._usage.snapshot()

    async def diff_review(self) -> Any:
        """Build a Git review model and enrich changed files with diagnostics."""

        review = await self.agent.git.review()
        paths = list(dict.fromkeys(item.path for item in review.files))
        diagnostics = await self.agent.lsp.diagnostics_for_paths(paths)
        for item in review.files:
            raw = diagnostics.get(item.path, "unavailable")
            if raw.startswith("ok —"):
                item.diagnostics = "clean"
            elif raw.startswith("unavailable") or raw == "not supported":
                item.diagnostics = raw
            else:
                issues = len([line for line in raw.splitlines() if line.strip()])
                item.diagnostics = f"{issues} issue{'s' if issues != 1 else ''}"
        return review

    async def revert_diff_file(self, path: str, scope: str) -> str:
        """Revert an explicitly confirmed review item as its own journal turn."""

        self.agent.journal.begin_turn()
        try:
            result = await self.agent.git.revert(path, scope)
        finally:
            self.agent.journal.end_turn()
            await self._persist_async()
        return result

    def _undo_last_turn_state(self) -> str:
        """Undo the latest reversible checkpoint without touching session storage."""

        turn = self.agent.journal.latest_turn()
        if turn:
            self.agent.journal.capture_post_bytes_before_undo(turn)
        undone = self.agent.journal.undo()
        return f"undid turn {undone.turn_id[:8]} ({len(undone.mutations)} files)"

    def undo_last_turn(self) -> str:
        """Undo the latest reversible checkpoint and persist it synchronously."""

        status = self._undo_last_turn_state()
        self._persist()
        return status

    async def undo_last_turn_async(self) -> str:
        status = await asyncio.to_thread(self._undo_last_turn_state)
        # SQLiteStorageManager is thread-affine; only the filesystem-heavy undo
        # runs in the worker, while snapshot persistence stays on this thread.
        await self._persist_async()
        return status

    def _require_idle_turn(self) -> None:
        """Refuse session switches while a turn is still running."""

        task = self._active_turn
        if task is not None and task is not asyncio.current_task() and not task.done():
            raise RuntimeError("a turn is still running; cancel it before switching sessions")

    async def start_new_session(
        self,
        workspace: Workspace | None = None,
        *,
        worktree_name: str = "",
    ) -> SessionMeta:
        """Persist current session and open a fresh one in-process."""
        self._require_idle_turn()
        self._clear_steer_state()
        await self._persist_async()
        if workspace is not None:
            self.workspace = workspace
        self._pending_worktree_name = worktree_name
        self._teardown_event_bridge()
        self._cancel_background_tasks()
        if self._agent is not None:
            with contextlib.suppress(Exception):
                await self._agent.close_tools()
        if self._storage is not None:
            self._storage.close()
            self._storage = None
        self._agent = None
        self._usage = UsageTracker()
        self.meta = None
        self._exit_requested = False
        return await self.start()

    async def switch_session(self, session_id: str) -> SessionMeta:
        """Persist current session and resume another."""
        self._require_idle_turn()
        self._clear_steer_state()
        await self._persist_async()
        self._teardown_event_bridge()
        self._cancel_background_tasks()
        if self._agent is not None:
            with contextlib.suppress(Exception):
                await self._agent.close_tools()
        if self._storage is not None:
            self._storage.close()
            self._storage = None
        self._agent = None
        self._usage = UsageTracker()
        meta = self.store.load_meta(session_id)
        self.workspace = self.store.workspace_for_resume(meta, self.workspace)
        self.meta = meta
        self._exit_requested = False
        return await self.start()

    def list_session_metas(self) -> list[SessionMeta]:
        return self.store.list_sessions(self.workspace)

    def worktree_manager(self) -> Any:
        from noah_code.worktree import WorktreeManager, worktree_storage_root

        return WorktreeManager(self.workspace.root, worktree_storage_root(self.config.session_dir))

    async def create_worktree_session(self, name: str | None = None) -> SessionMeta:
        """Create a linked worktree and start a new session there."""

        self._require_idle_turn()
        info = await asyncio.to_thread(self.worktree_manager().create, name)
        try:
            return await self.start_new_session(
                Workspace(root=info.directory), worktree_name=info.name
            )
        except Exception:
            with contextlib.suppress(Exception):
                self.worktree_manager().remove(info.name)
            raise

    def remove_worktree(self, name: str) -> Any:
        from noah_code.worktree import WorktreeError

        manager = self.worktree_manager()
        matches = [item for item in manager.list() if item.name == name or str(item.directory) == name]
        if matches and matches[0].directory.resolve() == self.workspace.root.resolve():
            raise WorktreeError("switch away from this worktree before removing it")
        return manager.remove(name)

    async def _handle_worktree(self, args: str) -> Literal["handled"]:
        from noah_code.worktree import WorktreeError

        parts = shlex.split(args) if args.strip() else []
        action = parts[0] if parts else "create"
        rest = parts[1:]
        if action == "create":
            name = rest[0] if rest else None
            try:
                meta = await self.create_worktree_session(name)
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"worktree {meta.worktree_name} · session {meta.session_id} · "
                    f"{meta.workspace_path}",
                )
            )
            return "handled"
        if action == "list":
            try:
                rows = self.worktree_manager().list()
            except WorktreeError as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            text = (
                "\n".join(f"{item.name}  {item.branch}  {item.directory}" for item in rows)
                or "(none)"
            )
            self.ui.render(_command_output(text))
            return "handled"
        if action == "remove":
            if not rest:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /worktree remove NAME"))
                return "handled"
            try:
                info = self.remove_worktree(rest[0])
            except WorktreeError as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            self.ui.render(HostEvent(HostEventKind.STATUS, f"removed worktree {info.name}"))
            return "handled"
        self.ui.render(
            HostEvent(
                HostEventKind.ERROR,
                "usage: /worktree [create [NAME]|list|remove NAME]",
            )
        )
        return "handled"

    def github_manager(self) -> Any:
        from noah_code.github import GithubManager

        return GithubManager(self.workspace.root)

    async def create_pull_request(
        self,
        title: str | None = None,
        body: str = "",
        base: str | None = None,
    ) -> Any:
        self._require_idle_turn()
        return await asyncio.to_thread(self.github_manager().create, title, body, base)

    async def push_pull_request_branch(self) -> str:
        self._require_idle_turn()
        return await asyncio.to_thread(self.github_manager().push)

    async def checkout_pull_request(self, number: int) -> str:
        self._require_idle_turn()
        return await asyncio.to_thread(self.github_manager().checkout, number)

    async def _handle_pr(self, args: str) -> Literal["handled"]:
        from noah_code.github import GithubError

        parts = shlex.split(args) if args.strip() else []
        if not parts:
            action, rest = "list", []
        elif parts[0].isdigit():
            action, rest = "view", parts[:1]
        else:
            action, rest = parts[0], parts[1:]
        try:
            if action == "list":
                rows = await asyncio.to_thread(self.github_manager().list)
                text = "\n".join(item.format_row() for item in rows) or "(none)"
                self.ui.render(_command_output(text))
                return "handled"
            if action == "view":
                number = int(rest[0]) if rest else None
                text = await asyncio.to_thread(self.github_manager().view, number)
                self.ui.render(_command_output(text or "(none)"))
                return "handled"
            if action == "create":
                title = " ".join(rest) or None
                info = await self.create_pull_request(title)
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"created #{info.number}  {info.title}  {info.url}",
                    )
                )
                return "handled"
            if action == "push":
                text = await self.push_pull_request_branch()
                self.ui.render(HostEvent(HostEventKind.STATUS, text))
                return "handled"
            if action == "checkout":
                if not rest or not rest[0].isdigit():
                    self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /pr checkout N"))
                    return "handled"
                branch = await self.checkout_pull_request(int(rest[0]))
                self.ui.render(
                    HostEvent(HostEventKind.STATUS, f"checked out #{rest[0]} as {branch}")
                )
                return "handled"
            if action == "comment":
                if len(rest) < 2 or not rest[0].isdigit():
                    self.ui.render(
                        HostEvent(HostEventKind.ERROR, "usage: /pr comment N TEXT")
                    )
                    return "handled"
                text = await asyncio.to_thread(
                    self.github_manager().comment, int(rest[0]), " ".join(rest[1:])
                )
                self.ui.render(HostEvent(HostEventKind.STATUS, text))
                return "handled"
        except (GithubError, RuntimeError, OSError, ValueError) as exc:
            self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return "handled"
        self.ui.render(
            HostEvent(
                HostEventKind.ERROR,
                "usage: /pr [list|view [N]|create [TITLE]|push|checkout N|comment N TEXT]",
            )
        )
        return "handled"

    async def _handle_plan(self, args: str) -> Literal["handled"]:
        from noah_code.project_notes import PlanStore

        store = PlanStore(self.workspace.root)
        verb = args.strip().split(None, 1)[0].lower() if args.strip() else "show"
        if verb in {"show", "read"}:
            self.ui.render(_command_output(store.read().strip() or "(no active plan)"))
            return "handled"
        if verb == "clear":
            store.clear()
            if self._agent is not None:
                self.agent.refresh_context_sources()
            self.ui.render(HostEvent(HostEventKind.STATUS, "plan cleared"))
            self.ui.set_status(self.status_prompt())
            return "handled"
        self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /plan [clear]"))
        return "handled"

    async def _handle_memory(self, args: str) -> Literal["handled"]:
        from noah_code.project_notes import MemoryStore, parse_memory_facts

        store = MemoryStore(self.workspace.root)
        parts = args.split(None, 1)
        verb = parts[0].lower() if parts and parts[0] else "show"
        rest = parts[1].strip() if len(parts) > 1 else ""
        if verb in {"show", "list", "read"}:
            self.ui.render(_command_output(store.read().strip() or "(no project memory yet)"))
            return "handled"
        if verb == "save":
            facts = parse_memory_facts(f"- {rest}" if rest else "")
            if not facts:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /memory save FACT"))
                return "handled"
            added = store.merge(facts)
            if self._agent is not None:
                self.agent.refresh_context_sources()
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"remembered {added[0]}" if added else "already remembered",
                )
            )
            return "handled"
        if verb == "forget":
            if not rest:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /memory forget TEXT"))
                return "handled"
            forgotten = store.forget(rest)
            if self._agent is not None:
                self.agent.refresh_context_sources()
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"forgot {rest}" if forgotten else "no matching memory",
                )
            )
            return "handled"
        if verb == "clear":
            store.clear()
            if self._agent is not None:
                self.agent.refresh_context_sources()
            self.ui.render(HostEvent(HostEventKind.STATUS, "memory cleared"))
            return "handled"
        self.ui.render(
            HostEvent(HostEventKind.ERROR, "usage: /memory [save FACT|forget TEXT|clear]")
        )
        return "handled"

    def list_skill_infos(self) -> list[Any]:
        """Return display metadata for all loaded and document skills."""

        if self._agent is None or not hasattr(self.agent, "skills"):
            return []
        from noah_code.skills_setup import list_skills

        return list_skills(self.agent.skills)

    def add_skill_from_path(self, path: str) -> Any:
        """Copy a compatible skill folder and discover it in this session."""

        from noah_code.skills_setup import add_skill

        return add_skill(path, registry=self.agent.skills)

    def list_mcp_infos(self) -> list[Any]:
        from noah_code.mcp_setup import list_mcp_servers

        return list_mcp_servers(self.workspace.root, self.config)

    def list_provider_infos(self) -> list[Any]:
        from noah_code.providers import list_providers

        model = self.meta.model if self.meta else self.config.model
        return list_providers(model)

    async def set_provider_api_key(self, provider: str, api_key: str) -> Any:
        """Activate a provider key and persist it in Noah's private auth file."""

        from noah_code.credentials import store_provider_api_key

        return await asyncio.to_thread(store_provider_api_key, provider, api_key)

    async def configure_provider(
        self,
        provider: str,
        model: str,
        *,
        alias: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        client_type: str = "completion",
        reasoning_effort: str | None = None,
    ) -> str:
        """Configure a provider, switch this session, and save the global default."""

        from noah_code.providers import (
            provider_preset,
            resolve_provider_model,
            save_custom_openai_provider,
        )

        if provider == "custom":
            if not alias or not base_url:
                raise ValueError("custom provider requires an alias and base URL")
            config_path = await asyncio.to_thread(
                save_custom_openai_provider,
                alias,
                model,
                base_url,
                api_key_env,
                client_type=client_type,
            )
            from nooa.unifiedllm import reload_registry

            await asyncio.to_thread(reload_registry)
            selected_model = alias
            credential_hint = api_key_env or "no API key"
            provider_label = "Custom OpenAI-compatible"
        else:
            from noah_code.providers import list_providers

            preset = provider_preset(provider)
            selected_model = resolve_provider_model(provider, model)
            credential_hint = (
                " or ".join(" + ".join(group) for group in preset.credential_groups) or "no API key"
            )
            provider_info = next(
                info for info in list_providers(selected_model) if info.key == provider
            )
            credential_hint += " [ready]" if provider_info.configured else " [missing]"
            provider_label = preset.label
            config_path = None

        selected_effort = reasoning_effort or self.config.reasoning_effort
        if selected_effort not in REASONING_EFFORTS:
            raise ValueError(
                "reasoning effort must be default, none, minimal, low, medium, high, or xhigh"
            )
        if self._agent is not None:
            await self._switch_model(selected_model, reasoning_effort=selected_effort)
        elif self.meta is not None:
            self.meta.model = selected_model
            self.meta.reasoning_effort = selected_effort
            self.store.save_meta(self.meta)
        default_path = save_user_default_model(selected_model)
        if reasoning_effort is not None:
            save_user_reasoning_effort(selected_effort)
        self.config.model = selected_model
        self.config.reasoning_effort = selected_effort
        suffix = f"; alias saved in {config_path}" if config_path else ""
        return (
            f"Using {provider_label}: {selected_model} (reasoning={selected_effort}). "
            f"Global default saved in {default_path}. "
            f"Credentials: {credential_hint}{suffix}"
        )

    async def connect_mcp_server(self, name: str) -> str:
        """Attach one configured MCP server to the live agent."""

        if name in self._mcp_attached:
            return f"MCP server {name} is already connected"
        from noah_code.mcp_setup import attach_mcp_server, load_mcp_servers, mcp_source_is_trusted

        servers, sources = load_mcp_servers(self.workspace.root, self.config)
        if name not in servers:
            raise KeyError(f"unknown MCP server: {name}")
        attr = await attach_mcp_server(
            self.agent,
            name,
            servers[name],
            engine=self.agent.engine,
            approvals=self.agent.approvals,
            trusted=mcp_source_is_trusted(sources.get(name, "")),
        )
        self._mcp_attached.add(name)
        self._mcp_errors.pop(name, None)
        return f"Connected MCP server {name} as self.{attr}"

    async def add_mcp_server(self, kind: str, name: str, target: str) -> str:
        """Persist and connect a simple STDIO or Streamable HTTP server."""

        from noah_code.mcp_setup import save_user_mcp_server

        normalized_kind = kind.strip().lower()
        if normalized_kind == "stdio":
            command = shlex.split(target)
            if not command:
                raise ValueError("STDIO server command cannot be empty")
            spec: dict[str, Any] = {
                "command": command[0],
                "args": command[1:],
                "transport": "stdio",
            }
        elif normalized_kind in {"http", "streamable-http"}:
            spec = {"url": target.strip(), "transport": "streamable-http"}
        else:
            raise ValueError("MCP transport must be stdio or http")
        path = await asyncio.to_thread(save_user_mcp_server, name, spec)
        try:
            status = await self.connect_mcp_server(name)
        except Exception as exc:
            return f"Saved MCP server {name} in {path}, but connection failed: {exc}"
        return f"Saved MCP server {name} in {path}. {status}"

    async def load_history_page(
        self,
        *,
        before: int | None = None,
        limit: int = 50,
    ) -> list[SessionEventRecord]:
        """Load persisted UI history without blocking the Textual event loop."""

        if self.meta is None:
            return []
        return await asyncio.to_thread(
            self.store.load_event_page,
            self.meta.session_id,
            before=before,
            limit=limit,
        )

    def cancel_active_turn(self) -> None:
        """Cancel the in-flight turn and pending approvals (Ctrl-C)."""
        self._clear_steer_state()
        if self._agent is not None:
            self._agent.approvals.cancel_all()
        task = self._active_turn
        if task is not None and not task.done():
            task.cancel()

    def _turn_running(self) -> bool:
        task = self._active_turn
        return task is not None and not task.done()

    def _clear_steer_state(self) -> None:
        self.steer_queue.clear()
        self._pending_attach_paths.clear()

    def take_pending_attaches(self) -> list[Path]:
        paths, self._pending_attach_paths = self._pending_attach_paths, []
        return paths

    def enqueue_steer(self, text: str, attach_paths: list[Path] | None = None) -> bool:
        """Queue a follow-up for the current turn. Returns True if oldest dropped."""

        paths = list(attach_paths or []) + self.take_pending_attaches()
        dropped = self.steer_queue.push(text, attach_paths=paths or None)
        if dropped:
            self.ui.render(HostEvent(HostEventKind.STATUS, "steer dropped oldest"))
        self.ui.set_status(self.status_prompt())
        return dropped

    async def handle_line(self, line: str) -> Literal["continue", "exit", "handled"]:
        slash = parse_slash(line)
        if slash:
            return await self._handle_slash(slash[0], slash[1])
        skill_prompt = await self._activate_explicit_skill(line)
        if skill_prompt is None:
            return "handled"
        line = skill_prompt
        self.ui.set_busy(True)
        self._active_turn = asyncio.current_task()
        try:
            await self._run_user_turn(line)
            return "continue"
        finally:
            self._active_turn = None
            self.ui.set_busy(False)
            self.ui.set_status(self.status_prompt())

    async def _activate_explicit_skill(self, line: str) -> str | None:
        """Resolve ``$name task``, approve it, and expose its instructions."""

        match = re.match(r"^\$([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$", line, re.DOTALL)
        if not match or self._agent is None:
            return line
        requested, task = match.groups()
        candidates = [
            info
            for info in self.list_skill_infos()
            if info.document_skill and requested in {info.name, info.registry_name}
        ]
        if not candidates:
            return line
        if not task.strip():
            self.ui.render(
                HostEvent(HostEventKind.ERROR, f"Add a task after ${candidates[0].name}")
            )
            return None
        info = candidates[0]
        decision = self.agent.engine.decide("skill", info.registry_name, tool="skill")
        try:
            await self.agent.approvals.require(decision)
            self.agent.skills.activate([info.registry_name])
        except Exception as exc:  # noqa: BLE001
            self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return None
        attr = nooa_compat.skill_attribute(self.agent.skills, info.registry_name)
        approved = getattr(self.agent, "_sandbox_approved_roots", None)
        if attr and isinstance(approved, set):
            approved.add(attr)
        return f"Use the ${info.name} skill instructions for this task:\n\n{task.strip()}"

    def _handle_slash_while_busy(
        self, name: str, args: str
    ) -> Literal["continue", "exit", "handled"] | None:
        """Allow read-only slash commands during a turn; block mutating ones."""

        if not self._turn_running() or name in SAFE_SLASH_WHILE_BUSY:
            return None
        if name in {"exit", "quit"}:
            return None
        if name == "attach":
            raw = args.strip().strip("\"'")
            if not raw:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /attach PATH"))
                return "handled"
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.workspace.root / path
            self._pending_attach_paths.append(path)
            self.ui.render(HostEvent(HostEventKind.STATUS, f"attach queued · {path}"))
            return "handled"
        self.ui.render(
            HostEvent(HostEventKind.STATUS, f"/{name} is blocked while a turn is running")
        )
        return "handled"

    async def _handle_slash(self, name: str, args: str) -> Literal["continue", "exit", "handled"]:
        busy = self._handle_slash_while_busy(name, args)
        if busy is not None:
            return busy
        agent = self.agent
        if name == "help":
            self.ui.render(_command_output(help_text(self._custom_commands)))
            return "handled"
        if name == "config":
            try:
                text = config_text(self.config, args)
            except KeyError:
                self.ui.render(
                    HostEvent(
                        HostEventKind.ERROR,
                        f"unknown configuration path {args.strip()!r}; use /config to list all",
                    )
                )
            else:
                self.ui.render(_command_output(text))
            return "handled"
        if name == "theme":
            requested = args.strip().lower()
            if not requested:
                choices = ", ".join(THEME_NAMES)
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"theme={self.config.ui.theme} available={choices}",
                    )
                )
                return "handled"
            try:
                selected = get_theme(requested).name
            except ValueError as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            path = save_user_theme(selected)
            self.config.ui.theme = selected
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"theme set to {selected} in {path}",
                    meta={"kind": "theme", "theme": selected},
                )
            )
            return "handled"
        if name in self._custom_commands:
            return await self._run_custom_command(name, args)
        if name == "exit" or name == "quit":
            self.cancel_active_turn()
            self._exit_requested = True
            return "exit"
        if name == "new":
            meta = await self.start_new_session()
            self.ui.render(
                HostEvent(HostEventKind.STATUS, f"started new session {meta.session_id}")
            )
            return "handled"
        if name == "worktree":
            return await self._handle_worktree(args)
        if name == "pr":
            return await self._handle_pr(args)
        if name == "plan":
            return await self._handle_plan(args)
        if name == "memory":
            return await self._handle_memory(args)
        if name == "skills":
            skills = getattr(agent, "skills", None)
            if skills is None:
                self.ui.render(HostEvent(HostEventKind.STATUS, "no skills registry"))
            else:
                try:
                    if args.strip().lower().startswith("add "):
                        source = args.strip()[4:].strip()
                        info = await asyncio.to_thread(self.add_skill_from_path, source)
                        text = f"Added ${info.name}\n{info.description}\n{info.source}"
                    elif args.strip():
                        text = "usage: /skills [add PATH]"
                    else:
                        from noah_code.skills_setup import format_skills

                        text = format_skills(skills)
                except Exception as exc:  # noqa: BLE001
                    text = f"skills error: {exc}"
                self.ui.render(_command_output(text if text else "(empty)"))
            return "handled"
        if name == "mcp":
            try:
                parts = shlex.split(args)
                if not parts:
                    rows = self.list_mcp_infos()
                    text = "MCP servers"
                    if rows:
                        for row in rows:
                            state = "connected" if row.name in self._mcp_attached else "available"
                            text += (
                                f"\n\n  {row.name}  [{state}]\n"
                                f"    {row.transport} · {row.target}\n    {row.source}"
                            )
                    else:
                        text += "\n\n  None configured"
                    text += "\n\nAdd with: /mcp add stdio NAME COMMAND…\n          /mcp add http NAME URL"
                elif parts[0] == "connect" and len(parts) == 2:
                    text = await self.connect_mcp_server(parts[1])
                elif parts[:2] == ["add", "stdio"] and len(parts) >= 4:
                    text = await self.add_mcp_server("stdio", parts[2], shlex.join(parts[3:]))
                elif parts[:2] == ["add", "http"] and len(parts) == 4:
                    text = await self.add_mcp_server("http", parts[2], parts[3])
                else:
                    text = (
                        "usage: /mcp [connect NAME | add stdio NAME COMMAND… | add http NAME URL]"
                    )
                self.ui.render(_command_output(text))
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"MCP error: {exc}"))
            return "handled"
        if name == "mode":
            mode = args.strip().lower()
            if not mode:
                self.ui.render(HostEvent(HostEventKind.STATUS, f"mode={agent.mode}"))
                return "handled"
            if mode not in {"build", "plan"}:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /mode [build|plan]"))
                return "handled"
            agent.set_mode(mode)
            if self.meta:
                self.meta.mode = mode
                self.store.save_meta(self.meta)
            self.ui.render(HostEvent(HostEventKind.STATUS, f"mode set to {mode}"))
            self.ui.set_status(self.status_prompt())
            return "handled"
        if name == "model":
            requested = args.strip()
            if not requested:
                default_model = user_default_model() or "(not configured)"
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"model={self.meta.model if self.meta else self.config.model} "
                        f"global_default={default_model}",
                    )
                )
                return "handled"
            if requested == "--global" or requested.startswith("--global "):
                global_model = requested.removeprefix("--global").strip()
                if not global_model:
                    self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /model --global MODEL"))
                    return "handled"
                await self._switch_model(global_model)
                path = save_user_default_model(global_model)
                self.config.model = global_model
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"global default model set to {global_model} in {path}",
                    )
                )
                return "handled"
            await self._switch_model(requested)
            return "handled"
        if name == "reasoning":
            requested = args.strip().lower()
            make_global = requested == "--global" or requested.startswith("--global ")
            if make_global:
                requested = requested.removeprefix("--global").strip()
            if not requested:
                effort = self.meta.reasoning_effort if self.meta else self.config.reasoning_effort
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"reasoning_effort={effort} (default lets the provider/model decide)",
                    )
                )
                return "handled"
            if requested not in REASONING_EFFORTS:
                self.ui.render(
                    HostEvent(
                        HostEventKind.ERROR,
                        "usage: /reasoning [--global] [default|none|minimal|low|medium|high|xhigh]",
                    )
                )
                return "handled"
            model = self.meta.model if self.meta else self.config.model
            await self._switch_model(model, reasoning_effort=requested)
            if make_global:
                path = save_user_reasoning_effort(requested)
                self.config.reasoning_effort = requested
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"global reasoning effort set to {requested} in {path}",
                    )
                )
            return "handled"
        if name == "providers":
            try:
                parts = shlex.split(args)
                if not parts:
                    from noah_code.providers import format_providers

                    text = format_providers(self.meta.model if self.meta else self.config.model)
                elif parts[0] == "use" and len(parts) == 3:
                    text = await self.configure_provider(parts[1], parts[2])
                else:
                    text = "usage: /providers [use PROVIDER MODEL]"
                self.ui.render(_command_output(text))
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"provider error: {exc}"))
            return "handled"
        if name == "session":
            current = self.meta
            if current is None:
                self.ui.render(_command_output("(no active session)"))
                return "handled"
            self.ui.render(
                _command_output(
                    f"id={current.session_id}\ntitle={current.title}\n"
                    f"mode={agent.mode}\nmodel={current.model}\n"
                    f"reasoning_effort={current.reasoning_effort}\n"
                    f"workspace={current.workspace_path}"
                    + (f"\nworktree={current.worktree_name}" if current.worktree_name else "")
                )
            )
            return "handled"
        if name == "sessions":
            if args.strip():
                sid = args.strip().split()[0]
                try:
                    await self.switch_session(sid)
                    self.ui.render(HostEvent(HostEventKind.STATUS, f"switched to session {sid}"))
                except Exception as exc:  # noqa: BLE001
                    self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            rows = self.store.list_sessions(self.workspace)
            text = (
                "\n".join(
                    f"{s.session_id}  {s.mode:5}  {s.title}"
                    + (f"  {s.worktree_name}" if s.worktree_name else "")
                    + ("  ← current" if self.meta and s.session_id == self.meta.session_id else "")
                    for s in rows
                )
                or "(none)"
            )
            text += "\n\nSwitch with: /sessions SESSION_ID"
            self.ui.render(_command_output(text))
            return "handled"
        if name == "todos":
            self.ui.render(_command_output(agent.todos.status() or "(no todos)"))
            return "handled"
        if name == "agents":
            listing = (
                agent.task.list() if getattr(agent, "task", None) is not None else "(no agents)"
            )
            self.ui.render(_command_output(listing))
            return "handled"
        if name == "attach":
            raw = args.strip().strip("\"'")
            if not raw:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /attach PATH"))
                return "handled"
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.workspace.root / path
            self.ui.set_busy(True)
            self._active_turn = asyncio.current_task()
            try:
                await self._run_user_turn("Please inspect the attached file.", attach_paths=[path])
            finally:
                self._active_turn = None
                self.ui.set_busy(False)
                self.ui.set_status(self.status_prompt())
            return "continue"
        if name == "status":
            reversible = agent.journal.last_turn_reversible() if agent.journal.can_undo() else True
            warn = ""
            if agent.journal.can_undo() and not reversible:
                warn = " last_turn=NOT_FULLY_REVERSIBLE(shell)"
            elif self._last_turn_shell_bypass:
                warn = " last_turn=shell_bypass"
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"mode={agent.mode} model={self.meta.model if self.meta else '?'} "
                    f"reasoning={self.meta.reasoning_effort if self.meta else '?'} "
                    f"session={self.meta.session_id if self.meta else '?'} "
                    f"title={self.meta.title if self.meta else '?'} "
                    f"undo={'yes' if agent.journal.can_undo() else 'no'} "
                    f"reversible={reversible}{warn}",
                )
            )
            return "handled"
        if name == "tokens":
            self.ui.render(_command_output(self.usage_snapshot().format()))
            evicted = nooa_compat.evicted_output_chars(self.agent)
            if evicted:
                self.ui.render(
                    _command_output(f"  compaction savings {evicted:,} chars (pointer eviction)")
                )
            return "handled"
        if name == "efficiency":
            profile = args.strip().lower()
            if not profile:
                profile = self.config.efficiency.profile
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"efficiency={profile} strategy={self.config.efficiency.strategy}",
                    )
                )
                return "handled"
            try:
                agent.set_efficiency_profile(profile)
            except ValueError as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            else:
                self.config.efficiency.profile = profile  # type: ignore[assignment]
                self.ui.render(HostEvent(HostEventKind.STATUS, f"efficiency set to {profile}"))
            return "handled"
        if name == "diff":
            try:
                review = await self.diff_review()
                if review.files:
                    lines = [
                        f"Changes · {len(review.files)} views · +{review.additions} -{review.deletions}"
                    ]
                    lines.extend(
                        f"  {item.scope[0].upper()} {item.path} +{item.additions} -{item.deletions} "
                        f"· {item.diagnostics}"
                        for item in review.files
                    )
                    text = "\n".join(lines)
                else:
                    text = "No staged or unstaged changes"
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"diff failed: {exc}"))
                return "handled"
            self.ui.render(HostEvent(HostEventKind.DIFF_REVIEW, text, meta={"review": review}))
            return "handled"
        if name == "undo":
            try:
                self.ui.render(HostEvent(HostEventKind.STATUS, await self.undo_last_turn_async()))
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return "handled"
        if name == "redo":
            try:
                redone = agent.journal.redo()
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"redid turn {redone.turn_id[:8]} ({len(redone.mutations)} files)",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return "handled"
        if name == "trace":
            self.ui.render(HostEvent(HostEventKind.STATUS, f"trace: {self._trace_info}"))
            return "handled"
        if name == "checkpoints":
            manager = getattr(self, "_checkpoints", None)
            if manager is None:
                self.ui.render(_command_output("(no checkpoint manager for this session)"))
                return "handled"
            entries = await asyncio.to_thread(manager.list)
            if not entries:
                enabled = self.config.checkpoints.enabled
                hint = "" if enabled else "\nEnable with [checkpoints] enabled=true or --checkpoint"
                self.ui.render(_command_output(f"(no checkpoints yet){hint}"))
                return "handled"
            rows = [
                f"{entry['ref']}\t{entry['commit'][:12]}"
                + (f" {entry['label']}" if entry["label"] else "")
                for entry in entries
            ]
            text = "\n".join(rows) + "\n\nRestore with: noah checkpoints restore REF"
            self.ui.render(_command_output(text))
            return "handled"
        if name == "compact":
            try:
                compacted = await agent.compact_history()
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"compact failed: {exc}"))
                return "handled"
            status = "history compacted" if compacted else "nothing eligible to compact"
            self.ui.render(HostEvent(HostEventKind.STATUS, status))
            return "handled"
        if name in {"continue"}:
            latest = self.store.latest_for_workspace(self.workspace)
            if latest is None:
                self.ui.render(
                    HostEvent(HostEventKind.ERROR, "no prior session for this workspace")
                )
                return "handled"
            if self.meta and latest.session_id == self.meta.session_id:
                self.ui.render(HostEvent(HostEventKind.STATUS, "already on the latest session"))
                return "handled"
            await self.switch_session(latest.session_id)
            self.ui.render(
                HostEvent(HostEventKind.STATUS, f"continued session {latest.session_id}")
            )
            return "handled"
        self.ui.render(HostEvent(HostEventKind.ERROR, f"unknown command /{name} - try /help"))
        return "handled"

    async def _run_custom_command(
        self, name: str, args: str
    ) -> Literal["continue", "exit", "handled"]:
        cmd = self._custom_commands[name]
        if cmd.mode in {"build", "plan"}:
            self.agent.set_mode(cmd.mode)
            if self.meta:
                self.meta.mode = cmd.mode
                self.store.save_meta(self.meta)
        if cmd.model:
            try:
                await self._switch_model(cmd.model)
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"model switch failed: {exc}"))
        rendered = cmd.render(args)
        if not rendered:
            self.ui.render(
                HostEvent(HostEventKind.ERROR, f"custom command /{name} produced empty body")
            )
            return "handled"
        self.ui.render(HostEvent(HostEventKind.STATUS, f"running /{name} ({cmd.source})"))
        self.ui.set_busy(True)
        self._active_turn = asyncio.current_task()
        try:
            await self._run_user_turn(rendered)
        finally:
            self._active_turn = None
            self.ui.set_busy(False)
            self.ui.set_status(self.status_prompt())
        return "continue"

    def _apply_runtime_llm_wrappers(self, llm: Any) -> Any:
        """Re-apply session budget and record/replay wraps after a client swap."""

        from noah_code.budget import SharedBudgetLLM
        from noah_code.llm_cache import resolve_cache_settings, wrap_with_cache
        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)
        guard = getattr(self, "_budget_guard", None)
        if guard is not None and guard.active:
            llm = SharedBudgetLLM(llm, guard)
        cache_mode, cache_dir = resolve_cache_settings()
        llm = wrap_with_cache(llm, cache_dir, cache_mode)
        if hasattr(llm, "stats"):
            self._llm_cache = llm
        return llm

    async def _switch_model(self, model: str, *, reasoning_effort: str | None = None) -> None:
        from noah_code.llm import get_llm_client, reasoning_overrides, sampling_overrides

        effort = reasoning_effort or (
            self.meta.reasoning_effort if self.meta else self.config.reasoning_effort
        )
        llm = await asyncio.to_thread(
            get_llm_client,
            model,
            **reasoning_overrides(effort),
            **sampling_overrides(self.config.sampling),
        )
        llm = self._apply_runtime_llm_wrappers(llm)
        self.agent.set_main_llm(
            llm,
            lightweight_follows_main=not bool(self.config.lightweight_model),
        )
        if self.meta:
            self.meta.model = model
            self.meta.reasoning_effort = effort
            self.agent.v.model = model
            self.store.save_meta(self.meta)
        self.ui.render(
            HostEvent(HostEventKind.STATUS, f"model set to {model} · reasoning={effort}")
        )
        self.ui.set_status(self.status_prompt())

    def _expand_user_text(self, text: str, attach_paths: list[Path] | None) -> Any:
        from noah_code.composer import expand_turn

        return expand_turn(text, self.workspace.root, attach_paths=attach_paths)

    def _deliver_expanded(self, agent: Any, expanded: Any) -> None:
        if expanded.images:
            agent.media.queue(expanded.images)
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"attached {len(expanded.images)} image(s) for show()",
                )
            )
        nooa_compat.queue_user_message(agent, expanded.text)

    def _apply_next_steer(self, agent: Any) -> bool:
        """Pop until one follow-up is queued. Returns False when the queue is empty."""

        while True:
            item = self.steer_queue.pop()
            if item is None:
                return False
            expanded = self._expand_user_text(item.text, list(item.attach_paths))
            if expansion_failed(item, expanded):
                self.ui.render(HostEvent(HostEventKind.STATUS, "steer dropped · could not expand"))
                continue
            self._deliver_expanded(agent, expanded)
            preview = " ".join(item.text.split())[:60]
            self.ui.render(HostEvent(HostEventKind.STATUS, f"steer applied · {preview}"))
            self.ui.set_status(self.status_prompt())
            return True

    async def _run_user_turn(
        self,
        text: str,
        *,
        attach_paths: list[Path] | None = None,
    ) -> HostResult:
        from nooa.interactive import RespondReason

        agent = self.agent
        # git status + instruction-file reads are subprocess/file work; keep
        # them off the UI event loop.
        await asyncio.to_thread(agent.refresh_context_sources)
        agent.inject_status_snapshot()
        agent.journal.begin_turn()
        self._deliver_expanded(agent, self._expand_user_text(text, attach_paths))

        if self.meta and self.meta.title == "untitled":
            if self.config.efficiency.deterministic_titles:
                self._set_session_title(_deterministic_title(text))
            else:
                # Keep a strong reference; the event loop only holds tasks weakly.
                self._title_task = asyncio.create_task(self._maybe_title(text))

        exit_code = 0
        explanation = ""
        try:
            while True:
                try:
                    wins = await agent.queue_manager.race()
                    notification: dict[str, list] = {}
                    for name, item in wins:
                        notification.setdefault(name, []).append(item)
                    result = await _handle_with_overflow_recovery(
                        agent, notification, render=self.ui.render
                    )
                    explanation = getattr(result, "explanation", "") or ""
                    kind = getattr(result, "kind", None)
                    self._sync_budget_cost()
                    self.ui.render(HostEvent(HostEventKind.STOP, _stop_text(kind, explanation)))
                    if kind == RespondReason.NEED_INPUT:
                        exit_code = 0
                    elif kind == RespondReason.WAIT and not agent.processes.has_running():
                        self.ui.render(
                            HostEvent(
                                HostEventKind.ERROR,
                                "Agent returned WAIT without a running background job; "
                                "start one with self.processes.start() or finish with DONE.",
                            )
                        )
                    else:
                        exit_code = 0
                except PermissionError as exc:
                    exit_code = 3
                    explanation = str(exc)
                    self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
                except asyncio.CancelledError:
                    exit_code = 130
                    explanation = "cancelled"
                    self.steer_queue.clear()
                    self.ui.render(HostEvent(HostEventKind.STATUS, "turn cancelled"))
                    raise
                except Exception as exc:
                    exit_code = 1
                    explanation = _friendly_agent_error(exc)
                    self.steer_queue.clear()
                    self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
                    logger.debug("handle() failed", exc_info=True)
                    break

                if exit_code == 1:
                    break
                if not self._apply_next_steer(agent):
                    break
        finally:
            agent.journal.end_turn()
            latest = agent.journal.latest_turn()
            self._last_turn_shell_bypass = bool(latest and latest.shell_may_bypass)
            if self._last_turn_shell_bypass:
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        "warning: this turn ran mutating shell commands; "
                        "file-journal undo may be incomplete",
                    )
                )
            if exit_code == 130:
                # The cancelled task cannot await, but the journal must be
                # finalized before it is serialized to the undo sidecar.
                with contextlib.suppress(Exception):
                    self._persist()
            else:
                await self._persist_async()
                label = " ".join(text.split())[:80] or "turn"
                await self._capture_checkpoint(label)
                if exit_code == 0:
                    self._memory_task = asyncio.create_task(self._maybe_remember(text))
        return HostResult(
            exit_code=exit_code,
            explanation=explanation,
            session_id=self.meta.session_id if self.meta else None,
        )

    async def _maybe_title(self, text: str) -> None:
        session_id = self.meta.session_id if self.meta else None
        try:
            title = await self.agent.name_session(text)
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("title generation failed", exc_info=True)
            return
        if session_id and (self.meta is None or self.meta.session_id != session_id):
            # The user switched sessions while the title call was in flight;
            # never write a stale title into the new session's meta.
            return
        title = (title or "").strip().strip('"')[:60]
        self._set_session_title(title)

    def _cancel_title_task(self) -> None:
        if self._title_task is not None and not self._title_task.done():
            self._title_task.cancel()
        self._title_task = None

    def _cancel_memory_task(self) -> None:
        if self._memory_task is not None and not self._memory_task.done():
            self._memory_task.cancel()
        self._memory_task = None

    def _cancel_background_tasks(self) -> None:
        self._cancel_title_task()
        self._cancel_memory_task()

    def _can_distill_memories(self) -> bool:
        if self._agent is None:
            return False
        llm: Any = self.agent._lightweight_llm
        seen: set[int] = set()
        while hasattr(llm, "_inner") and id(llm) not in seen:
            seen.add(id(llm))
            llm = llm._inner
        return type(llm).__name__ != "FakeLLMClient"

    async def _maybe_remember(self, text: str) -> None:
        if len(text.strip()) < 40 or not self._can_distill_memories():
            return
        session_id = self.meta.session_id if self.meta else None
        try:
            raw = await self.agent.distill_memories(text[:4000])
            added = self.agent.absorb_memories(raw or "")
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("memory distillation failed", exc_info=True)
            return
        if session_id and (self.meta is None or self.meta.session_id != session_id):
            return
        if not added:
            return
        self.agent.refresh_context_sources()
        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"remembered {len(added)} project convention(s)",
            )
        )

    def _set_session_title(self, title: str) -> None:
        selected = (title or "").strip().strip('"')[:60]
        if selected and self.meta:
            self.meta.title = selected
            self.agent.v.title = selected
            self.store.save_meta(self.meta)

    async def run_interactive(self) -> int:
        """Line-oriented console loop."""
        await self.start()
        interrupt_count = 0
        try:
            while not self._exit_requested:
                try:
                    line = await self.ui.prompt(self.status_prompt())
                except KeyboardInterrupt:
                    interrupt_count += 1
                    if interrupt_count >= 2:
                        self.ui.render(HostEvent(HostEventKind.STATUS, "exiting"))
                        return 130
                    self.ui.render(HostEvent(HostEventKind.STATUS, "Ctrl-C again to exit"))
                    continue
                if line is None:
                    break
                interrupt_count = 0
                line = line.strip()
                if not line:
                    continue
                try:
                    action = await self.handle_line(line)
                except KeyboardInterrupt:
                    interrupt_count += 1
                    self.cancel_active_turn()
                    self.ui.render(HostEvent(HostEventKind.STATUS, "turn cancelled"))
                    self._persist()
                    if interrupt_count >= 2:
                        return 130
                    continue
                except asyncio.CancelledError:
                    self.ui.render(HostEvent(HostEventKind.STATUS, "turn cancelled"))
                    self._persist()
                    continue
                if action == "exit":
                    break
            return 0
        finally:
            await self.close()

    async def run_tui(self, *, onboarding_required: bool = False) -> int:
        """Full-screen Textual UI. App owns input; host owns turns."""
        try:
            from noah_code.ui.textual_app import NoahCodeApp, TextualUI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Textual is required for the TUI. Install with: uv sync / pip install textual"
            ) from exc

        ui = TextualUI()
        self.ui = ui
        app = NoahCodeApp(self, ui, onboarding_required=onboarding_required)
        try:
            await app.run_async()
            return 0
        finally:
            await self.close()

    async def run_once(self, prompt: str) -> HostResult:
        await self.start()
        try:
            if self.config.auto_approve:

                async def _auto(req):  # noqa: ANN001
                    if req.decision.action == "deny":
                        return ApprovalChoice.REJECT
                    return ApprovalChoice.ONCE

                self.agent.approvals.set_handler(_auto)
            else:

                async def _reject(req):  # noqa: ANN001
                    return ApprovalChoice.REJECT

                self.agent.approvals.set_handler(_reject)

            result = await self._run_user_turn(prompt)
            return result
        finally:
            await self.close()
