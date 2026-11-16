"""Pure-Python host: dispatcher loop, persistence, approvals, slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noah_code import nooa_compat
from noah_code.approvals import ApprovalChoice
from noah_code.commands import config_text, help_text, parse_slash
from noah_code.config import (
    REASONING_EFFORTS,
    NoahCodeConfig,
    save_user_animations,
    save_user_default_model,
    save_user_reasoning_effort,
    save_user_theme,
    user_default_model,
)
from noah_code.custom_commands import CustomCommand, discover_custom_commands
from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEvent, HostEventKind
from noah_code.host_commands import HostCommandsMixin, _command_output
from noah_code.redaction import safe_error_message
from noah_code.sessions import SessionEventRecord, SessionMeta, SessionStore
from noah_code.steer import (
    SAFE_SLASH_WHILE_BUSY,
    STEER_QUEUE_CAP,
    SteerItem,
    SteerQueue,
    expansion_failed,
)
from noah_code.telemetry import AgentTelemetry, setup_agent_telemetry
from noah_code.themes import THEME_NAMES, get_theme
from noah_code.ui.console import ConsoleUI
from noah_code.ui.protocol import HostUI
from noah_code.usage import UsageSnapshot, UsageTracker
from noah_code.workspace import Workspace, open_workspace

if TYPE_CHECKING:
    from noah_code.agent import CodingAgent
    from noah_code.runtime_state import RunRecord, RuntimeStateStore, WorkspaceLease

__all__ = [
    "AgentHost",
    "HostResult",
    "REASONING_EFFORTS",
    "SAFE_SLASH_WHILE_BUSY",
    "THEME_NAMES",
    "_command_output",
    "_friendly_agent_error",
    "_handle_with_overflow_recovery",
    "_is_context_overflow",
    "_load_agent_runtime",
    "_stop_text",
    "config_text",
    "get_theme",
    "help_text",
    "parse_slash",
    "save_user_animations",
    "save_user_default_model",
    "save_user_reasoning_effort",
    "save_user_theme",
    "user_default_model",
]

logger = logging.getLogger(__name__)

_MEMORY_CUE = re.compile(
    r"(?:\bremember\b|\bfrom now on\b|\bgoing forward\b|\bfor this (?:project|repo)\b|"
    r"\bour (?:team|project) (?:uses?|prefers?|requires?)\b|\bwe (?:always|never|use|prefer)\b|"
    r"\b(?:always|never|prefer|must) (?:use|run|keep|avoid)\b|\bconvention\b|"
    r"\bstyle guide\b|\bpackage manager\b|^\s*MEMORY:)",
    re.IGNORECASE | re.MULTILINE,
)


def _friendly_agent_error(exc: Exception) -> str:
    """Turn framework/provider failures into bounded, actionable UI copy."""

    raw = safe_error_message(exc)
    iteration = re.search(r"Generation failed after (\d+) iterations \(max_iterations=(\d+)\)", raw)
    if iteration:
        used, limit = iteration.groups()
        return (
            f"Reached the iteration limit ({used}/{limit} turns). "
            "Continue with a narrower follow-up, or relaunch with "
            "`--max-iterations N` / NOAH_CODE_MAX_ITERATIONS to raise the cap."
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


def _turn_outcome(result: Any) -> tuple[str, str, str | None]:
    """Normalize NOOA typed turn results and legacy test stubs."""

    from nooa.interactive import Done, NeedInput, Waiting

    if isinstance(result, Done):
        return "DONE", result.explanation or "", result.message
    if isinstance(result, NeedInput):
        return "NEED_INPUT", result.reason or result.question or "", result.question
    if isinstance(result, Waiting):
        return "WAIT", result.explanation or "", result.message
    explanation = getattr(result, "explanation", "") or ""
    raw_kind = getattr(result, "kind", None)
    stop_kind = str(getattr(raw_kind, "value", raw_kind) or "").upper()
    return stop_kind, explanation, getattr(result, "message", None)


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


async def _aclose_llm_chain(llm: Any) -> None:
    """Best-effort close of wrapped LLM transports (httpx pools, Codex RPC)."""

    seen: set[int] = set()
    current = llm
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        aclose = getattr(current, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                result = aclose()
                if inspect.isawaitable(result):
                    await result
        current = getattr(current, "_inner", None)


@dataclass
class HostResult:
    exit_code: int
    explanation: str = ""
    session_id: str | None = None
    status: Literal["completed", "needs_input", "failed", "cancelled"] = "completed"
    run_id: str | None = None
    usage: dict[str, int | float] | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _BackgroundTaskOrigin:
    """Identity that a title or memory task is allowed to mutate."""

    agent: Any
    session_id: str
    workspace_root: Path


class AgentHost(HostCommandsMixin):
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
        client_mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.ui: HostUI = ui or ConsoleUI(markdown=config.ui.markdown)
        self.store = store or SessionStore(config.session_dir)
        self._llm = llm
        self._client_mcp_servers: dict[str, dict[str, Any]] = {
            str(key): dict(value)
            for key, value in (client_mcp_servers or {}).items()
            if isinstance(value, dict)
        }
        self._agent: CodingAgent | None = None
        self._storage: Any = None
        self._runtime: RuntimeStateStore | None = None
        self._workspace_lease: WorkspaceLease | None = None
        self._lease_shared = False
        self.meta = session_meta
        self._exit_requested = False
        self._title_task: asyncio.Task[Any] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._budget_guard: Any = None
        self._hooks: Any = None
        self._checkpoints: Any = None
        self.last_checkpoint: dict[str, Any] | None = None
        self.last_result: HostResult | None = None
        self._post_hook_tasks: list[asyncio.Task[Any]] = []
        self._trace_info = "session jsonl"
        self._telemetry = AgentTelemetry()
        self._active_turn: asyncio.Task[Any] | None = None
        self._session_lock = asyncio.Lock()
        self._last_live_state_sync = 0.0
        self._event_unsubs: list[Any] = []
        self._custom_commands: dict[str, CustomCommand] = {}
        self._last_turn_shell_bypass = False
        self._mcp_attached: set[str] = set()
        self._mcp_errors: dict[str, str] = {}
        self._usage = UsageTracker()
        self.steer_queue = SteerQueue()
        self.queue_paused = False
        self._wake_event = asyncio.Event()
        # Bounded poll interval for WAIT turns; re-checks job liveness each pass.
        self._wake_timeout_seconds = 30.0
        self._current_run_id: str | None = None
        self._interrupted_run: RunRecord | None = None
        self._pending_attach_paths: list[Path] = []
        self._active_context_paths: set[Path] = set()
        self._pending_worktree_name = ""
        self._last_agent_message = ""
        self.on_session_changed: Any = None  # optional UI callback

    @property
    def agent(self) -> CodingAgent:
        if self._agent is None:
            raise RuntimeError("host not started")
        return self._agent

    async def _setup_tracing_async(self, session_id: str) -> None:
        await asyncio.to_thread(
            self._telemetry.shutdown, self.config.tracing.export_timeout_millis
        )
        path = (
            Path(self.config.tracing.jsonl_dir).expanduser()
            if self.config.tracing.jsonl_dir
            else self._runtime.session_path / "traces"
            if self._runtime is not None
            else None
        )
        if path is not None:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        model = self.meta.model if self.meta is not None else self.config.model
        self._telemetry = await asyncio.to_thread(
            setup_agent_telemetry,
            self.config.tracing,
            session_id=session_id,
            model=model,
            jsonl_dir=str(path) if path is not None else None,
        )
        self._trace_info = self._telemetry.info

    def _acquire_lease_sync(self, session_id: str, *, shared: bool) -> WorkspaceLease:
        acquire: Any = self.store.acquire_workspace_lease
        try:
            return acquire(session_id, self.workspace, shared=shared)
        except TypeError:
            return acquire(session_id, self.workspace)

    async def _ensure_mode_lease(self, target_mode: str) -> None:
        """Upgrade a shared plan-mode lease before entering mutating build mode."""

        if (
            target_mode != "build"
            or not self.config.reliability.workspace_lease
            or self._workspace_lease is None
            or not self._lease_shared
            or self.meta is None
        ):
            return
        previous = self._workspace_lease
        previous.close()
        self._workspace_lease = None
        try:
            self._workspace_lease = await asyncio.to_thread(
                self._acquire_lease_sync, self.meta.session_id, shared=False
            )
            self._lease_shared = False
        except Exception:
            with contextlib.suppress(Exception):
                self._workspace_lease = await asyncio.to_thread(
                    self._acquire_lease_sync, self.meta.session_id, shared=True
                )
                self._lease_shared = True
            raise

    async def start(self) -> SessionMeta:
        from noah_code.worktree import infer_worktree_name, repo_id_for, worktree_storage_root

        loaded_meta_updated_at = self.meta.updated_at if self.meta is not None else None
        if self.meta is None:
            repo_id = await asyncio.to_thread(repo_id_for, self.workspace.root)
            wt_name = self._pending_worktree_name or await asyncio.to_thread(
                infer_worktree_name,
                self.workspace.root,
                worktree_storage_root(self.config.session_dir),
            )
            self.meta = await asyncio.to_thread(
                self.store.create,
                self.workspace,
                model=self.config.model,
                mode=self.config.mode,
                reasoning_effort=self.config.reasoning_effort,
                repo_id=repo_id,
                worktree_name=wt_name,
            )
            self._pending_worktree_name = ""
        else:
            self.store.verify_workspace(self.meta, self.workspace)
            inferred = await asyncio.to_thread(
                infer_worktree_name,
                self.workspace.root,
                worktree_storage_root(self.config.session_dir),
            )
            repo = await asyncio.to_thread(repo_id_for, self.workspace.root)
            dirty = False
            if repo and not self.meta.repo_id:
                self.meta.repo_id = repo
                dirty = True
            if inferred and not self.meta.worktree_name:
                self.meta.worktree_name = inferred
                dirty = True
            if dirty:
                await asyncio.to_thread(self.store.save_meta, self.meta)

        if self.config.reliability.workspace_lease and self._workspace_lease is None:
            shared = self.meta.mode == "plan" and not self.config.yolo
            self._workspace_lease = await asyncio.to_thread(
                self._acquire_lease_sync,
                self.meta.session_id,
                shared=shared,
            )
            self._lease_shared = shared
        try:
            return await self._start_with_lease(loaded_meta_updated_at)
        except BaseException:
            self._teardown_event_bridge()
            if self._agent is not None:
                with contextlib.suppress(Exception):
                    await _aclose_llm_chain(self._agent._llm)
                with contextlib.suppress(Exception):
                    await self._agent.close_tools()
                self._agent = None
            if self._storage is not None:
                with contextlib.suppress(Exception):
                    self._storage.close()
                self._storage = None
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._telemetry.shutdown)
            self._close_runtime()
            self._release_workspace_lease()
            raise

    def _close_runtime(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            close_fn = getattr(runtime, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()

    def _session_blob_dir(self) -> Path | None:
        if self.meta is None:
            return None
        with contextlib.suppress(Exception):
            return self.store._session_path(self.meta.session_id) / "blobs"
        return None

    async def _start_with_lease(self, loaded_meta_updated_at: float | None) -> SessionMeta:
        assert self.meta is not None
        self._runtime = await asyncio.to_thread(
            self.store.open_runtime,
            self.meta.session_id,
            max_events=self.config.reliability.max_runtime_events,
        )
        # Cold NOOA/LiteLLM import is independent of durable recovery. Start it
        # now so provider initialization overlaps SQLite and process cleanup.
        runtime_loader = asyncio.create_task(
            asyncio.to_thread(_load_agent_runtime),
            name="noah-runtime-loader",
        )
        try:
            checkpoint = await asyncio.to_thread(self._runtime.load_checkpoint)
            usage_state = await asyncio.to_thread(
                self._runtime.get_state,
                "live:usage",
                checkpoint.get("usage"),
            )
            if isinstance(usage_state, dict):
                self._usage.load_dict(usage_state)
            persisted_meta = checkpoint.get("meta")
            if (
                isinstance(persisted_meta, dict)
                and persisted_meta.get("session_id") == self.meta.session_id
            ):
                recovered_meta = SessionMeta.from_json(json.dumps(persisted_meta))
                if (
                    loaded_meta_updated_at is not None
                    and loaded_meta_updated_at > recovered_meta.updated_at
                ):
                    recovered_meta.mode = self.meta.mode
                    recovered_meta.model = self.meta.model
                    recovered_meta.reasoning_effort = self.meta.reasoning_effort
                self.meta = recovered_meta
            if self.meta.mode == "build":
                await self._ensure_mode_lease("build")
            recovered_files = await asyncio.to_thread(self._runtime.recover_file_operations)
            recovered_jobs = await asyncio.to_thread(
                self._runtime.recover_orphan_jobs,
                grace_seconds=self.config.processes.stop_grace_seconds,
            )
            interrupted_interactions = await asyncio.to_thread(
                self._runtime.interrupt_pending_interactions
            )
            self._interrupted_run = await asyncio.to_thread(
                self._runtime.latest_incomplete_run
            )
            agent_class, get_llm_client = await runtime_loader
        except BaseException:
            runtime_loader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runtime_loader
            raise
        if recovered_files:
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"recovered {len(recovered_files)} interrupted file operation(s); "
                    f"displaced contents saved in {self._runtime.recovery_dir}",
                )
            )
        if recovered_jobs:
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"cleaned {len(recovered_jobs)} orphaned background job(s)",
                )
            )
        if interrupted_interactions:
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"expired {interrupted_interactions} interrupted interaction(s)",
                )
            )
        from noah_code.llm import reasoning_overrides

        client_kwargs: dict[str, Any] = {}
        if self._llm is None:
            client_kwargs.update(self.config.sampling.overrides())
        llm = self._llm
        if llm is None:
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

        retry_config = self.config.reliability.retries
        fallback_clients = await self._build_fallback_clients(
            get_llm_client,
            self.meta.model,
            effort=self.meta.reasoning_effort,
            client_kwargs=client_kwargs,
        )

        shared_route = lightweight_llm is llm
        llm = self._wrap_resilient(llm, retry_config, fallback_clients)
        if shared_route:
            lightweight_llm = llm
        else:
            lightweight_llm = self._wrap_resilient(lightweight_llm, retry_config)

        from noah_code.budget import SharedBudgetLLM, _PrefixObserverOnly, wrap_with_budget
        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)
        lightweight_llm = wrap_conversational_replies(lightweight_llm)

        llm, self._budget_guard = wrap_with_budget(
            llm, self.config.budget, prefix_observer=self._usage
        )
        budget_state = await asyncio.to_thread(
            self._runtime.get_state,
            "live:budget",
            checkpoint.get("budget"),
        )
        if isinstance(budget_state, dict):
            self._budget_guard.load_state(budget_state)
        self._budget_guard.pause()
        if isinstance(llm, SharedBudgetLLM):
            lightweight_llm = SharedBudgetLLM(
                lightweight_llm, self._budget_guard, prefix_observer=self._usage
            )
        elif isinstance(llm, _PrefixObserverOnly):
            lightweight_llm = _PrefixObserverOnly(lightweight_llm, self._usage)

        meta = self.meta
        assert meta is not None
        await self._setup_tracing_async(meta.session_id)
        # SQLiteStorageManager binds its sqlite3 connection to the creating thread,
        # so open_storage and CodingAgent initialization must stay on the event-loop thread.
        self._storage = self.store.open_storage(meta.session_id)
        agent = agent_class(
            self.workspace,
            self.config,
            llm=llm,
            lightweight_llm=lightweight_llm,
            storage=self._storage,
            runtime=self._runtime,
            budget_guard=self._budget_guard,
            usage_tracker=self._usage,
            cache_namespace=f"noah:{meta.session_id}",
        )
        summarizers = nooa_compat.summarizers(agent)
        restored = self._storage.restore_latest_snapshot(agent)
        agent._summarizers = summarizers
        blob_dir = self._session_blob_dir()
        journal_obj: Any = agent.journal
        if blob_dir is not None and hasattr(journal_obj, "_blob_dir"):
            journal_obj._blob_dir = blob_dir
        if restored:
            agent.refresh_context_sources()
            agent._engine.load_session_rules(meta.permission_rules)
            journal_data = checkpoint.get("journal")
            if not isinstance(journal_data, dict):
                journal_data = await asyncio.to_thread(
                    self.store.load_journal, meta.session_id
                )
            try:
                journal_obj.load_dict(journal_data or meta.journal, blob_dir=blob_dir)
            except TypeError:
                journal_obj.load_dict(journal_data or meta.journal)
            todos = checkpoint.get("todos", meta.todos)
            if isinstance(todos, dict) and todos:
                agent.todos.from_dict(todos)
        mode_lit: Literal["build", "plan"] = "plan" if meta.mode == "plan" else "build"
        agent.set_mode(mode_lit)
        agent.v.mode = mode_lit
        agent.v.model = _clean_control_chars(meta.model)

        agent._approvals.set_handler(self.ui.ask_approval)
        agent.ask.set_handler(self.ui.ask_questions)
        agent._render_message = self._on_agent_message
        agent.ws.set_shell_chunk_handler(self._on_shell_chunk)
        agent.processes.set_lifecycle_handler(self._on_process_lifecycle)
        if getattr(agent, "task", None) is not None:
            agent.task.set_lifecycle_handler(self._on_subagent_lifecycle)

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
        self._checkpoints = await asyncio.to_thread(
            CheckpointManager,
            self.workspace.root,
            session_id,
            max_per_session=self.config.checkpoints.max_per_session,
        )
        if self.config.checkpoints.capture_before_mutation:

            async def _checkpoint_shell(command: str) -> None:
                label = "before shell · " + " ".join(command.split())[:60]
                await self._capture_checkpoint(label)

            agent.ws.set_mutation_checkpoint_handler(_checkpoint_shell)
        self._teardown_event_bridge()
        self._event_unsubs = install_event_bridge(
            agent,
            self._emit_with_hooks,
            self._usage,
            self._budget_guard,
            self._telemetry,
        )

        self._custom_commands = await asyncio.to_thread(
            discover_custom_commands, self.workspace.root
        )

        self._agent = agent

        assert self._runtime is not None
        queue_state = await asyncio.to_thread(self._runtime.get_state, "input_queue", {})
        self.queue_paused = bool(queue_state.get("paused", False))
        self._pending_attach_paths = [Path(path) for path in queue_state.get("attachments", [])]
        pending = await asyncio.to_thread(self._runtime.pending_inbox)
        order = {sequence: index for index, sequence in enumerate(queue_state.get("order", []))}
        for item in sorted(pending, key=lambda item: order.get(item.sequence, len(order))):
            dropped = self.steer_queue.push_with_dropped(
                item.text,
                attach_paths=[Path(path) for path in item.attach_paths],
                sequence=item.sequence,
            )
            if dropped is not None and dropped.sequence is not None:
                self._runtime.acknowledge_inbox(dropped.sequence, dropped=True)

        self._mcp_attached = set()
        self._mcp_errors = {}

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
                    extra_servers=self._client_mcp_servers or None,
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
        await self._run_lifecycle("session_start")
        return self.meta

    async def _run_lifecycle(self, event: str, **metadata: Any) -> None:
        hooks = self._hooks
        if hooks is None:
            return
        payload = {
            "session_id": self.meta.session_id if self.meta else "",
            "workspace": str(self.workspace.root),
            **metadata,
        }
        try:
            failures = await hooks.run_lifecycle(event, payload)
        except Exception as exc:
            failures = [safe_error_message(exc)]
        for failure in failures:
            self.ui.render(
                HostEvent(HostEventKind.STATUS, f"{event} hook: {safe_error_message(failure)}")
            )

    def _emit_with_hooks(self, event: HostEvent) -> None:
        """Render an event; schedule post-tool hooks for tool completions."""

        if self._runtime is not None and event.kind in {
            HostEventKind.TOOL_START,
            HostEventKind.TOOL_FINISH,
            HostEventKind.ERROR,
            HostEventKind.STOP,
        }:
            self._runtime.event(
                f"host.{event.kind.value}",
                {
                    "text": safe_error_message(event.text, limit=500),
                    "meta_keys": sorted(str(key) for key in event.meta),
                    "run_id": self._current_run_id,
                },
            )
            now = time.monotonic()
            terminal_event = event.kind in {HostEventKind.ERROR, HostEventKind.STOP}
            if terminal_event or now - self._last_live_state_sync >= 0.5:
                self._last_live_state_sync = now
                self._runtime.set_state("live:usage", self.usage_snapshot().to_dict())
                if self._budget_guard is not None:
                    self._runtime.set_state("live:budget", self._budget_guard.status())
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
            self.ui.render(
                HostEvent(HostEventKind.STATUS, f"hook warning: {safe_error_message(failure)}")
            )

    async def _flush_post_hooks(self, *, cancel: bool = False) -> None:
        tasks, self._post_hook_tasks = self._post_hook_tasks, []
        if cancel:
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _sync_budget_cost(self) -> None:
        guard = getattr(self, "_budget_guard", None)
        if guard is None or not guard.active:
            return
        guard.sync_cost_usd(self.usage_snapshot().cost_usd)

    async def _capture_checkpoint(self, label: str, *, skip_if_unchanged: bool = False) -> None:
        if getattr(self, "_skip_unchanged_checkpoint", False):
            skip_if_unchanged = True
            self._skip_unchanged_checkpoint = False
        manager = self._checkpoints
        if manager is None or not self.config.checkpoints.enabled:
            return
        previous_ref = self.last_checkpoint.get("ref") if isinstance(self.last_checkpoint, dict) else None
        try:
            try:
                snapshot = await asyncio.to_thread(
                    manager.capture, label, skip_if_unchanged=skip_if_unchanged
                )
            except TypeError:
                snapshot = await asyncio.to_thread(manager.capture, label)
        except Exception as exc:  # noqa: BLE001 - checkpoints are best-effort
            logger.debug("checkpoint capture failed", exc_info=True)
            self.ui.render(HostEvent(HostEventKind.STATUS, f"checkpoint failed: {exc}"))
            return
        if snapshot is not None:
            self.last_checkpoint = snapshot
            if skip_if_unchanged and previous_ref and snapshot.get("ref") == previous_ref:
                return
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    "◆",
                    meta={
                        "kind": "checkpoint",
                        "label": "Checkpoint saved",
                        "checkpoint": snapshot,
                    },
                )
            )

    def _teardown_event_bridge(self) -> None:
        for unsub in self._event_unsubs:
            with contextlib.suppress(Exception):
                unsub()
        self._event_unsubs = []

    def _on_shell_chunk(self, stream: str, text: str) -> None:
        self.ui.render(HostEvent(HostEventKind.SHELL_CHUNK, text, meta={"stream": stream}))

    def _on_process_lifecycle(
        self, job_id: str, name: str, message: str, terminal: bool = False
    ) -> None:
        """Push lifecycle changes to the UI without streaming logs into model context."""

        record: dict[str, Any] = {}
        if self._agent is not None:
            with contextlib.suppress(Exception):
                record = next(
                    (
                        item
                        for item in self.agent.processes.snapshot()
                        if item.get("id") == job_id
                    ),
                    {},
                )
        kind = str(record.get("kind", "process"))
        label = "terminal" if kind == "terminal" else "background job"
        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"{label} {name} · {message}",
                meta={
                    "kind": "background_job",
                    "job_id": job_id,
                    "job_kind": kind,
                    "name": name,
                    "state": record.get("state", "completed" if terminal else "running"),
                    "terminal": terminal,
                },
            )
        )
        if terminal:
            self._wake_event.set()

    def _on_subagent_lifecycle(self, activity: dict[str, Any]) -> None:
        """Expose delegated work as structured UI events and wake WAIT turns."""

        state = str(activity.get("state", "running"))
        agent = str(activity.get("agent", "agent"))
        prompt = " ".join(str(activity.get("prompt", "")).split())[:80]
        duration = float(activity.get("duration", 0.0) or 0.0)
        terminal = state in {"completed", "needs_input", "failed", "cancelled"}
        detail = f" · {prompt}" if prompt else ""
        elapsed = f" · {duration:.1f}s" if terminal else ""
        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"agent {agent} · {state}{elapsed}{detail}",
                meta={"kind": "subagent", **activity},
            )
        )
        if terminal:
            self._wake_event.set()

    def _on_agent_message(self, text: str, **_kwargs: Any) -> None:
        self._last_agent_message = str(text)
        self.ui.render(HostEvent(HostEventKind.MESSAGE, text))

    def _on_model_stream(self, event: Any) -> None:
        self.ui.render(
            HostEvent(
                HostEventKind.MODEL_STREAM,
                event.text,
                meta={
                    "phase": event.kind,
                    "call_id": event.call_id,
                    "model": event.model,
                    "attempt": event.attempt,
                    "error_type": event.error_type,
                },
            )
        )

    def _snapshot_meta_from_agent(self) -> SessionMeta | None:
        """Synchronize meta fields on the owning thread and return an isolated copy."""

        if self._agent is None or self.meta is None:
            return None
        self.meta.mode = self._agent.mode
        self.meta.model = getattr(self._agent.v, "model", self.meta.model) or self.meta.model
        self.meta.permission_rules = self._agent.engine.snapshot_session_rules()
        self.meta.journal = {}
        self.meta.todos = _json_safe(self._agent.todos.to_dict())
        with contextlib.suppress(Exception):
            title = getattr(self._agent.v, "title", None)
            if title:
                self.meta.title = str(title)
        return replace(
            self.meta,
            permission_rules=list(self.meta.permission_rules),
            todos=dict(self.meta.todos),
            journal={},
        )

    def _serialize_journal_data(self) -> dict:
        if self._agent is None:
            return {}
        blob_dir = self._session_blob_dir()
        journal_obj: Any = self._agent.journal
        try:
            return journal_obj.to_dict(blob_dir=blob_dir)
        except TypeError:
            return journal_obj.to_dict()

    def _persist_state(self) -> dict:
        """Serialize agent-owned state for persistence."""

        if self._agent is None or self.meta is None:
            return {}
        self._snapshot_meta_from_agent()
        return self._serialize_journal_data()

    def _write_persist_files(
        self, journal_data: dict, meta_snapshot: SessionMeta | None = None
    ) -> None:
        """Write meta.json and the undo sidecar. File/SQLite I/O only."""

        target_meta = meta_snapshot or self.meta
        if target_meta is None:
            return
        if self._runtime is not None:
            usage = self.usage_snapshot()
            budget = self._budget_guard.status() if self._budget_guard is not None else {}
            self._runtime.save_checkpoint(
                {
                    "meta": json.loads(target_meta.to_json()),
                    "journal": journal_data,
                    "todos": target_meta.todos,
                    "usage": usage.to_dict(),
                    "budget": budget,
                }
            )
        self.store.save_journal(target_meta.session_id, journal_data)
        self.store.save_meta(target_meta)
        if self.meta is not None and self.meta is not target_meta:
            self.meta.updated_at = target_meta.updated_at

    def _persist(self) -> None:
        if self._agent is None or self.meta is None or self._storage is None:
            return
        meta_snapshot = self._snapshot_meta_from_agent()
        journal_data = self._serialize_journal_data()
        self._write_persist_files(journal_data, meta_snapshot)
        self._storage.save_snapshot(self._agent)

    async def _persist_async(self) -> None:
        """Serialize and write session files off the UI event loop."""

        if self._agent is None or self.meta is None or self._storage is None:
            return
        meta_snapshot = self._snapshot_meta_from_agent()
        journal_data = await asyncio.to_thread(self._serialize_journal_data)
        await asyncio.to_thread(self._write_persist_files, journal_data, meta_snapshot)
        if self._agent is not None and self._storage is not None:
            self._storage.save_snapshot(self._agent)

    async def close(self) -> None:
        try:
            await self._cancel_background_tasks()
            await self._flush_post_hooks()
            await self._persist_async()
            if self._agent is not None:
                await self._run_lifecycle("session_end")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await self._cancel_background_tasks()
            self._teardown_event_bridge()
            if self._agent is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await _aclose_llm_chain(self._agent._llm)
                    if self._agent._lightweight_llm is not self._agent._llm:
                        await _aclose_llm_chain(self._agent._lightweight_llm)
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await self._agent.close_tools()
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            if self._agent is not None and self.config.tracing.enabled:
                try:
                    from nooa.tracing import flush_traces

                    flush_traces()
                except Exception:  # noqa: BLE001
                    logger.debug("trace flush failed", exc_info=True)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.to_thread(self._telemetry.shutdown)
            self._agent = None
            self._hooks = None
            self._close_runtime()
            self._release_workspace_lease()

    def _release_workspace_lease(self) -> None:
        lease, self._workspace_lease = self._workspace_lease, None
        self._lease_shared = False
        if lease is not None:
            lease.close()

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
        """List Git changes without waiting for individual patches or language servers."""
        return await self.agent.git.review(eager=False)

    async def diff_review_file(self, item: Any) -> None:
        await self.agent.git.review_file(item)

    async def diff_diagnostics(self, item: Any) -> None:
        """Editor diagnostics describe the current worktree, not executed test results."""
        diagnostics = await self.agent.lsp.diagnostics_for_paths([item.path])
        raw = diagnostics.get(item.path, "unavailable")
        if raw.startswith("ok —"):
            item.diagnostics = "clean"
        elif raw.startswith("unavailable") or raw == "not supported":
            item.diagnostics = raw
        else:
            issues = len([line for line in raw.splitlines() if line.strip()])
            item.diagnostics = f"{issues} issue{'s' if issues != 1 else ''}"

    def _require_review_idle(self) -> None:
        if self._turn_running():
            raise RuntimeError("Stop the active turn before reverting or undoing changes")

    async def revert_diff_file(
        self, path: str, scope: str, *, expected_revision: str | None = None
    ) -> str:
        """Revert an explicitly confirmed review item as its own journal turn."""

        self._require_review_idle()
        async with self._session_lock:
            self._active_turn = asyncio.current_task()
            self.ui.set_busy(True)
            self.agent.journal.begin_turn()
            try:
                result = await self.agent.git.revert(
                    path, scope, expected_revision=expected_revision
                )
            finally:
                self.agent.journal.end_turn()
                try:
                    await self._persist_async()
                finally:
                    self._active_turn = None
                    self.ui.set_busy(False)
            return result

    def _undo_last_turn_state(self, *, use_checkpoint: bool = False) -> str:
        """Undo the latest reversible turn or explicitly restore the latest Git checkpoint."""

        if use_checkpoint:
            manager = self._checkpoints
            if manager is None:
                raise RuntimeError("no Git checkpoint manager available for this session")
            entries = manager.list()
            if not entries:
                raise RuntimeError("no Git checkpoints recorded for this session")
            target = entries[-2] if len(entries) >= 2 else entries[-1]
            manager.restore(target["ref"])
            self._last_turn_shell_bypass = False
            return f"restored Git checkpoint {target['ref']} ({target['commit'][:12]})"

        turn = self.agent.journal.latest_turn()
        if turn:
            self.agent.journal.capture_post_bytes_before_undo(turn)
        try:
            undone = self.agent.journal.undo()
        except RuntimeError as exc:
            if (
                "shell mutations outside the file journal" in str(exc)
                and self._checkpoints is not None
                and self._checkpoints.list()
            ):
                raise RuntimeError(
                    f"{exc}; run '/undo --checkpoint' to restore the pre-command Git checkpoint"
                ) from exc
            raise
        return f"undid turn {undone.turn_id[:8]} ({len(undone.mutations)} files)"

    async def undo_last_turn_async(self, *, use_checkpoint: bool = False) -> str:
        self._require_review_idle()
        async with self._session_lock:
            self._active_turn = asyncio.current_task()
            self.ui.set_busy(True)
            undo_task = asyncio.create_task(
                asyncio.to_thread(self._undo_last_turn_state, use_checkpoint=use_checkpoint)
            )
            try:
                try:
                    status = await asyncio.shield(undo_task)
                except asyncio.CancelledError:
                    await undo_task
                    await self._persist_async()
                    raise
                await self._persist_async()
                return status
            finally:
                self._active_turn = None
                self.ui.set_busy(False)

    async def redo_last_turn_async(self) -> str:
        self._require_review_idle()
        async with self._session_lock:
            self._active_turn = asyncio.current_task()
            self.ui.set_busy(True)
            redo_task = asyncio.create_task(asyncio.to_thread(self.agent.journal.redo))
            try:
                try:
                    redone = await asyncio.shield(redo_task)
                except asyncio.CancelledError:
                    await redo_task
                    await self._persist_async()
                    raise
                await self._persist_async()
                return f"redid turn {redone.turn_id[:8]} ({len(redone.mutations)} files)"
            finally:
                self._active_turn = None
                self.ui.set_busy(False)

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
        async with self._session_lock:
            self._require_idle_turn()
            validated_workspace = (
                open_workspace(workspace.root) if workspace is not None else None
            )
            self._clear_steer_state()
            self._active_context_paths.clear()
            await self._cancel_background_tasks()
            await self._persist_async()
            if self._agent is not None:
                await self._run_lifecycle("session_end")
            if validated_workspace is not None:
                self.workspace = validated_workspace
            self._pending_worktree_name = worktree_name
            self._teardown_event_bridge()
            if self._agent is not None:
                with contextlib.suppress(Exception):
                    await _aclose_llm_chain(self._agent._llm)
                with contextlib.suppress(Exception):
                    await self._agent.close_tools()
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            self._agent = None
            self._close_runtime()
            self._release_workspace_lease()
            self._usage = UsageTracker()
            self.meta = None
            self._exit_requested = False
            return await self.start()

    async def switch_session(self, session_id: str) -> SessionMeta:
        """Persist current session and resume another."""
        self._require_idle_turn()
        async with self._session_lock:
            self._require_idle_turn()
            meta = await asyncio.to_thread(self.store.load_meta, session_id)
            rebound = await asyncio.to_thread(
                self.store.workspace_for_resume, meta, self.workspace
            )
            self._clear_steer_state()
            self._active_context_paths.clear()
            await self._cancel_background_tasks()
            await self._persist_async()
            if self._agent is not None:
                await self._run_lifecycle("session_end")
            self._teardown_event_bridge()
            if self._agent is not None:
                with contextlib.suppress(Exception):
                    await _aclose_llm_chain(self._agent._llm)
                with contextlib.suppress(Exception):
                    await self._agent.close_tools()
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            self._agent = None
            self._close_runtime()
            self._release_workspace_lease()
            self._usage = UsageTracker()
            self.workspace = rebound
            self.meta = meta
            self._exit_requested = False
            return await self.start()

    def list_session_metas(self) -> list[SessionMeta]:
        return self.store.list_sessions(self.workspace)

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

    def codex_login(self) -> Any:
        """Open a cancellable Codex-managed account login ceremony."""
        from noah_code.codex_account import CodexLogin

        return CodexLogin()

    async def configure_permission_mode(self, mode: str) -> str:
        """Save the onboarding choice before creating the agent or its permission engine."""
        from noah_code import config as _config_mod
        from noah_code.permission_modes import permission_mode_flags

        if self._agent is not None:
            raise RuntimeError("Choose permissions before starting a session")
        flags = permission_mode_flags(mode)
        path = await asyncio.to_thread(_config_mod.save_user_permission_mode, mode)
        self.config.auto_approve = flags["auto_approve"]
        self.config.yolo = flags["yolo"]
        return f"Permission mode set to {mode}; default saved in {path}"

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
            provider_infos = await asyncio.to_thread(list_providers, selected_model)
            provider_info = next(info for info in provider_infos if info.key == provider)
            credential_hint = provider_info.credential_hint
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
            await asyncio.to_thread(self.store.save_meta, self.meta)
        default_path = await asyncio.to_thread(save_user_default_model, selected_model)
        if reasoning_effort is not None:
            await asyncio.to_thread(save_user_reasoning_effort, selected_effort)
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

        servers, sources = await asyncio.to_thread(
            load_mcp_servers,
            self.workspace.root,
            self.config,
            extra_servers=self._client_mcp_servers or None,
        )
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

        import shlex

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
        """Stop the current turn, preserving follow-ups for explicit resumption."""
        self.queue_paused = True
        try:
            self._save_input_queue()
        except Exception:
            logger.warning("Could not persist the paused queue", exc_info=True)
        if self._agent is not None:
            self._agent.approvals.cancel_all()
        task = self._active_turn
        if task is not None and not task.done():
            task.cancel()

    def _turn_running(self) -> bool:
        task = self._active_turn
        return task is not None and not task.done()

    def _clear_steer_state(self, *, drop_durable: bool = False) -> None:
        queued = self.steer_queue.drain()
        if drop_durable and self._runtime is not None:
            for item in queued:
                if item.sequence is not None:
                    self._runtime.acknowledge_inbox(item.sequence, dropped=True)
        self._pending_attach_paths.clear()
        self.queue_paused = False
        if drop_durable:
            self._save_input_queue()

    def _save_input_queue(self) -> None:
        if self._runtime is not None:
            self._runtime.set_state(
                "input_queue",
                {
                    "paused": self.queue_paused,
                    "attachments": [str(path) for path in self._pending_attach_paths],
                    "order": [item.sequence for item in self.steer_queue.items()],
                },
            )

    def attach_context_path(self, raw_path: str | Path) -> Path:
        """Validate and queue a file or temporary clipboard image for the next prompt."""

        from noah_code.composer import validate_attachment

        path = validate_attachment(
            str(raw_path),
            self.workspace.root,
            engine=getattr(self._agent, "engine", None),
            max_file_bytes=self.config.max_file_bytes,
        )
        if path not in self._pending_attach_paths:
            self._pending_attach_paths.append(path)
            self._save_input_queue()
        return path

    def discard_queued_input(self) -> None:
        """Explicitly discard waiting prompts and pending attachments."""
        self._clear_steer_state(drop_durable=True)

    def resume_queued_input(self) -> None:
        self.queue_paused = False
        self._save_input_queue()
        self._wake_event.set()

    def take_pending_attaches(self) -> list[Path]:
        paths, self._pending_attach_paths = self._pending_attach_paths, []
        self._save_input_queue()
        return paths

    def pending_attach_paths(self) -> tuple[Path, ...]:
        """Return files waiting to be attached to the next queued prompt."""

        return tuple(self._pending_attach_paths)

    def active_context_paths(self) -> tuple[Path, ...]:
        """Return attachment paths already delivered in the active session."""

        return tuple(sorted(self._active_context_paths))

    def remove_pending_attach(self, index: int) -> Path | None:
        if index < 0 or index >= len(self._pending_attach_paths):
            return None
        path = self._pending_attach_paths.pop(index)
        self._save_input_queue()
        return path

    def remove_queued_steer(self, index: int) -> bool:
        item = self.steer_queue.remove(index)
        if item is None:
            return False
        if self._runtime is not None and item.sequence is not None:
            self._runtime.acknowledge_inbox(item.sequence, dropped=True)
        self._save_input_queue()
        return True

    def recall_queued_steer(self, index: int | None = None) -> SteerItem | None:
        """Move a waiting prompt and its attachments back to the composer."""

        item = self.steer_queue.pop_last() if index is None else self.steer_queue.remove(index)
        if item is None:
            return None
        if self._runtime is not None and item.sequence is not None:
            self._runtime.acknowledge_inbox(item.sequence, dropped=True)
        for path in item.attach_paths:
            if path not in self._pending_attach_paths:
                self._pending_attach_paths.append(path)
        self._save_input_queue()
        return item

    def move_queued_steer(self, index: int, delta: int) -> bool:
        moved = self.steer_queue.move(index, delta)
        if moved:
            self._save_input_queue()
        return moved

    def enqueue_steer(self, text: str, attach_paths: list[Path] | None = None) -> bool:
        """Queue a follow-up for the current turn. Returns True if oldest dropped."""

        paths = list(attach_paths or []) + self._pending_attach_paths
        items = self.steer_queue.items()
        sequence = (
            self._runtime.enqueue_inbox(
                text,
                paths or None,
                input_state={
                    "paused": self.queue_paused,
                    "attachments": [],
                    "order": [item.sequence for item in items[-(STEER_QUEUE_CAP - 1) :]],
                },
                discard_sequence=items[0].sequence if len(items) >= STEER_QUEUE_CAP else None,
            )
            if self._runtime is not None
            else None
        )
        self._pending_attach_paths.clear()
        dropped_item = self.steer_queue.push_with_dropped(
            text,
            attach_paths=paths or None,
            sequence=sequence,
        )
        if dropped_item is not None:
            self.ui.render(HostEvent(HostEventKind.STATUS, "steer dropped oldest"))
        if not self.queue_paused:
            self._wake_event.set()
        self.ui.set_status(self.status_prompt())
        return dropped_item is not None

    async def submit_prompt(self, text: str) -> HostResult | None:
        """Submit text without interpreting commands that change session identity."""
        self._require_idle_turn()
        async with self._session_lock:
            self._require_idle_turn()
            line = text
            self.ui.set_busy(True)
            self._active_turn = asyncio.current_task()
            try:
                skill_prompt = await self._activate_explicit_skill(line)
                if skill_prompt is None:
                    return HostResult(
                        1,
                        "skill activation failed",
                        self.meta.session_id if self.meta else None,
                        "failed",
                    )
                line = skill_prompt
                continuable = (
                    await asyncio.to_thread(self._runtime.latest_incomplete_run)
                    if self._runtime is not None
                    else None
                )
                run_id = (
                    continuable.run_id
                    if continuable and continuable.state == "waiting_user"
                    else None
                )
                attach_paths = self.take_pending_attaches()
                if run_id is None:
                    return await self._run_user_turn(line, attach_paths=attach_paths)
                return await self._run_user_turn(line, run_id=run_id, attach_paths=attach_paths)
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

    def _record_retry(self, payload: dict[str, Any]) -> None:
        """Emit retry telemetry to the runtime when it exists."""

        if self._runtime is not None:
            self._runtime.event("llm.retry", payload)
        self._telemetry.retry(payload)

    def _wrap_resilient(
        self,
        client: Any,
        retry_config: Any,
        fallbacks: Any = (),
    ) -> Any:
        """Apply the provider-retry/failover wrapper; used by start() and live switches."""

        from noah_code.llm import ResilientLLM

        return ResilientLLM(client, retry_config, fallbacks=fallbacks, on_retry=self._record_retry)

    async def _build_fallback_clients(
        self,
        get_llm_client: Any,
        model: str,
        *,
        effort: str,
        client_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Build fallback-route clients as configured, skipping the primary model."""

        from noah_code.llm import reasoning_overrides

        names = [
            name
            for name in self.config.reliability.retries.fallback_models
            if name and name != model
        ]
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        get_llm_client,
                        name,
                        **reasoning_overrides(effort),
                        **client_kwargs,
                    )
                    for name in names
                )
            )
        )

    def _apply_runtime_llm_wrappers(self, llm: Any) -> Any:
        """Re-apply conversational and session-budget wrappers after a client swap."""

        from noah_code.budget import SharedBudgetLLM, _PrefixObserverOnly
        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)

        guard = getattr(self, "_budget_guard", None)
        if guard is not None and guard.active:
            llm = SharedBudgetLLM(llm, guard, prefix_observer=self._usage)
        elif guard is not None:
            llm = _PrefixObserverOnly(llm, self._usage)
        return llm

    async def _switch_model(self, model: str, *, reasoning_effort: str | None = None) -> None:
        from noah_code.llm import get_llm_client, reasoning_overrides

        effort = reasoning_effort or (
            self.meta.reasoning_effort if self.meta else self.config.reasoning_effort
        )
        client_kwargs = self.config.sampling.overrides()
        llm = await asyncio.to_thread(
            get_llm_client,
            model,
            **reasoning_overrides(effort),
            **client_kwargs,
        )
        fallback_clients = await self._build_fallback_clients(
            get_llm_client,
            model,
            effort=effort,
            client_kwargs=client_kwargs,
        )
        llm = self._apply_runtime_llm_wrappers(
            self._wrap_resilient(llm, self.config.reliability.retries, fallback_clients)
        )
        previous_llm = self.agent._llm
        self.agent.set_main_llm(
            llm,
            lightweight_follows_main=not bool(self.config.lightweight_model),
        )
        if previous_llm is not llm:
            await _aclose_llm_chain(previous_llm)
        self._telemetry.model = model
        if self.meta:
            self.meta.model = model
            self.meta.reasoning_effort = effort
            self.agent.v.model = _clean_control_chars(model)
            await asyncio.to_thread(self.store.save_meta, self.meta)
        self.ui.render(
            HostEvent(HostEventKind.STATUS, f"model set to {model} · reasoning={effort}")
        )
        self.ui.set_status(self.status_prompt())

    def _expand_user_text(self, text: str, attach_paths: list[Path] | None) -> Any:
        from noah_code.composer import expand_turn

        return expand_turn(
            text,
            self.workspace.root,
            attach_paths=attach_paths,
            engine=getattr(self._agent, "engine", None),
            max_file_bytes=self.config.max_file_bytes,
        )

    def _deliver_expanded(self, agent: Any, expanded: Any) -> None:
        self._active_context_paths.update(Path(path).resolve() for path in expanded.paths)
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
        """Synchronously pop and apply the next non-interactive follow-up item."""

        if self.queue_paused:
            return False
        while True:
            item = self.steer_queue.pop()
            if item is None:
                return False
            expanded = self._expand_user_text(item.text, list(item.attach_paths))
            if expansion_failed(item, expanded):
                if self._runtime is not None and item.sequence is not None:
                    self._runtime.acknowledge_inbox(item.sequence, dropped=True)
                self.ui.render(HostEvent(HostEventKind.STATUS, "steer dropped · could not expand"))
                continue
            self._deliver_expanded(agent, expanded)
            if self._runtime is not None and item.sequence is not None:
                self._runtime.acknowledge_inbox(item.sequence)
            preview = " ".join(item.text.split())[:60]
            self.ui.render(HostEvent(HostEventKind.STATUS, f"steer applied · {preview}"))
            self.ui.set_status(self.status_prompt())
            return True

    async def _apply_next_steer_async(self, agent: Any) -> bool:
        """Pop until one follow-up (including explicit $skill activations) is queued."""

        if self.queue_paused:
            return False
        while True:
            item = self.steer_queue.pop()
            if item is None:
                return False
            skill_prompt = await self._activate_explicit_skill(item.text)
            if skill_prompt is None:
                if self._runtime is not None and item.sequence is not None:
                    self._runtime.acknowledge_inbox(item.sequence, dropped=True)
                self.ui.render(
                    HostEvent(HostEventKind.STATUS, "steer dropped · skill activation failed")
                )
                continue
            expanded = await asyncio.to_thread(
                self._expand_user_text, skill_prompt, list(item.attach_paths)
            )
            check_item = SteerItem(skill_prompt, item.attach_paths, item.sequence)
            if expansion_failed(check_item, expanded):
                if self._runtime is not None and item.sequence is not None:
                    self._runtime.acknowledge_inbox(item.sequence, dropped=True)
                self.ui.render(HostEvent(HostEventKind.STATUS, "steer dropped · could not expand"))
                continue
            self._deliver_expanded(agent, expanded)
            if self._runtime is not None and item.sequence is not None:
                self._runtime.acknowledge_inbox(item.sequence)
            preview = " ".join(item.text.split())[:60]
            self.ui.render(HostEvent(HostEventKind.STATUS, f"steer applied · {preview}"))
            self.ui.set_status(self.status_prompt())
            return True

    @staticmethod
    def _agent_has_running_background_work(agent: Any) -> bool:
        if agent.processes.has_running():
            return True
        task_tools = getattr(agent, "task", None)
        if task_tools is not None:
            has_running = getattr(task_tools, "has_running", None)
            if callable(has_running):
                return bool(has_running())
            jobs = getattr(task_tools, "_jobs", None)
            if isinstance(jobs, dict):
                return any(not job.done() for job in jobs.values())
        return False

    async def _wait_for_wake(self, agent: Any, *, latched: bool = False) -> bool:
        """Bounded poll for a background-job or background-subagent terminal wake."""

        while True:
            timeout = self._wake_timeout_seconds
            if self._budget_guard is not None:
                self._budget_guard.enforce()
                remaining = self._budget_guard.remaining_seconds()
                if remaining is not None:
                    timeout = min(timeout, remaining)
            if not self._agent_has_running_background_work(agent):
                if latched or self._wake_event.is_set():
                    self._wake_event.clear()
                    return True
                return False
            if self._wake_event.is_set():
                self._wake_event.clear()
                return True
            try:
                await asyncio.wait_for(
                    self._wake_event.wait(),
                    timeout=timeout,
                )
                self._wake_event.clear()
                return True
            except TimeoutError:
                continue

    async def _run_user_turn(
        self,
        text: str,
        *,
        attach_paths: list[Path] | None = None,
        run_id: str | None = None,
        recovery: bool = False,
        queued: bool = False,
    ) -> HostResult:
        agent = self.agent
        runtime = self._runtime
        if self.meta is not None:
            session_bytes = await asyncio.to_thread(
                self.store.session_size, self.meta.session_id
            )
            if session_bytes > self.config.reliability.session_max_bytes:
                raise RuntimeError(
                    "session storage quota exceeded "
                    f"({session_bytes:,} > "
                    f"{self.config.reliability.session_max_bytes:,} bytes); "
                    "archive or delete old sessions before continuing"
                )
        await asyncio.to_thread(agent.refresh_context_sources)
        agent.inject_status_snapshot()
        background_origin = self._background_task_origin(agent)
        if runtime is not None:
            if run_id is None:
                run_id = await asyncio.to_thread(runtime.begin_run, text)
            else:
                await asyncio.to_thread(
                    runtime.transition_run, run_id, "recovering" if recovery else "running"
                )
        self._current_run_id = run_id
        agent._current_run_id = run_id
        turn_observation = self._telemetry.start_turn(
            run_id=run_id,
            mode=agent.mode,
            recovery=recovery,
        )

        exit_code = 0
        explanation = ""
        run_state: Any = "running"
        telemetry_error_type = ""
        turn_result = HostResult(
            exit_code=0, session_id=self.meta.session_id if self.meta else None, run_id=run_id
        )
        self._budget_guard.resume()
        try:
            await self._run_lifecycle("turn_start", run_id=run_id, recovery=recovery)
            agent.journal.begin_turn()
            if recovery:
                nooa_compat.queue_system_message(
                    agent,
                    "The previous Noah process stopped before this run reached a terminal state. "
                    "Crash recovery has rolled back incomplete file writes and cleaned orphaned "
                    "processes. Reinspect the workspace, check completed effects before repeating "
                    f"them, and continue the original request:\n\n{text}",
                )
            elif queued:
                if not await self._apply_next_steer_async(agent):
                    exit_code = 1
                    run_state = "failed"
                    explanation = "No waiting prompt could be expanded"
                    return turn_result
            else:
                expanded = await asyncio.to_thread(self._expand_user_text, text, attach_paths)
                self._deliver_expanded(agent, expanded)

            if not recovery and self.meta and self.meta.title == "untitled":
                if self.config.efficiency.deterministic_titles:
                    self._set_session_title(_deterministic_title(text))
                else:
                    self._start_title_task(text, background_origin)

            while True:
                try:
                    wins = await agent.queue_manager.race()
                    notification: dict[str, list] = {}
                    for name, item in wins:
                        notification.setdefault(name, []).append(item)
                    if agent.mode == "build" and self._lease_shared:
                        await self._ensure_mode_lease("build")
                    from noah_code.model_streaming import model_stream

                    self._last_agent_message = ""
                    with model_stream(self._on_model_stream):
                        result = await _handle_with_overflow_recovery(
                            agent, notification, render=self.ui.render
                        )
                    stop_kind, explanation, reply_text = _turn_outcome(result)
                    if (
                        reply_text
                        and reply_text != self._last_agent_message
                        and callable(getattr(agent, "message", None))
                    ):
                        agent.message(reply_text)
                    self._sync_budget_cost()
                    self.ui.render(
                        HostEvent(
                            HostEventKind.STOP,
                            _stop_text(stop_kind, explanation),
                            meta={"kind": "agent_stop", "reason": stop_kind},
                        )
                    )
                    if stop_kind in {"NEED_INPUT", "GET_USER_INPUT"}:
                        exit_code = 0
                        run_state = "waiting_user"
                        if runtime is not None and run_id is not None:
                            await asyncio.to_thread(
                                runtime.transition_run,
                                run_id,
                                "waiting_user",
                                wake_kind="user",
                                wake_ref=explanation,
                            )
                    elif stop_kind == "WAIT":
                        if await self._apply_next_steer_async(agent):
                            continue
                        latched = self._wake_event.is_set()
                        self._wake_event.clear()
                        run_state = "waiting_process"
                        if runtime is not None and run_id is not None:
                            await asyncio.to_thread(
                                runtime.transition_run,
                                run_id,
                                "waiting_process",
                                wake_kind="process",
                                wake_ref=explanation,
                            )
                        woken = await self._wait_for_wake(agent, latched=latched)
                        if not woken:
                            exit_code = 1
                            run_state = "failed"
                            explanation = (
                                "Agent returned WAIT without a running background job; "
                                "start one with self.processes.start() or self.task.start(), "
                                "or finish with DONE."
                            )
                            self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
                            break
                        run_state = "running"
                        if runtime is not None and run_id is not None:
                            await asyncio.to_thread(runtime.transition_run, run_id, "running")
                        if await self._apply_next_steer_async(agent):
                            continue
                        agent.inject_status_snapshot(force=True)
                        nooa_compat.queue_system_message(
                            agent,
                            "Background work changed state. Inspect self.processes.status() "
                            "or self.task.status() and consume any relevant logs/results, "
                            "then continue the task.",
                        )
                        continue
                    elif stop_kind == "DONE":
                        exit_code = 0
                        run_state = "completed"
                    else:
                        raise RuntimeError(f"Agent returned an invalid stop reason: {result!r}")
                except PermissionError as exc:
                    exit_code = 3
                    run_state = "failed"
                    telemetry_error_type = type(exc).__name__
                    explanation = str(exc)
                    self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
                except asyncio.CancelledError:
                    exit_code = 130
                    run_state = "cancelled"
                    explanation = "cancelled"
                    self.queue_paused = True
                    with contextlib.suppress(Exception):
                        self._save_input_queue()
                    self.ui.render(HostEvent(HostEventKind.STATUS, "turn cancelled · queue paused"))
                    raise
                except Exception as exc:
                    exit_code = 1
                    run_state = "failed"
                    telemetry_error_type = type(exc).__name__
                    explanation = _friendly_agent_error(exc)
                    self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
                    logger.debug("handle() failed", exc_info=True)
                    break

                if exit_code == 1:
                    break
                if await self._apply_next_steer_async(agent):
                    run_state = "running"
                    if runtime is not None and run_id is not None:
                        await asyncio.to_thread(runtime.transition_run, run_id, "running")
                    continue
                if run_state != "waiting_user" and exit_code != 0:
                    run_state = "failed"
                break
        finally:
            try:
                await self._flush_post_hooks(cancel=exit_code == 130)
                agent.journal.end_turn()
                latest = agent.journal.latest_turn()
                self._last_turn_shell_bypass = bool(latest and latest.shell_may_bypass)
                if self._last_turn_shell_bypass:
                    self.ui.render(
                        HostEvent(
                            HostEventKind.STATUS,
                            "warning: this turn ran mutating shell commands; "
                            "file-journal undo may be incomplete (use /undo --checkpoint)",
                        )
                    )
                if exit_code == 130:
                    with contextlib.suppress(Exception):
                        self._persist()
                else:
                    await self._persist_async()
                    label = " ".join(text.split())[:80] or "turn"
                    turn_mutated = bool(latest and (latest.mutations or latest.shell_may_bypass))
                    self._skip_unchanged_checkpoint = (
                        not turn_mutated and self.last_checkpoint is not None
                    )
                    try:
                        await self._capture_checkpoint(label)
                    finally:
                        self._skip_unchanged_checkpoint = False
                    if exit_code == 0 and run_state == "completed":
                        self._track_background_task(
                            self._maybe_remember(text, background_origin),
                            name="noah-memory",
                        )
                if runtime is not None and run_id is not None:
                    await asyncio.to_thread(
                        runtime.transition_run,
                        run_id,
                        run_state,
                        wake_kind="user" if run_state == "waiting_user" else "",
                        wake_ref=explanation if run_state == "waiting_user" else "",
                        error=explanation if run_state == "failed" else "",
                    )
            except Exception as persist_error:
                run_state = "failed"
                exit_code = 1
                explanation = safe_error_message(persist_error)
                telemetry_error_type = type(persist_error).__name__
                if runtime is not None and run_id is not None:
                    with contextlib.suppress(Exception):
                        runtime.transition_run(run_id, "failed", error=str(persist_error))
                raise
            finally:
                self._budget_guard.pause()
                agent._current_run_id = None
                outcome = str(run_state)
                if outcome == "running":
                    outcome = "failed"
                    telemetry_error_type = telemetry_error_type or "AgentTurnError"
                self._telemetry.end_turn(
                    turn_observation,
                    outcome=outcome,
                    error_type=telemetry_error_type,
                )
                turn_result.exit_code = exit_code or (1 if outcome == "failed" else 0)
                turn_result.explanation = explanation
                turn_result.status = (
                    "needs_input"
                    if outcome == "waiting_user"
                    else "completed"
                    if outcome == "completed"
                    else "cancelled"
                    if outcome == "cancelled"
                    else "failed"
                )
                turn_result.usage = self.usage_snapshot().to_dict()
                self.last_result = turn_result
                self._current_run_id = None
                await self._run_lifecycle("turn_end", run_id=run_id, status=turn_result.status)
                ledger = getattr(agent.ws, "_verification", None)
                if ledger is not None:
                    with contextlib.suppress(Exception):
                        turn_result.checks = await ledger.snapshot(since=0.0)
        return turn_result

    def _background_task_origin(self, agent: Any) -> _BackgroundTaskOrigin:
        if self.meta is None:
            raise RuntimeError("cannot start background work without a session")
        return _BackgroundTaskOrigin(
            agent=agent,
            session_id=self.meta.session_id,
            workspace_root=self.workspace.root,
        )

    def _origin_is_current(self, origin: _BackgroundTaskOrigin) -> bool:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return False
        return bool(
            self._agent is origin.agent
            and self.meta is not None
            and self.meta.session_id == origin.session_id
            and self.workspace.root == origin.workspace_root
        )

    def _track_background_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        """Own a task strongly until it finishes so teardown can drain it."""

        task = asyncio.create_task(coroutine, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _start_title_task(self, text: str, origin: _BackgroundTaskOrigin) -> None:
        if self._title_task is not None and not self._title_task.done():
            return
        task = self._track_background_task(
            self._maybe_title(text, origin),
            name="noah-title",
        )
        self._title_task = task

        def clear_title(completed: asyncio.Task[Any]) -> None:
            if self._title_task is completed:
                self._title_task = None

        task.add_done_callback(clear_title)

    async def _cancel_background_tasks(self) -> None:
        """Cancel and drain all owned title/memory tasks except the caller."""

        current = asyncio.current_task()
        tasks = [
            task
            for task in tuple(self._background_tasks)
            if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.difference_update(task for task in tasks if task.done())
        if self._title_task is not None and self._title_task.done():
            self._title_task = None

    async def _maybe_title(self, text: str, origin: _BackgroundTaskOrigin) -> None:
        if not self._origin_is_current(origin):
            return
        try:
            title = await origin.agent.name_session(text)
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("title generation failed", exc_info=True)
            return
        if not self._origin_is_current(origin):
            return
        title = (title or "").strip().strip('"')[:60]
        await asyncio.to_thread(self._set_session_title, title)

    @staticmethod
    def _can_distill_memories(agent: Any) -> bool:
        if agent is None:
            return False
        llm: Any = agent._lightweight_llm
        seen: set[int] = set()
        while llm is not None and id(llm) not in seen:
            seen.add(id(llm))
            if hasattr(llm, "_inner"):
                llm = llm._inner
            elif getattr(llm, "_clients", None):
                llm = llm._clients[0]
            else:
                break
        return type(llm).__name__ != "FakeLLMClient"

    async def _maybe_remember(self, text: str, origin: _BackgroundTaskOrigin) -> None:
        efficiency = getattr(getattr(self, "config", None), "efficiency", None)
        memory_mode = getattr(efficiency, "memory_distillation", "heuristic")
        if (
            len(text.strip()) < 40
            or memory_mode == "off"
            or (memory_mode == "heuristic" and _MEMORY_CUE.search(text) is None)
            or not self._origin_is_current(origin)
            or not self._can_distill_memories(origin.agent)
        ):
            return
        try:
            raw = await origin.agent.distill_memories(text[:4000])
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("memory distillation failed", exc_info=True)
            return
        if not self._origin_is_current(origin):
            return
        try:
            added = await asyncio.to_thread(origin.agent.absorb_memories, raw or "")
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("memory absorption failed", exc_info=True)
            return
        if not added:
            return
        # Defer context refresh if another turn is already running so we never
        # mutate the system prompt prefix mid-generation.
        if not self._turn_running() and self._origin_is_current(origin):
            await asyncio.to_thread(origin.agent.refresh_context_sources)
        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"remembered {len(added)} project convention(s)",
            )
        )

    def _set_session_title(self, title: str) -> None:
        selected = _clean_control_chars(title).strip('"')[:60]
        if selected and self.meta:
            self.meta.title = selected
            self.store.save_meta(self.meta)

    async def resume_interrupted_run(self) -> HostResult | None:
        """Automatically continue a non-terminal run recovered at startup."""

        record = self._interrupted_run
        if (
            record is None
            or record.state == "waiting_user"
            or not self.config.reliability.auto_resume_interrupted_runs
        ):
            return None
        self._interrupted_run = None
        self.ui.render(
            HostEvent(
                HostEventKind.STATUS,
                f"resuming interrupted run {record.run_id[:8]} · previous state={record.state}",
            )
        )
        async with self._session_lock:
            self.ui.set_busy(True)
            self._active_turn = asyncio.current_task()
            try:
                return await self._run_user_turn(
                    record.user_text,
                    run_id=record.run_id,
                    recovery=True,
                )
            finally:
                self._active_turn = None
                self.ui.set_busy(False)
                self.ui.set_status(self.status_prompt())

    async def run_interactive(self) -> int:
        """Line-oriented console loop."""
        await self.start()
        await self.resume_interrupted_run()
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

    async def run_tui(
        self,
        *,
        onboarding_required: bool = False,
        permission_setup_required: bool = False,
    ) -> int:
        """Full-screen Textual UI. App owns input; host owns turns."""
        try:
            from noah_code.ui.textual_app import NoahCodeApp, TextualUI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Textual is required for the TUI but could not be imported.") from exc

        ui = TextualUI()
        self.ui = ui
        app = NoahCodeApp(
            self,
            ui,
            onboarding_required=onboarding_required,
            permission_setup_required=permission_setup_required,
        )
        try:
            await app.run_async(mouse=True)
            return 0
        finally:
            await self.close()

    async def run_once(self, prompt: str) -> HostResult:
        await self.start()
        ledger = getattr(self.agent.ws, "_verification", None)
        try:
            if self.config.auto_approve:

                async def _auto(req):  # noqa: ANN001
                    if req.decision.action == "deny":
                        return ApprovalChoice.REJECT
                    if req.decision.elevated_floor and not self.config.yolo:
                        return ApprovalChoice.REJECT
                    return ApprovalChoice.ONCE

                self.agent.approvals.set_handler(_auto)
            else:

                async def _reject(req):  # noqa: ANN001
                    return ApprovalChoice.REJECT

                self.agent.approvals.set_handler(_reject)

            await self.resume_interrupted_run()
            continuable = (
                await asyncio.to_thread(self._runtime.latest_incomplete_run)
                if self._runtime
                else None
            )
            run_id = (
                continuable.run_id
                if continuable and continuable.state == "waiting_user"
                else None
            )
            async with self._session_lock:
                self._active_turn = asyncio.current_task()
                try:
                    result = await self._run_user_turn(prompt, run_id=run_id)
                finally:
                    self._active_turn = None
            return result
        finally:
            try:
                await self.close()
            finally:
                if self.last_result is not None and ledger is not None:
                    with contextlib.suppress(Exception):
                        self.last_result.checks = await ledger.snapshot(since=0.0)


def _clean_control_chars(value: Any) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
