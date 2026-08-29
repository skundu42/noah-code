"""CodingAgent - InteractiveAgent for repository work."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import platform
import site
import sys
import sysconfig
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, Literal

from nooa import Agent, Context, hidden, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.interactive import InteractiveAgent, RespondReason, RespondResult
from nooa.runtime.restrictions import RESTRICTED_MODULES, RestrictionsConfig
from nooa.runtime.sandbox.config import FileRule, SandboxConfig, resolve_spec
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.strategies import CodeActStrategy
from nooa.strategies.codeact_lite import PlainCodeActBlockFormatter
from nooa.tools import TodoManager
from nooa.tools.shell_tools import ShellTools

from noah_code import nooa_compat
from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.macos_sandbox import build_macos_profile, macos_worker_main
from noah_code.permissions import PermissionEngine
from noah_code.predict import ISOLATED_PREDICT_CONTEXT, LeanPredictStrategy
from noah_code.secure_files import read_text_bounded
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.git_tools import GitTools
from noah_code.tools.github_tools import GithubTools
from noah_code.tools.lsp_tools import LSPTools
from noah_code.tools.media_tools import MediaTools
from noah_code.tools.memory_tools import MemoryTools
from noah_code.tools.plan_tools import PlanTools
from noah_code.tools.process_tools import ProcessTools
from noah_code.tools.question_tools import QuestionTools
from noah_code.tools.task_tools import TaskTools
from noah_code.tools.web_tools import WebTools
from noah_code.tools.workspace_tools import WorkspaceMutationCoordinator, WorkspaceTools
from noah_code.workspace import Workspace, WorkspaceError


def _interpreter_read_rules() -> tuple[FileRule, ...]:
    """Paths needed by the sandboxed interpreter, excluding host data such as /etc."""
    candidates: list[str] = [sys.prefix, sys.base_prefix, sys.exec_prefix]
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            candidates.append(value)
    with suppress(AttributeError):  # pragma: no cover - implementation-specific Python
        candidates.extend(site.getsitepackages())

    seen: set[str] = set()
    rules: list[FileRule] = []
    for raw in candidates:
        expanded = Path(raw).expanduser().absolute()
        for path in (str(expanded), str(expanded.resolve())):
            if path not in seen and Path(path).exists():
                seen.add(path)
                rules.append(FileRule(path=path, access="read"))
    if Path("/dev/null").exists():
        rules.append(FileRule(path="/dev/null", access="read_write"))
    return tuple(rules)


def _codeact_config(config: NoahCodeConfig) -> CodeActConfig:
    unsafe = config.unsafe_inprocess_code_execution
    restricted_imports = RESTRICTED_MODULES | frozenset({"nooa", "nooa_cli", "noah_code"})
    return CodeActConfig(
        # None = no iteration cap; budgets (tokens/cost/time) remain the brakes.
        max_iterations=config.max_iterations,
        cell_timeout=config.cell_timeout,
        execution_backend="inprocess" if unsafe else "sandbox",
        restrictions=RestrictionsConfig(restricted_imports=restricted_imports),
        sandbox=SandboxConfig(
            filesystem=True,
            workspace=None,
            allow=_interpreter_read_rules(),
            system_paths=False,
            network=False,
            max_memory_mb=512,
            max_cpu_seconds=max(1, int(config.cell_timeout)),
            require=True,
        ),
    )


class _PermissionSandboxedExecutor(SandboxedExecutor):
    """Broker only permission-gated agent capabilities back into the parent."""

    _EXACT_PATHS = frozenset(
        {
            ("git", "diff"),
            ("git", "log"),
            ("git", "status"),
            ("github", "list"),
            ("github", "view"),
            ("github", "create"),
            ("github", "push"),
            ("github", "checkout"),
            ("github", "comment"),
            ("plan", "read"),
            ("plan", "write"),
            ("plan", "enter"),
            ("plan", "exit_to_build"),
            ("memory", "list"),
            ("memory", "save"),
            ("memory", "forget"),
            ("message",),
            ("mode",),
            ("workspace_root",),
            ("ws", "edit"),
            ("ws", "inspect"),
            ("ws", "apply_patch"),
            ("ws", "apply_unified_diff"),
            ("ws", "list"),
            ("ws", "list_files"),
            ("ws", "read"),
            ("ws", "read_output"),
            ("ws", "replace"),
            ("ws", "run"),
            ("ws", "run_trusted_readonly"),
            ("ws", "search"),
            ("ws", "write"),
            ("ws", "write_file"),
            ("lsp", "changed_symbols"),
            ("lsp", "definition"),
            ("lsp", "diagnostics"),
            ("lsp", "document_symbols"),
            ("lsp", "hover"),
            ("lsp", "implementation"),
            ("lsp", "references"),
            ("lsp", "rename_preview"),
            ("lsp", "repository_map"),
            ("lsp", "workspace_symbols"),
            ("media", "consume"),
            ("media", "pending"),
            ("ask", "question"),
            ("web", "fetch"),
            ("web", "search"),
            ("task", "list"),
            ("task", "run"),
            ("task", "run_many"),
            ("task", "collaborate"),
            ("processes", "input"),
            ("processes", "open_terminal"),
            ("processes", "terminal_run"),
            ("processes", "terminal_status"),
            ("processes", "close_terminal"),
            ("processes", "logs"),
            ("processes", "start"),
            ("processes", "status"),
            ("processes", "stop"),
        }
    )
    _SAFE_SUBTREES = frozenset({("todos",), ("v",)})

    @classmethod
    def _path_allowed(cls, path: tuple[str, ...]) -> bool:
        if any(path[: len(prefix)] == prefix for prefix in cls._SAFE_SUBTREES):
            return True
        return path in cls._EXACT_PATHS or any(
            allowed[: len(path)] == path for allowed in cls._EXACT_PATHS
        )

    def _walk_path(self, path: list[str]) -> Any:
        normalized = tuple(path)
        approved_roots: set[str] = getattr(self._agent, "_sandbox_approved_roots", set())
        dynamically_allowed = bool(normalized and normalized[0] in approved_roots)
        if not normalized or not (self._path_allowed(normalized) or dynamically_allowed):
            display = ".".join(path) or "<root>"
            raise PermissionError(f"sandbox broker access denied: self.{display}")
        return super()._walk_path(path)


def _spawn_safe_local_agent(agent: Any) -> Any:
    """Empty instance of ``type(agent)`` for the spawned worker's local ``self``.

    The worker only needs the class (module globals and callable classification).
    Pickling the live agent into ``multiprocessing`` spawn copies Textual, MCP,
    and shell file descriptors into CPython's ``fds_to_keep`` list, which 3.12
    rejects as ``ValueError: bad value(s) in fds_to_keep``.
    """
    cls: Any = type(agent)
    try:
        return object.__new__(cls)
    except TypeError:
        return cls.__new__(cls)


_OMIT = object()
# ``spawnv_passfds`` is process-global; patch and restore it as one critical section.
_SPAWN_PASSFDS_LOCK = threading.RLock()


def _pipe_picklable(value: Any) -> bool:
    try:
        from multiprocessing.reduction import ForkingPickler

        ForkingPickler.dumps(value)
    except Exception:
        return False
    return True


def _spawn_safe_value(value: Any) -> Any:
    """Drop modules and other objects that cannot cross a spawn pipe."""

    if isinstance(value, types.ModuleType):
        return _OMIT
    if isinstance(value, dict):
        return {
            key: item
            for key, item in ((name, _spawn_safe_value(item)) for name, item in value.items())
            if item is not _OMIT
        }
    if isinstance(value, list | tuple):
        items = [item for item in (_spawn_safe_value(item) for item in value) if item is not _OMIT]
        return type(value)(items)
    if _pipe_picklable(value):
        return value
    return _OMIT


def _spawn_safe_framework_builtins(builtins: dict[str, Any]) -> dict[str, Any]:
    """Keep only pipe-picklable CodeAct builtins.

    ``filter_mro_module_globals(type(agent))`` rebuilds imported modules in the
    worker. Nested ``return_result`` is rebound after the child receives init.
    """
    return {
        name: item
        for name, item in ((key, _spawn_safe_value(value)) for key, value in builtins.items())
        if item is not _OMIT
    }


@contextmanager
def _unique_spawn_passfds() -> Iterator[None]:
    """Keep only open, unique inherit-FDs; CPython 3.12 rejects the rest."""
    import multiprocessing.util as util
    import os

    with _SPAWN_PASSFDS_LOCK:
        original = util.spawnv_passfds

        def _inherit_fds(passfds: Any) -> tuple[int, ...]:
            kept: list[int] = []
            for fd in sorted({int(fd) for fd in passfds}):
                if fd < 0:
                    continue
                try:
                    os.fstat(fd)
                except OSError:
                    continue
                kept.append(fd)
            return tuple(kept)

        def _spawnv(path: Any, args: Any, passfds: Any) -> Any:
            return original(path, args, _inherit_fds(passfds))

        util.spawnv_passfds = _spawnv
        try:
            yield
        finally:
            util.spawnv_passfds = original


class _MacOSPermissionSandboxedExecutor(_PermissionSandboxedExecutor):
    """NOOA brokered worker contained with macOS's native sandbox profile."""

    def __init__(
        self,
        agent: Any,
        config: SandboxConfig,
        *,
        cell_timeout: float | None,
        framework_builtins: dict[str, Any] | None = None,
        restrictions: Any = None,
    ) -> None:
        # NOOA probes specifically for Linux Landlock/seccomp. Preserve
        # its worker and broker implementation, but install equivalent native
        # macOS guards in the child before NOOA enters its execution loop.
        worker_config = config.model_copy(
            update={
                "filesystem": False,
                "network": True,
                "max_memory_mb": 0,
                "max_cpu_seconds": 0,
            }
        )
        self._agent = agent
        self._config = worker_config
        self._cell_timeout = cell_timeout
        self._framework_builtins = framework_builtins or {}
        self._restrictions = restrictions
        self._spec = resolve_spec(worker_config)
        self._degraded: list[str] = []
        # Forking a multithreaded Textual process can deadlock in inherited
        # locks and is deprecated on macOS/Python 3.12. This custom executor's
        # initialization payload is spawn-safe, so start a clean interpreter.
        self._ctx = mp.get_context("spawn")
        self._proc: mp.process.BaseProcess | None = None
        self._conn: Any = None
        self._lock = asyncio.Lock()
        self._req_id = 0
        self._closed = False
        self._disabled = False
        self._macos_profile = build_macos_profile(rule.path for rule in config.allow)
        self._macos_max_memory_mb = config.max_memory_mb
        self._macos_max_cpu_seconds = config.max_cpu_seconds

    def _start_worker(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=macos_worker_main,
            args=(
                child_conn,
                self._macos_profile,
                self._macos_max_memory_mb,
                self._macos_max_cpu_seconds,
            ),
            daemon=True,
            name="nooa-macos-sandbox-worker",
        )
        with _unique_spawn_passfds():
            proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc
        try:
            # Send the worker payload over the pipe after spawn. Pickling it as
            # Process args registers every Connection/socket in CPython's
            # fds_to_keep list (duplicates, closed handles, MCP stdio).
            parent_conn.send(
                {
                    "agent": _spawn_safe_local_agent(self._agent),
                    "framework_builtins": _spawn_safe_framework_builtins(
                        self._framework_builtins
                    ),
                    "restrictions": self._restrictions,
                    "spec": self._spec,
                }
            )
        except Exception:
            self._terminate_worker()
            raise


def _cache_first_block_order(base: list[str] | None) -> list[str]:
    """Stable-prefix-first ordering: static instructions before volatile blocks."""

    stable_blocks = (
        "repo_instructions",
        "agents",
        "subagent",
        "workspace",
        "active_plan",
        "project_memory",
    )
    return [
        *(block for block in (base or []) if block not in stable_blocks),
        *stable_blocks,
    ]


def _summarization_token_limit(config: NoahCodeConfig, llm: Any) -> int:
    """Resolve a practical history budget without overriding an explicit limit."""

    if config.summarization.max_tokens is not None:
        return config.summarization.max_tokens
    context_window = getattr(llm, "context_window", None)
    ratio_limit = (
        int(context_window * config.summarization.trigger_ratio)
        if context_window
        else config.efficiency.context_token_budget
    )
    return min(ratio_limit, config.efficiency.context_token_budget)


_NOAH_CODEACT_INSTRUCTIONS = """## Noah CodeAct

Work through tool calls; plain assistant prose cannot run code or end a turn.
Use `execute_python(code)` for work and `return_result(...)` to finish. Python
cell state persists. Parameters and `self` are preloaded. Every `self.ws.*`
call is async: always `await` it before iterating or accessing the result.
`self.message(text)` is synchronous—never await it.

### Tools

- Explore with focused `self.ws.list/search/read` and `self.lsp` queries. Edit a
  Match with `self.ws.replace`. `self.ws.edit(path, old, new)` takes exactly
  three arguments; two-argument calls are invalid. Prefer one atomic
  `self.ws.apply_patch(changes)` for a
  coherent batch; `self.ws.apply_unified_diff` accepts git-style hunks.
- Common shapes: `await ws.search(pattern, path=".", paths=None, regex=True)`
  returns iterable Matches plus `.stdout`; `await ws.read(path, lines=None)`
  returns text with `.text`/`.content`; `lines=60` reads the first 60 lines;
  `await ws.list(pattern="**/*", path=".")` returns paths.
- Run commands with `await self.ws.run(command)` and inspect
  returncode/stdout/stderr. `read_only=True` skips approval only for commands
  the engine recognizes as read-only (Git inspection, search, listing, text
  filters). It is rejected for pytest, uv, Python, builds, and mutations.
- Use `self.processes.start/logs/status/input/stop` for long jobs and consume
  logs by cursor. Persistent shells use `open_terminal`, `terminal_run`,
  `terminal_status`, and `close_terminal`; raw terminal input is blocked.
- `self.web.fetch/search` are read-only. Use `self.github` for PR operations,
  not shell commands. Ask choices through `self.ask.question`.
- Manage plans through `self.plan.write/enter/exit_to_build`; never call
  `set_mode`. Save standing conventions with `self.memory.save`.
- Delegate bounded units with `self.task.run`, independent units with
  `run_many`, and lead synthesis with `collaborate`.
- If `self.media` has pending images, `show()` each consumed image first.

### Workflow and safety

Understand the requested end state before acting. For a conversational answer,
call `self.message(...)` and finish without editing. Otherwise inspect repository
instructions and nearby code, use todos only for multi-step work, preserve
unrelated changes, make the smallest coherent edit, and validate proportional
to risk. Never claim a command passed unless its successful result was observed.
In plan mode do not mutate; follow an active plan without expanding scope.
Do not commit, push, publish, or create external resources unless explicitly
asked. Do not read secrets or expose sensitive values. `--yolo` applies only
when the host was explicitly launched for a throwaway workspace.

Sandboxed cells forbid host/system imports (including os, sys, subprocess,
shutil, nooa, and noah_code), dynamic execution, input, and attaching callables
to `self`. Use dedicated tools instead; use `doc(self)` for API help.

End with exactly one RespondResult: DONE, NEED_INPUT, or WAIT (WAIT requires a
registered running job). In code call
`return_result(RespondReason.DONE, explanation="...")`; never pass a bare string,
which has the wrong type.
Show user-facing text first with synchronous `self.message("...")`.
"""

_NOAH_EXECUTION_CONTEXT = """## Execution Context

Inside `execute_python`, method parameters and `self` are already in scope and
state persists across cells. Always available: `print`, `pprint`, `doc`,
`return_result`, `RespondReason`, `asyncio`, and `typing`. Inspect only the API
you need with `doc(self)`; nested sandbox proxies are not introspectable. Noah's
internal module imports are intentionally not part of the agent-facing contract.
"""

_OBSERVABILITY_EVENT_TYPES = (
    "LLMCallStart",
    "LLMCallEnd",
    "LLMComplete",
    "Reasoning",
    "Error",
)


def _forward_observability_events(source: Any, target: Any) -> list[Any]:
    """Forward runtime evidence without copying model-visible history."""

    if source is target:
        return []

    def forward(event: Any) -> None:
        target.add(event, record=False)

    return [source.on(event_type, forward) for event_type in _OBSERVABILITY_EVENT_TYPES]


class _AuxiliaryPredictor(Agent):
    """Stateless one-shot helpers for non-coding model work."""

    def __init__(self, llm: Any, *, cache_namespace: str) -> None:
        super().__init__(llm=llm)
        self._agent_id = cache_namespace

    @strategy(
        LeanPredictStrategy(PredictConfig(output_serialization="tool_call")),
        context=ISOLATED_PREDICT_CONTEXT,
    )
    async def name_session(self, user_message: str) -> str:
        """Create a specific 2-5 word coding-session title. Return only the title.

        Do not include quotes, punctuation, generic words like "task", or claims
        beyond the user's request. Keep useful identifiers when present.
        """

        ...

    @strategy(
        LeanPredictStrategy(PredictConfig(output_serialization="tool_call")),
        context=ISOLATED_PREDICT_CONTEXT,
    )
    async def distill_result(self, transcript: str) -> str:
        """Compress one subagent transcript into an evidence-preserving parent report.

        Return only useful sections as short plain lines without markdown headers:
        Findings - concrete answers with exact paths and symbols.
        Changes - files edited and what changed.
        Validation - commands and exact observed outcomes.
        Open - unresolved problems or user questions.

        Omit empty sections, chatter, raw output, and superseded attempts. Never
        invent success. Preserve identifiers, commands, and numbers. Maximum 1200
        characters.
        """

        ...

    @strategy(
        LeanPredictStrategy(PredictConfig(output_serialization="tool_call")),
        context=ISOLATED_PREDICT_CONTEXT,
    )
    async def distill_memories(self, turn: str) -> str:
        """Extract only standing project conventions that should persist across sessions.

        Return EMPTY when there is none; otherwise return at most eight `MEMORY:`
        lines. Include conventions such as package managers, forbidden paths, or
        PR-title style. Never include secrets, task status, or one-off bugs.
        """

        ...


class _PermissionCodeActStrategy(CodeActStrategy):
    async def strategy_instructions(self, runtime: Any) -> str:
        """Use one stable Noah-specific contract instead of generic duplicate guidance."""

        _ = runtime
        return _NOAH_CODEACT_INSTRUCTIONS

    async def execution_context(self, runtime: Any) -> str:
        """Expose the supported cell surface, not Noah's implementation imports."""

        _ = runtime
        return _NOAH_EXECUTION_CONTEXT

    async def sandbox_context(self, runtime: Any) -> str:
        """Describe actionable sandbox limits without volatile interpreter paths."""

        _ = runtime
        sandbox = self.config.sandbox
        return (
            "Code cells run in an isolated worker: "
            f"{self.config.cell_timeout:g}s wall-clock, "
            f"{sandbox.max_cpu_seconds:g}s CPU, {sandbox.max_memory_mb} MB memory. "
            "Direct workspace filesystem access and network sockets are unavailable; "
            "use `self.*` tools. Returned values must be picklable. "
            "Treat `self.<attr>` values as copies: persist changes through methods or reassignment. "
            "Keep live objects in cell state and return compact summaries."
        )

    def _build_builtins(self, runtime: Any, call: Any) -> dict[str, Any]:
        builtins = super()._build_builtins(runtime, call)
        # InteractiveAgent documents this exact inline return pattern. Keep the
        # enum explicit in case module-context filtering changes upstream.
        builtins["RespondReason"] = RespondReason
        return builtins

    def get_block_order(self) -> list[str]:
        """NOOA consults whichever strategy instance renders a request."""

        return _cache_first_block_order(super().get_block_order())

    def _create_sandbox_executor(self, runtime: Any, call: Any, builtins: dict[str, Any]) -> Any:
        framework_builtins = {**builtins, "_call": call}
        executor_type = (
            _MacOSPermissionSandboxedExecutor
            if platform.system() == "Darwin"
            else _PermissionSandboxedExecutor
        )
        return executor_type(
            runtime.agent,
            self.config.sandbox,
            cell_timeout=self.config.cell_timeout,
            framework_builtins=framework_builtins,
            restrictions=self.config.restrictions,
        )


class _LeanPermissionCodeActStrategy(_PermissionCodeActStrategy):
    """Compact NOOA event rendering while retaining cross-turn history."""

    @property
    def name(self) -> str:
        return "NOAH_LEAN_CODEACT"

    async def execute(self, runtime: Any, call: Any) -> Any:
        original = runtime.agent.render_config
        event_format = nooa_compat.truncation_event_format(runtime.agent)
        runtime.agent.render_config = original.model_copy(
            update={"block_formatter": PlainCodeActBlockFormatter(event_format=event_format)}
        )
        try:
            return await super().execute(runtime, call)
        finally:
            runtime.agent.render_config = original


class _AdaptivePermissionCodeActStrategy(_PermissionCodeActStrategy):
    """Resolve NOOA strategy limits from the live agent efficiency profile."""

    @property
    def name(self) -> str:
        return "NOAH_ADAPTIVE_CODEACT"

    async def execute(self, runtime: Any, call: Any) -> Any:
        config = runtime.agent._config
        strategy_type = (
            _LeanPermissionCodeActStrategy
            if config.efficiency.strategy == "lean"
            else _PermissionCodeActStrategy
        )
        delegate = strategy_type(config=_codeact_config(config))
        return await delegate.execute(runtime, call)


class CodingAgent(InteractiveAgent):
    """Repository coding agent for noah-code.

    Inspect, plan, edit, and verify inside a single workspace. Prefer
    focused reads and Match-based edits. Never claim tests passed unless
    observed. Mode and permissions are enforced in tool code, not only prompts.
    """

    workspace_root: str = "."
    mode: Literal["build", "plan"] = "build"

    def __init__(
        self,
        workspace: Workspace,
        config: NoahCodeConfig,
        *,
        llm: Any = None,
        lightweight_llm: Any = None,
        storage: Any = None,
        engine: PermissionEngine | None = None,
        approvals: ApprovalBroker | None = None,
        journal: SnapshotJournal | None = None,
        runtime: Any = None,
        coordinator: WorkspaceMutationCoordinator | None = None,
        budget_guard: Any = None,
        usage_tracker: Any = None,
        cache_namespace: str | None = None,
        observability_event_manager: Any = None,
        nested: bool = False,
        nested_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)
        super().__init__(llm=llm, storage=storage, **kwargs)
        if cache_namespace:
            # NOOA derives prompt_cache_key from this id. A durable Noah
            # session therefore keeps provider cache affinity across restart.
            self._agent_id = cache_namespace
        self._lightweight_llm = lightweight_llm or self._llm
        self.workspace_root = str(workspace.root)
        self.mode = config.mode
        self._config = config
        self._nested = nested

        self._engine = engine or PermissionEngine(
            config.permission_rules,
            mode=config.mode,
            auto_approve=config.auto_approve,
            yolo=config.yolo,
        )
        self._engine.mode = config.mode
        self._approvals = approvals or ApprovalBroker(
            self._engine,
            runtime=runtime,
            timeout_seconds=config.reliability.interaction_timeout_seconds,
        )
        self._approvals.set_runtime(
            runtime,
            timeout_seconds=config.reliability.interaction_timeout_seconds,
        )
        self._journal = journal or SnapshotJournal(blob_limit=config.undo_blob_limit)
        self._runtime = runtime
        self._coordinator = coordinator or WorkspaceMutationCoordinator()
        self._budget_guard = budget_guard
        self._usage_tracker = usage_tracker
        self._observability_event_manager = (
            observability_event_manager
            if observability_event_manager is not None
            else self.event_manager
        )
        self._observability_unsubs = _forward_observability_events(
            self.event_manager,
            self._observability_event_manager,
        )

        self.lsp = LSPTools(
            workspace,
            self._engine,
            self._approvals,
            enabled=config.lsp.enabled,
            timeout=config.lsp.timeout_seconds,
            server_overrides=config.lsp.servers,
            max_symbols=config.lsp.max_symbols,
            max_file_bytes=config.max_file_bytes,
        )

        shell = ShellTools(cwd=str(workspace.root))
        self._shell: Annotated[ShellTools, hidden] = shell

        self.ws = WorkspaceTools(
            workspace,
            shell,
            self._engine,
            self._approvals,
            self._journal,
            max_output_chars=config.max_output_chars,
            max_output_lines=config.efficiency.max_output_lines,
            max_search_results=config.efficiency.max_search_results,
            max_file_results=config.efficiency.max_file_results,
            output_retention_hours=config.efficiency.tool_output_retention_hours,
            default_timeout=config.command_timeout,
            lsp=self.lsp,
            runtime=runtime,
            coordinator=self._coordinator,
            output_store_root=runtime.artifact_dir if runtime is not None else None,
            output_store_max_bytes=config.reliability.artifact_max_bytes,
        )
        self.processes = ProcessTools(
            self.ws,
            max_jobs=config.processes.max_jobs,
            max_runtime_seconds=config.processes.max_runtime_seconds,
            max_buffer_chars=config.processes.max_buffer_chars,
            stop_grace_seconds=config.processes.stop_grace_seconds,
            runtime=runtime,
        )
        self.todos = TodoManager()
        self.git = GitTools(self.ws)
        self.github = GithubTools(
            workspace.root,
            self._engine,
            self._approvals,
            runtime=runtime,
        )
        self.web = WebTools(self._engine, self._approvals)
        self.ask = QuestionTools(
            self._engine,
            self._approvals,
            runtime=runtime,
            timeout_seconds=config.reliability.interaction_timeout_seconds,
        )
        self.plan = PlanTools(workspace.root, self, self.ask, self._engine)
        self.memory = MemoryTools(
            workspace.root,
            on_change=self.refresh_context_sources,
        )
        self.media = MediaTools()
        self._sandbox_approved_roots: set[str] = set()
        if not nested:
            self.task = TaskTools(
                workspace,
                self._engine,
                self._approvals,
                parent=self,
            )

        from noah_code.skills_setup import install_skills

        self._skills_status = "" if nested else install_skills(self, workspace.root, config)

        # Bounded live context - not full trees/diffs.
        #
        # Only cache-stable content lives in the system prompt. Volatile state
        # (todos, git, background jobs) is injected as append-only events at
        # each user-turn boundary via inject_status_snapshot(); mutating a
        # mid-prefix block would invalidate provider prompt-cache for the
        # entire conversation on every change.
        self.context["workspace"] = Context(
            expr="f'workspace={self.workspace_root}\\nmode={self.mode}'"
        )
        self._git_summary_value = self._git_summary()
        if nested_prompt:
            self.context["subagent"] = Context(nested_prompt, prefix=True)
        if not nested:
            self.context["agents"] = Context(expr="self.task.list()")

        if config.summarization.policy != "none":
            from nooa.config.summarizer_config import TokenBudgetConfig

            from noah_code.summarization import CodingSessionSummarizer

            summarizer = CodingSessionSummarizer.install(
                self,
                llm=self._lightweight_llm,
                config=TokenBudgetConfig(
                    max_tokens=_summarization_token_limit(config, self._llm),
                    preserve_recent=config.summarization.preserve_recent,
                    target_chars=config.summarization.target_chars,
                ),
            )
            summarizer._agent_id = f"{self.agent_id}:summarize"
            self._observability_unsubs.extend(
                _forward_observability_events(
                    summarizer.event_manager,
                    self._observability_event_manager,
                )
            )

        # Discover AGENTS.md / README hints without dumping trees.
        self._repo_instructions_value = self._repo_instructions()
        self.context["repo_instructions"] = Context(
            self._repo_instructions_value,
            prefix=True,
        )
        self._active_plan_value = self._active_plan()
        self.context["active_plan"] = Context(self._active_plan_value, prefix=True)
        self._project_memory_value = self._project_memory()
        self.context["project_memory"] = Context(self._project_memory_value, prefix=True)

    @hidden
    def refresh_context_sources(self) -> None:
        """Refresh cacheable instructions at a safe user-turn boundary."""

        self._git_summary_value = self._git_summary()
        current = self._repo_instructions()
        if current != self._repo_instructions_value:
            self._repo_instructions_value = current
            self.context["repo_instructions"] = Context(current, prefix=True)
        plan = self._active_plan()
        if plan != getattr(self, "_active_plan_value", None):
            self._active_plan_value = plan
            self.context["active_plan"] = Context(plan, prefix=True)
        memory = self._project_memory()
        if memory != getattr(self, "_project_memory_value", None):
            self._project_memory_value = memory
            self.context["project_memory"] = Context(memory, prefix=True)

    @hidden
    def inject_status_snapshot(self, *, force: bool = False) -> bool:
        """Append volatile workspace state to the event stream.

        Appending keeps the request prefix byte-stable: providers cache the
        unchanged head and only process the new tail. Returns True when a
        snapshot was injected. ``force`` injects even when nothing changed
        since the last snapshot (used at turn starts after resume).
        """

        from nooa.events import Feedback

        sections: list[str] = []
        git_summary = self._git_summary_value
        if git_summary and not git_summary.startswith("(not a git repo"):
            sections.append(f"[git]\n{git_summary}")
        todos_status = self.todos.status()
        if todos_status and todos_status != "(no todos)":
            sections.append(f"[todos]\n{todos_status}")
        jobs_summary = self.processes.summary()
        if jobs_summary and jobs_summary != "(no background jobs)":
            sections.append(f"[background_jobs]\n{jobs_summary}")
        snapshot = "\n\n".join(sections)
        if len(snapshot) > 2000:
            snapshot = snapshot[:1990] + "\n…"
        if not snapshot:
            return False
        if not force and snapshot == getattr(self, "_last_status_snapshot", None):
            return False
        self._last_status_snapshot = snapshot
        self.event_manager.add(Feedback(content=f"[workspace status]\n{snapshot}"))
        return True

    @hidden
    def sync_model_limits(self) -> None:
        """Keep the configured compaction ratio after a live model switch."""

        from nooa.config.summarizer_config import TokenBudgetConfig

        maximum = _summarization_token_limit(self._config, self._llm)
        for summarizer in nooa_compat.summarizers(self):
            current = summarizer.config
            summarizer.config = TokenBudgetConfig(
                max_tokens=maximum,
                preserve_recent=current.preserve_recent,
                target_chars=current.target_chars,
            )

    @hidden
    def set_main_llm(self, llm: Any, *, lightweight_follows_main: bool) -> None:
        """Switch the primary model and keep compaction routing coherent."""

        from noah_code.llm_replies import wrap_conversational_replies

        llm = wrap_conversational_replies(llm)
        self._llm = llm
        if lightweight_follows_main:
            self._lightweight_llm = llm
            nooa_compat.rebind_summarizer_llms(self, llm)
        self.sync_model_limits()

    @hidden
    def set_efficiency_profile(self, profile: str) -> None:
        """Apply a live tool-output efficiency profile."""

        if profile not in {"fast", "balanced", "deep"}:
            raise ValueError("profile must be fast, balanced, or deep")
        self._config.efficiency.profile = profile  # type: ignore[assignment]
        self.ws.set_efficiency_profile(profile)

    @hidden
    async def compact_history(self) -> bool:
        """Compact the oldest eligible history now, at an explicit turn boundary."""

        return await nooa_compat.compact_summarizers(self)

    @hidden
    async def _run_auxiliary_predict(self, route: str, value: str) -> str:
        """Run one helper against fresh history and a stable session-local cache route."""

        predictor = _AuxiliaryPredictor(
            self._lightweight_llm,
            cache_namespace=f"{self.agent_id}:aux:{route}",
        )
        unsubs = _forward_observability_events(
            predictor.event_manager,
            self._observability_event_manager,
        )
        try:
            method = getattr(predictor, route)
            return str(await method(value))
        finally:
            for unsubscribe in unsubs:
                unsubscribe()

    @hidden
    async def name_session(self, user_message: str) -> str:
        """Generate a compact title without adding helper events to coding history."""

        return await self._run_auxiliary_predict("name_session", user_message)

    @hidden
    async def distill_result(self, transcript: str) -> str:
        """Condense a large child result without inheriting the child's work history."""

        return await self._run_auxiliary_predict("distill_result", transcript)

    @hidden
    def _git_summary(self) -> str:
        import subprocess

        unavailable = "(not a git repo or empty status)"
        try:
            status = subprocess.run(
                ["git", "status", "--short", "--branch"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if status.returncode != 0:
                return unavailable
            return (status.stdout or "").strip() or unavailable
        except (OSError, subprocess.SubprocessError):
            return unavailable

    @hidden
    def _repo_instructions(self) -> str:
        root = Path(self.workspace_root).resolve()
        chunks: list[str] = []
        for name in ("AGENTS.md", "CLAUDE.md", ".noah-code/instructions.md"):
            try:
                result = read_text_bounded(root, name, max_bytes=16 * 1024)
            except (OSError, WorkspaceError):
                # Repository-controlled links and parent swaps must not turn
                # trusted context assembly into an external-file read.
                continue
            text = result.text
            if result.truncated or len(text) > 4000:
                text = text[:4000] + "\n...(truncated)..."
            chunks.append(f"## {name}\n{text}")
        return "\n\n".join(chunks) if chunks else "(no repository instruction files found)"

    @hidden
    def _active_plan(self) -> str:
        from noah_code.project_notes import PlanStore

        text = PlanStore(Path(self.workspace_root)).read().strip()
        if not text:
            return "(no active plan)"
        if len(text) > 4000:
            text = text[:4000] + "\n...(truncated)..."
        return f"## Active plan (.noah-code/plan.md)\n{text}\n\nFollow this plan in build mode. Do not expand scope."

    @hidden
    def _project_memory(self) -> str:
        from noah_code.project_notes import MemoryStore

        text = MemoryStore(Path(self.workspace_root)).read().strip()
        if not text:
            return "(no project memory yet)"
        if len(text) > 4000:
            text = text[:4000] + "\n...(truncated)..."
        return f"## Project memory (.noah-code/memory.md)\n{text}"

    @hidden
    def absorb_memories(self, raw: str) -> list[str]:
        from noah_code.project_notes import MemoryStore, parse_distilled_memories

        return MemoryStore(Path(self.workspace_root)).merge(parse_distilled_memories(raw))

    @hidden
    async def distill_memories(self, turn: str) -> str:
        """Extract durable conventions without polluting the coding conversation."""

        return await self._run_auxiliary_predict("distill_memories", turn)

    @hidden
    @property
    def engine(self) -> PermissionEngine:
        return self._engine

    @hidden
    @property
    def approvals(self) -> ApprovalBroker:
        return self._approvals

    @hidden
    @property
    def journal(self) -> SnapshotJournal:
        return self._journal

    @hidden
    def set_mode(self, mode: Literal["build", "plan"]) -> None:
        self.mode = mode
        self._engine.mode = mode
        self.v.mode = mode

    @hidden
    @strategy(
        _AdaptivePermissionCodeActStrategy(
            config=CodeActConfig(max_iterations=40, cell_timeout=120.0)
        )
    )
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """Handle one conversational turn for a coding task.

        Process every notification item as one turn. Follow the Noah CodeAct
        contract and repository context, fulfill the requested end state, and
        return a valid RespondResult with a concise, evidence-based explanation.
        """
        ...

    @hidden
    async def close_tools(self) -> None:
        """Close every owned shell, LSP server, and background process."""

        for unsubscribe in self._observability_unsubs:
            unsubscribe()
        self._observability_unsubs.clear()
        await asyncio.gather(
            self.processes.close(),
            self.lsp.close(),
            self.ws.close(),
            return_exceptions=True,
        )
