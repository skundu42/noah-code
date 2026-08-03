"""CodingAgent - InteractiveAgent for repository work."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import platform
import site
import sys
import sysconfig
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any, Literal

from nooa import Context, hidden, strategy
from nooa.config import CodeActConfig
from nooa.interactive import (
    InteractiveAgent,
    RespondResult,
    SummarizationConfig,
    install_summarizer,
)
from nooa.runtime.restrictions import RESTRICTED_MODULES, RestrictionsConfig
from nooa.runtime.sandbox.config import FileRule, SandboxConfig, resolve_spec
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.strategies import CodeActStrategy
from nooa.tools import TodoManager
from nooa.tools.shell_tools import ShellTools

from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.macos_sandbox import build_macos_profile, macos_worker_main
from noah_code.permissions import PermissionEngine
from noah_code.snapshots import SnapshotJournal
from noah_code.tools.git_tools import GitTools
from noah_code.tools.workspace_tools import WorkspaceTools
from noah_code.workspace import Workspace


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
            ("message",),
            ("mode",),
            ("workspace_root",),
            ("ws", "list_files"),
            ("ws", "read"),
            ("ws", "replace"),
            ("ws", "run"),
            ("ws", "search"),
            ("ws", "write_file"),
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
        approved_roots = getattr(self._agent, "_sandbox_approved_roots", set())
        dynamically_allowed = bool(normalized and normalized[0] in approved_roots)
        if not normalized or not (self._path_allowed(normalized) or dynamically_allowed):
            display = ".".join(path) or "<root>"
            raise PermissionError(f"sandbox broker access denied: self.{display}")
        return super()._walk_path(path)


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
        # NOOA 0.0.8 probes specifically for Linux Landlock/seccomp. Preserve
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
        self._ctx = mp.get_context(config.start_method)
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
        init = {
            "agent": self._agent,
            "framework_builtins": self._framework_builtins,
            "restrictions": self._restrictions,
            "spec": self._spec,
        }
        proc = self._ctx.Process(
            target=macos_worker_main,
            args=(
                child_conn,
                init,
                self._macos_profile,
                self._macos_max_memory_mb,
                self._macos_max_cpu_seconds,
            ),
            daemon=True,
            name="nooa-macos-sandbox-worker",
        )
        proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc


class _PermissionCodeActStrategy(CodeActStrategy):
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
        storage: Any = None,
        engine: PermissionEngine | None = None,
        approvals: ApprovalBroker | None = None,
        journal: SnapshotJournal | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, storage=storage, **kwargs)
        self.workspace_root = str(workspace.root)
        self.mode = config.mode
        self._config = config

        self._engine = engine or PermissionEngine(
            config.permission_rules,
            mode=config.mode,
            auto_approve=config.auto_approve,
        )
        self._engine.mode = config.mode
        self._approvals = approvals or ApprovalBroker(self._engine)
        self._journal = journal or SnapshotJournal(blob_limit=config.undo_blob_limit)

        shell = ShellTools(cwd=str(workspace.root))
        self._shell: Annotated[ShellTools, hidden] = shell

        self.ws = WorkspaceTools(
            workspace,
            shell,
            self._engine,
            self._approvals,
            self._journal,
            max_output_chars=config.max_output_chars,
            default_timeout=config.command_timeout,
        )
        self.todos = TodoManager()
        self.git = GitTools(self.ws)
        self._sandbox_approved_roots: set[str] = set()

        from noah_code.skills_setup import install_skills

        self._skills_status = install_skills(self, workspace.root, config)

        # Apply instance CodeAct limits without mutating other agents' class attrs.
        # CodeActConfig is frozen, so replace the strategy object on this method
        # only when values differ from the class decorator defaults.
        desired = _codeact_config(config)
        current = getattr(type(self).handle, "_strategy_override", None)
        if current is None or getattr(current, "config", None) != desired:
            # Bound on the unbound function object - acceptable for a single-process CLI.
            type(self).handle._strategy_override = _PermissionCodeActStrategy(config=desired)

        # Bounded live context - not full trees/diffs.
        self.context["workspace"] = Context(
            expr="f'workspace={self.workspace_root}\\nmode={self.mode}'"
        )
        self.context["todos"] = Context(expr="self.todos.status()")
        self.context["git"] = Context(expr="self._git_summary()")

        if config.summarization.policy != "none":
            install_summarizer(
                SummarizationConfig(
                    policy=config.summarization.policy,
                    max_tokens=config.summarization.max_tokens,
                    preserve_recent=config.summarization.preserve_recent,
                    target_chars=config.summarization.target_chars,
                ),
                self,
            )

        # Discover AGENTS.md / README hints without dumping trees.
        self.context["repo_instructions"] = Context(expr="self._repo_instructions()")

    @hidden
    def _git_summary(self) -> str:
        import subprocess

        try:
            status = subprocess.run(
                ["git", "status", "--short", "--branch"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            out = (status.stdout or status.stderr or "").strip()
            lines = out.splitlines()
            if len(lines) > 30:
                return "\n".join(lines[:30]) + f"\n...[{len(lines) - 30} more]"
            return out or "(not a git repo or empty status)"
        except (OSError, subprocess.SubprocessError):
            return "(git status unavailable)"

    @hidden
    def _repo_instructions(self) -> str:
        root = Path(self.workspace_root)
        chunks: list[str] = []
        for name in ("AGENTS.md", "CLAUDE.md", ".noah-code/instructions.md"):
            path = root / name
            if path.is_file():
                text = path.read_text(errors="replace")
                if len(text) > 4000:
                    text = text[:4000] + "\n...(truncated)..."
                chunks.append(f"## {name}\n{text}")
        return "\n\n".join(chunks) if chunks else "(no repository instruction files found)"

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
        _PermissionCodeActStrategy(config=CodeActConfig(max_iterations=40, cell_timeout=120.0))
    )
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """Handle one conversational turn for a coding task.

        Read all user messages, slash-command results, and system messages
        in the notification. Understand the requested end state before editing.

        Workflow:
        - Inspect relevant repository instructions and nearby code first.
        - Prefer ``self.ws.search`` / focused ``self.ws.read`` over dumping large files.
        - Use ``self.todos`` for genuinely multi-step tasks; keep todos current.
        - Make the smallest coherent change with Match-based ``self.ws.replace``.
        - Preserve unrelated user modifications.
        - Run validation proportional to risk (focused tests, not entire suites).
        - Never claim a command or test passed unless its successful result was observed.
        - Report blockers concretely via ``self.message(...)``.
        - In plan mode (see ``self.mode``), do not modify files or run mutating commands;
          return an evidence-based plan with file references.
        - Do not commit, push, publish, or create external resources unless explicitly asked.
        - Do not read secrets or expose sensitive environment values.

        Return exactly one valid RespondResult:
        - DONE - request complete
        - NEED_INPUT - user input genuinely required
        - WAIT - a registered background job is still running
        """
        ...
