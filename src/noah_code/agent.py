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
from nooa.config import CodeActConfig, PredictConfig
from nooa.interactive import InteractiveAgent, RespondReason, RespondResult
from nooa.runtime.restrictions import RESTRICTED_MODULES, RestrictionsConfig
from nooa.runtime.sandbox.config import FileRule, SandboxConfig, resolve_spec
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.strategies.codeact_lite import PlainCodeActBlockFormatter
from nooa.tools import TodoManager
from nooa.tools.shell_tools import ShellTools

from noah_code import nooa_compat
from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.macos_sandbox import build_macos_profile, macos_worker_main
from noah_code.permissions import PermissionEngine
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
            ("ws", "list"),
            ("ws", "list_files"),
            ("ws", "read"),
            ("ws", "read_output"),
            ("ws", "replace"),
            ("ws", "run"),
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
            ("processes", "input"),
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


def _cache_first_block_order(base: list[str] | None) -> list[str]:
    """Stable-prefix-first ordering: static instructions before volatile blocks."""

    return [
        *(base or []),
        "repo_instructions",
        "agents",
        "subagent",
        "workspace",
        "active_plan",
        "project_memory",
    ]


class _PermissionCodeActStrategy(CodeActStrategy):
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

    def get_block_order(self) -> list[str]:
        return _cache_first_block_order(super().get_block_order())

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

    def get_block_order(self) -> list[str]:
        return _cache_first_block_order(super().get_block_order())

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
        nested: bool = False,
        nested_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, storage=storage, **kwargs)
        self._lightweight_llm = lightweight_llm or self._llm
        self.workspace_root = str(workspace.root)
        self.mode = config.mode
        self._config = config
        self._nested = nested

        self._engine = engine or PermissionEngine(
            config.permission_rules,
            mode=config.mode,
            auto_approve=config.auto_approve,
        )
        self._engine.mode = config.mode
        self._approvals = approvals or ApprovalBroker(self._engine)
        self._journal = journal or SnapshotJournal(blob_limit=config.undo_blob_limit)

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
        )
        self.processes = ProcessTools(
            self.ws,
            max_jobs=config.processes.max_jobs,
            max_runtime_seconds=config.processes.max_runtime_seconds,
            max_buffer_chars=config.processes.max_buffer_chars,
            stop_grace_seconds=config.processes.stop_grace_seconds,
        )
        self.todos = TodoManager()
        self.git = GitTools(self.ws)
        self.github = GithubTools(workspace.root, self._engine, self._approvals)
        self.web = WebTools(self._engine, self._approvals)
        self.ask = QuestionTools(self._engine, self._approvals)
        self.plan = PlanTools(workspace.root, self, self.ask, self._engine, self._approvals)
        self.memory = MemoryTools(
            workspace.root,
            self._engine,
            self._approvals,
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

            max_tokens = config.summarization.max_tokens
            context_window = getattr(self._llm, "context_window", None)
            if max_tokens is None and context_window:
                max_tokens = int(context_window * config.summarization.trigger_ratio)
            CodingSessionSummarizer.install(
                self,
                llm=self._lightweight_llm,
                config=TokenBudgetConfig(
                    max_tokens=max_tokens or 100_000,
                    preserve_recent=config.summarization.preserve_recent,
                    target_chars=config.summarization.target_chars,
                ),
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

        context_window = getattr(self._llm, "context_window", None)
        maximum = self._config.summarization.max_tokens or (
            int(context_window * self._config.summarization.trigger_ratio)
            if context_window
            else 100_000
        )
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
    @strategy(
        PredictStrategy(PredictConfig(output_serialization="tool_call")),
        llm=lambda agent: agent._lightweight_llm,
    )
    async def name_session(self, user_message: str) -> str:
        """Generate an ultra-short 2-5 word coding-session title.

        Conversation starts with: {user_message}
        """

        ...

    @hidden
    @strategy(
        PredictStrategy(PredictConfig(output_serialization="tool_call")),
        llm=lambda agent: agent._lightweight_llm,
    )
    async def distill_result(self, transcript: str) -> str:
        """Compress one subagent work transcript into a compact report for its parent.

        Transcript begins: {transcript}

        Return only these sections when they contain real information, as short
        plain lines without markdown headers:
        Findings - concrete answers with file paths and symbol names.
        Changes - files edited and what changed in each.
        Validation - commands run and their exact observed outcomes.
        Open - unresolved problems or questions that need the user.

        Preserve exact paths, identifiers, commands, and numbers. Omit chatter,
        raw tool output, and superseded attempts. Never claim success that the
        transcript does not show. Stay under 1200 characters.
        """

        ...

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
    @strategy(
        PredictStrategy(PredictConfig(output_serialization="tool_call")),
        llm=lambda agent: agent._lightweight_llm,
    )
    async def distill_memories(self, turn: str) -> str:
        """Extract standing project conventions from this turn.

        Turn begins: {turn}

        Return EMPTY if nothing should persist across sessions.
        Otherwise return one MEMORY: line per standing convention such as
        package manager, forbidden paths, or PR title style.
        Never include secrets, task status, or one-off bugs. At most 8 lines.
        """

        ...

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

        Read all user messages, slash-command results, and system messages
        in the notification. Understand the requested end state before editing.

        Minimal tool cookbook:
        - ``await self.ws.list("**/*.py")`` lists files.
        - ``await self.ws.search("symbol")`` returns locations as text.
        - ``match = await self.ws.read("path.py", lines=(10, 30))`` returns an
          editable Match; ``await self.ws.replace(match, "replacement")`` edits it.
        - ``await self.ws.edit("path.py", "unique old text", "new text")`` is the
          simple string-edit form. ``await self.ws.write("new.py", content)`` creates files.
        - Prefer ``await self.ws.apply_patch(changes)`` for coherent edits. Each change is
          ``{"path": ..., "old": exact_text_or_None, "new": replacement_or_None}``;
          one call validates and atomically commits the full batch.
        - ``await self.ws.apply_unified_diff(diff_text)`` applies a unified diff
          (git-style ---/+++/@@ hunks) with the same atomic verification.
        - Use ``self.lsp`` for definitions, implementations, references, symbols, hover,
          diagnostics, rename previews, and compact repository maps before broad searches.
        - Use ``self.processes.start/logs/status/input/stop`` for servers, watchers, and
          long-running commands. Consume logs by cursor; do not poll without new work.
        - ``result = await self.ws.run("pytest -q")`` runs validation; inspect
          ``result.returncode``, ``result.stdout``, and ``result.stderr``.
        - ``await self.web.fetch(url)`` reads a page; ``await self.web.search(query)``
          searches the public web. Both ask for approval by default.
        - ``await self.github.list()`` / ``view(number)`` inspect pull requests.
        - ``await self.github.create(title, body)`` pushes HEAD and opens a PR.
        - ``await self.github.push()`` updates the remote branch.
        - ``await self.github.checkout(number)`` checks out ``pr/<number>``.
        - ``await self.github.comment(number, text)`` comments on a PR.
        - ``await self.ask.question(header, prompt, options)`` pauses for a user choice.
        - ``await self.plan.write(markdown)`` pins ``.noah-code/plan.md`` (allowed in plan mode).
        - ``await self.plan.exit_to_build()`` asks to switch to build after a plan exists.
        - ``await self.plan.enter()`` asks to switch to plan mode. Do not call ``set_mode``.
        - ``await self.memory.save(fact)`` remembers a standing project convention.
        - ``await self.task.run("explore", "...")`` or ``"general"`` runs a nested
          NOOA subagent with isolated history; results arrive condensed. Use
          ``await self.task.run_many([("explore", "..."), ...])`` to fan out
          bounded independent units concurrently.
        - ``self.task.list()`` shows markdown agents.
        - If ``self.media.pending()`` is non-empty, ``show()`` each ``self.media.consume()``
          image before reasoning. ``show`` is a CodeAct builtin; do not import ``nooa``.

        Workflow:
        - If the user attached images, show them first.
        - Inspect relevant repository instructions and nearby code first.
        - Prefer ``self.ws.search`` / focused ``self.ws.read`` over dumping large files.
        - Delegate bounded research or parallel units with ``self.task.run``.
        - Use ``self.todos`` for genuinely multi-step tasks; keep todos current.
        - Make the smallest coherent change, preferring one atomic ``self.ws.apply_patch``.
        - Preserve unrelated user modifications.
        - Run validation proportional to risk (focused tests, not entire suites).
        - Never claim a command or test passed unless its successful result was observed.
        - Report blockers concretely via ``self.message(...)``.
        - In plan mode (see ``self.mode``), do not modify files or run mutating commands.
          Write the checklist with ``self.plan.write`` then ``self.plan.exit_to_build``.
        - If an active plan is loaded, implement that plan; do not expand scope.
        - Do not commit, push, publish, or create external resources unless explicitly asked.
          When asked to open or update a pull request, use ``self.github`` — never ``git push``
          or mutating ``gh pr`` through the shell.
        - Do not read secrets or expose sensitive environment values.

        Return exactly one valid RespondResult:
        - DONE - request complete
        - NEED_INPUT - user input genuinely required
        - WAIT - a registered background job is still running
        """
        ...

    @hidden
    async def close_tools(self) -> None:
        """Close every owned shell, LSP server, and background process."""

        await asyncio.gather(
            self.processes.close(),
            self.lsp.close(),
            self.ws.close(),
            return_exceptions=True,
        )
