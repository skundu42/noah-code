"""Slash-command, worktree, PR, notes, and team handlers for AgentHost."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from noah_code import nooa_compat
from noah_code.commands import config_text, help_text, parse_slash
from noah_code.config import REASONING_EFFORTS
from noah_code.events import HostEvent, HostEventKind
from noah_code.redaction import safe_error_message
from noah_code.steer import SAFE_SLASH_WHILE_BUSY
from noah_code.themes import THEME_NAMES, get_theme
from noah_code.workspace import Workspace

if TYPE_CHECKING:
    from noah_code.agent import CodingAgent
    from noah_code.config import NoahCodeConfig
    from noah_code.custom_commands import CustomCommand
    from noah_code.sessions import SessionMeta, SessionStore
    from noah_code.steer import SteerQueue
    from noah_code.ui.protocol import HostUI
    from noah_code.usage import UsageSnapshot


def _command_output(text: str) -> HostEvent:
    """Return exact, preformatted command output for every UI frontend."""

    return HostEvent(
        HostEventKind.MESSAGE,
        text,
        meta={"format": "plain", "source": "command"},
    )


class HostCommandsMixin:
    """Command-routing surface mixed into :class:`noah_code.host.AgentHost`."""

    workspace: Workspace
    config: NoahCodeConfig
    ui: HostUI
    store: SessionStore
    meta: SessionMeta | None
    steer_queue: SteerQueue
    queue_paused: bool
    _agent: CodingAgent | None
    _runtime: Any
    _checkpoints: Any
    _active_turn: asyncio.Task[Any] | None
    _session_lock: asyncio.Lock
    _exit_requested: bool
    _trace_info: str
    _custom_commands: dict[str, CustomCommand]
    _last_turn_shell_bypass: bool
    _mcp_attached: set[str]
    _mcp_errors: dict[str, str]
    _pending_attach_paths: list[Path]

    if TYPE_CHECKING:

        @property
        def agent(self) -> CodingAgent: ...
        def status_prompt(self) -> str: ...
        def usage_snapshot(self) -> UsageSnapshot: ...
        def _turn_running(self) -> bool: ...
        def _require_idle_turn(self) -> None: ...
        def _require_review_idle(self) -> None: ...
        def _save_input_queue(self) -> None: ...
        def discard_queued_input(self) -> None: ...
        def resume_queued_input(self) -> None: ...
        def pending_attach_paths(self) -> tuple[Path, ...]: ...
        def active_context_paths(self) -> tuple[Path, ...]: ...
        def enqueue_steer(
            self, text: str, attach_paths: list[Path] | None = None
        ) -> bool: ...
        def cancel_active_turn(self) -> None: ...
        async def submit_prompt(self, text: str) -> Any: ...
        async def start_new_session(
            self, workspace: Workspace | None = None, *, worktree_name: str = ""
        ) -> SessionMeta: ...
        async def switch_session(self, session_id: str) -> SessionMeta: ...
        async def _switch_model(
            self, model: str, *, reasoning_effort: str | None = None
        ) -> None: ...
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
        ) -> str: ...
        async def connect_mcp_server(self, name: str) -> str: ...
        async def add_mcp_server(self, kind: str, name: str, target: str) -> str: ...
        def list_mcp_infos(self) -> list[Any]: ...
        def list_skill_infos(self) -> list[Any]: ...
        def add_skill_from_path(self, path: str) -> Any: ...
        async def diff_review(self) -> Any: ...
        async def undo_last_turn_async(self, *, use_checkpoint: bool = False) -> str: ...
        async def redo_last_turn_async(self) -> str: ...
        async def _persist_async(self) -> None: ...
        async def _run_lifecycle(self, event: str, **metadata: Any) -> None: ...
        async def _run_user_turn(
            self,
            text: str,
            *,
            attach_paths: list[Path] | None = None,
            run_id: str | None = None,
            recovery: bool = False,
            queued: bool = False,
        ) -> Any: ...

    def work_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Return presentation-safe live/recent agent and process work."""

        if self._agent is None:
            return {"agents": [], "jobs": []}
        agents: list[dict[str, Any]] = []
        task = getattr(self.agent, "task", None)
        if task is not None:
            with contextlib.suppress(Exception):
                agents = task.snapshot(limit=20)
        jobs: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            jobs = self.agent.processes.snapshot(limit=20)
        return {"agents": agents, "jobs": jobs}

    def work_status_text(self) -> str:
        snapshot = self.work_snapshot()
        agents = snapshot["agents"]
        jobs = snapshot["jobs"]
        lines = ["Work ledger", "", "Agents"]
        if not agents:
            lines.append(
                "  No delegated work yet. Start /team build OBJECTIVE or choose a team with F9."
            )
            lines.append(
                "  Noah can also use task.run_many or task.collaborate for custom assignments."
            )
        for item in agents:
            mode = "read" if item.get("readonly") else "write"
            lines.append(
                f"  {item['id']}  [{item['state']:<9}] {item['agent']} · {mode} · "
                f"{float(item.get('duration', 0.0)):.1f}s"
            )
            if item.get("prompt"):
                lines.append(f"    {str(item['prompt'])[:100]}")
        lines.extend(["", "Terminals and jobs"])
        if not jobs:
            lines.append("  No sessions. Noah can use processes.open_terminal or processes.start.")
        for item in jobs:
            kind = "terminal" if item.get("kind") == "terminal" else "job"
            lines.append(
                f"  {item['id']}  [{item['state']:<9}] {kind} · {item['name']} · "
                f"{float(item.get('elapsed', 0.0)):.1f}s"
            )
        return "\n".join(lines)

    def context_snapshot(self) -> list[dict[str, str]]:
        """Return a display-safe inventory of context sources for the UI."""

        from noah_code.project_notes import MEMORY_RELATIVE, PLAN_RELATIVE, MemoryStore, PlanStore

        rows: list[dict[str, str]] = []
        root = self.workspace.root
        for relative in ("AGENTS.md", "CLAUDE.md", ".noah-code/instructions.md"):
            path = root / relative
            if path.is_file():
                rows.append(
                    {
                        "kind": "instruction",
                        "label": relative,
                        "detail": f"Repository instructions · {path}",
                    }
                )
        if PlanStore(root).exists():
            rows.append(
                {
                    "kind": "plan",
                    "label": PLAN_RELATIVE,
                    "detail": "Pinned project plan included at each turn boundary",
                }
            )
        if MemoryStore(root).exists():
            rows.append(
                {
                    "kind": "memory",
                    "label": MEMORY_RELATIVE,
                    "detail": "Durable project conventions included in model context",
                }
            )
        for path in self.active_context_paths():
            rows.append(
                {
                    "kind": "attachment",
                    "label": path.name,
                    "detail": f"Attached in this session · {path}",
                }
            )
        for path in self.pending_attach_paths():
            rows.append(
                {
                    "kind": "pending",
                    "label": path.name,
                    "detail": f"Waiting for the next prompt · {path}",
                }
            )
        if self._agent is not None:
            with contextlib.suppress(Exception):
                for info in self.list_skill_infos():
                    if info.active:
                        rows.append(
                            {
                                "kind": "skill",
                                "label": info.name,
                                "detail": f"Active skill · {info.source}",
                            }
                        )
        for name in sorted(self._mcp_attached):
            rows.append(
                {
                    "kind": "mcp",
                    "label": name,
                    "detail": "Connected MCP server tools available to Noah",
                }
            )
        return rows

    def context_status_text(self) -> str:
        rows = self.context_snapshot()
        if not rows:
            return "Context sources\n  Workspace defaults only"
        lines = ["Context sources"]
        lines.extend(f"  {row['kind']:<11} {row['label']}\n    {row['detail']}" for row in rows)
        return "\n".join(lines)

    def worktree_manager(self) -> Any:
        from noah_code.worktree import WorktreeManager, worktree_storage_root

        return WorktreeManager(self.workspace.root, worktree_storage_root(self.config.session_dir))

    async def create_worktree_session(
        self, name: str | None = None, *, include_dirty: bool = False
    ) -> SessionMeta:
        """Create a linked worktree and start a new session there."""

        self._require_idle_turn()
        manager = self.worktree_manager()
        try:
            info = await asyncio.to_thread(manager.create, name, include_dirty=include_dirty)
        except TypeError:
            info = await asyncio.to_thread(manager.create, name)
        try:
            meta = await self.start_new_session(
                Workspace(root=info.directory), worktree_name=info.name
            )
            await self._run_lifecycle(
                "worktree_created", name=info.name, directory=str(info.directory)
            )
            return meta
        except Exception:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.worktree_manager().remove, info.name)
            raise

    def remove_worktree(self, name: str) -> Any:
        from noah_code.worktree import WorktreeError

        manager = self.worktree_manager()
        matches = [
            item for item in manager.list() if item.name == name or str(item.directory) == name
        ]
        if matches and matches[0].directory.resolve() == self.workspace.root.resolve():
            raise WorktreeError("switch away from this worktree before removing it")
        return manager.remove(name)

    async def _handle_worktree(self, args: str) -> Literal["handled"]:
        from noah_code.worktree import WorktreeError

        parts = shlex.split(args) if args.strip() else []
        action = parts[0] if parts else "create"
        rest = parts[1:]
        if action == "create":
            include_dirty = "--dirty" in rest
            filtered = [item for item in rest if item != "--dirty"]
            name = filtered[0] if filtered else None
            try:
                meta = await self.create_worktree_session(name, include_dirty=include_dirty)
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
                rows = await asyncio.to_thread(self.worktree_manager().list)
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
                info = await asyncio.to_thread(self.remove_worktree, rest[0])
            except WorktreeError as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            self.ui.render(HostEvent(HostEventKind.STATUS, f"removed worktree {info.name}"))
            return "handled"
        self.ui.render(
            HostEvent(
                HostEventKind.ERROR,
                "usage: /worktree [create [--dirty] [NAME]|list|remove NAME]",
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
            content = await asyncio.to_thread(store.read)
            self.ui.render(_command_output(content.strip() or "(no active plan)"))
            return "handled"
        if verb == "clear":
            await asyncio.to_thread(store.clear)
            if self._agent is not None:
                await asyncio.to_thread(self.agent.refresh_context_sources)
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
            content = await asyncio.to_thread(store.read)
            self.ui.render(_command_output(content.strip() or "(no project memory yet)"))
            return "handled"
        if verb == "save":
            facts = parse_memory_facts(f"- {rest}" if rest else "")
            if not facts:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /memory save FACT"))
                return "handled"
            added = await asyncio.to_thread(store.merge, facts)
            if self._agent is not None:
                await asyncio.to_thread(self.agent.refresh_context_sources)
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
            forgotten = await asyncio.to_thread(store.forget, rest)
            if self._agent is not None:
                await asyncio.to_thread(self.agent.refresh_context_sources)
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"forgot {rest}" if forgotten else "no matching memory",
                )
            )
            return "handled"
        if verb == "clear":
            await asyncio.to_thread(store.clear)
            if self._agent is not None:
                await asyncio.to_thread(self.agent.refresh_context_sources)
            self.ui.render(HostEvent(HostEventKind.STATUS, "memory cleared"))
            return "handled"
        self.ui.render(
            HostEvent(HostEventKind.ERROR, "usage: /memory [save FACT|forget TEXT|clear]")
        )
        return "handled"

    def team_status_text(self) -> str:
        """Describe coordinated workflows without starting model work."""

        from noah_code.teams import team_workflows

        lines = ["Agent teams", ""]
        for workflow in team_workflows():
            mode = "read-only" if workflow.readonly else "build mode required"
            phases = " → ".join(phase.title for phase in workflow.phases)
            lines.extend(
                [
                    f"  /team {workflow.name} OBJECTIVE · {mode}",
                    f"    {workflow.description}",
                    f"    {phases}",
                ]
            )
        lines.extend(
            [
                "",
                "Example: /team review Find regressions in the current changes",
                "Track progress with /work or F4. Teams share this session's budget and approvals.",
            ]
        )
        return "\n".join(lines)

    def prepare_team_prompt(self, args: str) -> str:
        """Validate a team request and route it through the normal agent lifecycle."""

        from noah_code.teams import get_team_workflow

        parts = args.strip().split(maxsplit=1)
        if not parts:
            raise ValueError("usage: /team <build|review|investigate> OBJECTIVE")
        workflow = get_team_workflow(parts[0].lower())
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError(f"Add an objective: /team {workflow.name} OBJECTIVE")
        if not workflow.readonly and self.agent.mode != "build":
            raise ValueError(
                "The build team requires build mode. Use /mode build first, "
                "or choose /team review or /team investigate for read-only work."
            )
        objective = parts[1].strip()
        access_instruction = (
            "This is a read-only request: do not modify files or switch to build mode. "
            if workflow.readonly
            else ""
        )
        return (
            f"Run the {workflow.title} team workflow for the objective below. "
            f"Call await self.task.team(objective, workflow={workflow.name!r}) with that "
            "objective, then inspect the returned reports and provide one consolidated answer "
            "with observed validation and any unresolved blockers. Respect the active mode, "
            "repository instructions, and existing user changes. "
            f"{access_instruction}\n\n"
            f"Objective:\n{objective}"
        )

    async def _handle_team(self, args: str) -> Literal["continue", "exit", "handled"]:
        if not args.strip():
            self.ui.render(_command_output(self.team_status_text()))
            return "handled"
        try:
            prompt = self.prepare_team_prompt(args)
        except ValueError as exc:
            self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return "handled"
        another_turn_running = (
            self._turn_running() and self._active_turn is not asyncio.current_task()
        )
        if another_turn_running or self.queue_paused:
            self.enqueue_steer(prompt)
            state = "paused" if self.queue_paused else "queued"
            self.ui.render(HostEvent(HostEventKind.STATUS, f"team {state} · /work to track progress"))
            return "handled"
        return await self.handle_line(prompt)

    async def handle_line(self, line: str) -> Literal["continue", "exit", "handled"]:
        slash = parse_slash(line)
        if slash:
            if slash[0] == "team":
                return await self._handle_team(slash[1])
            return await self._handle_slash(slash[0], slash[1])
        result = await self.submit_prompt(line)
        return "continue" if result is not None else "handled"

    def _handle_slash_while_busy(
        self, name: str, args: str
    ) -> Literal["continue", "exit", "handled"] | None:
        """Allow read-only slash commands during a turn; block mutating ones."""

        if not (self._turn_running() or self._session_lock.locked()) or name in SAFE_SLASH_WHILE_BUSY:
            return None
        if name in {"exit", "quit"}:
            return None
        if name == "attach":
            from noah_code.composer import validate_attachment

            raw = args.strip().strip("\"'")
            if not raw:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /attach PATH"))
                return "handled"
            try:
                path = validate_attachment(
                    raw,
                    self.workspace.root,
                    engine=self.agent.engine,
                    max_file_bytes=self.config.max_file_bytes,
                )
            except (PermissionError, ValueError) as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            self._pending_attach_paths.append(path)
            self._save_input_queue()
            self.ui.render(HostEvent(HostEventKind.STATUS, f"attach queued · {path}"))
            return "handled"
        self.ui.render(
            HostEvent(HostEventKind.STATUS, f"/{name} is blocked while a turn is running")
        )
        return "handled"

    async def _handle_slash(self, name: str, args: str) -> Literal["continue", "exit", "handled"]:
        from noah_code import host as _host_mod

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
            path = await asyncio.to_thread(_host_mod.save_user_theme, selected)
            self.config.ui.theme = selected
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"theme set to {selected} in {path}",
                    meta={"kind": "theme", "theme": selected},
                )
            )
            return "handled"
        if name == "animations":
            requested = args.strip().lower()
            if not requested:
                state = "on" if self.config.ui.animations else "off"
                self.ui.render(HostEvent(HostEventKind.STATUS, f"animations={state}"))
                return "handled"
            if requested not in {"on", "off"}:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /animations [on|off]"))
                return "handled"
            enabled = requested == "on"
            path = await asyncio.to_thread(_host_mod.save_user_animations, enabled)
            self.config.ui.animations = enabled
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    f"animations set to {requested} · saved in {path}",
                    meta={"kind": "animations", "enabled": enabled},
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
                    rows = await asyncio.to_thread(self.list_mcp_infos)
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
                    text += (
                        "\n\nAdd with: /mcp add stdio NAME COMMAND…\n"
                        "          /mcp add http NAME URL"
                    )
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
            upgrade_lease = getattr(self, "_ensure_mode_lease", None)
            if callable(upgrade_lease):
                try:
                    await upgrade_lease(mode)
                except Exception as exc:  # noqa: BLE001
                    self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                    return "handled"
            mode_lit: Literal["build", "plan"] = "plan" if mode == "plan" else "build"
            agent.set_mode(mode_lit)
            if self.meta:
                self.meta.mode = mode_lit
                await asyncio.to_thread(self.store.save_meta, self.meta)
            self.ui.render(HostEvent(HostEventKind.STATUS, f"mode set to {mode}"))
            self.ui.set_status(self.status_prompt())
            return "handled"
        if name == "model":
            requested = args.strip()
            if not requested:
                default_model = _host_mod.user_default_model() or "(not configured)"
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
                path = await asyncio.to_thread(_host_mod.save_user_default_model, global_model)
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
                path = await asyncio.to_thread(_host_mod.save_user_reasoning_effort, requested)
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

                    text = await asyncio.to_thread(
                        format_providers, self.meta.model if self.meta else self.config.model
                    )
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
            rows = await asyncio.to_thread(self.store.list_sessions, self.workspace)
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
        if name == "tasks":
            parts = args.strip().split(maxsplit=2)
            tasks = getattr(agent, "task", None)
            try:
                if tasks is None:
                    text = "(no child tasks)"
                elif not parts or len(parts) == 1:
                    text = tasks.status(parts[0] if parts else "")
                elif parts[0] == "cancel" and len(parts) == 2:
                    text = await tasks.cancel(parts[1])
                elif parts[0] == "diff" and len(parts) == 2 and hasattr(tasks, "diff"):
                    text = await tasks.diff(parts[1])
                elif parts[0] == "apply" and len(parts) == 2 and hasattr(tasks, "apply"):
                    self._require_review_idle()
                    async with self._session_lock:
                        self._active_turn = asyncio.current_task()
                        self.ui.set_busy(True)
                        agent.journal.begin_turn()
                        try:
                            text = await tasks.apply(parts[1])
                        finally:
                            agent.journal.end_turn()
                            try:
                                await self._persist_async()
                            finally:
                                self._active_turn = None
                                self.ui.set_busy(False)
                elif parts[0] == "cleanup" and len(parts) == 2 and hasattr(tasks, "cleanup"):
                    text = await tasks.cleanup(parts[1])
                elif parts[0] == "follow" and len(parts) == 3:
                    if self._turn_running():
                        raise RuntimeError(
                            "wait for the current turn before following up with a child"
                        )
                    async with self._session_lock:
                        self._active_turn = asyncio.current_task()
                        self.ui.set_busy(True)
                        agent.journal.begin_turn()
                        try:
                            text = await tasks.follow_up(parts[1], parts[2])
                        finally:
                            agent.journal.end_turn()
                            try:
                                await self._persist_async()
                            finally:
                                self._active_turn = None
                                self.ui.set_busy(False)
                else:
                    text = (
                        "usage: /tasks [ID | cancel ID | diff ID | apply ID | "
                        "cleanup ID | follow ID PROMPT]"
                    )
                self.ui.render(_command_output(text))
            except Exception as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, safe_error_message(exc)))
            return "handled"
        if name == "work":
            self.ui.render(_command_output(self.work_status_text()))
            return "handled"
        if name == "queue":
            if args.strip() == "discard":
                self.discard_queued_input()
            elif args.strip() == "resume":
                self.resume_queued_input()
                if not self._turn_running() and self.steer_queue.items():
                    async with self._session_lock:
                        self.ui.set_busy(True)
                        self._active_turn = asyncio.current_task()
                        try:
                            await self._run_user_turn(self.steer_queue.items()[0].text, queued=True)
                        finally:
                            self._active_turn = None
                            self.ui.set_busy(False)
                            self.ui.set_status(self.status_prompt())
                    return "continue"
            queued = self.steer_queue.items()
            attached = self.pending_attach_paths()
            state = "paused" if self.queue_paused else "waiting"
            lines = [
                f"Queued prompts: {len(queued)} · {state}",
                f"Pending attachments: {len(attached)}",
            ]
            for index, item in enumerate(queued, start=1):
                preview = " ".join(item.text.split())[:80]
                suffix = f" · {len(item.attach_paths)} file(s)" if item.attach_paths else ""
                lines.append(f"  {index}. {preview}{suffix}")
            for index, path in enumerate(attached, start=1):
                lines.append(f"  A{index}. {path}")
            self.ui.render(_command_output("\n".join(lines)))
            return "handled"
        if name == "terminals":
            self.ui.render(_command_output(await agent.processes.terminal_status()))
            return "handled"
        if name == "attach":
            from noah_code.composer import validate_attachment

            raw = args.strip().strip("\"'")
            if not raw:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /attach PATH"))
                return "handled"
            try:
                path = validate_attachment(
                    raw,
                    self.workspace.root,
                    engine=self.agent.engine,
                    max_file_bytes=self.config.max_file_bytes,
                )
            except (PermissionError, ValueError) as exc:
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
                return "handled"
            async with self._session_lock:
                self.ui.set_busy(True)
                self._active_turn = asyncio.current_task()
                try:
                    await self._run_user_turn(
                        "Please inspect the attached file.", attach_paths=[path]
                    )
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
        if name == "context":
            self.ui.render(_command_output(self.context_status_text()))
            return "handled"
        if name == "timeline":
            self.ui.render(
                HostEvent(
                    HostEventKind.STATUS,
                    "Open the task timeline with F2 in the TUI",
                )
            )
            return "handled"
        if name == "health":
            health = (
                await asyncio.to_thread(self._runtime.health)
                if self._runtime is not None
                else {}
            )
            rows = ["Durable runtime health"]
            rows.extend(f"  {key}  {value}" for key, value in health.items())
            self.ui.render(_command_output("\n".join(rows)))
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
                        f"Changes · {len(review.files)} views · select a file to load its patch"
                    ]
                    lines.extend(
                        f"  {item.scope[0].upper()} {item.path} · {item.status}"
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
            use_checkpoint = args.strip().lower() in {"--checkpoint", "checkpoint"}
            try:
                status = await self.undo_last_turn_async(use_checkpoint=use_checkpoint)
                self.ui.render(HostEvent(HostEventKind.STATUS, status))
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, str(exc)))
            return "handled"
        if name == "redo":
            try:
                status = await self.redo_last_turn_async()
                self.ui.render(HostEvent(HostEventKind.STATUS, status))
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
                hint = "" if enabled else "\nEnable with [checkpoints] enabled=true"
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
            self._require_review_idle()
            async with self._session_lock:
                self._active_turn = asyncio.current_task()
                self.ui.set_busy(True)
                try:
                    compacted = await agent.compact_history()
                    if compacted:
                        await self._persist_async()
                except Exception as exc:  # noqa: BLE001
                    self.ui.render(HostEvent(HostEventKind.ERROR, f"compact failed: {exc}"))
                    return "handled"
                finally:
                    self._active_turn = None
                    self.ui.set_busy(False)
            status = "history compacted" if compacted else "nothing eligible to compact"
            self.ui.render(HostEvent(HostEventKind.STATUS, status))
            return "handled"
        if name == "continue":
            latest = await asyncio.to_thread(self.store.latest_for_workspace, self.workspace)
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
        rendered = cmd.render(args)
        if not rendered:
            self.ui.render(
                HostEvent(HostEventKind.ERROR, f"custom command /{name} produced empty body")
            )
            return "handled"
        prior_mode = self.agent.mode
        prior_model = self.meta.model if self.meta else self.config.model
        project_source = cmd.source.startswith("project:")
        target_mode: Literal["build", "plan"] | None = None
        if cmd.mode in {"build", "plan"}:
            if project_source and cmd.mode == "build" and prior_mode == "plan":
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"project command /{name} cannot leave plan mode; staying in plan",
                    )
                )
            else:
                target_mode = cmd.mode  # type: ignore[assignment]
        target_model = None if project_source else cmd.model
        mode_changed = False
        model_changed = False
        if target_mode is not None and target_mode != prior_mode:
            self.agent.set_mode(target_mode)
            mode_changed = True
            if self.meta:
                self.meta.mode = target_mode
                await asyncio.to_thread(self.store.save_meta, self.meta)
        if target_model and target_model != prior_model:
            try:
                await self._switch_model(target_model)
                model_changed = True
            except Exception as exc:  # noqa: BLE001
                self.ui.render(HostEvent(HostEventKind.ERROR, f"model switch failed: {exc}"))
        self.ui.render(HostEvent(HostEventKind.STATUS, f"running /{name} ({cmd.source})"))
        async with self._session_lock:
            self.ui.set_busy(True)
            self._active_turn = asyncio.current_task()
            try:
                await self._run_user_turn(rendered)
            finally:
                if mode_changed and self._agent is not None:
                    with contextlib.suppress(Exception):
                        self.agent.set_mode(prior_mode)
                    if self.meta:
                        self.meta.mode = prior_mode
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(self.store.save_meta, self.meta)
                if model_changed and self._agent is not None:
                    with contextlib.suppress(Exception):
                        await self._switch_model(prior_model)
                self._active_turn = None
                self.ui.set_busy(False)
                self.ui.set_status(self.status_prompt())
        return "continue"
