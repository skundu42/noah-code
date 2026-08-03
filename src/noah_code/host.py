"""Pure-Python host: dispatcher loop, persistence, approvals, slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nooa.interactive import RespondReason, apply_model_limits
from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

from noah_code.agent import CodingAgent
from noah_code.approvals import ApprovalChoice
from noah_code.commands import config_text, help_text, parse_slash
from noah_code.config import NoahCodeConfig, save_user_default_model, user_default_model
from noah_code.custom_commands import CustomCommand, discover_custom_commands
from noah_code.event_bridge import install_event_bridge
from noah_code.events import HostEvent, HostEventKind
from noah_code.sessions import SessionEventRecord, SessionMeta, SessionStore
from noah_code.ui.console import ConsoleUI
from noah_code.ui.protocol import HostUI
from noah_code.workspace import Workspace

logger = logging.getLogger(__name__)


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
            return {str(k): _json_safe(v) for k, v in value.items()}  # type: ignore[arg-type]
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
        self._storage = None
        self.meta = session_meta
        self._cancel_turn = False
        self._exit_requested = False
        self._last_stop = ""
        self._trace_info = "auto (viewer if reachable)"
        self._active_turn: asyncio.Task[Any] | None = None
        self._event_unsubs: list[Any] = []
        self._custom_commands: dict[str, CustomCommand] = {}
        self._last_turn_shell_bypass = False
        self.on_session_changed: Any = None  # optional UI callback

    @property
    def agent(self) -> CodingAgent:
        if self._agent is None:
            raise RuntimeError("host not started")
        return self._agent

    def _setup_tracing(self, session_id: str) -> None:
        if not self.config.tracing.enabled:
            return
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
        if self.meta is None:
            self.meta = self.store.create(
                self.workspace, model=self.config.model, mode=self.config.mode
            )
        else:
            self.store.verify_workspace(self.meta, self.workspace)

        self._setup_tracing(self.meta.session_id)
        self._storage = self.store.open_storage(self.meta.session_id)

        llm = self._llm
        if llm is None:
            from nooa.unifiedllm import get_llm_client

            llm = get_llm_client(self.meta.model)

        agent = CodingAgent(
            self.workspace,
            self.config,
            llm=llm,
            storage=self._storage,
        )
        # Restore snapshot if present.
        restored = self._storage.restore_latest_snapshot(agent)
        if restored:
            # Re-bind host-owned nosnapshot infrastructure after restore.
            agent._engine.mode = getattr(agent, "mode", self.meta.mode)  # type: ignore[attr-defined]
            agent._engine.load_session_rules(self.meta.permission_rules)
            agent.journal.load_dict(self.meta.journal)
            if self.meta.todos:
                agent.todos.from_dict(self.meta.todos)
        else:
            agent.set_mode(self.meta.mode)  # type: ignore[arg-type]
            agent.v.mode = self.meta.mode
            agent.v.model = self.meta.model

        agent._approvals.set_handler(self.ui.ask_approval)
        agent._render_message = self._on_agent_message
        agent.ws.set_shell_chunk_handler(self._on_shell_chunk)

        self._teardown_event_bridge()
        self._event_unsubs = install_event_bridge(agent, self.ui.render)

        self._custom_commands = discover_custom_commands(self.workspace.root)

        self._agent = agent

        # Optional MCP (best-effort; never blocks startup on missing extra).
        try:
            from noah_code.mcp_setup import install_mcp

            mcp_status = await install_mcp(
                agent,
                self.workspace.root,
                self.config,
                engine=agent.engine,
                approvals=agent.approvals,
            )
            self.ui.render(HostEvent(HostEventKind.STATUS, mcp_status))
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

    def _teardown_event_bridge(self) -> None:
        for unsub in self._event_unsubs:
            with contextlib.suppress(Exception):
                unsub()
        self._event_unsubs = []

    def _on_shell_chunk(self, stream: str, text: str) -> None:
        self.ui.render(HostEvent(HostEventKind.SHELL_CHUNK, text, meta={"stream": stream}))

    def _on_agent_message(self, text: str, **_kwargs: Any) -> None:
        self.ui.render(HostEvent(HostEventKind.MESSAGE, text))

    def _persist(self) -> None:
        if self._agent is None or self.meta is None or self._storage is None:
            return
        self.meta.mode = self._agent.mode
        self.meta.model = getattr(self._agent.v, "model", self.meta.model) or self.meta.model
        self.meta.permission_rules = self._agent.engine.snapshot_session_rules()
        self.meta.journal = self._agent.journal.to_dict()
        self.meta.todos = _json_safe(self._agent.todos.to_dict())
        with contextlib.suppress(Exception):
            title = getattr(self._agent.v, "title", None)
            if title:
                self.meta.title = str(title)
        self.store.save_meta(self.meta)
        self._storage.save_snapshot(self._agent)

    async def close(self) -> None:
        try:
            self._persist()
        finally:
            self._teardown_event_bridge()
            if self._agent is not None:
                try:
                    await self._agent.ws.close()
                except Exception:  # noqa: BLE001
                    logger.debug("shell close failed", exc_info=True)
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            try:
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
        return f"noah [{mode}|{model}|{sid}{title}]"

    async def start_new_session(self) -> SessionMeta:
        """Persist current session and open a fresh one in-process."""
        self._persist()
        self._teardown_event_bridge()
        if self._agent is not None:
            with contextlib.suppress(Exception):
                await self._agent.ws.close()
        if self._storage is not None:
            self._storage.close()
            self._storage = None
        self._agent = None
        self.meta = None
        self._exit_requested = False
        return await self.start()

    async def switch_session(self, session_id: str) -> SessionMeta:
        """Persist current session and resume another."""
        self._persist()
        self._teardown_event_bridge()
        if self._agent is not None:
            with contextlib.suppress(Exception):
                await self._agent.ws.close()
        if self._storage is not None:
            self._storage.close()
            self._storage = None
        self._agent = None
        meta = self.store.load_meta(session_id)
        self.store.verify_workspace(meta, self.workspace)
        self.meta = meta
        self._exit_requested = False
        return await self.start()

    def list_session_metas(self) -> list[SessionMeta]:
        return self.store.list_sessions(self.workspace)

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
        self._cancel_turn = True
        if self._agent is not None:
            self._agent.approvals.cancel_all()
        task = self._active_turn
        if task is not None and not task.done():
            task.cancel()

    async def handle_line(self, line: str) -> Literal["continue", "exit", "handled"]:
        slash = parse_slash(line)
        if slash:
            return await self._handle_slash(slash[0], slash[1])
        self._cancel_turn = False
        self.ui.set_busy(True)
        self._active_turn = asyncio.current_task()
        try:
            await self._run_user_turn(line)
            return "continue"
        finally:
            self._active_turn = None
            self.ui.set_busy(False)
            self.ui.set_status(self.status_prompt())

    async def _handle_slash(self, name: str, args: str) -> Literal["continue", "exit", "handled"]:
        agent = self.agent
        if name == "help":
            self.ui.render(HostEvent(HostEventKind.MESSAGE, help_text(self._custom_commands)))
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
                self.ui.render(HostEvent(HostEventKind.MESSAGE, f"```text\n{text}\n```"))
            return "handled"
        if name in self._custom_commands:
            return await self._run_custom_command(name, args)
        if name == "exit" or name == "quit":
            self._exit_requested = True
            return "exit"
        if name == "new":
            meta = await self.start_new_session()
            self.ui.render(
                HostEvent(HostEventKind.STATUS, f"started new session {meta.session_id}")
            )
            return "handled"
        if name == "skills":
            skills = getattr(agent, "skills", None)
            if skills is None:
                self.ui.render(HostEvent(HostEventKind.STATUS, "no skills registry"))
            else:
                try:
                    text = skills.status()
                except Exception as exc:  # noqa: BLE001
                    text = f"skills error: {exc}"
                self.ui.render(HostEvent(HostEventKind.MESSAGE, text or "(empty)"))
            return "handled"
        if name == "mode":
            mode = args.strip().lower()
            if not mode:
                self.ui.render(HostEvent(HostEventKind.STATUS, f"mode={agent.mode}"))
                return "handled"
            if mode not in {"build", "plan"}:
                self.ui.render(HostEvent(HostEventKind.ERROR, "usage: /mode [build|plan]"))
                return "handled"
            agent.set_mode(mode)  # type: ignore[arg-type]
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
                    self.ui.render(
                        HostEvent(HostEventKind.ERROR, "usage: /model --global MODEL")
                    )
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
        if name == "session":
            self.ui.render(
                HostEvent(
                    HostEventKind.MESSAGE,
                    f"id={self.meta.session_id}\ntitle={self.meta.title}\n"
                    f"mode={agent.mode}\nmodel={self.meta.model}\n"
                    f"workspace={self.meta.workspace_path}",
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
                    + ("  ← current" if self.meta and s.session_id == self.meta.session_id else "")
                    for s in rows
                )
                or "(none)"
            )
            text += "\n\nSwitch with: /sessions SESSION_ID"
            self.ui.render(HostEvent(HostEventKind.MESSAGE, text))
            return "handled"
        if name == "todos":
            self.ui.render(HostEvent(HostEventKind.MESSAGE, agent.todos.status() or "(no todos)"))
            return "handled"
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
                    f"session={self.meta.session_id if self.meta else '?'} "
                    f"title={self.meta.title if self.meta else '?'} "
                    f"undo={'yes' if agent.journal.can_undo() else 'no'} "
                    f"reversible={reversible}{warn}",
                )
            )
            return "handled"
        if name == "diff":
            try:
                text = await agent.git.diff()
            except Exception as exc:  # noqa: BLE001
                text = f"diff failed: {exc}"
            self.ui.render(HostEvent(HostEventKind.MESSAGE, f"```\n{text}\n```"))
            return "handled"
        if name == "undo":
            try:
                turn = agent.journal._turns[-1] if agent.journal.can_undo() else None
                if turn:
                    agent.journal.capture_post_bytes_before_undo(turn)
                undone = agent.journal.undo()
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        f"undid turn {undone.turn_id[:8]} ({len(undone.mutations)} files)",
                    )
                )
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
        if name == "compact":
            for summarizer in getattr(agent, "_summarizers", []):
                try:
                    await summarizer.summarize()  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    self.ui.render(HostEvent(HostEventKind.ERROR, f"compact failed: {exc}"))
                    return "handled"
            self.ui.render(HostEvent(HostEventKind.STATUS, "compaction requested"))
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
            self.agent.set_mode(cmd.mode)  # type: ignore[arg-type]
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

    async def _switch_model(self, model: str) -> None:
        from nooa.unifiedllm import get_llm_client

        llm = get_llm_client(model)
        self.agent._llm = llm
        apply_model_limits(self.agent)
        if self.meta:
            self.meta.model = model
            self.agent.v.model = model
            self.store.save_meta(self.meta)
        self.ui.render(HostEvent(HostEventKind.STATUS, f"model set to {model}"))
        self.ui.set_status(self.status_prompt())

    async def _run_user_turn(self, text: str) -> HostResult:
        agent = self.agent
        agent.journal.begin_turn()
        agent._user_messages_in.put(text)

        if self.meta and self.meta.title == "untitled":
            asyncio.create_task(self._maybe_title(text))

        exit_code = 0
        explanation = ""
        try:
            wins = await agent.queue_manager.race()
            notification: dict[str, list] = {}
            for name, item in wins:
                notification.setdefault(name, []).append(item)
            result = await agent.handle(notification)
            explanation = getattr(result, "explanation", "") or ""
            kind = getattr(result, "kind", None)
            self._last_stop = explanation
            self.ui.render(HostEvent(HostEventKind.STOP, f"{kind}: {explanation}"))
            if kind == RespondReason.NEED_INPUT:
                exit_code = 0
        except PermissionError as exc:
            exit_code = 3
            explanation = str(exc)
            self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
        except asyncio.CancelledError:
            exit_code = 130
            explanation = "cancelled"
            self.ui.render(HostEvent(HostEventKind.STATUS, "turn cancelled"))
            raise
        except Exception as exc:
            exit_code = 1
            explanation = f"agent failure: {exc}"
            self.ui.render(HostEvent(HostEventKind.ERROR, explanation))
            logger.exception("handle() failed")
        finally:
            agent.journal.end_turn()
            self._last_turn_shell_bypass = bool(
                agent.journal._turns and agent.journal._turns[-1].shell_may_bypass
            )
            if self._last_turn_shell_bypass:
                self.ui.render(
                    HostEvent(
                        HostEventKind.STATUS,
                        "warning: this turn ran mutating shell commands; "
                        "file-journal undo may be incomplete",
                    )
                )
            self._persist()
        return HostResult(
            exit_code=exit_code,
            explanation=explanation,
            session_id=self.meta.session_id if self.meta else None,
        )

    async def _maybe_title(self, text: str) -> None:
        try:
            title = await self.agent.name_session(text)
            title = (title or "").strip().strip('"')[:60]
            if title and self.meta:
                self.meta.title = title
                self.agent.v.title = title
                self.store.save_meta(self.meta)
        except Exception:  # noqa: BLE001 - never fail the main task
            logger.debug("title generation failed", exc_info=True)

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

    async def run_tui(self) -> int:
        """Full-screen Textual UI. App owns input; host owns turns."""
        try:
            from noah_code.ui.textual_app import NoahCodeApp, TextualUI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Textual is required for the TUI. Install with: uv sync / pip install textual"
            ) from exc

        ui = TextualUI()
        self.ui = ui
        await self.start()
        app = NoahCodeApp(self, ui)
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
