"""MCP configuration, persistence, and attachment for CodingAgent."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import shlex
import tempfile
import types
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.permissions import PermissionCategory, PermissionEngine

_RESERVED_AGENT_ATTRS = frozenset(
    {
        "agent_id",
        "approvals",
        "ask",
        "context",
        "engine",
        "event_manager",
        "events",
        "git",
        "github",
        "journal",
        "lsp",
        "media",
        "memory",
        "message",
        "mode",
        "plan",
        "processes",
        "producers",
        "queue_manager",
        "skills",
        "task",
        "todos",
        "tools",
        "user_messages",
        "v",
        "vars",
        "web",
        "workspace_root",
        "ws",
        "_approvals",
        "_budget_guard",
        "_config",
        "_coordinator",
        "_engine",
        "_journal",
        "_llm",
        "_lightweight_llm",
        "_mcp_stdio_pools",
        "_observability_event_manager",
        "_observability_unsubs",
        "_runtime",
        "_sandbox_approved_roots",
        "_shell",
        "_usage_tracker",
    }
)
_MUTATING_MCP_TOOL = re.compile(
    r"^(?:create|update|delete|remove|send|post|put|patch|comment|reply|push|"
    r"merge|close|write|set|add|upload|publish|execute|run|trigger|cancel|approve|reject)(?:_|$)",
    re.IGNORECASE,
)
_READ_ONLY_BROWSER_TOOLS = frozenset(
    {"browser_snapshot", "browser_console_messages", "browser_network_requests"}
)
_STDIO_TRANSPORT_ERROR_NAMES = frozenset(
    {
        "ClosedResourceError",
        "BrokenResourceError",
        "EndOfStream",
    }
)
_STDIO_TRANSPORT_ERROR_MARKERS = (
    "broken pipe",
    "eof",
    "end of stream",
    "endofstream",
    "closed resource",
    "closedresourceerror",
    "brokenresourceerror",
    "connection closed",
    "connection reset",
    "process exited",
    "not running",
)


def browser_tool_mutates(name: str, arguments: dict[str, Any] | None = None) -> bool:
    """Fail closed for browser actions, including arbitrary script evaluation.

    Even navigation can trigger server-side changes. The narrow read catalog
    stops being read-only when asked to save output to a file.
    """

    name = name.lower()
    if not name.startswith("browser_"):
        return False
    return name not in _READ_ONLY_BROWSER_TOOLS or bool((arguments or {}).get("filename"))


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    found: list[BaseException] = []
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return found


def _is_stdio_transport_error(exc: BaseException) -> bool:
    for item in _iter_exception_chain(exc):
        if isinstance(
            item,
            (BrokenPipeError, EOFError, ConnectionError, ProcessLookupError),
        ):
            return True
        if type(item).__name__ in _STDIO_TRANSPORT_ERROR_NAMES:
            return True
        text = str(item).lower()
        if any(marker in text for marker in _STDIO_TRANSPORT_ERROR_MARKERS):
            return True
    return False


class _PersistentStdioSession:
    """Keep one stdio MCP server process/session alive across tool calls.

    Runs ``connect_to_server()`` inside a dedicated task so ``anyio`` task
    groups are always entered and exited on the same asyncio task.
    """

    def __init__(
        self, connect_factory: Callable[[], AbstractAsyncContextManager[Any]]
    ) -> None:
        self._connect_factory = connect_factory
        self._lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[Any] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def is_connected(self) -> bool:
        return (
            self._runner_task is not None
            and not self._runner_task.done()
            and self._ready is not None
            and self._ready.done()
            and not self._ready.cancelled()
            and self._ready.exception() is None
        )

    @property
    def runner_failed(self) -> bool:
        return self._runner_task is not None and self._runner_task.done()

    async def _run(
        self,
        ready: asyncio.Future[Any],
        stop_event: asyncio.Event,
    ) -> None:
        try:
            async with self._connect_factory() as session:
                if not ready.done():
                    ready.set_result(session)
                await stop_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                raise

    async def _ensure_session_locked(self) -> Any:
        if self.is_connected and self._ready is not None:
            return self._ready.result()
        await self._teardown_locked()
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[Any] = loop.create_future()
        stop_event = asyncio.Event()
        task = asyncio.create_task(self._run(ready, stop_event))
        self._runner_task = task
        self._ready = ready
        self._stop_event = stop_event
        try:
            return await ready
        except BaseException:
            await self._teardown_locked()
            raise

    async def _teardown_locked(self) -> None:
        task = self._runner_task
        stop_event = self._stop_event
        self._runner_task = None
        self._ready = None
        self._stop_event = None
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            if not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
                except (TimeoutError, BaseException):
                    task.cancel()
                    with suppress(BaseException):
                        await task
            else:
                with suppress(BaseException):
                    task.result()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Any]:
        async with self._lock:
            session = await self._ensure_session_locked()
            try:
                yield session
            except BaseException as exc:
                if self.runner_failed or _is_stdio_transport_error(exc):
                    await self._teardown_locked()
                raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._teardown_locked()


def _install_persistent_stdio_pool(tool: Any) -> _PersistentStdioSession | None:
    client = getattr(tool, "_client", None)
    if client is None or getattr(client, "transport", None) != "stdio":
        return None
    connect = getattr(client, "connect_to_server", None)
    if not callable(connect) or inspect.iscoroutinefunction(connect):
        return None
    existing = getattr(client, "_noah_stdio_pool", None)
    if isinstance(existing, _PersistentStdioSession):
        return existing
    pool = _PersistentStdioSession(connect)
    client.connect_to_server = pool.connect
    with suppress(Exception):
        client._noah_stdio_pool = pool
    with suppress(Exception):
        tool._noah_stdio_pool = pool
    return pool


async def close_mcp_servers(agent: Any) -> None:
    """Close all persistent stdio MCP sessions attached to ``agent``."""

    pools: list[_PersistentStdioSession] = []
    seen: set[int] = set()
    tracked = getattr(agent, "_mcp_stdio_pools", None)
    if isinstance(tracked, dict):
        for pool in list(tracked.values()):
            if isinstance(pool, _PersistentStdioSession) and id(pool) not in seen:
                seen.add(id(pool))
                pools.append(pool)
        tracked.clear()
    elif isinstance(tracked, list):
        for pool in list(tracked):
            if isinstance(pool, _PersistentStdioSession) and id(pool) not in seen:
                seen.add(id(pool))
                pools.append(pool)
        tracked.clear()

    agent_dict = getattr(agent, "__dict__", None)
    if isinstance(agent_dict, dict):
        for value in list(agent_dict.values()):
            pool = getattr(value, "_noah_stdio_pool", None)
            if pool is None:
                client = getattr(value, "_client", None)
                pool = getattr(client, "_noah_stdio_pool", None)
            if isinstance(pool, _PersistentStdioSession) and id(pool) not in seen:
                seen.add(id(pool))
                pools.append(pool)

    if pools:
        await asyncio.gather(*(pool.aclose() for pool in pools), return_exceptions=True)


_REAL_PATH_HOME = Path.home


def _resolve_user_home(home: Path | None = None) -> Path:
    if home is not None:
        return home.expanduser()
    if Path.home != _REAL_PATH_HOME:
        return Path.home().expanduser()
    from noah_code.skills_setup import _default_user_home

    return _default_user_home()


@dataclass(frozen=True)
class MCPPreset:
    """Built-in one-click MCP server preset."""

    key: str
    label: str
    transport: str
    target: str
    description: str
    env_var: str | None = None

    def to_spec(self) -> dict[str, Any]:
        if self.transport == "stdio":
            parts = shlex.split(self.target)
            return {
                "command": parts[0],
                "args": parts[1:],
                "transport": "stdio",
            }
        return {
            "url": self.target,
            "transport": "streamable-http",
        }


MCP_PRESETS: tuple[MCPPreset, ...] = (
    MCPPreset(
        "context7",
        "Context7 Docs",
        "stdio",
        "npx -y @upstash/context7-mcp@latest",
        "Up-to-date library & framework docs",
        None,
    ),
    MCPPreset(
        "playwright",
        "Playwright Browser",
        "stdio",
        "npx -y @playwright/mcp@latest",
        "Browser automation & inspection",
        None,
    ),
    MCPPreset(
        "filesystem",
        "Filesystem",
        "stdio",
        "npx -y @modelcontextprotocol/server-filesystem .",
        "Scoped local file operations",
        None,
    ),
    MCPPreset(
        "fetch",
        "Web Fetch",
        "stdio",
        "uvx mcp-server-fetch",
        "Fetch & convert web pages to Markdown",
        None,
    ),
    MCPPreset(
        "memory",
        "Knowledge Graph Memory",
        "stdio",
        "npx -y @modelcontextprotocol/server-memory",
        "Persistent entity/relation graph memory",
        None,
    ),
    MCPPreset(
        "sequential-thinking",
        "Sequential Thinking",
        "stdio",
        "npx -y @modelcontextprotocol/server-sequential-thinking",
        "Structured step-by-step problem decomposition",
        None,
    ),
    MCPPreset(
        "github",
        "GitHub",
        "stdio",
        "npx -y @modelcontextprotocol/server-github",
        "Repos, issues, PRs, and code search",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
    ),
    MCPPreset(
        "postgres",
        "PostgreSQL",
        "stdio",
        "npx -y @modelcontextprotocol/server-postgres",
        "Inspect schemas & run read-only SQL",
        "POSTGRES_URL",
    ),
    MCPPreset(
        "brave-search",
        "Brave Search",
        "stdio",
        "npx -y @modelcontextprotocol/server-brave-search",
        "Web & local search via Brave API",
        "BRAVE_API_KEY",
    ),
    MCPPreset(
        "linear",
        "Linear",
        "http",
        "https://mcp.linear.app/mcp",
        "Issues, projects, and cycles",
        None,
    ),
    MCPPreset(
        "sentry",
        "Sentry",
        "http",
        "https://mcp.sentry.dev/mcp",
        "Production errors, traces, and issues",
        None,
    ),
    MCPPreset(
        "cloudflare-docs",
        "Cloudflare Docs",
        "http",
        "https://docs.mcp.cloudflare.com/mcp",
        "Search Cloudflare developer documentation",
        None,
    ),
)


def list_mcp_presets() -> tuple[MCPPreset, ...]:
    """Return built-in one-click MCP server presets."""

    return MCP_PRESETS


def get_mcp_preset(key: str) -> MCPPreset:
    """Look up a built-in MCP preset by key (case-insensitive)."""

    normalized = key.strip().lower()
    for preset in MCP_PRESETS:
        if preset.key == normalized:
            return preset
    raise KeyError(f"unknown MCP preset: {key}")


def parse_mcp_add_target(tokens: list[str] | tuple[str, ...]) -> tuple[str, str, str]:
    """Parse ``[stdio|http] NAME TARGET...`` or shorthand ``NAME URL_OR_CMD...``."""

    parts = [part for part in tokens if part]
    if len(parts) < 2:
        raise ValueError("usage: mcp add [stdio|http] NAME URL_OR_COMMAND…")
    first = parts[0].lower()
    if first == "stdio":
        if len(parts) < 3:
            raise ValueError("usage: mcp add stdio NAME COMMAND…")
        return "stdio", parts[1], shlex.join(parts[2:])
    if first in {"http", "streamable-http"}:
        if len(parts) != 3:
            raise ValueError("usage: mcp add http NAME URL")
        return "http", parts[1], parts[2]
    name = parts[0]
    rest = parts[1:]
    if len(rest) == 1 and rest[0].lower().startswith(("http://", "https://")):
        return "http", name, rest[0]
    return "stdio", name, shlex.join(rest)


@dataclass(frozen=True)
class MCPServerInfo:
    name: str
    transport: str
    target: str
    source: str


@dataclass(frozen=True)
class MCPInstallResult:
    attached: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.attached and not self.errors:
            return "mcp: none configured"
        parts = [f"attached={list(self.attached)}"] if self.attached else []
        if self.errors:
            parts.append(f"errors={list(self.errors)}")
        return "mcp: " + " ".join(parts)


def mcp_config_paths(workspace: Path, *, home: Path | None = None) -> list[Path]:
    user_home = _resolve_user_home(home)
    return [
        user_home / ".config" / "noah-code" / "mcp.json",
        workspace / ".mcp.json",
        workspace / ".noah-code" / "mcp.json",
    ]


def mcp_source_is_trusted(source: str, *, home: Path | None = None) -> bool:
    """Return True when the MCP catalog entry came from user-owned configuration."""

    if source == "user config.toml":
        return True
    user_home = _resolve_user_home(home)
    trusted = user_home / ".config" / "noah-code" / "mcp.json"
    try:
        return Path(source).expanduser().resolve() == trusted.resolve()
    except OSError:
        return False


def _server_mapping(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        return {}
    candidate = data.get("mcpServers", data.get("servers", data))
    if not isinstance(candidate, dict):
        return {}
    return {str(name): dict(spec) for name, spec in candidate.items() if isinstance(spec, dict)}


def _normalize_oauth_scope(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list | tuple | set):
        joined = " ".join(str(part).strip() for part in value if str(part).strip())
        return joined or None
    return None


def _normalized_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    kind = str(normalized.pop("type", "") or "").lower()
    if kind in {"http", "streamable-http"}:
        normalized.setdefault("transport", "streamable-http")
    elif kind == "sse":
        normalized.setdefault("transport", "sse")
    elif kind == "stdio":
        normalized.setdefault("transport", "stdio")
    if normalized.get("url") and not normalized.get("transport"):
        normalized["transport"] = "streamable-http"

    if "oauth_scopes" in normalized:
        raw_scopes = normalized.pop("oauth_scopes")
        if normalized.get("oauth_scope") is None:
            mapped_scope = _normalize_oauth_scope(raw_scopes)
            if mapped_scope is not None:
                normalized["oauth_scope"] = mapped_scope
    elif "oauth_scope" in normalized:
        mapped_scope = _normalize_oauth_scope(normalized["oauth_scope"])
        if mapped_scope is not None:
            normalized["oauth_scope"] = mapped_scope
        else:
            normalized.pop("oauth_scope", None)

    if "oauth_callback_port" in normalized:
        raw_port = normalized.pop("oauth_callback_port")
        if not normalized.get("oauth_redirect_uri") and raw_port is not None:
            port = int(raw_port)
            if not (0 <= port <= 65535):
                raise ValueError("oauth_callback_port must be between 0 and 65535")
            normalized["oauth_redirect_uri"] = f"http://127.0.0.1:{port}/callback"

    allowed = {
        "args",
        "command",
        "env",
        "headers",
        "oauth_client_id",
        "oauth_manual",
        "oauth_open_browser",
        "oauth_redirect_uri",
        "oauth_scope",
        "oauth_timeout",
        "transport",
        "url",
    }
    return {key: value for key, value in normalized.items() if key in allowed}


def load_mcp_servers(
    workspace: Path,
    config: NoahCodeConfig,
    *,
    home: Path | None = None,
    extra_servers: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Merge Claude/Cursor-style JSON, untrusted session requests, and trusted Noah config."""

    servers: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    if isinstance(extra_servers, dict):
        for name, spec in _server_mapping(extra_servers).items():
            servers[name] = spec
            sources[name] = "client session"
    for path in mcp_config_paths(workspace, home=home):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for name, spec in _server_mapping(data).items():
            if spec.get("disabled") is True or spec.get("enabled") is False:
                servers.pop(name, None)
                sources.pop(name, None)
                continue
            servers[name] = _normalized_spec(spec)
            sources[name] = str(path)

    # Trusted user config has highest precedence and accepts either
    # ``mcp.servers`` or a direct mapping for backwards compatibility.
    for name, spec in _server_mapping(config.mcp).items():
        if spec.get("disabled") is True or spec.get("enabled") is False:
            servers.pop(name, None)
            sources.pop(name, None)
            continue
        servers[name] = _normalized_spec(spec)
        sources[name] = "user config.toml"
    return servers, sources


def list_mcp_servers(
    workspace: Path,
    config: NoahCodeConfig,
    *,
    home: Path | None = None,
) -> list[MCPServerInfo]:
    servers, sources = load_mcp_servers(workspace, config, home=home)
    rows: list[MCPServerInfo] = []
    for name, spec in sorted(servers.items()):
        transport = str(spec.get("transport") or ("stdio" if spec.get("command") else "http"))
        target = str(spec.get("url") or spec.get("command") or "not configured")
        rows.append(MCPServerInfo(name, transport, target, sources.get(name, "unknown")))
    return rows


def save_user_mcp_server(
    name: str,
    spec: dict[str, Any],
    *,
    home: Path | None = None,
) -> Path:
    """Persist one server in a portable ``mcpServers`` JSON file."""

    clean_name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", clean_name):
        raise ValueError("server name must use letters, numbers, '.', '_' or '-'")
    normalized = _normalized_spec(spec)
    if not normalized.get("command") and not normalized.get("url"):
        raise ValueError("MCP server requires a command or URL")
    if normalized.get("url"):
        parsed = urlparse(str(normalized["url"]))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MCP URL must be an absolute http:// or https:// endpoint")

    user_home = _resolve_user_home(home)
    path = user_home / ".config" / "noah-code" / "mcp.json"
    existing: dict[str, dict[str, Any]] = {}
    document: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            if not isinstance(loaded, dict):
                raise ValueError("top-level value must be an object")
            document = loaded
            existing = _server_mapping(document)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read existing MCP config: {exc}") from exc
    if clean_name in existing:
        raise FileExistsError(f"MCP server already exists: {clean_name}")
    existing[clean_name] = normalized

    path.parent.mkdir(parents=True, exist_ok=True)
    document["mcpServers"] = existing
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".mcp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(payload)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def remove_user_mcp_server(name: str, *, home: Path | None = None) -> bool:
    """Remove ``name`` from the user's ``~/.config/noah-code/mcp.json`` if present."""

    clean_name = name.strip()
    if not clean_name:
        return False
    user_home = _resolve_user_home(home)
    path = user_home / ".config" / "noah-code" / "mcp.json"
    if not path.is_file():
        return False
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing MCP config: {exc}") from exc
    if not isinstance(loaded, dict):
        return False
    existing = _server_mapping(loaded)
    if clean_name not in existing:
        return False
    existing.pop(clean_name, None)
    document = dict(loaded)
    document.pop("servers", None)
    document["mcpServers"] = existing
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".mcp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(payload)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def discover_external_mcp_servers(
    *, home: Path | None = None
) -> dict[str, tuple[dict[str, Any], str]]:
    """Inspect Claude Code, Claude Desktop, and Cursor user MCP configs safely."""

    user_home = _resolve_user_home(home)
    candidates: tuple[tuple[Path, str, bool], ...] = (
        (
            user_home
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json",
            "Claude Desktop",
            False,
        ),
        (
            user_home / ".config" / "Claude" / "claude_desktop_config.json",
            "Claude Desktop",
            False,
        ),
        (user_home / ".cursor" / "mcp.json", "Cursor", False),
        (user_home / ".claude" / "mcp.json", "Claude Code", False),
        (user_home / ".claude.json", "Claude Code", True),
    )
    discovered: dict[str, tuple[dict[str, Any], str]] = {}
    for config_path, label, require_servers_key in candidates:
        if not config_path.is_file():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if require_servers_key and "mcpServers" not in data and "servers" not in data:
            continue
        for raw_name, raw_spec in _server_mapping(data).items():
            clean_name = raw_name.strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", clean_name):
                continue
            if raw_spec.get("disabled") is True or raw_spec.get("enabled") is False:
                continue
            try:
                normalized = _normalized_spec(raw_spec)
            except ValueError:
                continue
            if not normalized.get("command") and not normalized.get("url"):
                continue
            if normalized.get("url"):
                parsed = urlparse(str(normalized["url"]))
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
            discovered[clean_name] = (normalized, f"{label} ({config_path})")
    return discovered


async def attach_mcp_server(
    agent: Any,
    name: str,
    spec: dict[str, Any],
    *,
    engine: PermissionEngine,
    approvals: ApprovalBroker,
    trusted: bool = True,
    startup: bool = False,
) -> str:
    """Permission-check and attach a single configured MCP server."""

    if trusted:
        decision = engine.decide(PermissionCategory.MCP, name, tool="mcp")
        if startup and decision.action == "ask":
            decision = replace(
                decision,
                action="allow",
                reason=f"{decision.reason} (configured MCP)",
            )
    else:
        saved_auto = engine.auto_approve
        saved_yolo = engine.yolo
        engine.auto_approve = False
        engine.yolo = False
        try:
            decision = engine.decide(PermissionCategory.MCP, name, tool="mcp")
        finally:
            engine.auto_approve = saved_auto
            engine.yolo = saved_yolo
        if decision.action == "ask" and (saved_auto or saved_yolo or startup):
            raise PermissionError(f"workspace MCP servers cannot be auto-approved: {name}")
    await approvals.require(decision)
    try:
        from nooa.mcp import MCPManager
    except ImportError as exc:
        raise RuntimeError("MCP support is not installed; run: uv sync --extra mcp") from exc

    tool = await asyncio.to_thread(
        MCPManager.create_from_server,
        name,
        **_normalized_spec(spec),
    )
    stdio_pool = _install_persistent_stdio_pool(tool)
    runtime = getattr(agent, "_runtime", None)
    if callable(getattr(tool, "_call_tool", None)):
        original_call = tool._call_tool
        browser_effect_lock = asyncio.Lock()

        async def _call_once(
            tool_name: str,
            clean: dict[str, Any],
            preserve_none: set[str] | frozenset[str],
        ) -> Any:
            if preserve_none:
                return await original_call(tool_name, clean, _preserve_none=preserve_none)
            return await original_call(tool_name, clean)

        async def _call_underlying(
            tool_name: str,
            clean: dict[str, Any],
            preserve_none: set[str] | frozenset[str],
        ) -> Any:
            was_connected = stdio_pool.is_connected if stdio_pool is not None else False
            try:
                return await _call_once(tool_name, clean, preserve_none)
            except Exception as exc:
                if (
                    stdio_pool is not None
                    and was_connected
                    and _is_stdio_transport_error(exc)
                ):
                    await stdio_pool.aclose()
                    return await _call_once(tool_name, clean, preserve_none)
                raise

        async def _invoke(
            tool_name: str,
            clean: dict[str, Any],
            *,
            browser_mutation: bool,
            preserve_none: set[str] | frozenset[str] = frozenset(),
        ) -> Any:
            if runtime is None or (
                not browser_mutation and _MUTATING_MCP_TOOL.match(tool_name) is None
            ):
                return await _call_underlying(tool_name, clean, preserve_none)
            active_run_id = getattr(agent, "_current_run_id", None)
            effect_request: Any = (
                {"arguments": clean, "run_id": active_run_id}
                if not browser_mutation and active_run_id
                else clean
            )
            effect_key, cached, result, recovering = runtime.begin_effect(
                "mcp",
                f"{name}.{tool_name}",
                effect_request,
            )
            if cached:
                if not browser_mutation:
                    return result
                # A repeated click or navigation is a fresh explicit action,
                # not an idempotent operation. Reopen the completed journal
                # record; an interruption here safely blocks the next retry.
                runtime.fail_effect(effect_key, "starting another browser invocation")
                runtime.begin_effect("mcp", f"{name}.{tool_name}", effect_request)
            if recovering:
                runtime.fail_effect(
                    effect_key,
                    "ambiguous after process interruption; automatic replay refused",
                )
                raise RuntimeError(
                    f"MCP mutation {name}.{tool_name} may already have completed before a "
                    "crash; inspect the remote system before retrying with changed arguments"
                )
            try:
                value = await _call_underlying(tool_name, clean, preserve_none)
            except Exception as exc:
                runtime.fail_effect(effect_key, str(exc))
                raise
            runtime.complete_effect(effect_key, value)
            return value

        async def _durable_mcp_call(
            _self: Any,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
            *,
            _preserve_none: set[str] | frozenset[str] = frozenset(),
        ) -> Any:
            preserve = set(_preserve_none)
            clean = {
                key: value
                for key, value in (arguments or {}).items()
                if value is not None or key in preserve
            }
            browser_mutation = browser_tool_mutates(tool_name, clean)
            is_mutating = browser_mutation or _MUTATING_MCP_TOOL.match(tool_name) is not None
            decision = engine.decide(
                PermissionCategory.MCP, f"{name}.{tool_name}", tool=f"mcp.{tool_name}"
            )
            if is_mutating and engine.mode == "plan":
                reason = (
                    f"plan mode forbids browser mutations ({name}.{tool_name})"
                    if browser_mutation
                    else f"plan mode forbids mutating MCP calls ({name}.{tool_name})"
                )
                decision = replace(
                    decision,
                    action="deny",
                    reason=reason,
                )
                await approvals.require(decision)
            elif (
                browser_mutation
                or not trusted
                or decision.action == "deny"
                or (decision.matching_rule is not None and decision.matching_rule.pattern != "*")
            ):
                await approvals.require(decision)
            if browser_mutation:
                # Serialize browser effects so an identical concurrent action
                # cannot confuse an in-flight operation with crash recovery.
                async with browser_effect_lock:
                    return await _invoke(
                        tool_name,
                        clean,
                        browser_mutation=True,
                        preserve_none=preserve,
                    )
            return await _invoke(
                tool_name,
                clean,
                browser_mutation=False,
                preserve_none=preserve,
            )

        tool._call_tool = types.MethodType(_durable_mcp_call, tool)
    attr = re_attr(name)
    if stdio_pool is not None:
        with contextlib.suppress(Exception):
            pools = getattr(agent, "_mcp_stdio_pools", None)
            if not isinstance(pools, dict):
                pools = {}
                agent._mcp_stdio_pools = pools
            previous = pools.pop(attr, None)
            if isinstance(previous, _PersistentStdioSession) and previous is not stdio_pool:
                await previous.aclose()
            pools[attr] = stdio_pool
    setattr(agent, attr, tool)
    approved = getattr(agent, "_sandbox_approved_roots", None)
    if isinstance(approved, set):
        approved.add(attr)
    return attr


async def install_mcp(
    agent: Any,
    workspace: Path,
    config: NoahCodeConfig,
    *,
    engine: PermissionEngine,
    approvals: ApprovalBroker,
    home: Path | None = None,
    startup: bool = False,
    extra_servers: dict[str, dict[str, Any]] | None = None,
) -> MCPInstallResult:
    """Attach all configured servers without failing startup on one bad server."""

    servers, sources = load_mcp_servers(
        workspace, config, home=home, extra_servers=extra_servers
    )
    attached: list[str] = []
    errors: list[str] = []
    for name, spec in servers.items():
        trusted = mcp_source_is_trusted(sources.get(name, ""), home=home)
        if startup and not trusted:
            errors.append(f"{name}: workspace MCP servers cannot be auto-approved: {name}")
            continue
        try:
            await attach_mcp_server(
                agent,
                name,
                spec,
                engine=engine,
                approvals=approvals,
                trusted=trusted,
                startup=startup,
            )
            attached.append(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    return MCPInstallResult(tuple(attached), tuple(errors))


def re_attr(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"mcp_{cleaned}"
    if not cleaned:
        return "mcp_server"
    if cleaned in _RESERVED_AGENT_ATTRS or cleaned.startswith("_"):
        trimmed = cleaned.lstrip("_") or "server"
        cleaned = f"mcp_{trimmed}"
    return cleaned
