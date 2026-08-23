"""MCP configuration, persistence, and attachment for CodingAgent."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.permissions import PermissionCategory, PermissionEngine

_RESERVED_AGENT_ATTRS = frozenset(
    {
        "approvals",
        "context",
        "engine",
        "git",
        "journal",
        "lsp",
        "message",
        "mode",
        "processes",
        "skills",
        "todos",
        "v",
        "workspace_root",
        "ws",
        "_approvals",
        "_config",
        "_engine",
        "_journal",
        "_llm",
        "_sandbox_approved_roots",
        "_shell",
    }
)
_MUTATING_MCP_TOOL = re.compile(
    r"^(?:create|update|delete|remove|send|post|put|patch|comment|reply|push|"
    r"merge|close|write|set|add|upload|publish|execute|run|trigger|cancel|approve|reject)(?:_|$)",
    re.IGNORECASE,
)


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
    user_home = (home or Path.home()).expanduser()
    return [
        user_home / ".config" / "noah-code" / "mcp.json",
        workspace / ".mcp.json",
        workspace / ".noah-code" / "mcp.json",
    ]


def mcp_source_is_trusted(source: str, *, home: Path | None = None) -> bool:
    """Return True when the MCP catalog entry came from user-owned configuration."""

    if source == "user config.toml":
        return True
    user_home = (home or Path.home()).expanduser()
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
    allowed = {
        "args",
        "command",
        "env",
        "headers",
        "oauth_callback_port",
        "oauth_client_id",
        "oauth_scopes",
        "transport",
        "url",
    }
    return {key: value for key, value in normalized.items() if key in allowed}


def load_mcp_servers(
    workspace: Path,
    config: NoahCodeConfig,
    *,
    home: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Merge Claude/Cursor-style JSON and trusted Noah configuration."""

    servers: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
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

    user_home = (home or Path.home()).expanduser()
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
        saved = engine.auto_approve
        engine.auto_approve = False
        try:
            decision = engine.decide(PermissionCategory.MCP, name, tool="mcp")
        finally:
            engine.auto_approve = saved
        if decision.action == "ask" and (saved or startup):
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
    runtime = getattr(agent, "_runtime", None)
    if runtime is not None:
        original_call = tool._call_tool

        async def _durable_mcp_call(
            _self: Any,
            tool_name: str,
            arguments: dict[str, Any] | None = None,
        ) -> Any:
            clean = {key: value for key, value in (arguments or {}).items() if value is not None}
            if _MUTATING_MCP_TOOL.match(tool_name) is None:
                return await original_call(tool_name, clean)
            effect_key, cached, result, recovering = runtime.begin_effect(
                "mcp",
                f"{name}.{tool_name}",
                clean,
            )
            if cached:
                return result
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
                value = await original_call(tool_name, clean)
            except Exception as exc:
                runtime.fail_effect(effect_key, str(exc))
                raise
            runtime.complete_effect(effect_key, value)
            return value

        tool._call_tool = types.MethodType(_durable_mcp_call, tool)
    attr = re_attr(name)
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
) -> MCPInstallResult:
    """Attach all configured servers without failing startup on one bad server."""

    servers, sources = load_mcp_servers(workspace, config, home=home)
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
