"""Optional MCP server attachment for CodingAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noah_code.approvals import ApprovalBroker
from noah_code.config import NoahCodeConfig
from noah_code.permissions import PermissionCategory, PermissionEngine


def mcp_config_paths(workspace: Path) -> list[Path]:
    return [
        workspace / ".mcp.json",
        workspace / ".noah-code" / "mcp.json",
        Path.home() / ".config" / "noah-code" / "mcp.json",
    ]


async def install_mcp(
    agent: Any,
    workspace: Path,
    config: NoahCodeConfig,
    *,
    engine: PermissionEngine,
    approvals: ApprovalBroker,
) -> str:
    """Attach configured MCP servers as agent attributes when available.

    Permission category ``mcp`` is checked before first use via a thin wrapper
    only if we can gate at attach time; otherwise servers are attached and
    documented as requiring the mcp permission policy.
    """
    servers = dict(config.mcp.get("servers") or {})
    mcp_file: Path | None = None
    for path in mcp_config_paths(workspace):
        if path.is_file():
            mcp_file = path
            break

    if not servers and mcp_file is None:
        return "mcp: none configured"

    try:
        from nooa.mcp import MCPManager
    except ImportError:
        return "mcp: nooa[mcp] not installed"

    decision = engine.decide(PermissionCategory.MCP, "*")
    if decision.denied:
        return f"mcp: denied ({decision.reason})"
    if decision.needs_ask:
        try:
            await approvals.require(decision)
        except PermissionError as exc:
            return f"mcp: {exc}"

    attached: list[str] = []
    try:
        names = list(servers.keys()) if servers else MCPManager.list_servers(mcp_file=mcp_file)
    except Exception as exc:  # noqa: BLE001
        return f"mcp: list failed ({exc})"

    for name in names:
        try:
            spec = servers.get(name, {})
            tool = MCPManager.create_from_server(
                name,
                mcp_file=mcp_file,
                **{k: v for k, v in spec.items() if k != "name"},
            )
            # Sanitize attribute name.
            attr = re_attr(name)
            setattr(agent, attr, tool)
            approved = getattr(agent, "_sandbox_approved_roots", None)
            if isinstance(approved, set):
                approved.add(attr)
            attached.append(attr)
        except Exception as exc:  # noqa: BLE001
            return f"mcp: attached={attached} error on {name}: {exc}"
    return f"mcp: attached={attached}"


def re_attr(name: str) -> str:
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"mcp_{cleaned}"
    return cleaned or "mcp_server"
