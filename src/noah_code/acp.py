"""Agent Client Protocol v1 over newline-delimited JSON-RPC stdio.

Wire shapes follow the official v1 schema, not the draft v2 protocol:
https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/v1/schema.json
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from noah_code import __version__
from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.permissions import PermissionDecision
from noah_code.redaction import safe_error_message
from noah_code.service import AgentService, ServiceError, ServiceSession

MAX_FRAME_BYTES = 1_048_576

ToolKind = Literal["read", "edit", "search", "execute", "fetch", "think", "other"]


class RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RPCError(-32602, f"{name} must be a nonempty string")
    return value


def _infer_tool_kind(
    title: str,
    meta: Mapping[str, Any] | None = None,
    *,
    category: str = "",
) -> ToolKind:
    cat = category.strip().lower()
    if cat == "edit":
        return "edit"
    if cat in {"read", "lsp"}:
        return "read"
    if cat == "bash":
        return "execute"
    if cat == "websearch":
        return "search"
    if cat == "webfetch":
        return "fetch"

    segments = [seg.strip() for seg in title.split("·") if seg.strip()]
    verbs = [seg.split(None, 1)[0].rstrip(":").lower() for seg in segments if seg]
    if any(v in {"edit", "write", "replace", "patch", "apply_patch"} for v in verbs):
        return "edit"
    if any(v in {"bash", "process", "run", "shell", "terminal"} for v in verbs):
        return "execute"
    if any(v in {"fetch", "webfetch"} for v in verbs):
        return "fetch"
    if any(v in {"grep", "glob", "search", "websearch", "find"} for v in verbs):
        return "search"
    if any(v in {"read", "inspect", "lsp", "git"} for v in verbs):
        return "read"
    if any(v in {"think", "todos", "plan", "reasoning"} for v in verbs):
        return "think"

    tool_name = str((meta or {}).get("tool") or title).strip().lower()
    if tool_name in {"write", "write_file", "edit", "replace", "apply_patch", "apply_unified_diff"}:
        return "edit"
    if tool_name in {"read", "ws_read", "inspect"} or tool_name.startswith("lsp"):
        return "read"
    if tool_name in {"search", "grep", "glob", "list", "list_files", "websearch"}:
        return "search"
    if tool_name in {"fetch", "webfetch"}:
        return "fetch"
    if tool_name in {"bash", "sh", "run", "terminal_run"} or tool_name.startswith("process"):
        return "execute"
    if tool_name in {"think", "todos", "plan"}:
        return "think"
    return "other"


def _prompt_text(blocks: Any) -> str:
    if not isinstance(blocks, list) or not blocks or len(blocks) > 100:
        raise RPCError(-32602, "prompt must contain 1 to 100 content blocks")
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RPCError(-32602, "invalid prompt content block")
        btype = block.get("type")
        if btype == "text":
            parts.append(_string(block, "text"))
        elif btype == "resource_link":
            # References stay references. Host tools perform any subsequent
            # reads/fetches with the existing path and network permission gates.
            parts.append(
                f"Referenced resource: {_string(block, 'name')}\nURI: {_string(block, 'uri')}"
            )
        elif btype == "resource":
            res = block.get("resource") if isinstance(block.get("resource"), dict) else block
            if not isinstance(res, dict):
                raise RPCError(-32602, "invalid resource content block")
            uri = _string(res, "uri")
            mime = res.get("mimeType")
            mime_label = f", {mime.strip()}" if isinstance(mime, str) and mime.strip() else ""
            res_text = res.get("text")
            res_blob = res.get("blob")
            if isinstance(res_text, str):
                parts.append(f"Embedded resource ({uri}{mime_label}):\n{res_text[:120_000]}")
            elif isinstance(res_blob, str) and res_blob.strip():
                try:
                    raw = base64.b64decode(res_blob, validate=True)
                except ValueError as exc:
                    raise RPCError(-32602, "invalid base64 resource blob") from exc
                parts.append(
                    f"Embedded binary resource ({uri}{mime_label}, {len(raw)} bytes)"
                )
            else:
                raise RPCError(-32602, "resource block must contain text or blob")
        elif btype == "image":
            mime_type = _string(block, "mimeType")
            if not mime_type.lower().startswith("image/"):
                raise RPCError(-32602, "image mimeType must start with 'image/'")
            data = block.get("data")
            image_uri = block.get("uri")
            if isinstance(data, str) and data.strip():
                try:
                    raw = base64.b64decode(data, validate=True)
                except ValueError as exc:
                    raise RPCError(-32602, "invalid base64 image data") from exc
                uri_note = (
                    f", uri={image_uri.strip()}"
                    if isinstance(image_uri, str) and image_uri.strip()
                    else ""
                )
                parts.append(f"[Image: {mime_type}, {len(raw)} bytes{uri_note}]")
            elif isinstance(image_uri, str) and image_uri.strip():
                parts.append(f"[Image: {mime_type}, uri={image_uri.strip()}]")
            else:
                raise RPCError(-32602, "image block requires base64 data or uri")
        else:
            raise RPCError(
                -32602,
                "only text, resource_link, resource, and image prompt blocks are supported",
            )
    return "\n\n".join(parts)


def _mcp_servers(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    servers = params.get("mcpServers")
    if not isinstance(servers, list) or len(servers) > 32:
        raise RPCError(-32602, "mcpServers must be an array of at most 32 servers")
    result: dict[str, dict[str, Any]] = {}
    for server in servers:
        if not isinstance(server, dict):
            raise RPCError(-32602, "invalid MCP server entry")
        stype = server.get("type")
        name = _string(server, "name")
        if name in result:
            raise RPCError(-32602, "MCP names must be unique")
        if stype in {None, "stdio"}:
            command = _string(server, "command")
            if not Path(command).is_absolute():
                raise RPCError(-32602, "stdio MCP commands must be absolute paths")
            args, env = server.get("args"), server.get("env", [])
            if (
                not isinstance(args, list)
                or any(not isinstance(item, str) for item in args)
                or not isinstance(env, list)
            ):
                raise RPCError(-32602, "invalid stdio MCP args or env")
            environment: dict[str, str] = {}
            for item in env:
                if not isinstance(item, dict) or not isinstance(item.get("value"), str):
                    raise RPCError(-32602, "invalid MCP environment variable")
                key = _string(item, "name")
                if "=" in key or "\0" in key or "\0" in item["value"]:
                    raise RPCError(-32602, "invalid MCP environment variable")
                environment[key] = item["value"]
            result[name] = {
                "transport": "stdio",
                "command": command,
                "args": args,
                "env": environment,
                "trusted": False,
            }
        elif stype in {"http", "sse"}:
            url = _string(server, "url")
            parsed = urlparse(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or any(ch in url for ch in "\r\n\0")
            ):
                raise RPCError(-32602, "MCP URL must be an absolute http(s) endpoint")
            raw_headers = server.get("headers", [])
            if not isinstance(raw_headers, list):
                raise RPCError(-32602, "invalid MCP HTTP headers")
            headers: dict[str, str] = {}
            for item in raw_headers:
                if not isinstance(item, dict) or not isinstance(item.get("value"), str):
                    raise RPCError(-32602, "invalid MCP HTTP header")
                hname = _string(item, "name")
                hval = item["value"]
                if any(ch in hname or ch in hval for ch in "\r\n\0"):
                    raise RPCError(-32602, "invalid MCP HTTP header")
                headers[hname] = hval
            result[name] = {
                "transport": "streamable-http" if stype == "http" else "sse",
                "url": url,
                "headers": headers,
                "trusted": False,
            }
        else:
            raise RPCError(-32602, "only stdio, http, and sse MCP servers are supported")
    return result


class ACPConnection:
    """One ACP client owns one service; EOF closes its sessions and pending work."""

    def __init__(self, service: AgentService) -> None:
        self.service = service
        self.initialized = False
        self._writer: Any = None
        self._write_lock = asyncio.Lock()
        self._requests: dict[str | int, asyncio.Task[Any]] = {}
        self._permission_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._loading: set[str] = set()
        self._streamed_reasoning: dict[str, set[str]] = {}

    async def send(self, value: dict[str, Any]) -> None:
        data = (
            json.dumps({"jsonrpc": "2.0", **value}, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode()
        async with self._write_lock:
            self._writer.write(data)
            async with asyncio.timeout(30):
                await self._writer.drain()

    async def update(self, session_id: str, update: dict[str, Any]) -> None:
        await self.send(
            {"method": "session/update", "params": {"sessionId": session_id, "update": update}}
        )

    async def _message(
        self, session_id: str, text: str, *, kind: str = "agent_message_chunk"
    ) -> None:
        # Keep output frames bounded even for historical messages.
        for start in range(0, max(len(text), 1), 32_000):
            await self.update(
                session_id,
                {
                    "sessionUpdate": kind,
                    "content": {"type": "text", "text": text[start : start + 32_000]},
                },
            )

    async def _approval(self, session_id: str, request: ApprovalRequest) -> ApprovalChoice:
        request_id = f"noah-permission-{uuid.uuid4().hex}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._permission_responses[request_id] = future
        decision = request.decision
        title = f"{decision.category}: {decision.target}"[:8000]
        tool_kind = _infer_tool_kind(
            title,
            {"tool": getattr(decision, "tool", "")},
            category=str(decision.category),
        )
        try:
            await self.send(
                {
                    "id": request_id,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": request.id,
                            "title": title,
                            "kind": tool_kind,
                            "status": "pending",
                        },
                        "options": [
                            {"optionId": "once", "name": "Allow once", "kind": "allow_once"},
                            {
                                "optionId": "session",
                                "name": "Allow for this session",
                                "kind": "allow_always",
                            },
                            {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
                        ],
                    },
                }
            )
            response = await future
            outcome = response.get("outcome")
            if isinstance(outcome, dict) and outcome.get("outcome") == "selected":
                try:
                    return ApprovalChoice(str(outcome.get("optionId", "")))
                except (ValueError, TypeError):
                    pass
            return ApprovalChoice.REJECT
        finally:
            self._permission_responses.pop(request_id, None)

    async def _questions(self, session_id: str, prompts: list[Any]) -> Any:
        text = "\n\n".join(
            f"{item.header}: {item.prompt}\n" + "\n".join(f"- {option}" for option in item.options)
            for item in prompts
        )
        await self._message(session_id, text)
        # Do not fabricate a user answer or misuse permission options for a
        # question. The model sees this tool error and can end its current turn.
        raise PermissionError(
            "ACP client has no structured-question interface. Wait for a normal user reply."
        )

    def _bind_ui(self, session: ServiceSession) -> None:
        session_id = session.meta.session_id
        session.ui.approval_handler = lambda request: self._approval(session_id, request)
        session.ui.question_handler = lambda prompts: self._questions(session_id, prompts)

    async def _replay(self, session: ServiceSession) -> None:
        # The host's pages run backwards. First retain only page cursors, then
        # stream pages oldest-first; never hold the whole transcript in memory.
        cursors: list[int | None] = []
        before: int | None = None
        while True:
            rows = await session.host.load_history_page(before=before, limit=200)
            if not rows:
                break
            cursors.append(before)
            next_before = rows[0].insertion_order
            if before is not None and next_before >= before:
                raise RPCError(-32603, "session history pagination did not advance")
            before = next_before
        for cursor in reversed(cursors):
            for row in await session.host.load_history_page(before=cursor, limit=200):
                payload = row.payload
                if row.event_type == "Task":
                    await self._message(
                        session.meta.session_id,
                        str(payload.get("prompt", "")),
                        kind="user_message_chunk",
                    )
                elif row.event_type in {"Message", "AgentMessage", "AssistantEvent"}:
                    await self._message(session.meta.session_id, str(payload.get("content", "")))
                elif row.event_type == "Summary":
                    await self._message(
                        session.meta.session_id,
                        "Context summary:\n"
                        + str(payload.get("content", payload.get("summary", ""))),
                    )
                elif row.event_type == "ToolCallEvent":
                    result = payload.get("result")
                    result = result if isinstance(result, dict) else {}
                    failed = result.get("result_status") in {"failed", "error", "fail"}
                    title = str(payload.get("name", "Recorded tool call"))
                    await self.update(
                        session.meta.session_id,
                        {
                            "sessionUpdate": "tool_call",
                            "toolCallId": row.event_id,
                            "title": title,
                            "kind": _infer_tool_kind(title, {"tool": title}),
                            "status": "failed" if failed else "completed",
                        },
                    )

    async def _event(self, session_id: str, event: dict[str, Any]) -> None:
        kind, text = event["kind"], event.get("text", "")
        if kind == "message":
            await self._message(session_id, text)
        elif kind == "model_stream":
            meta = event.get("meta")
            if not isinstance(meta, dict):
                return
            phase = meta.get("phase")
            call_id = str(meta.get("call_id") or "_active")
            if phase == "reasoning" and text:
                self._streamed_reasoning.setdefault(session_id, set()).add(call_id)
                await self._message(session_id, text, kind="agent_thought_chunk")
            elif phase in {"error", "cancel"}:
                self._streamed_reasoning.get(session_id, set()).discard(call_id)
        elif kind == "reasoning":
            streamed = self._streamed_reasoning.get(session_id)
            if streamed:
                streamed.pop()
                return
            await self._message(session_id, text, kind="agent_thought_chunk")
        elif kind == "error" and text.strip():
            await self._message(session_id, f"Error: {text}")
        elif kind in {"tool_start", "tool_finish"}:
            meta = event.get("meta", {})
            tool_id = meta.get("activity_id")
            if not tool_id:
                return
            if kind == "tool_start":
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": tool_id,
                        "title": text,
                        "kind": _infer_tool_kind(text, meta),
                        "status": "in_progress",
                    },
                )
            else:
                failed = meta.get("result_status") in {"failed", "error", "fail"}
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tool_id,
                        "status": "failed" if failed else "completed",
                    },
                )
        elif kind == "shell_chunk" and event.get("meta", {}).get("activity_id"):
            await self.update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": event["meta"]["activity_id"],
                    "content": [{"type": "content", "content": {"type": "text", "text": text}}],
                },
            )
        elif kind == "replay_gap":
            await self._message(
                session_id,
                "[Note: some intermediate stream updates were skipped due to high output volume.]",
            )

    async def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if method == "initialize":
            version = params.get("protocolVersion")
            if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                raise RPCError(-32602, "protocolVersion must be a positive integer")
            if self.initialized:
                raise RPCError(-32600, "connection is already initialized")
            self.initialized = True
            return {
                "protocolVersion": 1,
                "agentInfo": {"name": "noah-code", "title": "Noah Code", "version": __version__},
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {
                        "image": True,
                        "audio": False,
                        "embeddedContext": True,
                    },
                    "mcpCapabilities": {"http": True, "sse": True},
                },
                "authMethods": [],
            }
        if not self.initialized:
            raise RPCError(-32600, "initialize must be called first")
        if method in {"session/new", "session/load"}:
            from noah_code.workspace import open_workspace

            cwd = _string(params, "cwd")
            if not Path(cwd).is_absolute() or params.get("additionalDirectories"):
                raise RPCError(
                    -32602, "cwd must be absolute; additional directories are unsupported"
                )
            session_id = _string(params, "sessionId") if method == "session/load" else None
            mcp_servers = _mcp_servers(params)
            workspace_root = open_workspace(cwd).root
            for existing in list(self.service.sessions.values()):
                same_session = session_id is not None and existing.meta.session_id == session_id
                same_workspace = existing.host.workspace.root == workspace_root
                if same_session or same_workspace:
                    if existing.active or existing.meta.session_id in self._loading:
                        raise RPCError(-32000, "session is busy")
                    await self.service.close_session(existing.meta.session_id)
            approved_mcp: set[str] = set()

            async def _open_approval(sid: str, request: ApprovalRequest) -> ApprovalChoice:
                choice = await self._approval(sid, request)
                if choice != ApprovalChoice.REJECT:
                    approved_mcp.add(str(request.decision.target))
                return choice

            session = await self.service.open_session(
                cwd=cwd,
                session_id=session_id,
                mcp_servers=mcp_servers,
                approval_handler=_open_approval,
            )
            self._bind_ui(session)
            host_cfg = getattr(session.host, "config", self.service.config)
            if not (getattr(host_cfg, "auto_approve", False) or getattr(host_cfg, "yolo", False)):
                for srv_name, srv_spec in mcp_servers.items():
                    if (
                        srv_spec.get("transport") == "stdio"
                        and srv_spec.get("trusted") is False
                        and srv_name not in approved_mcp
                    ):
                        decision = PermissionDecision(
                            "mcp",
                            f"{srv_name} ({srv_spec.get('command', '')})",
                            "ask",
                            None,
                            "client-supplied stdio MCP server requires approval",
                            srv_name,
                            tool="mcp",
                        )
                        req = ApprovalRequest(
                            f"mcp-{uuid.uuid4().hex}",
                            decision,
                            time.time(),
                            asyncio.get_running_loop().create_future(),
                        )
                        choice = await self._approval(session.meta.session_id, req)
                        if choice == ApprovalChoice.REJECT:
                            await self.service.close_session(session.meta.session_id)
                            raise RPCError(
                                -32000, f"rejected client-supplied stdio MCP server: {srv_name}"
                            )
            if method == "session/load":
                session_id = session.meta.session_id
                if session.active or session_id in self._loading:
                    raise RPCError(-32000, "session is busy")
                self._loading.add(session_id)
                try:
                    await self._replay(session)
                finally:
                    self._loading.discard(session_id)
                return {}
            return {"sessionId": session.meta.session_id}
        if method == "session/set_mode":
            session_id = _string(params, "sessionId")
            raw_mode = params.get("modeId") if "modeId" in params else params.get("mode")
            if not isinstance(raw_mode, str) or raw_mode.strip().lower() not in {"build", "plan"}:
                raise RPCError(-32602, "modeId must be 'build' or 'plan'")
            mode = raw_mode.strip().lower()
            session = self.service.get(session_id)
            async with session.lock:
                if session.active:
                    raise RPCError(-32000, "session is busy")
                if getattr(session.host, "_agent", None) is not None:
                    session.host.agent.set_mode(mode)
                session.meta.mode = mode
                if getattr(session.host, "meta", None) is not None:
                    session.host.meta.mode = mode
                await asyncio.to_thread(self.service.store.save_meta, session.meta)
            return {}
        if method == "session/set_model":
            session_id = _string(params, "sessionId")
            raw_model = params.get("modelId") if "modelId" in params else params.get("model")
            if (
                not isinstance(raw_model, str)
                or not raw_model.strip()
                or any(ch.isspace() for ch in raw_model.strip())
            ):
                raise RPCError(-32602, "modelId must be a nonempty model identifier")
            model = raw_model.strip()
            session = self.service.get(session_id)
            async with session.lock:
                if session.active:
                    raise RPCError(-32000, "session is busy")
                switch_fn = getattr(session.host, "_switch_model", None)
                if callable(switch_fn) and getattr(session.host, "_agent", None) is not None:
                    await switch_fn(model)
                else:
                    session.meta.model = model
                    if getattr(session.host, "meta", None) is not None:
                        session.host.meta.model = model
                    if hasattr(session.host, "config"):
                        session.host.config.model = model
                    await asyncio.to_thread(self.service.store.save_meta, session.meta)
            return {}
        if method == "session/cancel":
            await self.service.cancel(_string(params, "sessionId"))
            return None
        if method == "session/prompt":
            session_id = _string(params, "sessionId")
            if session_id in self._loading:
                raise RPCError(-32000, "session history is still loading")
            session = self.service.get(session_id)
            after = session.ui.sequence
            self._streamed_reasoning.pop(session_id, None)
            receipt = self.service.submit(session_id, _prompt_text(params.get("prompt")))
            try:
                async for event in session.ui.stream(after):
                    if (
                        event["kind"] == "turn_complete"
                        and event["request_id"] == receipt["request_id"]
                    ):
                        result = event["result"]
                        if result["status"] == "failed":
                            raise RPCError(-32000, safe_error_message(result["explanation"]))
                        return {
                            "stopReason": (
                                "cancelled" if result["status"] == "cancelled" else "end_turn"
                            )
                        }
                    await self._event(session_id, event)
            finally:
                self._streamed_reasoning.pop(session_id, None)
            raise RPCError(-32000, "session closed during prompt")
        raise RPCError(-32601, "method not found")

    async def _dispatch(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        try:
            params = message.get("params", {})
            if not isinstance(params, dict):
                raise RPCError(-32602, "params must be an object")
            result = await self.handle(message["method"], params)
            if "id" in message:
                await self.send({"id": request_id, "result": result or {}})
        except RPCError as exc:
            if "id" in message:
                await self.send(
                    {"id": request_id, "error": {"code": exc.code, "message": str(exc)}}
                )
        except (ServiceError, ValueError, OSError, RuntimeError) as exc:
            if "id" in message:
                await self.send(
                    {
                        "id": request_id,
                        "error": {"code": -32000, "message": safe_error_message(exc)},
                    }
                )
        except Exception:
            if "id" in message:
                await self.send(
                    {"id": request_id, "error": {"code": -32603, "message": "internal agent error"}}
                )
        finally:
            if isinstance(request_id, (str, int)):
                self._requests.pop(request_id, None)

    async def run(self, reader: asyncio.StreamReader, writer: Any) -> None:
        self._writer = writer
        try:
            while True:
                try:
                    raw = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await self.send(
                        {
                            "id": None,
                            "error": {"code": -32600, "message": "request frame too large"},
                        }
                    )
                    break
                if not raw:
                    break
                if len(raw) > MAX_FRAME_BYTES:
                    await self.send(
                        {
                            "id": None,
                            "error": {"code": -32600, "message": "request frame too large"},
                        }
                    )
                    break
                try:
                    message = json.loads(raw)
                except (ValueError, UnicodeError):
                    await self.send(
                        {"id": None, "error": {"code": -32700, "message": "parse error"}}
                    )
                    continue
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    await self.send(
                        {"id": None, "error": {"code": -32600, "message": "invalid request"}}
                    )
                    continue
                request_id = message.get("id")
                if "method" not in message:
                    future = (
                        self._permission_responses.get(request_id)
                        if isinstance(request_id, str)
                        else None
                    )
                    if future is not None and not future.done():
                        result = message.get("result")
                        future.set_result(result if isinstance(result, dict) else {})
                    continue
                if not isinstance(message["method"], str) or (
                    "id" in message
                    and (not isinstance(request_id, (str, int)) or isinstance(request_id, bool))
                ):
                    await self.send(
                        {"id": None, "error": {"code": -32600, "message": "invalid request"}}
                    )
                    continue
                if request_id in self._requests:
                    await self.send(
                        {
                            "id": request_id,
                            "error": {"code": -32600, "message": "duplicate active request id"},
                        }
                    )
                    continue
                # Notifications other than cancellation are unsupported. Do not
                # execute a request-shaped mutation that has no response channel.
                if "id" not in message and message["method"] != "session/cancel":
                    continue
                if len(self._tasks) >= 64:
                    if "id" in message:
                        await self.send(
                            {
                                "id": request_id,
                                "error": {"code": -32000, "message": "too many active requests"},
                            }
                        )
                    continue
                task = asyncio.create_task(self._dispatch(message))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                if isinstance(request_id, (str, int)):
                    self._requests[request_id] = task
                # Let initialization and prompt registration execute before
                # processing the next already-buffered client message.
                await asyncio.sleep(0)
        finally:
            for future in self._permission_responses.values():
                if not future.done():
                    future.cancel()
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.service.close()


async def run_stdio(service: AgentService) -> None:
    """Run ACP on standard pipes; all incidental library output goes to stderr."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
    protocol = asyncio.StreamReaderProtocol(reader)
    input_transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    factory: Callable[[], Any] = asyncio.streams.FlowControlMixin
    output_transport, output_protocol = await loop.connect_write_pipe(factory, sys.stdout.buffer)
    writer = asyncio.StreamWriter(output_transport, output_protocol, None, loop)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            await ACPConnection(service).run(reader, writer)
    finally:
        input_transport.close()
        output_transport.close()
