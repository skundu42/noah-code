"""Agent Client Protocol v1 over newline-delimited JSON-RPC stdio.

Wire shapes follow the official v1 schema, not the draft v2 protocol:
https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/v1/schema.json
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from collections.abc import Callable
from typing import Any

from noah_code import __version__
from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.redaction import safe_error_message
from noah_code.service import AgentService, ServiceError, ServiceSession

MAX_FRAME_BYTES = 1_048_576


class RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RPCError(-32602, f"{name} must be a nonempty string")
    return value


def _prompt_text(blocks: Any) -> str:
    if not isinstance(blocks, list) or not blocks or len(blocks) > 100:
        raise RPCError(-32602, "prompt must contain 1 to 100 content blocks")
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RPCError(-32602, "invalid prompt content block")
        if block.get("type") == "text":
            parts.append(_string(block, "text"))
        elif block.get("type") == "resource_link":
            # References stay references. Host tools perform any subsequent
            # reads/fetches with the existing path and network permission gates.
            parts.append(
                f"Referenced resource: {_string(block, 'name')}\nURI: {_string(block, 'uri')}"
            )
        else:
            raise RPCError(-32602, "only text and resource_link prompt blocks are supported")
    return "\n\n".join(parts)


def _mcp_servers(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from pathlib import Path

    servers = params.get("mcpServers")
    if not isinstance(servers, list) or len(servers) > 32:
        raise RPCError(-32602, "mcpServers must be an array of at most 32 stdio servers")
    result: dict[str, dict[str, Any]] = {}
    for server in servers:
        if not isinstance(server, dict) or server.get("type") not in {None, "stdio"}:
            raise RPCError(-32602, "only stdio MCP servers are supported")
        name, command = _string(server, "name"), _string(server, "command")
        if name in result or not Path(command).is_absolute():
            raise RPCError(-32602, "MCP names must be unique and commands must be absolute paths")
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
        result[name] = {"transport": "stdio", "command": command, "args": args, "env": environment}
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
        try:
            await self.send(
                {
                    "id": request_id,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": request.id,
                            "title": f"{decision.category}: {decision.target}"[:8000],
                            "kind": "other",
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
                elif row.event_type in {"Message", "AssistantEvent"}:
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
                    await self.update(
                        session.meta.session_id,
                        {
                            "sessionUpdate": "tool_call",
                            "toolCallId": row.event_id,
                            "title": str(payload.get("name", "Recorded tool call")),
                            "kind": "other",
                            "status": "failed" if failed else "completed",
                        },
                    )

    async def _event(self, session_id: str, event: dict[str, Any]) -> None:
        kind, text = event["kind"], event.get("text", "")
        if kind == "message":
            await self._message(session_id, text)
        elif kind == "reasoning":
            await self._message(session_id, text, kind="agent_thought_chunk")
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
                        "kind": "other",
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
            raise RPCError(
                -32603, "client could not keep up with session updates; reload the session"
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
                        "image": False,
                        "audio": False,
                        "embeddedContext": False,
                    },
                    "mcpCapabilities": {"http": False, "sse": False},
                },
                "authMethods": [],
            }
        if not self.initialized:
            raise RPCError(-32600, "initialize must be called first")
        if method in {"session/new", "session/load"}:
            from pathlib import Path

            cwd = _string(params, "cwd")
            if not Path(cwd).is_absolute() or params.get("additionalDirectories"):
                raise RPCError(
                    -32602, "cwd must be absolute; additional directories are unsupported"
                )
            session_id = _string(params, "sessionId") if method == "session/load" else None
            session = await self.service.open_session(
                cwd=cwd, session_id=session_id, mcp_servers=_mcp_servers(params)
            )
            self._bind_ui(session)
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
        if method == "session/cancel":
            await self.service.cancel(_string(params, "sessionId"))
            return None
        if method == "session/prompt":
            session_id = _string(params, "sessionId")
            if session_id in self._loading:
                raise RPCError(-32000, "session history is still loading")
            session = self.service.get(session_id)
            after = session.ui.sequence
            receipt = self.service.submit(session_id, _prompt_text(params.get("prompt")))
            async for event in session.ui.stream(after):
                if (
                    event["kind"] == "turn_complete"
                    and event["request_id"] == receipt["request_id"]
                ):
                    result = event["result"]
                    if result["status"] == "failed":
                        raise RPCError(-32000, safe_error_message(result["explanation"]))
                    return {
                        "stopReason": "cancelled" if result["status"] == "cancelled" else "end_turn"
                    }
                await self._event(session_id, event)
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
