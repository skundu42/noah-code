"""Owned agent sessions and a small authenticated HTTP/SSE transport.

The service is an adapter around AgentHost: it never implements tools, permissions,
or a second agent loop. HTTP connections can disappear without terminating work.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from noah_code.approvals import ApprovalChoice, ApprovalRequest
from noah_code.config import NoahCodeConfig
from noah_code.events import HostEvent
from noah_code.host import AgentHost, HostResult
from noah_code.redaction import safe_error_message
from noah_code.sessions import SessionError, SessionEventRecord, SessionMeta, SessionStore
from noah_code.workspace import Workspace, open_workspace

MAX_REQUEST_BYTES = 1_048_576
MAX_EVENT_TEXT = 32_000
MAX_PENDING_INTERACTIONS = 32


class ServiceError(Exception):
    """A bounded, client-facing service error."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _summary(meta: SessionMeta) -> dict[str, Any]:
    # Never serialize the entire configuration or remembered permission rules.
    return {
        key: getattr(meta, key)
        for key in (
            "session_id",
            "workspace_path",
            "title",
            "mode",
            "model",
            "created_at",
            "updated_at",
        )
    }


def _history_event(record: SessionEventRecord) -> dict[str, Any] | None:
    """Project transcript fields; never expose raw provider/debug event payloads."""
    source = record.payload
    payload: dict[str, Any]
    if record.event_type == "Task":
        payload = {"prompt": str(source.get("prompt", ""))}
    elif record.event_type in {"Message", "AssistantEvent", "Reasoning"}:
        payload = {"content": str(source.get("content", ""))}
    elif record.event_type == "Summary":
        payload = {"content": str(source.get("content", source.get("summary", "")))}
    elif record.event_type == "Error":
        payload = {"content": safe_error_message(str(source.get("content", "")), limit=4000)}
    elif record.event_type == "ToolCallEvent":
        result = source.get("result")
        result = result if isinstance(result, dict) else {}
        payload = {
            "name": str(source.get("name", "tool")),
            "result": {
                "result_status": str(result.get("result_status", "recorded")),
            },
        }
        if error := result.get("error"):
            payload["result"]["error"] = safe_error_message(str(error), limit=4000)
    else:
        return None
    return {
        "insertion_order": record.insertion_order,
        "event_id": record.event_id,
        "event_type": record.event_type,
        "payload": payload,
    }


class ServiceUI:
    """HostUI with bounded replay and interactions independent of connections."""

    def __init__(self, *, history_limit: int = 1024) -> None:
        if not 1 <= history_limit <= 10_000:
            raise ValueError("history_limit must be between 1 and 10000")
        self.events: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self.sequence = 0
        self.changed = asyncio.Event()
        self.closed = False
        self.busy = False
        self.pending: dict[str, tuple[dict[str, Any], asyncio.Future[Any]]] = {}
        self.approval_handler: Callable[[ApprovalRequest], Awaitable[ApprovalChoice]] | None = None
        self.question_handler: Callable[[list[Any]], Awaitable[Any]] | None = None
        self._loop = asyncio.get_running_loop()

    def publish(self, kind: str, **data: Any) -> None:
        if self.closed:
            return
        self.sequence += 1
        self.events.append({"sequence": self.sequence, "kind": kind, **data})
        self.changed.set()

    def render(self, event: HostEvent) -> None:
        # Event callbacks may originate in a tool's worker thread. Only the
        # owning event loop touches replay state and wakes stream consumers.
        meta = {
            key: value[:2000] if isinstance(value, str) else value
            for key, value in event.meta.items()
            if key
            in {
                "activity_id",
                "tool",
                "state",
                "result_status",
                "stream",
                "source",
                "kind",
                "phase",
                "call_id",
                "model",
                "attempt",
                "error_type",
            }
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        text = str(event.text)
        if str(event.kind) == "error":
            text = safe_error_message(text, limit=MAX_EVENT_TEXT)

        def emit() -> None:
            for start in range(0, max(len(text), 1), MAX_EVENT_TEXT):
                self.publish(str(event.kind), text=text[start : start + MAX_EVENT_TEXT], meta=meta)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self._loop:
            emit()
        elif not self._loop.is_closed():
            self._loop.call_soon_threadsafe(emit)

    def set_status(self, text: str) -> None:
        self.publish("status", text=text[:MAX_EVENT_TEXT])

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.publish("busy", busy=busy)

    async def prompt(self, status: str) -> None:
        raise RuntimeError("service prompts must be submitted through the API")

    async def ask_approval(self, request: ApprovalRequest) -> ApprovalChoice:
        if self.approval_handler is not None:
            return await self.approval_handler(request)
        decision = request.decision
        value = await self._interact(
            request.id,
            {
                "type": "approval",
                "category": str(decision.category),
                "target": decision.target[:MAX_EVENT_TEXT],
                "reason": decision.reason[:4000],
                "choices": [item.value for item in ApprovalChoice],
            },
        )
        return ApprovalChoice(value["choice"])

    async def ask_questions(self, prompts: list[Any]) -> Any:
        if self.question_handler is not None:
            return await self.question_handler(prompts)
        from noah_code.tools.question_tools import QuestionAnswer

        value = await self._interact(
            uuid.uuid4().hex,
            {
                "type": "question",
                "prompts": [asdict(prompt) for prompt in prompts],
            },
        )
        return QuestionAnswer(value["selections"], value["custom"])

    async def _interact(self, interaction_id: str, detail: dict[str, Any]) -> Any:
        if self.closed or len(self.pending) >= MAX_PENDING_INTERACTIONS:
            raise PermissionError("service interaction limit reached or session closed")
        future = self._loop.create_future()
        self.pending[interaction_id] = (detail, future)
        self.publish("interaction", interaction_id=interaction_id, **detail)
        try:
            return await future
        finally:
            self.pending.pop(interaction_id, None)
            self.publish("interaction_closed", interaction_id=interaction_id)

    def answer(self, interaction_id: str, answer: dict[str, Any]) -> None:
        item = self.pending.get(interaction_id)
        if item is None or item[1].done():
            raise ServiceError("interaction is no longer pending", 404)
        detail, future = item
        if detail["type"] == "approval":
            if answer.get("choice") not in {item.value for item in ApprovalChoice}:
                raise ServiceError("choice must be once, session, or reject")
            result = {"choice": answer["choice"]}
        else:
            selections, custom = answer.get("selections", []), answer.get("custom", "")
            available = {option for prompt in detail["prompts"] for option in prompt["options"]}
            if (
                not isinstance(selections, list)
                or len(selections) > 100
                or any(not isinstance(item, str) or item not in available for item in selections)
                or not isinstance(custom, str)
                or len(custom) > MAX_EVENT_TEXT
            ):
                raise ServiceError("invalid question answer")
            result = {"selections": selections, "custom": custom}
        future.set_result(result)

    async def stream(self, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        if after < 0 or after > self.sequence:
            raise ServiceError("event cursor is outside this session's stream")
        while True:
            self.changed.clear()
            if self.events and after < self.events[0]["sequence"] - 1:
                after = self.events[0]["sequence"] - 1
                yield {"sequence": after, "kind": "replay_gap", "oldest_sequence": after + 1}
            for item in tuple(self.events):
                if item["sequence"] > after:
                    after = item["sequence"]
                    yield item
            if self.closed:
                return
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=15)
            except TimeoutError:
                yield {"sequence": after, "kind": "heartbeat"}

    def close(self) -> None:
        self.publish("session_closed")
        self.closed = True
        for _detail, future in self.pending.values():
            if not future.done():
                future.cancel()
        self.changed.set()


@dataclass
class ServiceSession:
    host: Any
    ui: ServiceUI
    meta: SessionMeta
    task: asyncio.Task[Any] | None = None
    request_id: str | None = None
    result: HostResult | None = None
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

    def snapshot(self) -> dict[str, Any]:
        return {
            **_summary(self.meta),
            "active": self.active,
            "request_id": self.request_id,
            "sequence": self.ui.sequence,
            "interactions": [
                {"interaction_id": key, **detail}
                for key, (detail, _future) in self.ui.pending.items()
            ],
            "result": asdict(self.result) if self.result is not None else None,
        }


class AgentService:
    """Own hosts until explicit close; transport disconnects do not cancel HTTP work."""

    def __init__(
        self,
        workspace: Workspace,
        config: NoahCodeConfig,
        *,
        host_factory: Callable[..., Any] = AgentHost,
        max_sessions: int = 8,
        history_limit: int = 1024,
    ) -> None:
        if not 1 <= max_sessions <= 64:
            raise ValueError("max_sessions must be between 1 and 64")
        self.workspace, self.config = workspace, config
        self.store = SessionStore(config.session_dir)
        self.sessions: dict[str, ServiceSession] = {}
        self._host_factory = host_factory
        self._max_sessions, self._history_limit = max_sessions, history_limit
        self._lock = asyncio.Lock()
        self._closed = False

    async def open_session(
        self,
        *,
        cwd: str | None = None,
        session_id: str | None = None,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
    ) -> ServiceSession:
        async with self._lock:
            if self._closed:
                raise ServiceError("service is closed", 503)
            workspace = open_workspace(cwd or self.workspace.root)
            if session_id in self.sessions:
                existing = self.sessions[session_id]
                if existing.host.workspace.root != workspace.root:
                    raise ServiceError("session belongs to a different workspace", 409)
                if mcp_servers:
                    raise ServiceError("close the active session before changing MCP servers", 409)
                return existing
            if len(self.sessions) >= self._max_sessions:
                raise ServiceError("active session limit reached; close a session first", 409)
            if any(item.host.workspace.root == workspace.root for item in self.sessions.values()):
                raise ServiceError(
                    "workspace already active; close it or use a separate worktree", 409
                )
            meta = await asyncio.to_thread(self.store.load_meta, session_id) if session_id else None
            if meta is not None:
                self.store.verify_workspace(meta, workspace)
            config = self.config.model_copy(deep=True)
            if mcp_servers:
                # The authenticated client supplied these commands, as it would
                # in trusted user configuration. Never save their credentials.
                from noah_code.mcp_setup import load_mcp_servers

                configured, _sources = load_mcp_servers(workspace.root, config)
                config.mcp = {**configured, **mcp_servers}
                config.efficiency.lazy_mcp = False
            ui = ServiceUI(history_limit=self._history_limit)
            host = self._host_factory(workspace, config, ui=ui, session_meta=meta, store=self.store)
            try:
                meta = await host.start()
                if mcp_servers and not set(mcp_servers).issubset(host._mcp_attached):
                    raise ServiceError("one or more requested MCP servers could not connect", 409)
            except BaseException:
                ui.close()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await host.close()
                raise
            session = ServiceSession(host, ui, meta)
            self.sessions[meta.session_id] = session
            ui.publish("session_opened", session_id=meta.session_id)
            return session

    def get(self, session_id: str) -> ServiceSession:
        session = self.sessions.get(session_id)
        if session is None or session.closed:
            raise ServiceError("session is not open; create or load it first", 404)
        return session

    async def list_sessions(self) -> list[dict[str, Any]]:
        metas = await asyncio.to_thread(self.store.list_sessions)
        return [{**_summary(meta), "open": meta.session_id in self.sessions} for meta in metas]

    def submit(self, session_id: str, prompt: str, *, queue: bool = False) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256_000:
            raise ServiceError("prompt must contain 1 to 256000 characters")
        session = self.get(session_id)
        if session.active:
            if not queue:
                raise ServiceError("session is busy; use queue=true or cancel it", 409)
            from noah_code.steer import STEER_QUEUE_CAP

            if len(session.host.steer_queue.items()) >= STEER_QUEUE_CAP:
                raise ServiceError("prompt queue is full", 409)
            session.host.enqueue_steer(prompt)
            session.ui.publish("prompt_queued")
            return {"status": "queued", "request_id": session.request_id}
        session.request_id = uuid.uuid4().hex
        session.result = None
        session.task = asyncio.create_task(self._run(session, prompt))
        return {"status": "running", "request_id": session.request_id}

    async def _run(self, session: ServiceSession, prompt: str | None) -> HostResult:
        try:
            result = (
                await session.host.resume_interrupted_run()
                if prompt is None
                else await session.host.submit_prompt(prompt)
            )
            session.result = result or HostResult(0, session_id=session.meta.session_id)
            if session.result.status == "failed":
                session.result.explanation = safe_error_message(session.result.explanation)
        except asyncio.CancelledError:
            session.result = HostResult(130, "cancelled", session.meta.session_id, "cancelled")
        except Exception as exc:
            session.result = HostResult(
                1, safe_error_message(exc), session.meta.session_id, "failed"
            )
            session.ui.publish("error", text=session.result.explanation)
        session.ui.publish(
            "turn_complete", request_id=session.request_id, result=asdict(session.result)
        )
        return session.result

    def recover(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session.active:
            raise ServiceError("session is busy", 409)
        session.request_id = uuid.uuid4().hex
        session.result = None
        session.task = asyncio.create_task(self._run(session, None))
        return {"status": "running", "request_id": session.request_id}

    async def cancel(self, session_id: str) -> None:
        session = self.get(session_id)
        if session.active:
            session.host.cancel_active_turn()
            assert session.task is not None
            session.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.task
            if session.result is None:
                # A just-created task can be cancelled before _run starts.
                session.result = HostResult(130, "cancelled", session_id, "cancelled")
                session.ui.publish(
                    "turn_complete", request_id=session.request_id, result=asdict(session.result)
                )

    async def close_session(self, session_id: str) -> None:
        session = self.get(session_id)
        async with session.lock:
            if session.closed:
                return
            session.closed = True
            try:
                if session.active:
                    session.host.cancel_active_turn()
                    assert session.task is not None
                    session.task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await session.task
                await session.host.close()
            finally:
                session.ui.close()
                self.sessions.pop(session_id, None)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            results = await asyncio.gather(
                *(self.close_session(key) for key in list(self.sessions)),
                return_exceptions=True,
            )
            if any(isinstance(result, BaseException) for result in results):
                raise ServiceError("one or more sessions failed to close cleanly", 500)


class HTTPService:
    """Bounded HTTP/1.1 JSON requests and SSE, one request per connection."""

    def __init__(self, service: AgentService, token: str) -> None:
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 1024
            or not token.isascii()
            or any(char.isspace() for char in token)
        ):
            raise ValueError("HTTP token must contain 32 to 1024 non-whitespace ASCII characters")
        self.service = service
        self._authorization = ("Bearer " + token).encode("ascii")
        self.server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[Any]] = set()
        self._closing = False

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> HTTPService:
        if self.server is not None or self._closing:
            raise RuntimeError("HTTP service cannot be started twice")
        self.server = await asyncio.start_server(self._handle, host, port, limit=16_384)
        return self

    async def close(self) -> None:
        self._closing = True
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self._connections)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.service.close()

    async def serve_forever(self) -> None:
        if self.server is None:
            raise RuntimeError("HTTP service has not been started")
        try:
            await self.server.serve_forever()
        finally:
            await self.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        if self._closing or len(self._connections) >= 64:
            writer.close()
            return
        self._connections.add(task)
        streaming = False
        try:
            async with asyncio.timeout(15):
                raw = await reader.readuntil(b"\r\n\r\n")
                lines = raw.decode("iso-8859-1").split("\r\n")
                method, target, version = lines[0].split(" ")
                if version != "HTTP/1.1" or not target.startswith("/"):
                    raise ServiceError("expected an HTTP/1.1 origin-form request")
                headers: dict[str, str] = {}
                for line in lines[1:-2]:
                    key, separator, value = line.partition(":")
                    key = key.lower()
                    if not separator or key in headers or key.strip() != key:
                        raise ServiceError("invalid or duplicate HTTP header")
                    headers[key] = value.strip()
                authorization = headers.get("authorization", "").encode("iso-8859-1")
                if not hmac.compare_digest(authorization, self._authorization):
                    raise ServiceError("authentication required", 401)
                if "origin" in headers:
                    raise ServiceError("browser-origin requests are not supported", 403)
                if "transfer-encoding" in headers:
                    raise ServiceError("chunked request bodies are not supported")
                length = int(headers.get("content-length", "0"))
                if not 0 <= length <= MAX_REQUEST_BYTES:
                    raise ServiceError("request body too large", 413)
                body = await reader.readexactly(length)
            parts = urlsplit(target)
            path = parts.path.rstrip("/").split("/")
            query = parse_qs(parts.query)
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                raise ServiceError("request body must be a JSON object")
            if method == "GET" and len(path) == 5 and path[4] == "events":
                self._prefix(path)
                ui = self.service.get(path[3]).ui
                cursor = query.get("after", [headers.get("last-event-id", "0")])[0]
                after = int(cursor)
                if not 0 <= after <= ui.sequence:
                    raise ServiceError("invalid event cursor")
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Cache-Control: no-store\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                streaming = True
                disconnected = asyncio.create_task(reader.read(1))
                try:
                    iterator = ui.stream(after).__aiter__()
                    while not disconnected.done():
                        next_event = asyncio.ensure_future(anext(iterator))
                        try:
                            done, _pending = await asyncio.wait(
                                {next_event, disconnected},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if disconnected in done:
                                break
                            event = next_event.result()
                            payload = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
                            writer.write(f"id: {event['sequence']}\ndata: {payload}\n\n".encode())
                            async with asyncio.timeout(15):
                                await writer.drain()
                        except StopAsyncIteration:
                            break
                        finally:
                            if not next_event.done():
                                next_event.cancel()
                            await asyncio.gather(next_event, return_exceptions=True)
                finally:
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
                return
            status, result = await self._route(method, path, query, data)
            await self._respond(writer, status, result)
        except ServiceError as exc:
            if not streaming:
                await self._respond(writer, exc.status, {"error": safe_error_message(exc)})
        except (ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            if not streaming:
                await self._respond(writer, 400, {"error": "invalid request"})
        except TimeoutError:
            if not streaming:
                await self._respond(writer, 408, {"error": "request timed out"})
        except (SessionError, OSError, RuntimeError) as exc:
            if not streaming:
                await self._respond(writer, 409, {"error": safe_error_message(exc)})
        except Exception:
            if not streaming:
                await self._respond(writer, 500, {"error": "internal service error"})
        finally:
            self._connections.discard(task)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    @staticmethod
    def _prefix(path: list[str]) -> None:
        if path[:3] != ["", "v1", "sessions"]:
            raise ServiceError("endpoint not found", 404)

    async def _route(
        self,
        method: str,
        path: list[str],
        query: dict[str, list[str]],
        data: dict[str, Any],
    ) -> tuple[int, Any]:
        self._prefix(path)
        if len(path) == 3:
            if method == "GET":
                return 200, {"sessions": await self.service.list_sessions()}
            if method == "POST":
                for key in ("cwd", "session_id"):
                    if key in data and not isinstance(data[key], str):
                        raise ServiceError(f"{key} must be a string")
                session = await self.service.open_session(
                    cwd=data.get("cwd"),
                    session_id=data.get("session_id"),
                )
                return 201, session.snapshot()
        if len(path) >= 4:
            session_id = path[3]
            session = self.service.get(session_id)
            if len(path) == 4:
                if method == "GET":
                    return 200, session.snapshot()
                if method == "DELETE":
                    await self.service.close_session(session_id)
                    return 200, {"status": "closed"}
            if len(path) == 5:
                action = path[4]
                if method == "POST" and action == "prompt":
                    if not isinstance(data.get("queue", False), bool):
                        raise ServiceError("queue must be a boolean")
                    return 202, self.service.submit(
                        session_id, data.get("prompt", ""), queue=data.get("queue", False)
                    )
                if method == "POST" and action == "cancel":
                    await self.service.cancel(session_id)
                    return 200, {"status": "cancelled"}
                if method == "POST" and action == "recover":
                    return 202, self.service.recover(session_id)
                if method == "GET" and action == "history":
                    before = int(query["before"][0]) if "before" in query else None
                    limit = int(query.get("limit", ["50"])[0])
                    rows = await session.host.load_history_page(before=before, limit=limit)
                    return 200, {
                        "events": [
                            item for row in rows if (item := _history_event(row)) is not None
                        ],
                        "before": rows[0].insertion_order if rows else None,
                    }
            if len(path) == 6 and path[4] == "interactions" and method == "POST":
                session.ui.answer(path[5], data)
                return 200, {"status": "answered"}
        raise ServiceError("endpoint not found", 404)

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
        reason = {
            200: "OK",
            201: "Created",
            202: "Accepted",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            408: "Request Timeout",
            409: "Conflict",
            413: "Payload Too Large",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status, "Error")
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\n"
            "Connection: close\r\n\r\n".encode()
            + body
        )
        with contextlib.suppress(ConnectionError, TimeoutError):
            async with asyncio.timeout(15):
                await writer.drain()


async def serve_http(
    service: AgentService,
    *,
    token: str,
    host: str = "127.0.0.1",
    port: int = 0,
) -> HTTPService:
    """Start an HTTP endpoint; caller owns serve_forever()/close()."""
    return await HTTPService(service, token).start(host, port)
