"""Lazy Language Server Protocol navigation and compact repository symbols."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from nooa import Skill, hidden, spec

from noah_code.approvals import ApprovalBroker
from noah_code.credentials import scrubbed_subprocess_env
from noah_code.permissions import PermissionCategory, PermissionEngine, is_secret_path
from noah_code.workspace import Workspace, parse_porcelain_z

_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".rb": "ruby",
}

_SERVER_CANDIDATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "python": (
        ("basedpyright-langserver", "--stdio"),
        ("pyright-langserver", "--stdio"),
        ("pylsp",),
    ),
    "javascript": (("typescript-language-server", "--stdio"),),
    "javascriptreact": (("typescript-language-server", "--stdio"),),
    "typescript": (("typescript-language-server", "--stdio"),),
    "typescriptreact": (("typescript-language-server", "--stdio"),),
    "go": (("gopls",),),
    "rust": (("rust-analyzer",),),
    "c": (("clangd",),),
    "cpp": (("clangd",),),
    "java": (("jdtls",),),
    "ruby": (("ruby-lsp",),),
}

_SYMBOL_KINDS = {
    2: "module",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    23: "struct",
    26: "type",
}

_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


@dataclass(frozen=True)
class RepositorySymbol:
    path: str
    line: int
    kind: str
    name: str
    detail: str = ""

    def render(self) -> str:
        suffix = f" — {self.detail}" if self.detail else ""
        return f"{self.path}:{self.line}  {self.kind} {self.name}{suffix}"


def _path_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"unsupported LSP URI: {uri}")
    path_part = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        path_part = f"//{parsed.netloc}{path_part}"
    return Path(path_part)


def _is_secret_uri(uri: str, root: Path) -> bool:
    try:
        path = _uri_path(uri)
    except ValueError:
        return False
    try:
        rel = path.resolve().relative_to(root.resolve())
        return is_secret_path(rel)
    except (OSError, ValueError):
        return is_secret_path(path.name)


def _position_for_file(resolved: Path, line: int, column: int) -> dict[str, int]:
    if line < 1 or column < 1:
        raise ValueError("line and column are 1-indexed and must be positive")
    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"line": line - 1, "character": column - 1}
    if 1 <= line <= len(lines):
        prefix = lines[line - 1][: column - 1]
        utf16_col = len(prefix.encode("utf-16-le")) // 2
        return {"line": line - 1, "character": utf16_col}
    return {"line": line - 1, "character": column - 1}


def _location_text(item: dict[str, Any], root: Path) -> str:
    target = item.get("targetUri") or item.get("uri") or ""
    target_range = item.get("targetSelectionRange") or item.get("range") or {}
    start = target_range.get("start", {})
    try:
        path = _uri_path(target)
        display = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    except (ValueError, TypeError):
        display = str(target)
    return f"{display}:{int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}"


class _LSPClient:
    """Small JSON-RPC stdio client with owned process lifecycle."""

    def __init__(self, root: Path, command: tuple[str, ...], *, timeout: float) -> None:
        self.root = root
        self.command = command
        self.timeout = timeout
        self.process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._diagnostic_events: dict[str, asyncio.Event] = {}
        self._open_documents: dict[str, tuple[tuple[int, int], int, Path, str]] = {}
        self._closed = False

    def _healthy(self) -> bool:
        """True only while the transport can still deliver responses."""

        return (
            self.process is not None
            and self.process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Kill a process and its whole session group, then reap it."""

        if process.returncode is None:
            with contextlib.suppress(OSError):
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)

    async def _cancel_pump_tasks(self) -> None:
        reader, self._reader_task = self._reader_task, None
        stderr, self._stderr_task = self._stderr_task, None
        bg = list(getattr(self, "_background_tasks", ()))
        if hasattr(self, "_background_tasks"):
            self._background_tasks.clear()
        for task in (reader, stderr, *bg):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("LSP client is closed")
        if self._healthy():
            return
        async with self._start_lock:
            if self._healthy():
                return
            stale, self.process = self.process, None
            await self._cancel_pump_tasks()
            reopen: list[tuple[Path, str]] = [
                (doc_path, doc_lang)
                for _stamp, _version, doc_path, doc_lang in self._open_documents.values()
            ]
            self._open_documents.clear()
            self._diagnostics.clear()
            self._diagnostic_events.clear()
            if stale is not None:
                # Never drop a live handle: the old server would leak.
                await self._terminate(stale)
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.root,
                env=scrubbed_subprocess_env(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
            self._reader_task = asyncio.create_task(self._read_loop(), name="noah-lsp-reader")
            self._stderr_task = asyncio.create_task(self._drain_stderr(), name="noah-lsp-stderr")
            initialize = {
                "processId": os.getpid(),
                "clientInfo": {"name": "noah-code", "version": "0.2"},
                "rootUri": _path_uri(self.root),
                "workspaceFolders": [{"uri": _path_uri(self.root), "name": self.root.name}],
                "capabilities": {
                    "workspace": {
                        "workspaceFolders": True,
                        "symbol": {"dynamicRegistration": False},
                    },
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "implementation": {"linkSupport": True},
                        "references": {},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "rename": {"prepareSupport": True},
                        "diagnostic": {},
                        "synchronization": {"didSave": True},
                    },
                },
            }
            try:
                await self._request_started(
                    "initialize", initialize, timeout=max(self.timeout, 10.0)
                )
                await self._notify_started("initialized", {})
            except Exception:
                process, self.process = self.process, None
                if process is not None:
                    await self._terminate(process)
                await self._cancel_pump_tasks()
                raise
            for doc_path, doc_lang in reopen:
                if doc_path.is_file():
                    with contextlib.suppress(Exception):
                        await self.open_document(doc_path, doc_lang)

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        await self.start()
        try:
            return await self._request_started(method, params, timeout=timeout)
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            if self._healthy():
                raise
            await self.start()
            return await self._request_started(method, params, timeout=timeout)

    async def _request_started(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout or self.timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self.start()
        await self._notify_started(method, params)

    async def _notify_started(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def open_document(self, path: Path, language: str) -> str:
        uri = _path_uri(path)
        previous = self._open_documents.get(uri)
        if not path.is_file():
            if previous is not None:
                self._open_documents.pop(uri, None)
                self._diagnostics.pop(uri, None)
                self._diagnostic_events.pop(uri, None)
                with contextlib.suppress(Exception):
                    await self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
            raise FileNotFoundError(f"document does not exist: {path}")
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if previous and previous[0] == stamp:
            return uri
        text = path.read_text(errors="replace")
        version = (previous[1] + 1) if previous else 1
        # Clear before sending so synchronous/fast diagnostics are never missed.
        self._diagnostic_events.setdefault(uri, asyncio.Event()).clear()
        if previous:
            await self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            await self.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": language,
                        "version": version,
                        "text": text,
                    }
                },
            )
        self._open_documents[uri] = (stamp, version, path, language)
        return uri

    async def sync_document(self, path: Path) -> None:
        uri = _path_uri(path)
        previous = self._open_documents.get(uri)
        if previous is None:
            return
        if not path.is_file():
            self._open_documents.pop(uri, None)
            self._diagnostics.pop(uri, None)
            self._diagnostic_events.pop(uri, None)
            with contextlib.suppress(Exception):
                await self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
            return
        language = previous[3] if len(previous) >= 4 else "plaintext"
        with contextlib.suppress(Exception):
            await self.open_document(path, language)

    async def wait_diagnostics(self, uri: str, timeout: float = 0.35) -> list[dict[str, Any]]:
        event = self._diagnostic_events.setdefault(uri, asyncio.Event())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=timeout)
        return self._diagnostics.get(uri, [])

    async def _send(self, message: dict[str, Any]) -> None:
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.returncode is not None
        ):
            raise RuntimeError(f"LSP server exited: {' '.join(self.command)}")
        body = json.dumps(message, separators=(",", ":")).encode()
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        async with self._write_lock:
            self.process.stdin.write(frame)
            await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        stream = self.process.stdout
        try:
            while True:
                headers: dict[str, str] = {}
                while True:
                    line = await stream.readline()
                    if not line:
                        raise EOFError
                    if line in {b"\r\n", b"\n"}:
                        break
                    key, _, value = line.decode(errors="replace").partition(":")
                    headers[key.lower().strip()] = value.strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                message = json.loads(await stream.readexactly(length))
                self._handle_message(message)
        except (EOFError, asyncio.CancelledError):
            pass
        except Exception as exc:  # noqa: BLE001
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("LSP server closed"))

    def _schedule_send(self, message: dict[str, Any]) -> None:
        if not self._healthy():
            return
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            if not hasattr(self, "_background_tasks"):
                self._background_tasks: set[asyncio.Task[Any]] = set()
            task = loop.create_task(self._send_response_safe(message))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _send_response_safe(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await self._send(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str) and "id" in message and message["id"] is not None:
            params = message.get("params") or {}
            msg_id = message["id"]
            if method == "workspace/configuration":
                items = params.get("items") or []
                reply: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": [None] * len(items),
                }
            elif method in {
                "client/registerCapability",
                "client/unregisterCapability",
                "window/workDoneProgress/create",
                "workspace/semanticTokens/refresh",
                "workspace/diagnostic/refresh",
                "workspace/codeLens/refresh",
                "workspace/inlayHint/refresh",
            }:
                reply = {"jsonrpc": "2.0", "id": msg_id, "result": None}
            else:
                reply = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            self._schedule_send(reply)
            return
        try:
            request_id = int(message["id"])
        except (KeyError, TypeError, ValueError):
            request_id = None
        future = self._pending.pop(request_id, None) if request_id is not None else None
        if future is not None:
            # wait_for cancels the future on timeout before the caller can pop
            # it; a response landing in that window must not crash the loop.
            if future.done():
                return
            if "error" in message:
                future.set_exception(RuntimeError(str(message["error"])))
            else:
                future.set_result(message.get("result"))
            return
        if method == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            uri = str(params.get("uri", ""))
            self._diagnostics[uri] = list(params.get("diagnostics") or [])
            self._diagnostic_events.setdefault(uri, asyncio.Event()).set()

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            while await self.process.stderr.read(4096):
                pass
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - draining is best-effort
            pass

    async def close(self) -> None:
        if self._closed:
            return
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(Exception):
                # Use the already-started transport directly: setting
                # ``_closed`` first makes the public request path reject its
                # own graceful-shutdown request.
                await self._request_started("shutdown", {}, timeout=1.0)
            with contextlib.suppress(Exception):
                await self._notify_started("exit", {})
        self._closed = True
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(OSError):
                if os.name != "nt":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
        if self.process is not None:
            # Escalate to a process-group SIGKILL and reap; the server may have
            # spawned children of its own that would otherwise linger.
            await self._terminate(self.process)
        await self._cancel_pump_tasks()


_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "switch",
        "catch",
        "synchronized",
        "return",
        "sizeof",
        "alignof",
        "decltype",
        "throw",
        "new",
        "delete",
        "case",
        "do",
        "try",
    }
)

_RUST_DECL_RE = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:(?:default|const|async|unsafe|extern(?:\s+\"[^\"]*\")?)\s+)*"
    r"(?P<kind>fn|struct|enum|trait|type|mod)\s+"
    r"(?P<name>[A-Za-z_]\w*)",
    re.MULTILINE,
)

_GO_METHOD_RE = re.compile(
    r"^\s*func\s*\(\s*(?:[A-Za-z_]\w*\s+)?\*?\s*(?P<recv>[A-Za-z_]\w*)(?:\[[^\]]*\])?\s*\)\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(",
    re.MULTILINE,
)
_GO_FUNC_RE = re.compile(
    r"^\s*func\s+(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(",
    re.MULTILINE,
)
_GO_TYPE_RE = re.compile(
    r"^\s*type\s+(?P<name>[A-Za-z_]\w*)(?:\s*\[[^\]]*\])?\s+(?:=\s*)?(?P<tkind>struct|interface)\b"
    r"|^\s*type\s+(?P<alias>[A-Za-z_]\w*)(?:\s*\[[^\]]*\])?\s+",
    re.MULTILINE,
)

_TS_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<default>default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*"
    r"(?P<name>[A-Za-z_$][\w$]*)?\s*(?:<[^>\n]*>)?\s*\("
)
_TS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<default>default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s*"
    r"(?P<name>[A-Za-z_$][\w$]*)?"
)
_TS_TYPE_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:const\s+)?"
    r"(?P<kind>interface|type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)"
)
_TS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)"
    r"(?:\s*:[^=\n]+)?\s*=\s*(?:async\s+)?"
    r"(?:function\b|(?:<[^>\n]+>\s*)?(?:\([^)\n]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=\n>]+)?\s*=>)"
)
_TS_DEFAULT_ARROW_RE = re.compile(
    r"^\s*export\s+default\s+(?:async\s+)?(?:\([^)\n]*\)|[A-Za-z_$][\w$]*)\s*=>"
)
_TS_EXPORT_DEFAULT_IDENT_RE = re.compile(
    r"^\s*export\s+default\s+(?P<name>[A-Za-z_$][\w$]*)\s*;?\s*$"
)
_TS_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|readonly|static|abstract|override|async|get|set)\s+)*"
    r"\*?\s*(?P<name>[A-Za-z_$#][\w$]*)\s*(?:<[^>\n]*>)?\s*\([^;\n]*\)(?:\s*:[^{\n]+)?\s*\{"
)
_TS_ARROW_PROP_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|readonly|static|override)\s+)*"
    r"(?P<name>[A-Za-z_$#][\w$]*)(?:\s*:[^=\n]+)?\s*=\s*(?:async\s+)?"
    r"(?:<[^>\n]+>\s*)?(?:\([^)\n]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=\n>]+)?\s*=>"
)

_RUBY_CONTAINER_RE = re.compile(r"^\s*(?P<kind>class|module)\s+(?P<name>[A-Za-z_][\w:]*)")
_RUBY_DEF_RE = re.compile(r"^\s*def\s+(?:self\.)?(?P<name>[A-Za-z_]\w*[?!=]?)")

_C_TYPE_CONTAINER_RE = re.compile(
    r"^\s*(?:template\s*<[^>]*>\s*)?"
    r"(?:(?:public|private|protected|internal|static|final|abstract|sealed|non-sealed)\s+)*"
    r"(?P<kind>class|struct|interface|enum|record|namespace)\s+"
    r"(?P<name>[A-Za-z_]\w*)"
)
_C_TYPED_FUNC_RE = re.compile(
    r"^\s*(?:@[A-Za-z_][\w.]*(?:\([^)]*\))?\s+)*"
    r"(?:template\s*<[^>]*>\s*)?"
    r"(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|"
    r"strictfp|default|inline|virtual|explicit|constexpr|consteval|constinit|extern|friend)\s+)*"
    r"(?:(?:const|volatile|unsigned|signed|short|long|struct|enum|typename)\s+)*"
    r"(?P<ret>[A-Za-z_][\w:.]*\s*(?:<[^>{;]*>)?(?:\s*[\*&\[\]]+)*)\s+"
    r"(?:(?P<qual>[A-Za-z_]\w*)::)?(?P<name>~?[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^{;]*)\)"
    r"(?:\s*(?:const|volatile|noexcept(?:\([^)]*\))?|override|final|"
    r"throws\s+[A-Za-z0-9_.,\s]+))*\s*\{"
)
_C_CTOR_RE = re.compile(
    r"^\s*(?:@[A-Za-z_][\w.]*(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|private|protected|explicit|constexpr|inline)\s+)*"
    r"(?P<name>~?[A-Za-z_]\w*)\s*\((?P<params>[^{;]*)\)"
    r"(?:\s*(?::[^{;]+|noexcept(?:\([^)]*\))?|throws\s+[A-Za-z0-9_.,\s]+))*\s*\{"
)


def _strip_line_strings_and_comments(line: str) -> str:
    """Strip single-line comments and string literals for lightweight brace counting."""
    out: list[str] = []
    in_quote: str | None = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            in_quote = ch
            i += 1
            continue
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _format_python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    num_defaults = len(args.defaults)
    default_start = len(positional) - num_defaults

    def fmt_arg(arg: ast.arg, has_default: bool = False, prefix: str = "") -> str:
        item = f"{prefix}{arg.arg}"
        if arg.annotation is not None:
            with contextlib.suppress(Exception):
                ann = ast.unparse(arg.annotation)
                if len(ann) <= 24:
                    item += f": {ann}"
        if has_default:
            item += "=..."
        return item

    for idx, arg in enumerate(positional):
        parts.append(fmt_arg(arg, has_default=idx >= default_start))
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append(fmt_arg(args.vararg, prefix="*"))
    elif args.kwonlyargs:
        parts.append("*")
    for kw_arg, kw_def in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        parts.append(fmt_arg(kw_arg, has_default=kw_def is not None))
    if args.kwarg is not None:
        parts.append(fmt_arg(args.kwarg, prefix="**"))

    ret_suffix = ""
    if node.returns is not None:
        with contextlib.suppress(Exception):
            ret_str = ast.unparse(node.returns)
            if len(ret_str) <= 28:
                ret_suffix = f" -> {ret_str}"
    full = f"({', '.join(parts)}){ret_suffix}"
    if len(full) <= 88:
        return full
    bare_names = [a.arg for a in positional]
    if args.vararg is not None:
        bare_names.append(f"*{args.vararg.arg}")
    bare_names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg is not None:
        bare_names.append(f"**{args.kwarg.arg}")
    compact = f"({', '.join(bare_names)}){ret_suffix}"
    return compact if len(compact) <= 88 else compact[:85] + "...)"


def _walk_python_stmts(
    stmts: list[ast.stmt],
    rel: str,
    class_stack: tuple[str, ...],
    found: list[RepositorySymbol],
) -> None:
    for stmt in stmts:
        if isinstance(stmt, ast.ClassDef):
            qualname = ".".join((*class_stack, stmt.name))
            detail = ""
            if stmt.bases:
                with contextlib.suppress(Exception):
                    bases = ", ".join(ast.unparse(b) for b in stmt.bases)
                    if bases and len(bases) <= 60:
                        detail = f"({bases})"
            found.append(RepositorySymbol(rel, stmt.lineno, "class", qualname, detail))
            _walk_python_stmts(stmt.body, rel, (*class_stack, stmt.name), found)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join((*class_stack, stmt.name))
            kind = "method" if class_stack else "function"
            found.append(
                RepositorySymbol(rel, stmt.lineno, kind, qualname, _format_python_signature(stmt))
            )
        elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            _walk_python_stmts(stmt.body, rel, class_stack, found)
            _walk_python_stmts(stmt.orelse, rel, class_stack, found)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _walk_python_stmts(stmt.body, rel, class_stack, found)
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            _walk_python_stmts(stmt.body, rel, class_stack, found)
            for handler in stmt.handlers:
                _walk_python_stmts(handler.body, rel, class_stack, found)
            _walk_python_stmts(stmt.orelse, rel, class_stack, found)
            _walk_python_stmts(stmt.finalbody, rel, class_stack, found)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                _walk_python_stmts(case.body, rel, class_stack, found)


class RepositoryMap:
    """Lazy, mtime-keyed declaration map with cross-file reference ranking."""

    def __init__(self, root: Path, *, max_files: int = 2500, max_file_bytes: int = 512_000) -> None:
        self.root = root
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self._cache: dict[Path, tuple[int, int, list[RepositorySymbol]]] = {}
        self._token_cache: dict[Path, tuple[int, int, frozenset[str]]] = {}

    def _load_file(self, path: Path) -> tuple[list[RepositorySymbol], frozenset[str]]:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached_syms = self._cache.get(path)
        cached_toks = self._token_cache.get(path)
        if cached_syms and cached_syms[:2] == key and cached_toks and cached_toks[:2] == key:
            return cached_syms[2], cached_toks[2]
        if stat.st_size > self.max_file_bytes or path.suffix.lower() not in _LANGUAGES:
            symbols: list[RepositorySymbol] = []
            tokens: frozenset[str] = frozenset()
        else:
            text = path.read_text(errors="replace")
            tokens = frozenset(_IDENT_RE.findall(text))
            symbols = (
                cached_syms[2]
                if cached_syms and cached_syms[:2] == key
                else self._extract(path, text)
            )
        self._cache[path] = (*key, symbols)
        self._token_cache[path] = (*key, tokens)
        return symbols, tokens

    def symbols_for(self, path: Path) -> list[RepositorySymbol]:
        symbols, _tokens = self._load_file(path)
        return symbols

    @staticmethod
    def _module_stem(rel_path: str) -> str:
        p = Path(rel_path)
        stem = p.stem
        if stem in {"__init__", "index", "mod", "lib", "main"} and len(p.parts) >= 2:
            return p.parts[-2]
        return stem

    def build(self, query: str = "", *, limit: int = 300) -> list[RepositorySymbol]:
        if limit <= 0:
            return []
        lowered = query.strip().casefold()
        file_symbols: dict[str, list[RepositorySymbol]] = {}
        file_tokens: dict[str, frozenset[str]] = {}
        file_stems: dict[str, str] = {}

        for path in self._files():
            if is_secret_path(path) or path.suffix.lower() not in _LANGUAGES:
                continue
            with contextlib.suppress(OSError, UnicodeError, SyntaxError):
                rel = str(path.relative_to(self.root))
                syms, idents = self._load_file(path)
                file_tokens[rel] = idents
                file_stems[rel] = self._module_stem(rel)
                matched = [
                    symbol
                    for symbol in syms
                    if (
                        not lowered
                        or lowered in f"{symbol.name} {symbol.path} {symbol.kind}".casefold()
                    )
                ]
                if matched:
                    file_symbols[rel] = matched

        if not file_symbols:
            return []

        doc_freq: Counter[str] = Counter()
        for idents in file_tokens.values():
            doc_freq.update(idents)

        scored_candidates: list[tuple[float, float, RepositorySymbol]] = []
        for rel, syms in file_symbols.items():
            idents = file_tokens.get(rel, frozenset())
            stem = file_stems.get(rel, "")
            stem_refs = (
                float(max(0, doc_freq[stem] - (1 if stem in idents else 0)))
                if len(stem) >= 3
                else 0.0
            )
            per_file: list[tuple[float, RepositorySymbol]] = []
            for sym in syms:
                short_name = sym.name.rsplit(".", 1)[-1]
                owner_name = sym.name.split(".", 1)[0] if "." in sym.name else ""
                if short_name.startswith("__") and short_name.endswith("__") and owner_name:
                    sym_refs = (
                        max(0, doc_freq[owner_name] - (1 if owner_name in idents else 0)) * 0.5
                    )
                else:
                    sym_refs = float(
                        max(0, doc_freq[short_name] - (1 if short_name in idents else 0))
                    )
                    if owner_name:
                        sym_refs += 0.25 * max(
                            0, doc_freq[owner_name] - (1 if owner_name in idents else 0)
                        )
                kind_bonus = (
                    2.0
                    if sym.kind
                    in {"class", "interface", "struct", "trait", "enum", "type", "module"}
                    else (1.0 if sym.kind == "function" else 0.5)
                )
                query_bonus = 0.0
                if lowered:
                    name_cf = sym.name.casefold()
                    short_cf = short_name.casefold()
                    path_cf = sym.path.casefold()
                    if short_cf == lowered or name_cf == lowered:
                        query_bonus = 1000.0
                    elif short_cf.startswith(lowered) or name_cf.startswith(lowered):
                        query_bonus = 500.0
                    elif lowered in short_cf or lowered in name_cf:
                        query_bonus = 250.0
                    elif stem.casefold() == lowered:
                        query_bonus = 150.0
                    elif lowered in path_cf:
                        query_bonus = 50.0
                    else:
                        query_bonus = 10.0
                raw_score = query_bonus + sym_refs * 10.0 + stem_refs * 3.0 + kind_bonus
                per_file.append((raw_score, sym))

            per_file.sort(key=lambda item: (-item[0], item[1].line, item[1].name))
            for rank_in_file, (raw_score, sym) in enumerate(per_file):
                effective_score = raw_score - 0.3 * rank_in_file
                scored_candidates.append((effective_score, raw_score, sym))

        scored_candidates.sort(
            key=lambda item: (-item[0], -item[1], item[2].path, item[2].line, item[2].name)
        )
        selected = scored_candidates[:limit]

        grouped: dict[str, list[RepositorySymbol]] = {}
        file_top_score: dict[str, float] = {}
        for _effective, raw_score, sym in selected:
            grouped.setdefault(sym.path, []).append(sym)
            if raw_score > file_top_score.get(sym.path, float("-inf")):
                file_top_score[sym.path] = raw_score

        ordered_files = sorted(grouped, key=lambda rel: (-file_top_score[rel], rel))
        output: list[RepositorySymbol] = []
        for rel in ordered_files:
            output.extend(sorted(grouped[rel], key=lambda item: (item.line, item.name)))
        return output

    def _files(self) -> list[Path]:
        try:
            result = subprocess.run(
                ["git", "ls-files", "-co", "--exclude-standard", "-z"],
                cwd=self.root,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                names = [
                    name for name in result.stdout.decode(errors="replace").split("\0") if name
                ]
                files: list[Path] = []
                for name in names:
                    if is_secret_path(name):
                        continue
                    candidate = self.root / name
                    with contextlib.suppress(OSError):
                        resolved = candidate.resolve()
                        rel = resolved.relative_to(self.root)
                        if (
                            not candidate.is_symlink()
                            and not is_secret_path(rel)
                            and resolved.is_file()
                        ):
                            files.append(resolved)
                            if len(files) >= self.max_files:
                                break
                return files
        except (OSError, subprocess.SubprocessError):
            pass
        ignored = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build"}
        files = []
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if ignored.intersection(rel.parts) or is_secret_path(rel):
                continue
            files.append(path)
            if len(files) >= self.max_files:
                break
        return files

    def _extract(self, path: Path, text: str) -> list[RepositorySymbol]:
        rel = str(path.relative_to(self.root))
        suffix = path.suffix.lower()
        if suffix in {".py", ".pyi"}:
            tree = ast.parse(text)
            found: list[RepositorySymbol] = []
            _walk_python_stmts(tree.body, rel, (), found)
            return sorted(found, key=lambda item: (item.line, item.name))
        if suffix == ".rs":
            return self._extract_rust(rel, text)
        if suffix == ".go":
            return self._extract_go(rel, text)
        if suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
            return self._extract_ts_js(rel, text)
        if suffix == ".rb":
            return self._extract_ruby(rel, text)
        if suffix in {".java", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}:
            return self._extract_c_family(rel, text)
        return []

    @staticmethod
    def _extract_rust(rel: str, text: str) -> list[RepositorySymbol]:
        kind_map = {"fn": "function", "mod": "module"}
        found = [
            RepositorySymbol(
                rel,
                text.count("\n", 0, match.start()) + 1,
                kind_map.get(match.group("kind"), match.group("kind")),
                match.group("name"),
            )
            for match in _RUST_DECL_RE.finditer(text)
        ]
        return sorted(found, key=lambda item: (item.line, item.name))

    @staticmethod
    def _extract_go(rel: str, text: str) -> list[RepositorySymbol]:
        found: list[RepositorySymbol] = []
        for match in _GO_METHOD_RE.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            found.append(
                RepositorySymbol(
                    rel, lineno, "method", f"{match.group('recv')}.{match.group('name')}"
                )
            )
        for match in _GO_FUNC_RE.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            found.append(RepositorySymbol(rel, lineno, "function", match.group("name")))
        for match in _GO_TYPE_RE.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            if match.group("name"):
                found.append(
                    RepositorySymbol(rel, lineno, match.group("tkind"), match.group("name"))
                )
            elif match.group("alias"):
                found.append(RepositorySymbol(rel, lineno, "type", match.group("alias")))
        return sorted(found, key=lambda item: (item.line, item.name))

    @staticmethod
    def _extract_ts_js(rel: str, text: str) -> list[RepositorySymbol]:
        found: list[RepositorySymbol] = []
        brace_depth = 0
        class_stack: list[tuple[str, int]] = []
        for lineno, raw_line in enumerate(text.splitlines(), 1):
            cleaned = _strip_line_strings_and_comments(raw_line)
            while class_stack and brace_depth < class_stack[-1][1]:
                class_stack.pop()

            matched_line = False
            if m := _TS_CLASS_RE.match(cleaned):
                name = m.group("name") or ("default" if m.group("default") else "")
                if name:
                    found.append(RepositorySymbol(rel, lineno, "class", name))
                    opens = cleaned.count("{") - cleaned.count("}")
                    if "{" in cleaned:
                        class_stack.append((name, brace_depth + max(opens, 1)))
                    matched_line = True
            elif m := _TS_FUNC_RE.match(cleaned):
                name = m.group("name") or ("default" if m.group("default") else "")
                if name:
                    found.append(RepositorySymbol(rel, lineno, "function", name))
                    matched_line = True
            elif m := _TS_TYPE_DECL_RE.match(cleaned):
                found.append(RepositorySymbol(rel, lineno, m.group("kind"), m.group("name")))
                matched_line = True
            elif m := _TS_ARROW_RE.match(cleaned):
                if not class_stack:
                    found.append(RepositorySymbol(rel, lineno, "function", m.group("name")))
                    matched_line = True
            elif _TS_DEFAULT_ARROW_RE.match(cleaned):
                found.append(RepositorySymbol(rel, lineno, "function", "default"))
                matched_line = True
            elif m := _TS_EXPORT_DEFAULT_IDENT_RE.match(cleaned):
                found.append(RepositorySymbol(rel, lineno, "variable", m.group("name")))
                matched_line = True

            if not matched_line and class_stack and brace_depth == class_stack[-1][1]:
                cls_name = class_stack[-1][0]
                if (m := _TS_METHOD_RE.match(cleaned)) or (m := _TS_ARROW_PROP_RE.match(cleaned)):
                    mname = m.group("name")
                    if mname not in _CONTROL_KEYWORDS:
                        found.append(RepositorySymbol(rel, lineno, "method", f"{cls_name}.{mname}"))

            brace_depth += cleaned.count("{") - cleaned.count("}")
            if brace_depth < 0:
                brace_depth = 0
            while class_stack and brace_depth < class_stack[-1][1]:
                class_stack.pop()
        return sorted(found, key=lambda item: (item.line, item.name))

    @staticmethod
    def _extract_ruby(rel: str, text: str) -> list[RepositorySymbol]:
        found: list[RepositorySymbol] = []
        containers: list[str] = []
        depth_stack: list[str] = []
        block_openers = re.compile(
            r"^\s*(?:class|module|def|if|unless|case|while|until|for|begin)\b|\bdo(?:\s*\|[^|]*\|)?\s*$"
        )
        for lineno, raw_line in enumerate(text.splitlines(), 1):
            stripped = raw_line.split("#", 1)[0].strip()
            if not stripped:
                continue
            if m := _RUBY_CONTAINER_RE.match(stripped):
                kind = m.group("kind")
                name = m.group("name").rsplit("::", 1)[-1]
                qual = ".".join((*containers, name)) if containers else name
                found.append(RepositorySymbol(rel, lineno, kind, qual))
                containers.append(name)
                depth_stack.append("container")
                continue
            if m := _RUBY_DEF_RE.match(stripped):
                mname = m.group("name")
                if containers:
                    found.append(
                        RepositorySymbol(rel, lineno, "method", f"{'.'.join(containers)}.{mname}")
                    )
                else:
                    found.append(RepositorySymbol(rel, lineno, "function", mname))
                if not stripped.endswith("end"):
                    depth_stack.append("other")
                continue
            if stripped == "end" or stripped.startswith("end "):
                if depth_stack:
                    popped = depth_stack.pop()
                    if popped == "container" and containers:
                        containers.pop()
                continue
            if block_openers.search(stripped) and not stripped.endswith("end"):
                depth_stack.append("other")
        return sorted(found, key=lambda item: (item.line, item.name))

    @staticmethod
    def _extract_c_family(rel: str, text: str) -> list[RepositorySymbol]:
        found: list[RepositorySymbol] = []
        brace_depth = 0
        type_stack: list[tuple[str, str, int]] = []
        lines = text.splitlines()
        total = len(lines)
        for idx, raw_line in enumerate(lines):
            lineno = idx + 1
            cleaned = _strip_line_strings_and_comments(raw_line)
            while type_stack and brace_depth < type_stack[-1][2]:
                type_stack.pop()

            stripped = cleaned.strip()
            if stripped and not stripped.startswith("#"):
                probe = cleaned
                if "(" in cleaned and "{" not in cleaned and ";" not in cleaned and idx + 1 < total:
                    probe = cleaned + " " + _strip_line_strings_and_comments(lines[idx + 1])

                if m := _C_TYPE_CONTAINER_RE.match(cleaned):
                    raw_kind = m.group("kind")
                    kind = (
                        "class"
                        if raw_kind == "record"
                        else ("module" if raw_kind == "namespace" else raw_kind)
                    )
                    name = m.group("name")
                    if "{" in probe or not stripped.endswith(";"):
                        found.append(RepositorySymbol(rel, lineno, kind, name))
                        if "{" in probe:
                            opens = cleaned.count("{") - cleaned.count("}")
                            if opens > 0:
                                type_stack.append((name, kind, brace_depth + opens))
                elif m := _C_TYPED_FUNC_RE.match(probe):
                    ret = m.group("ret").strip()
                    name = m.group("name")
                    qual = m.group("qual")
                    if name not in _CONTROL_KEYWORDS and ret not in _CONTROL_KEYWORDS:
                        if qual:
                            found.append(RepositorySymbol(rel, lineno, "method", f"{qual}.{name}"))
                        elif type_stack and type_stack[-1][1] in {
                            "class",
                            "struct",
                            "interface",
                            "enum",
                        }:
                            found.append(
                                RepositorySymbol(
                                    rel, lineno, "method", f"{type_stack[-1][0]}.{name}"
                                )
                            )
                        else:
                            found.append(RepositorySymbol(rel, lineno, "function", name))
                elif (
                    type_stack
                    and type_stack[-1][1] in {"class", "struct"}
                    and (m := _C_CTOR_RE.match(probe))
                ):
                    cname = m.group("name")
                    owner = type_stack[-1][0]
                    if cname in {owner, f"~{owner}"}:
                        found.append(RepositorySymbol(rel, lineno, "method", f"{owner}.{cname}"))

            brace_depth += cleaned.count("{") - cleaned.count("}")
            if brace_depth < 0:
                brace_depth = 0
            while type_stack and brace_depth < type_stack[-1][2]:
                type_stack.pop()
        return sorted(found, key=lambda item: (item.line, item.name))


class LSPTools(Skill):
    """Navigate code semantically; servers launch only for the requested language."""

    def __init__(
        self,
        workspace: Workspace,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        timeout: float = 5.0,
        enabled: bool = True,
        server_overrides: dict[str, list[str]] | None = None,
        max_symbols: int = 300,
        max_file_bytes: int = 512_000,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._engine = engine
        self._approvals = approvals
        self._timeout = timeout
        self._enabled = enabled
        self._overrides = server_overrides or {}
        self._max_symbols = max_symbols
        self._clients: dict[tuple[str, ...], _LSPClient] = {}
        self._request_cache: dict[tuple[str, str, tuple[int, int], str], Any] = {}
        self._map = RepositoryMap(workspace.root, max_file_bytes=max_file_bytes)

    async def definition(
        self,
        path: Annotated[str, spec(description="Workspace file")],
        line: Annotated[int, spec(description="1-indexed line")],
        column: Annotated[int, spec(description="1-indexed column")],
    ) -> str:
        """Find the definition at a source position."""
        return await self._locations("textDocument/definition", path, line, column)

    async def implementation(
        self,
        path: Annotated[str, spec(description="Workspace file")],
        line: Annotated[int, spec(description="1-indexed line")],
        column: Annotated[int, spec(description="1-indexed column")],
    ) -> str:
        """Find implementations at a source position."""
        return await self._locations("textDocument/implementation", path, line, column)

    async def references(
        self,
        path: Annotated[str, spec(description="Workspace file")],
        line: Annotated[int, spec(description="1-indexed line")],
        column: Annotated[int, spec(description="1-indexed column")],
        include_declaration: Annotated[bool, spec(description="Include the declaration")] = True,
    ) -> str:
        """Find references at a source position."""
        resolved, language, client, uri = await self._document(path)
        result = await self._cached_request(
            client,
            "textDocument/references",
            resolved,
            {
                "textDocument": {"uri": uri},
                "position": _position_for_file(resolved, line, column),
                "context": {"includeDeclaration": include_declaration},
            },
        )
        del language
        return self._render_locations(result)

    async def hover(
        self,
        path: Annotated[str, spec(description="Workspace file")],
        line: Annotated[int, spec(description="1-indexed line")],
        column: Annotated[int, spec(description="1-indexed column")],
    ) -> str:
        """Return type and documentation at a source position."""
        resolved, _language, client, uri = await self._document(path)
        result = await self._cached_request(
            client,
            "textDocument/hover",
            resolved,
            {"textDocument": {"uri": uri}, "position": _position_for_file(resolved, line, column)},
        )
        if not result:
            return "(no hover information)"
        contents = result.get("contents", result) if isinstance(result, dict) else result
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return str(contents.get("value") or contents.get("language") or contents)
        return "\n".join(
            str(item.get("value", item)) if isinstance(item, dict) else str(item)
            for item in contents
        )

    async def document_symbols(
        self, path: Annotated[str, spec(description="Workspace file")]
    ) -> str:
        """List declarations in one file, using LSP with a parser fallback."""
        resolved = await self._authorize_path(path)
        try:
            _resolved, _language, client, uri = await self._document(path, resolved=resolved)
            result = await self._cached_request(
                client,
                "textDocument/documentSymbol",
                resolved,
                {"textDocument": {"uri": uri}},
            )
            rows = self._flatten_symbols(result or [], resolved)
            if rows:
                return "\n".join(rows[: self._max_symbols])
        except RuntimeError:
            pass
        fallback = self._map.symbols_for(resolved)
        return "\n".join(item.render() for item in fallback[: self._max_symbols]) or "(no symbols)"

    async def workspace_symbols(
        self,
        query: Annotated[str, spec(description="Symbol name or fuzzy query")] = "",
    ) -> str:
        """Search workspace declarations without building an embedding index."""
        await self._authorize_target(query or "*")
        lsp_rows: list[str] = []
        tried: set[str] = set()
        for path in await asyncio.to_thread(self._map._files):  # noqa: SLF001 - bounded git listing
            language = _LANGUAGES.get(path.suffix.lower())
            if language is None or language in tried or self._command(language) is None:
                continue
            tried.add(language)
            try:
                client = await self._client(language)
                result = await client.request("workspace/symbol", {"query": query})
                for item in list(result or []):
                    location = item.get("location", item)
                    loc_uri = str(location.get("targetUri") or location.get("uri") or "")
                    if loc_uri and _is_secret_uri(loc_uri, self._workspace.root):
                        continue
                    prefix = f"{_SYMBOL_KINDS.get(int(item.get('kind', 0)), 'symbol')} "
                    lsp_rows.append(
                        f"{_location_text(location, self._workspace.root)}  {prefix}{item.get('name', '?')}"
                    )
                    if len(lsp_rows) >= self._max_symbols:
                        return "\n".join(lsp_rows)
            except Exception:  # noqa: BLE001 - fall back to the local map per language
                pass
            if len(tried) >= 4:
                break
        if lsp_rows:
            return "\n".join(lsp_rows)
        rows = await asyncio.to_thread(self._map.build, query, limit=self._max_symbols)
        return "\n".join(item.render() for item in rows) or "(no symbols)"

    async def repository_map(
        self,
        query: Annotated[str, spec(description="Optional symbol/path filter")] = "",
    ) -> str:
        """Return a compact, lazy repository declaration map."""
        return await self.workspace_symbols(query)

    async def diagnostics(self, path: Annotated[str, spec(description="Workspace file")]) -> str:
        """Return current LSP diagnostics for a file."""
        resolved, _language, client, uri = await self._document(path)
        diagnostics: list[dict[str, Any]] = []
        try:
            result = await client.request(
                "textDocument/diagnostic", {"textDocument": {"uri": uri}}, timeout=1.0
            )
            if isinstance(result, dict):
                diagnostics = list(result.get("items") or [])
        except (RuntimeError, TimeoutError):
            diagnostics = await client.wait_diagnostics(uri)
        if not diagnostics:
            return "ok — no diagnostics"
        rel = self._workspace.relpath(resolved)
        rows = []
        for item in diagnostics[:100]:
            start = item.get("range", {}).get("start", {})
            severity = _SEVERITY.get(int(item.get("severity", 3)), "info")
            message = " ".join(str(item.get("message", "diagnostic")).split())
            rows.append(
                f"{rel}:{int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1} "
                f"[{severity}] {message}"
            )
        return "\n".join(rows)

    async def rename_preview(
        self,
        path: Annotated[str, spec(description="Workspace file")],
        line: Annotated[int, spec(description="1-indexed line")],
        column: Annotated[int, spec(description="1-indexed column")],
        new_name: Annotated[str, spec(description="Proposed new symbol name")],
    ) -> str:
        """Preview, but never apply, a workspace symbol rename."""
        resolved, _language, client, uri = await self._document(path)
        position = _position_for_file(resolved, line, column)
        with contextlib.suppress(RuntimeError):
            await client.request(
                "textDocument/prepareRename",
                {"textDocument": {"uri": uri}, "position": position},
            )
        result = await self._cached_request(
            client,
            "textDocument/rename",
            resolved,
            {"textDocument": {"uri": uri}, "position": position, "newName": new_name},
            extra=new_name,
        )
        if not isinstance(result, dict):
            return "(rename unavailable)"
        changes = result.get("changes") or {}
        document_changes = result.get("documentChanges") or []
        rows: list[str] = []
        for target_uri, edits in changes.items():
            if _is_secret_uri(target_uri, self._workspace.root):
                continue
            rows.extend(self._render_text_edits(target_uri, edits))
        for change in document_changes:
            if "edits" in change:
                doc_uri = str(change.get("textDocument", {}).get("uri", ""))
                if doc_uri and _is_secret_uri(doc_uri, self._workspace.root):
                    continue
                rows.extend(
                    self._render_text_edits(doc_uri, change["edits"])
                )
        return "Rename preview (not applied):\n" + ("\n".join(rows) or "(no edits)")

    async def changed_symbols(self) -> str:
        """Summarize declarations in Git-changed files."""
        await self._authorize_target("changed-files")
        paths = await asyncio.to_thread(self._changed_paths)
        if not paths:
            return "(no changed source files)"
        sections: list[str] = []
        for path in paths[:50]:
            if path.suffix.lower() not in _LANGUAGES or not path.is_file():
                continue
            rel = self._workspace.relpath(path)
            if is_secret_path(rel):
                continue
            try:
                symbols = await self.document_symbols(rel)
            except Exception as exc:  # noqa: BLE001
                symbols = f"unavailable — {exc}"
            sections.append(f"## {rel}\n{symbols}")
        return "\n\n".join(sections) or "(no changed source symbols)"

    @hidden
    async def sync_paths(self, paths: list[str | Path]) -> None:
        """Notify active LSP clients when tracked workspace files change or vanish."""
        if not self._clients:
            return
        for raw in paths[:50]:
            try:
                resolved = self._workspace.resolve(raw)
            except ValueError:
                continue
            uri = _path_uri(resolved)
            for client in list(self._clients.values()):
                if client._healthy() and uri in client._open_documents:
                    await client.sync_document(resolved)

    @hidden
    async def diagnostics_for_paths(self, paths: list[str]) -> dict[str, str]:
        async def one(path: str) -> tuple[str, str]:
            if Path(path).suffix.lower() not in _LANGUAGES:
                return path, "not supported"
            try:
                return path, await self.diagnostics(path)
            except Exception as exc:  # noqa: BLE001
                return path, f"unavailable — {exc}"

        return dict(await asyncio.gather(*(one(path) for path in paths[:20])))

    async def _locations(self, method: str, path: str, line: int, column: int) -> str:
        resolved, _language, client, uri = await self._document(path)
        result = await self._cached_request(
            client,
            method,
            resolved,
            {"textDocument": {"uri": uri}, "position": _position_for_file(resolved, line, column)},
        )
        return self._render_locations(result)

    async def _document(
        self, path: str, *, resolved: Path | None = None
    ) -> tuple[Path, str, _LSPClient, str]:
        resolved = resolved or await self._authorize_path(path)
        language = _LANGUAGES.get(resolved.suffix.lower())
        if language is None:
            raise RuntimeError(f"no LSP language mapping for {resolved.suffix or resolved.name}")
        client = await self._client(language)
        uri = await client.open_document(resolved, language)
        return resolved, language, client, uri

    async def _client(self, language: str) -> _LSPClient:
        command = self._command(language)
        if command is None:
            names = ", ".join(candidate[0] for candidate in _SERVER_CANDIDATES.get(language, ()))
            raise RuntimeError(
                f"no {language} language server found"
                + (f"; install one of: {names}" if names else "")
            )
        client = self._clients.get(command)
        if client is None:
            client = _LSPClient(self._workspace.root, command, timeout=self._timeout)
            self._clients[command] = client
        await client.start()
        return client

    def _command(self, language: str) -> tuple[str, ...] | None:
        override = self._overrides.get(language)
        if override:
            resolved = shutil.which(override[0]) or override[0]
            return (resolved, *override[1:])
        for candidate in _SERVER_CANDIDATES.get(language, ()):
            if found := shutil.which(candidate[0]):
                return (found, *candidate[1:])
        return None

    async def _cached_request(
        self,
        client: _LSPClient,
        method: str,
        path: Path,
        params: dict[str, Any],
        *,
        extra: str = "",
    ) -> Any:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        key = (" ".join(client.command), method, stamp, json.dumps(params, sort_keys=True) + extra)
        if key not in self._request_cache:
            self._request_cache[key] = await client.request(method, params)
        return self._request_cache[key]

    async def _authorize_path(self, path: str) -> Path:
        resolved = self._workspace.resolve(path)
        rel = self._workspace.relpath(resolved)
        await self._approvals.require(
            self._engine.decide(PermissionCategory.READ, rel, tool="lsp")
        )
        await self._authorize_target(rel)
        return resolved

    async def _authorize_target(self, target: str) -> None:
        if not self._enabled:
            raise RuntimeError("LSP tools are disabled in Noah configuration")
        await self._approvals.require(
            self._engine.decide(PermissionCategory.LSP, target, tool="lsp")
        )

    @staticmethod
    def _position(line: int, column: int) -> dict[str, int]:
        if line < 1 or column < 1:
            raise ValueError("line and column are 1-indexed and must be positive")
        return {"line": line - 1, "character": column - 1}

    def _render_locations(self, result: Any) -> str:
        if not result:
            return "(no locations)"
        items = result if isinstance(result, list) else [result]
        visible: list[str] = []
        for item in items:
            uri = str(item.get("targetUri") or item.get("uri") or "")
            if uri and _is_secret_uri(uri, self._workspace.root):
                continue
            visible.append(_location_text(item, self._workspace.root))
            if len(visible) >= 100:
                break
        return "\n".join(visible) or "(no locations)"

    def _flatten_symbols(self, symbols: list[dict[str, Any]], path: Path) -> list[str]:
        rel = self._workspace.relpath(path)
        rows: list[str] = []

        def visit(items: list[dict[str, Any]], prefix: str = "") -> None:
            for item in items:
                location = (
                    item.get("selectionRange")
                    or item.get("range")
                    or item.get("location", {}).get("range", {})
                )
                start = location.get("start", {})
                name = str(item.get("name", "?"))
                kind = _SYMBOL_KINDS.get(int(item.get("kind", 0)), "symbol")
                detail = str(item.get("detail", "")).strip()
                suffix = f" — {detail}" if detail else ""
                rows.append(f"{rel}:{int(start.get('line', 0)) + 1}  {kind} {prefix}{name}{suffix}")
                visit(list(item.get("children") or []), prefix=f"{prefix}{name}.")

        visit(symbols)
        return rows

    def _render_text_edits(self, uri: str, edits: list[dict[str, Any]]) -> list[str]:
        try:
            path = _uri_path(uri)
            display = (
                self._workspace.relpath(path)
                if path.is_relative_to(self._workspace.root)
                else str(path)
            )
        except (ValueError, TypeError):
            display = uri
        rows = []
        for edit in edits:
            start = edit.get("range", {}).get("start", {})
            replacement = " ".join(str(edit.get("newText", "")).split())
            rows.append(
                f"{display}:{int(start.get('line', 0)) + 1}:{int(start.get('character', 0)) + 1}"
                f" -> {replacement[:120]}"
            )
        return rows

    def _changed_paths(self) -> list[Path]:
        try:
            prefix_res = subprocess.run(
                ["git", "rev-parse", "--show-prefix"],
                cwd=self._workspace.root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=5,
                check=False,
            )
            prefix = prefix_res.stdout.decode(errors="replace").strip() if prefix_res.returncode == 0 else ""
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z", "--", "."],
                cwd=self._workspace.root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return []
            _, entries = parse_porcelain_z(result.stdout.decode(errors="replace"))
            paths: list[Path] = []
            for entry in entries:
                name = entry.path
                if prefix:
                    if not name.startswith(prefix):
                        continue
                    name = name[len(prefix) :]
                if not name:
                    continue
                path = (self._workspace.root / name).resolve()
                if path.is_relative_to(self._workspace.root):
                    paths.append(path)
            return list(dict.fromkeys(paths))
        except (OSError, subprocess.SubprocessError):
            return []

    @hidden
    async def close(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self._clients.values()), return_exceptions=True
        )
        self._clients.clear()
