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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

from nooa import Skill, hidden, spec

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine, is_secret_path
from noah_code.workspace import Workspace

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
    return Path(unquote(parsed.path))


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
        self._write_lock = asyncio.Lock()
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._diagnostic_events: dict[str, asyncio.Event] = {}
        self._open_documents: dict[str, tuple[int, int]] = {}
        self._closed = False

    async def start(self) -> None:
        if self.process is not None:
            return
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        self._reader_task = asyncio.create_task(self._read_loop(), name="noah-lsp-reader")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="noah-lsp-stderr")
        initialize = {
            "processId": os.getpid(),
            "clientInfo": {"name": "noah-code", "version": "0.1"},
            "rootUri": _path_uri(self.root),
            "workspaceFolders": [{"uri": _path_uri(self.root), "name": self.root.name}],
            "capabilities": {
                "workspace": {"workspaceFolders": True, "symbol": {"dynamicRegistration": False}},
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
        await self.request("initialize", initialize, timeout=max(self.timeout, 10.0))
        await self.notify("initialized", {})

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        await self.start()
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
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def open_document(self, path: Path, language: str) -> str:
        uri = _path_uri(path)
        stat = path.stat()
        stamp = stat.st_mtime_ns
        previous = self._open_documents.get(uri)
        if previous and previous[0] == stamp:
            return uri
        text = path.read_text(errors="replace")
        version = (previous[1] + 1) if previous else 1
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
        self._open_documents[uri] = (stamp, version)
        self._diagnostic_events.setdefault(uri, asyncio.Event()).clear()
        return uri

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
                if (
                    "id" in message
                    and (future := self._pending.get(int(message["id"]))) is not None
                ):
                    if "error" in message:
                        future.set_exception(RuntimeError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
                elif message.get("method") == "textDocument/publishDiagnostics":
                    params = message.get("params", {})
                    uri = str(params.get("uri", ""))
                    self._diagnostics[uri] = list(params.get("diagnostics") or [])
                    self._diagnostic_events.setdefault(uri, asyncio.Event()).set()
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

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while await self.process.stderr.read(4096):
            pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(Exception):
                await self.request("shutdown", {}, timeout=1.0)
            with contextlib.suppress(Exception):
                await self.notify("exit", {})
            with contextlib.suppress(ProcessLookupError):
                if os.name != "nt":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            if self.process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


class RepositoryMap:
    """Lazy, mtime-keyed declaration map with no embedding index."""

    def __init__(self, root: Path, *, max_files: int = 2500, max_file_bytes: int = 512_000) -> None:
        self.root = root
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self._cache: dict[Path, tuple[int, int, list[RepositorySymbol]]] = {}

    def symbols_for(self, path: Path) -> list[RepositorySymbol]:
        stat = path.stat()
        cached = self._cache.get(path)
        key = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[:2] == key:
            return cached[2]
        if stat.st_size > self.max_file_bytes or path.suffix.lower() not in _LANGUAGES:
            symbols: list[RepositorySymbol] = []
        else:
            text = path.read_text(errors="replace")
            symbols = self._extract(path, text)
        self._cache[path] = (*key, symbols)
        return symbols

    def build(self, query: str = "", *, limit: int = 300) -> list[RepositorySymbol]:
        lowered = query.casefold()
        symbols: list[RepositorySymbol] = []
        for path in self._files():
            if is_secret_path(path) or path.suffix.lower() not in _LANGUAGES:
                continue
            with contextlib.suppress(OSError, UnicodeError, SyntaxError):
                for symbol in self.symbols_for(path):
                    if (
                        not lowered
                        or lowered in f"{symbol.name} {symbol.path} {symbol.kind}".casefold()
                    ):
                        symbols.append(symbol)
                        if len(symbols) >= limit:
                            return symbols
        return symbols

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
                    candidate = self.root / name
                    with contextlib.suppress(OSError):
                        resolved = candidate.resolve()
                        if (
                            not candidate.is_symlink()
                            and resolved.is_relative_to(self.root)
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
            if ignored.intersection(path.relative_to(self.root).parts):
                continue
            files.append(path)
            if len(files) >= self.max_files:
                break
        return files

    def _extract(self, path: Path, text: str) -> list[RepositorySymbol]:
        rel = str(path.relative_to(self.root))
        if path.suffix.lower() in {".py", ".pyi"}:
            tree = ast.parse(text)
            found: list[RepositorySymbol] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    found.append(RepositorySymbol(rel, node.lineno, kind, node.name))
            return sorted(found, key=lambda item: (item.line, item.name))
        pattern = re.compile(
            r"^\s*(?:(?:export|public|private|protected|static|async|final|abstract)\s+)*"
            r"(?P<kind>class|interface|enum|struct|trait|type|function|fn|def|func)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)",
            re.MULTILINE,
        )
        return [
            RepositorySymbol(
                rel,
                text.count("\n", 0, match.start()) + 1,
                {"fn": "function", "func": "function", "def": "function"}.get(
                    match.group("kind"), match.group("kind")
                ),
                match.group("name"),
            )
            for match in pattern.finditer(text)
        ]


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
        self._request_cache: dict[tuple[str, str, int, str], Any] = {}
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
                "position": self._position(line, column),
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
            {"textDocument": {"uri": uri}, "position": self._position(line, column)},
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
        position = self._position(line, column)
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
            rows.extend(self._render_text_edits(target_uri, edits))
        for change in document_changes:
            if "edits" in change:
                rows.extend(
                    self._render_text_edits(
                        change.get("textDocument", {}).get("uri", ""), change["edits"]
                    )
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
            try:
                symbols = await self.document_symbols(rel)
            except Exception as exc:  # noqa: BLE001
                symbols = f"unavailable — {exc}"
            sections.append(f"## {rel}\n{symbols}")
        return "\n\n".join(sections) or "(no changed source symbols)"

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
            {"textDocument": {"uri": uri}, "position": self._position(line, column)},
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
        stamp = path.stat().st_mtime_ns
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
        return "\n".join(_location_text(item, self._workspace.root) for item in items[:100])

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
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z"],
                cwd=self._workspace.root,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return []
            entries = [
                entry for entry in result.stdout.decode(errors="replace").split("\0") if entry
            ]
            paths: list[Path] = []
            index = 0
            while index < len(entries):
                entry = entries[index]
                name = entry[3:] if len(entry) >= 4 else ""
                if entry[:2].strip() in {"R", "C"} and index + 1 < len(entries):
                    index += 1
                    name = entries[index]
                path = (self._workspace.root / name).resolve()
                if path.is_relative_to(self._workspace.root):
                    paths.append(path)
                index += 1
            return list(dict.fromkeys(paths))
        except (OSError, subprocess.SubprocessError):
            return []

    @hidden
    async def close(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self._clients.values()), return_exceptions=True
        )
        self._clients.clear()
