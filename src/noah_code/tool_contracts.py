"""Bounded, broker-safe documentation for an explicitly exposed tool surface.

Sandbox proxies cannot reveal the wrapped method's signature through ``doc``.
This catalog resolves trusted methods once in the host and retains only text.
Queries never traverse an object, import a module, or evaluate an expression.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from nooa import Skill, hidden

MAX_HELP_CHARS = 16_000
MAX_HINT_CHARS = 1_600
_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?")
_ALIASES = {"PatchChanges": "list[dict[str, str | None]]", "InspectTargets": "list[str]"}


@dataclass(frozen=True)
class _Details:
    returns: str
    notes: str = ""
    examples: tuple[str, ...] = ()


_LIST = _Details(
    "list[str]: workspace-relative path strings; iterate or join the strings directly.",
    "A bounded listing may end with a truncation marker. Each entry is a string.",
    ('paths = await self.ws.list("**/*.py"); print("\\n".join(paths))',),
)
_WRITE = _Details(
    "FileWrite with .path, .message, .diff, .new_text; print(result) for the edit report.",
    "Creates or overwrites the whole file. Read an existing file before replacing its content. "
    "Python files are syntax-checked before writes.",
    ('print(await self.ws.write("notes.txt", "Hello\\n"))',),
)
_EDIT = _Details(
    "FileWrite with .path, .message, .diff, .new_text; print(result) for the edit report.",
    "The old text must match exactly once, including whitespace. Pass exactly path, old, new.",
    ('print(await self.ws.edit("example.py", "return 1", "return 2"))',),
)
_DETAILS = {
    "ws.list": _LIST,
    "ws.list_files": _Details(
        _LIST.returns, _LIST.notes, ('print(await self.ws.list_files("**/*.py"))',)
    ),
    "ws.read": _Details(
        "WorkspaceMatch with .text (raw file text), .path, .start, .end; "
        "oversized output is WorkspaceText (str) with .text/.content and a managed preview.",
        "Print result.text to preserve source whitespace. A complete Match is an edit anchor; "
        "a truncated preview is not. lines=60 means first 60 lines; lines=(10, 30) is inclusive.",
        (
            'anchor = await self.ws.read("example.py"); print(anchor.text)',
            'print((await self.ws.read("example.py", lines=(1, 40))).text)',
        ),
    ),
    "ws.search": _Details(
        "SearchResult with .stdout, .stderr, .returncode, .matches; also iterable over Match "
        "anchors with .text, .path, .start, .end.",
        "An empty match sequence means no matches. Use regex=False for a literal search.",
        (
            'result = await self.ws.search("return 1", paths=["example.py"], regex=False); '
            "print(result.stdout)",
        ),
    ),
    "ws.replace": _Details(
        _WRITE.returns,
        "Two forms: replace(match, new_text) replaces a complete read/search Match; "
        "replace(path, old, new) replaces exactly one literal occurrence. A Match retains the "
        "original preimage; reuse it instead of retyping existing indentation. Refresh stale "
        "anchors with read(). A read preview cannot be used as an anchor. Match form accepts "
        "exactly two arguments. Python files are syntax-checked before writes.",
        (
            'anchor = await self.ws.read("example.py"); '
            'print(await self.ws.replace(anchor, anchor.text.replace("return 1", "return 2")))',
            'print(await self.ws.replace("example.py", "return 1", "return 2"))',
        ),
    ),
    "ws.edit": _EDIT,
    "ws.write": _WRITE,
    "ws.write_file": _Details(
        _WRITE.returns, _WRITE.notes, ('print(await self.ws.write_file("notes.txt", "Hello\\n"))',)
    ),
    "ws.apply_patch": _Details(
        "str: atomic batch report and diagnostics.",
        "changes is a list of dictionaries with exactly path, old, new. All paths must be "
        "distinct. Update: old is nonempty exact text occurring once and new is replacement "
        "text. Create: old=None, new is content, and the file must not exist. Delete: new=None "
        "and old equals the entire current file. Capture existing text from read().text. "
        "Every change is preflighted before the batch writes anything.",
        (
            'print(await self.ws.apply_patch([{"path": "example.py", "old": "return 1", '
            '"new": "return 2"}]))',
            'print(await self.ws.apply_patch([{"path": "new.txt", "old": None, '
            '"new": "Hello\\n"}]))',
            'old = (await self.ws.read("obsolete.txt")).text; '
            'print(await self.ws.apply_patch([{"path": "obsolete.txt", "old": old, "new": None}]))',
        ),
    ),
    "ws.apply_unified_diff": _Details(
        "str: atomic batch report and diagnostics.",
        "Accepts git-style ---/+++/@@ hunks with verified context; /dev/null denotes creates "
        "or deletes. A context mismatch aborts before writes.",
        (
            "print(await self.ws.apply_unified_diff("
            '"--- a/notes.txt\\n+++ b/notes.txt\\n@@ -1 +1 @@\\n-Hello\\n+Goodbye\\n"))',
        ),
    ),
    "ws.run": _Details(
        "ShellResult with .returncode, .stdout, .stderr, .timed_out; success requires returncode=0.",
        "Await the command and inspect its result. Tests/builds/Python use the default "
        "read_only=False; read_only=True is reserved for recognized read-only shell commands.",
        (
            'result = await self.ws.run("python -m unittest discover -s tests"); '
            "print(result.returncode, result.stdout, result.stderr)",
        ),
    ),
    "ws.inspect": _Details(
        "WorkspaceText (str) with .text/.content: bounded combined search/read output.",
        "Supply files, searches, or symbols=True. Use read() when you need an edit anchor.",
        ('print(await self.ws.inspect(files=["example.py"], searches=["TODO"]))',),
    ),
    "ws.read_output": _Details(
        "WorkspaceText (str) with .text/.content: a bounded slice of stored output.",
        "Use the output ID from a truncated tool result and an inclusive line range.",
    ),
    "ws.checks": _Details(
        "str: recorded verification commands, status, and staleness after edits.",
        "Run checks again when edits make earlier evidence stale.",
        ("print(await self.ws.checks())",),
    ),
    "message": _Details(
        "None: sends user-facing text.",
        "Synchronous: call without await.",
        ('self.message("The tests passed.")',),
    ),
}


class _Annotation:
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


class _PlainAnnotations(ast.NodeTransformer):
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "Annotated":
            first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
            return self.visit(first)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _ALIASES:
            return ast.parse(_ALIASES[node.id], mode="eval").body
        return node


def _annotation(value: Any) -> Any:
    if value is inspect.Signature.empty:
        return value
    text = value if isinstance(value, str) else inspect.formatannotation(value)
    with contextlib.suppress(SyntaxError, ValueError):
        text = ast.unparse(_PlainAnnotations().visit(ast.parse(text, mode="eval")))
    return _Annotation(text)


def _signature(function: Any) -> str:
    signature = inspect.signature(function)
    parameters = [
        parameter.replace(annotation=_annotation(parameter.annotation))
        for name, parameter in signature.parameters.items()
        if name not in {"self", "cls"}
    ]
    return str(
        signature.replace(
            parameters=parameters,
            return_annotation=_annotation(signature.return_annotation),
        )
    )


@dataclass(frozen=True)
class _Contract:
    name: str
    signature: str
    asynchronous: bool
    summary: str
    details: _Details

    def render(self, *, brief: bool = False) -> str:
        kind = "async; await required" if self.asynchronous else "sync; do not await"
        lines = [f"self.{self.name}{self.signature} [{kind}]", self.summary]
        if self.details.returns:
            lines.append(f"Returns: {self.details.returns}")
        if self.details.notes:
            lines.append(self.details.notes)
        examples = self.details.examples[:1] if brief else self.details.examples
        lines.extend(f"Example: {example}" for example in examples)
        return "\n".join(line for line in lines if line)


class ToolContracts(Skill):
    """Safe tool reference: synchronous list() and help() return plain text.

    Use ``print(self.tools.help("ws"))`` before editing. Tool references are
    captured at construction from the host's broker allowlist, then discarded.
    """

    def __init__(
        self,
        tools: Mapping[str, Any],
        *,
        allowed_paths: Collection[tuple[str, ...]],
    ) -> None:
        super().__init__()
        contracts: dict[str, _Contract] = {}
        sources = {**tools, "tools": self}
        paths = set(allowed_paths) | {("tools", "list"), ("tools", "help")}
        for path in sorted(paths):
            name = ".".join(path)
            if not _NAME.fullmatch(name) or len(path) not in {1, 2}:
                continue
            owner = sources.get(path[0])
            if owner is None:
                continue
            function = owner if len(path) == 1 else getattr(owner, path[1], None)
            if not callable(function):
                continue
            try:
                signature = _signature(function)
            except (TypeError, ValueError):
                continue
            summary = (inspect.getdoc(function) or "").split("\n\n", 1)[0]
            summary = " ".join(summary.split())[:400]
            contracts[name] = _Contract(
                name,
                signature,
                inspect.iscoroutinefunction(function),
                summary,
                _DETAILS.get(name, _Details("")),
            )
        self._contracts = contracts

    def list(self, group: str = "") -> str:
        """List exposed tool names. Optional group narrows the index, for example 'ws'."""
        query = self._query(group)
        if query is None:
            return "Use a tool group such as 'ws' or an empty string."
        groups: dict[str, list[str]] = {}
        for name, contract in self._contracts.items():
            root, _, method = name.partition(".")
            if query and query != root:
                continue
            label = (method or root) + (" [async]" if contract.asynchronous else " [sync]")
            groups.setdefault(root, []).append(label)
        text = "\n".join(f"{root}: {', '.join(methods)}" for root, methods in groups.items())
        if len(text) > 8_000:
            text = text[:8_000] + "\nMore tools omitted; choose a group."
        return (text or "Unknown tool group.") + (
            "\nUse print(self.tools.help('ws.read')) for a method, or help('ws') for a group."
        )

    def help(self, name: str = "ws") -> str:
        """Return exact signatures, async flags, results and examples for a tool or group."""
        query = self._query(name)
        if not query:
            return self.list()
        contracts = [
            contract
            for key, contract in self._contracts.items()
            if key == query or key.startswith(query + ".")
        ]
        if not contracts:
            return "Unknown tool. Use print(self.tools.list()) for exposed names."
        parts: list[str] = []
        length = 0
        for contract in contracts:
            rendered = contract.render()
            if length + len(rendered) + 2 > MAX_HELP_CHARS - 150:
                parts.append(
                    "More tools omitted; request self.tools.help('group.method') individually."
                )
                break
            parts.append(rendered)
            length += len(rendered) + 2
        return "\n\n".join(parts)

    @hidden
    def hint(self, name: str | tuple[str, ...]) -> str:
        """Host-only compact recovery guidance to append to a failed broker call."""
        query = self._query(".".join(name) if isinstance(name, tuple) else name)
        contract = self._contracts.get(query or "")
        if contract is None:
            return "Use print(self.tools.list()) for exposed tool names."
        suffix = f"\nReference: print(self.tools.help('{contract.name}'))"
        return contract.render(brief=True)[: MAX_HINT_CHARS - len(suffix)] + suffix

    @staticmethod
    def _query(value: Any) -> str | None:
        if not isinstance(value, str) or len(value) > 100:
            return None
        name = value.strip().removeprefix("self.")
        return name if not name or _NAME.fullmatch(name) else None
