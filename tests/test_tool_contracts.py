"""Tool discovery must cross the sandbox as text, without reflecting host objects."""

from __future__ import annotations

import ast
import gc
import inspect
import platform
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nooa.runtime.restrictions import RestrictionsConfig
from nooa.runtime.sandbox.config import SandboxConfig

from noah_code.agent import (
    _interpreter_read_rules,
    _MacOSPermissionSandboxedExecutor,
    _PermissionSandboxedExecutor,
)
from noah_code.tool_contracts import _DETAILS, MAX_HELP_CHARS, MAX_HINT_CHARS, ToolContracts
from noah_code.tools.workspace_tools import WorkspaceTools
from test_workspace_tools import _make_ws


def workspace_contracts() -> ToolContracts:
    return ToolContracts(
        {"ws": WorkspaceTools}, allowed_paths=_PermissionSandboxedExecutor._EXACT_PATHS
    )


def test_workspace_contracts_have_actual_signatures_shapes_and_edit_examples() -> None:
    catalog = workspace_contracts()
    assert (
        "self.ws.read(path: str, lines: tuple[int, int] | int | None = None)"
        " -> WorkspaceMatch | WorkspaceText [async; await required]"
    ) in catalog.help("self.ws.read")
    assert "Annotated" not in catalog.help("ws")
    assert "spec(" not in catalog.help("ws")
    assert "list[str]" in catalog.help("ws.list")
    assert "join(paths)" in catalog.help("ws.list")
    assert "replace(match, new_text)" in catalog.help("ws.replace")
    assert "replace(path, old, new)" in catalog.help("ws.replace")
    patch = catalog.help("ws.apply_patch")
    assert "changes: list[dict[str, str | None]]" in patch
    assert "exactly path, old, new" in patch
    assert "file must not exist" in patch
    assert "None" in patch
    assert len(catalog.help("ws")) < MAX_HELP_CHARS
    assert "[sync; do not await]" in catalog.help("tools.help")


def test_catalog_retains_text_only_and_queries_cannot_resolve_new_attributes() -> None:
    class Source:
        async def exposed(self, path: str, *, limit: int = 3) -> str:
            """Documented read."""
            return path

        def private(self) -> str:
            return "secret-host-internal"

    source = Source()
    reference = weakref.ref(source)
    catalog = ToolContracts({"sample": source}, allowed_paths={("sample", "exposed")})
    del source
    gc.collect()
    assert reference() is None
    assert "limit: int = 3" in catalog.help("sample.exposed")
    for query in ("sample.private", "sample.__dict__", "sample.exposed.__globals__", "os.system"):
        assert "secret-host-internal" not in catalog.help(query)
        assert "secret-host-internal" not in catalog.hint(query)
    assert "private" not in catalog.list()


def test_signature_is_captured_from_public_method_without_evaluating_annotations() -> None:
    def never_evaluate_this():
        raise AssertionError("Annotations are documentation, not executable code")

    class Source:
        def method(self, value: never_evaluate_this(), /, *, exact: bool = False) -> str:
            return ""

    catalog = ToolContracts({"sample": Source()}, allowed_paths={("sample", "method")})
    text = catalog.help("sample.method")
    assert "never_evaluate_this()" in text
    assert "/, *, exact: bool = False" in text
    assert "[sync; do not await]" in text


def test_recovery_hint_has_correct_contract_and_is_bounded() -> None:
    catalog = workspace_contracts()
    hint = catalog.hint(("ws", "apply_patch"))
    assert "exactly path, old, new" in hint
    assert "print(self.tools.help('ws.apply_patch'))" in hint
    assert len(hint) <= MAX_HINT_CHARS
    for query in ("_contracts", "ws.read.__call__", "x" * 101, None):
        assert len(catalog.help(query)) <= MAX_HELP_CHARS
        assert len(catalog.hint(query)) <= MAX_HINT_CHARS


@pytest.mark.parametrize(
    ("tool", "example"),
    [
        (name, example)
        for name, details in _DETAILS.items()
        if name.startswith("ws.") and name != "ws.run"
        for example in details.examples
    ],
)
async def test_workspace_examples_execute_against_real_tools(tool, example, tmp_path) -> None:
    (tmp_path / "example.py").write_text("def example():\n    return 1\n")
    (tmp_path / "notes.txt").write_text("Hello\n")
    (tmp_path / "obsolete.txt").write_text("Obsolete\n")
    ws = _make_ws(tmp_path)
    try:
        result = eval(
            compile(example, f"contract:{tool}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
            {"self": SimpleNamespace(ws=ws)},
        )
        if inspect.isawaitable(result):
            await result
        compile((tmp_path / "example.py").read_text(), "example.py", "exec")
    finally:
        await ws.close()


async def test_broker_returns_plain_help_text_and_denies_contract_internals() -> None:
    catalog = workspace_contracts()
    executor = object.__new__(_PermissionSandboxedExecutor)
    executor._agent = SimpleNamespace(tools=catalog)
    executor._max_error = 2000
    result = await executor._dispatch_tool_call(
        {"kind": "call", "path": ["tools", "help"], "args": ["ws.replace"], "kwargs": {}}
    )
    assert result["ok"] is True
    assert isinstance(result["result"], str)
    assert "replace(match, new_text)" in result["result"]
    for path in (["tools", "_contracts"], ["tools", "hint"], ["tools", "help", "__globals__"]):
        result = await executor._dispatch_tool_call({"kind": "attr", "path": path})
        assert result["ok"] is False


@pytest.mark.parametrize(
    ("method", "args", "expected"),
    [
        ("edit", ["example.py", "replacement"], "self.ws.edit(path:"),
        ("apply_patch", [[{"path": "example.py", "op": "replace"}]], "exactly path, old, new"),
    ],
)
async def test_broker_errors_include_actionable_contract(
    tmp_path: Path, method: str, args: list[Any], expected: str,
) -> None:
    ws = _make_ws(tmp_path)
    executor = object.__new__(_PermissionSandboxedExecutor)
    executor._agent = SimpleNamespace(ws=ws, tools=workspace_contracts())
    executor._max_error = 2000
    try:
        result = await executor._dispatch_tool_call(
            {"kind": "call", "path": ["ws", method], "args": args}
        )
    finally:
        await ws.close()
    assert result["ok"] is False
    assert expected in result["call_hint"]
    assert f"self.tools.help('ws.{method}')" in result["call_hint"]
    assert len(result["call_hint"]) <= MAX_HINT_CHARS


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires native macOS sandbox")
async def test_actual_worker_can_discover_tools_without_reflecting_the_proxy() -> None:
    executor = _MacOSPermissionSandboxedExecutor(
        SimpleNamespace(tools=workspace_contracts()),
        SandboxConfig(
            filesystem=True,
            allow=_interpreter_read_rules(),
            system_paths=False,
            network=False,
            max_cpu_seconds=10,
            require=True,
        ),
        cell_timeout=15,
        restrictions=RestrictionsConfig(),
    )
    try:
        result = await executor.run_cell("print(self.tools.help('ws.replace'))")
        assert result.error is None
        assert "replace(match, new_text)" in result.stdout
        assert "async; await required" in result.stdout
        blocked = await executor.run_cell("print(self.tools.hint('ws.replace'))")
        assert blocked.error is not None
        assert "denied" in str(blocked.error)
    finally:
        await executor.aclose()
