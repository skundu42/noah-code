"""Lazy LSP and repository-map tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from noah_code.approvals import ApprovalBroker
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.lsp_tools import LSPTools, RepositoryMap
from noah_code.workspace import Workspace


def test_repository_map_is_compact_and_invalidates_by_mtime(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("class First:\n    pass\n\ndef helper():\n    return 1\n")
    repository = RepositoryMap(tmp_path)

    initial = repository.build()
    assert [(item.kind, item.name) for item in initial] == [
        ("class", "First"),
        ("function", "helper"),
    ]

    source.write_text("class Second:\n    pass\n")
    refreshed = repository.build()
    assert [item.name for item in refreshed] == ["Second"]


@pytest.mark.asyncio
async def test_lsp_navigation_and_rename_preview_use_one_lazy_client(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("def target(value: int) -> int:\n    return value\n")
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tools = LSPTools(workspace, engine, ApprovalBroker(engine))

    class FakeClient:
        command = ("fake-lsp",)

        async def open_document(self, path, language):  # noqa: ANN001, ANN202
            assert path == source
            assert language == "python"
            return source.as_uri()

        async def request(self, method, params, *, timeout=None):  # noqa: ANN001, ANN202
            del timeout
            if method in {"textDocument/definition", "textDocument/implementation"}:
                return [{"uri": source.as_uri(), "range": {"start": {"line": 0, "character": 4}}}]
            if method == "textDocument/hover":
                return {"contents": {"kind": "markdown", "value": "`target(value: int) -> int`"}}
            if method == "textDocument/prepareRename":
                return {"range": params["position"], "placeholder": "target"}
            if method == "textDocument/rename":
                return {
                    "changes": {
                        source.as_uri(): [
                            {
                                "range": {"start": {"line": 0, "character": 4}},
                                "newText": params["newName"],
                            }
                        ]
                    }
                }
            raise AssertionError(method)

    fake = FakeClient()
    tools._client = AsyncMock(return_value=fake)  # type: ignore[method-assign]

    assert await tools.definition("module.py", 1, 5) == "module.py:1:5"
    assert "target(value: int)" in await tools.hover("module.py", 1, 5)
    preview = await tools.rename_preview("module.py", 1, 5, "renamed")
    assert "Rename preview (not applied)" in preview
    assert "module.py:1:5 -> renamed" in preview
    assert source.read_text().startswith("def target")
    assert tools._client.await_count == 3


@pytest.mark.asyncio
async def test_lsp_respects_disabled_configuration(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n")
    workspace = Workspace(tmp_path.resolve())
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    tools = LSPTools(workspace, engine, ApprovalBroker(engine), enabled=False)

    with pytest.raises(RuntimeError, match="disabled"):
        await tools.document_symbols("module.py")
