"""Shared test fixtures."""

from __future__ import annotations

import pytest

from noah_code.mcp_setup import MCPInstallResult


@pytest.fixture(autouse=True)
def stub_eager_mcp_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host.start() from connecting the developer's real MCP servers."""

    async def _install(*_args: object, **_kwargs: object) -> MCPInstallResult:
        return MCPInstallResult()

    monkeypatch.setattr("noah_code.mcp_setup.install_mcp", _install)
