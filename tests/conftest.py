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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Re-enable sockets for integration tests; they exist to hit the network.

    The default suite runs with ``--disable-socket`` (see pyproject addopts),
    which would otherwise make every integration-marked test skip/fail.
    """

    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(pytest.mark.enable_socket)
