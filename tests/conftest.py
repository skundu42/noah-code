"""Shared test fixtures."""

from __future__ import annotations

import pytest

from noah_code.mcp_setup import MCPInstallResult


@pytest.fixture(autouse=True)
def isolate_user_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep tests isolated from the developer's ~/.config, ~/.local, and ~/.agents."""

    env_root = tmp_path_factory.mktemp("user-env")
    user_home = env_root / "home"
    user_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NOAH_CODE_CONFIG", str(env_root / "config" / "config.toml"))
    monkeypatch.setenv("XDG_DATA_HOME", str(env_root / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(env_root / "state"))
    monkeypatch.setattr("noah_code.skills_setup._default_user_home", lambda: user_home)


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
