"""Explicit browser opt-in using Microsoft's Playwright MCP server.

    https://github.com/microsoft/playwright-mcp

The preset is persisted through the normal MCP configuration path. Constructing
or saving it does not install packages, launch a process, or open a browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noah_code.mcp_setup import save_user_mcp_server

# Verified against the official npm package on 2026-09-26. Keep the version exact
# so enabling browser tools cannot silently execute a different package release.
PLAYWRIGHT_MCP_VERSION = "0.0.82"
PLAYWRIGHT_MCP_PACKAGE = f"@playwright/mcp@{PLAYWRIGHT_MCP_VERSION}"
BROWSER_CHOICES = ("chrome", "firefox", "webkit", "msedge")
DEFAULT_BROWSER_SERVER_NAME = "browser"


def browser_server_spec(*, browser: str = "chrome", headless: bool = True) -> dict[str, Any]:
    """Build a pinned stdio server with an isolated in-memory browser profile."""

    if browser not in BROWSER_CHOICES:
        raise ValueError(f"browser must be one of: {', '.join(BROWSER_CHOICES)}")
    args = ["-y", PLAYWRIGHT_MCP_PACKAGE, "--isolated", "--browser", browser]
    if headless:
        args.append("--headless")
    return {"transport": "stdio", "command": "npx", "args": args}


def configure_browser(
    *,
    home: Path | None = None,
    name: str = DEFAULT_BROWSER_SERVER_NAME,
    browser: str = "chrome",
    headless: bool = True,
) -> Path:
    """Enable browser tools in user-owned MCP config after an explicit setup action.

    The next normal MCP connection starts the server and may download the pinned
    npm package. Existing entries are never overwritten. Requires Node.js 18+
    with npx and Noah's MCP extra when the server is connected, not when saved.
    """

    return save_user_mcp_server(
        name,
        browser_server_spec(browser=browser, headless=headless),
        home=home,
    )
