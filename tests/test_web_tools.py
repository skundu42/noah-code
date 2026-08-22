"""Permission-gated webfetch and websearch."""

from __future__ import annotations

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.web_tools import WebTools


async def _always_once(_req):
    return ApprovalChoice.ONCE


def _web(*, auto: bool = True, mode: str = "build", transport=None) -> WebTools:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode=mode, auto_approve=auto)  # type: ignore[arg-type]
    approvals = ApprovalBroker(engine, handler=_always_once)
    return WebTools(engine, approvals, transport=transport)


class _FakeTransport:
    def __init__(self, responses: dict[str, tuple[str, str]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        self.calls.append(url)
        _ = timeout, max_bytes
        if url not in self.responses:
            raise LookupError(url)
        return self.responses[url]


@pytest.mark.asyncio
async def test_fetch_strips_html_and_bounds_output() -> None:
    transport = _FakeTransport(
        {
            "https://example.com/docs": (
                "text/html",
                "<html><head><title>Docs</title></head><body><h1>Install</h1><p>pip install x</p></body></html>",
            )
        }
    )
    web = _web(transport=transport)

    text = await web.fetch("https://example.com/docs")

    assert "Install" in text
    assert "pip install x" in text
    assert "<html>" not in text
    assert transport.calls == ["https://example.com/docs"]


@pytest.mark.asyncio
async def test_fetch_rejects_non_http_urls() -> None:
    web = _web(transport=_FakeTransport({}))
    with pytest.raises(ValueError, match="http"):
        await web.fetch("file:///etc/passwd")


@pytest.mark.asyncio
async def test_fetch_asks_then_denies_without_handler() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    web = WebTools(engine, ApprovalBroker(engine), transport=_FakeTransport({}))
    with pytest.raises(PermissionError):
        await web.fetch("https://example.com")


@pytest.mark.asyncio
async def test_search_returns_ranked_results() -> None:
    html = """
    <html><body>
      <a class="result__a" href="https://docs.python.org/3/library/asyncio.html">asyncio</a>
      <a class="result__a" href="https://docs.python.org/3/library/json.html">json</a>
    </body></html>
    """
    transport = _FakeTransport({"https://html.duckduckgo.com/html/?q=asyncio": ("text/html", html)})
    web = _web(transport=transport)

    text = await web.search("asyncio")

    assert "asyncio" in text
    assert "docs.python.org" in text
    assert "json" in text
