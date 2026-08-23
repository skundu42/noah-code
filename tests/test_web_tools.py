"""Permission-gated webfetch and websearch."""

from __future__ import annotations

import socket

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.web_tools import WebTools, _PublicWebTransport, _validated_public_target


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
async def test_fetch_rejects_url_credentials() -> None:
    web = _web(transport=_FakeTransport({}))
    with pytest.raises(ValueError, match="credentials"):
        await web.fetch("https://operator:password@example.com/private")


def test_public_transport_rejects_private_and_link_local_ip_literals() -> None:
    transport = _PublicWebTransport()

    for url in (
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://224.0.0.1/service",
        "http://[::1]/admin",
        "http://[ff02::1]/service",
        "http://[64:ff9b::127.0.0.1]/service",
    ):
        with pytest.raises(ValueError, match="public network"):
            transport.fetch(url, timeout=0.1, max_bytes=100)


def test_public_target_rejects_dns_with_any_private_answer(monkeypatch) -> None:
    def mixed_answers(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 80)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mixed_answers)

    with pytest.raises(ValueError, match="public network"):
        _validated_public_target("http://public.example/resource")


def test_public_transport_revalidates_redirect_targets(monkeypatch) -> None:
    class RedirectResponse:
        status = 302

        @staticmethod
        def getheader(name: str):  # noqa: ANN205
            return "http://127.0.0.1/private" if name == "Location" else None

        @staticmethod
        def read(_limit: int) -> bytes:
            return b""

    class RedirectConnection:
        def request(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        @staticmethod
        def getresponse() -> RedirectResponse:
            return RedirectResponse()

        def close(self) -> None:
            return None

    def public_answer(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    monkeypatch.setattr(
        "noah_code.tools.web_tools._open_pinned_connection",
        lambda *_args, **_kwargs: RedirectConnection(),
    )

    with pytest.raises(ValueError, match="public network"):
        _PublicWebTransport().fetch(
            "http://public.example/redirect",
            timeout=0.1,
            max_bytes=100,
        )


@pytest.mark.asyncio
async def test_default_fetch_does_not_need_an_approval_handler() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    transport = _FakeTransport({"https://example.com": ("text/plain", "hello")})
    web = WebTools(engine, ApprovalBroker(engine), transport=transport)

    assert "hello" in await web.fetch("https://example.com")
    assert transport.calls == ["https://example.com"]


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
