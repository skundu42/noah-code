"""Permission-gated webfetch and websearch."""

from __future__ import annotations

import socket

import pytest

from noah_code.approvals import ApprovalBroker, ApprovalChoice
from noah_code.config import DEFAULT_PERMISSION_RULES
from noah_code.permissions import PermissionEngine
from noah_code.tools.web_tools import (
    _DEFAULT_BACKOFF_SECONDS,
    _MAX_RATE_LIMIT_RETRIES,
    _MAX_RETRY_AFTER_SECONDS,
    WebTools,
    _PublicWebTransport,
    _validated_public_target,
)


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
        self.timeouts: list[float] = []

    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        self.calls.append(url)
        self.timeouts.append(timeout)
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


# ---------------------------------------------------------------------------
# Scripted HTTP transport harness (no real sockets).
# ---------------------------------------------------------------------------


class _ScriptedResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self._headers = {key.lower(): value for key, value in headers.items()}
        self._body = body

    def getheader(self, name: str):  # noqa: ANN201
        return self._headers.get(name.lower())

    def read(self, limit: int) -> bytes:
        return self._body[:limit]


_PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 80))]


def _install_scripted_http(monkeypatch, handler) -> list[tuple[str, float]]:
    """Serve responses from ``handler(connection)`` for every pinned request.

    Returns the request targets observed, paired with the timeout each
    connection was constructed with.
    """

    observed: list[tuple[str, float]] = []

    class ScriptedConnection:
        def __init__(self, target, address, timeout) -> None:  # noqa: ANN001
            self.target = target
            self.timeout = timeout

        def request(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def getresponse(self) -> _ScriptedResponse:
            observed.append((self.target.request_target, self.timeout))
            status, headers, body = handler(self)
            return _ScriptedResponse(status, headers, body)

        def close(self) -> None:
            return None

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: list(_PUBLIC_DNS))
    monkeypatch.setattr(
        "noah_code.tools.web_tools._open_pinned_connection",
        lambda target, address, timeout: ScriptedConnection(target, address, timeout),
    )
    return observed


def test_public_transport_follows_relative_redirect(monkeypatch) -> None:
    seen = _install_scripted_http(
        monkeypatch,
        lambda conn: (
            (302, {"Location": "/next?page=1"}, b"")
            if conn.target.request_target == "/start"
            else (200, {"Content-Type": "text/plain"}, b"final page")
        ),
    )

    content_type, body = _PublicWebTransport().fetch(
        "http://public.example/start", timeout=0.5, max_bytes=1000
    )

    assert content_type == "text/plain"
    assert body == "final page"
    assert [target for target, _timeout in seen] == ["/start", "/next?page=1"]


def test_public_transport_redirect_loop_hits_limit(monkeypatch) -> None:
    _install_scripted_http(
        monkeypatch,
        lambda conn: (302, {"Location": "/loop"}, b""),
    )

    with pytest.raises(ValueError, match="redirect limit exceeded"):
        _PublicWebTransport().fetch("http://public.example/loop", timeout=0.5, max_bytes=100)


def test_public_transport_raises_last_connection_error(monkeypatch) -> None:
    _install_scripted_http(monkeypatch, lambda conn: (_ for _ in ()).throw(OSError("refused")))

    with pytest.raises(OSError, match="refused"):
        _PublicWebTransport().fetch("http://public.example/x", timeout=0.5, max_bytes=100)


def test_public_transport_truncates_body_to_max_bytes(monkeypatch) -> None:
    _install_scripted_http(
        monkeypatch,
        lambda conn: (200, {"Content-Type": "text/plain"}, b"0123456789"),
    )

    _content_type, body = _PublicWebTransport().fetch(
        "http://public.example/big", timeout=0.5, max_bytes=4
    )

    assert body == "0123"


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("text/plain; charset=utf-99", "héllo"),
        ("text/plain; charset=", "héllo"),
    ],
)
def test_public_transport_bad_charset_falls_back_to_utf8(
    monkeypatch, content_type, expected
) -> None:
    _install_scripted_http(
        monkeypatch,
        lambda conn: (200, {"Content-Type": content_type}, expected.encode("utf-8")),
    )

    _content_type, body = _PublicWebTransport().fetch(
        "http://public.example/i18n", timeout=0.5, max_bytes=100
    )

    assert body == expected


def _rate_limited_handler(state: dict[str, int], retry_after: str | None, final_ok: bool):
    def handler(conn):  # noqa: ANN001, ANN202
        state["count"] += 1
        if state["count"] <= state["limit"]:
            headers = {"Retry-After": retry_after} if retry_after is not None else {}
            return (429, headers, b"slow down")
        if final_ok:
            return (200, {"Content-Type": "text/plain"}, b"recovered")
        return (429, {}, b"still limited")

    return handler


def test_rate_limit_backs_off_then_recovers(monkeypatch) -> None:
    state = {"count": 0, "limit": 1}
    _install_scripted_http(monkeypatch, _rate_limited_handler(state, "1.5", final_ok=True))
    sleeps: list[float] = []
    monkeypatch.setattr("noah_code.tools.web_tools._sleep", sleeps.append)

    content_type, body = _PublicWebTransport().fetch(
        "http://public.example/api", timeout=0.5, max_bytes=100
    )

    assert body == "recovered"
    assert sleeps == [1.5]
    assert state["count"] == 2


def test_rate_limit_caps_retry_after(monkeypatch) -> None:
    state = {"count": 0, "limit": 1}
    _install_scripted_http(monkeypatch, _rate_limited_handler(state, "9999", final_ok=True))
    sleeps: list[float] = []
    monkeypatch.setattr("noah_code.tools.web_tools._sleep", sleeps.append)

    _PublicWebTransport().fetch("http://public.example/api", timeout=0.5, max_bytes=100)

    assert sleeps == [_MAX_RETRY_AFTER_SECONDS]


def test_rate_limit_default_backoff_when_header_missing_or_invalid(monkeypatch) -> None:
    state = {"count": 0, "limit": 1}
    _install_scripted_http(monkeypatch, _rate_limited_handler(state, None, final_ok=True))
    sleeps: list[float] = []
    monkeypatch.setattr("noah_code.tools.web_tools._sleep", sleeps.append)

    _PublicWebTransport().fetch("http://public.example/api", timeout=0.5, max_bytes=100)

    assert sleeps == [_DEFAULT_BACKOFF_SECONDS]

    state = {"count": 0, "limit": 1}
    _install_scripted_http(monkeypatch, _rate_limited_handler(state, "later", final_ok=True))
    sleeps.clear()
    _PublicWebTransport().fetch("http://public.example/api", timeout=0.5, max_bytes=100)

    assert sleeps == [_DEFAULT_BACKOFF_SECONDS]


def test_rate_limit_gives_up_after_bounded_retries(monkeypatch) -> None:
    state = {"count": 0, "limit": 99}
    _install_scripted_http(monkeypatch, _rate_limited_handler(state, "1", final_ok=False))
    monkeypatch.setattr("noah_code.tools.web_tools._sleep", lambda _seconds: None)

    with pytest.raises(OSError, match="rate limited"):
        _PublicWebTransport().fetch("http://public.example/api", timeout=0.5, max_bytes=100)
    assert state["count"] == _MAX_RATE_LIMIT_RETRIES + 1


def test_html_text_drops_noscript_with_nested_tags() -> None:
    from noah_code.tools.web_tools import _HTMLText

    parser = _HTMLText()
    parser.feed("<p>before</p><noscript><p>Please enable JavaScript.</p></noscript><p>after</p>")

    text = parser.text()

    assert "before" in text and "after" in text
    assert "enable JavaScript" not in text


def test_html_text_keeps_skipping_script_containing_markup_like_text() -> None:
    from noah_code.tools.web_tools import _HTMLText

    parser = _HTMLText()
    parser.feed('<script>var a = "</p>";</script><style>p{color:red}</style><p>visible</p>')

    text = parser.text()

    assert "var a" not in text
    assert "color:red" not in text
    assert "visible" in text
