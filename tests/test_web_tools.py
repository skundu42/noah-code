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
    _PinnedHTTPConnection,
    _PinnedHTTPSConnection,
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
        self.max_bytes_values: list[int] = []

    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        self.calls.append(url)
        self.timeouts.append(timeout)
        self.max_bytes_values.append(max_bytes)
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


# ---------------------------------------------------------------------------
# WebTools-level contract tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configured_timeout_and_max_bytes_reach_the_transport() -> None:
    transport = _FakeTransport(
        {
            "https://example.com/docs": ("text/plain", "hello"),
            "https://html.duckduckgo.com/html/?q=hi": ("text/plain", ""),
        }
    )
    web = _web(transport=transport)
    web._timeout = 3.25
    web._max_bytes = 1234

    await web.fetch("https://example.com/docs")
    await web.search("hi")

    assert transport.timeouts == [3.25, 3.25]
    assert transport.max_bytes_values == [1234, 1234]


def _plain_transport(text: str) -> _FakeTransport:
    return _FakeTransport({"https://example.com/big": ("text/plain", text)})


@pytest.mark.asyncio
async def test_fetch_truncates_oversized_text_with_marker() -> None:
    from noah_code.tools.web_tools import _MAX_CHARS

    transport = _plain_transport("x" * (_MAX_CHARS + 500))
    web = _web(transport=transport)

    text = await web.fetch("https://example.com/big")

    assert len(text) == _MAX_CHARS + len("\n...(truncated)...")
    assert text.endswith("...(truncated)...")


@pytest.mark.asyncio
async def test_search_rejects_empty_query() -> None:
    web = _web(transport=_FakeTransport({}))

    with pytest.raises(ValueError, match="query is required"):
        await web.search("   ")


def _search_html(links: list[tuple[str, str]]) -> str:
    body = "".join(f'<a href="{href}">{label}</a>' for label, href in links)
    return f"<html><body>{body}</body></html>"


@pytest.mark.asyncio
async def test_search_dedupes_urls() -> None:
    html = _search_html(
        [
            ("first", "https://a.example/1"),
            ("second", "https://a.example/1"),
            ("third", "https://b.example/2"),
        ]
    )
    transport = _FakeTransport({"https://html.duckduckgo.com/html/?q=x": ("text/html", html)})
    web = _web(transport=transport)

    text = await web.search("x")

    assert text.count("https://a.example/1") == 1
    assert "third" in text


@pytest.mark.asyncio
async def test_search_caps_results_at_eight() -> None:
    links = [(f"label{i}", f"https://e.example/{i}") for i in range(12)]
    transport = _FakeTransport(
        {"https://html.duckduckgo.com/html/?q=x": ("text/html", _search_html(links))}
    )
    web = _web(transport=transport)

    text = await web.search("x")
    results = [line for line in text.splitlines() if line and line[0].isdigit()]
    assert len(results) == 8
    assert all(f"https://e.example/{i}" not in text for i in range(8, 12))


@pytest.mark.asyncio
async def test_search_reports_when_nothing_matches() -> None:
    transport = _FakeTransport(
        {"https://html.duckduckgo.com/html/?q=x": ("text/html", "<html></html>")}
    )
    web = _web(transport=transport)

    assert await web.search("x") == "No search results for 'x'."


# ---------------------------------------------------------------------------
# URL validation and DNS-resolution defenses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://exa\x00mple.com/",
        "ftp://example.com/file",
        "not a url",
        "https://operator:pw@example.com/",
        "https://example.com:70000/",
        "https://example.com:notaport/",
        "http://[fe80::1%25eth0]/",
    ],
)
def test_require_http_url_rejects_malicious_and_malformed_urls(url: str) -> None:
    from noah_code.tools.web_tools import _require_http_url

    with pytest.raises(ValueError):
        _require_http_url(url)


def test_public_target_rejects_overlong_hostname_label() -> None:
    with pytest.raises(ValueError, match="hostname is invalid"):
        _validated_public_target("http://" + "a" * 64 + ".example.com/")


def test_public_target_reports_unresolvable_host(monkeypatch) -> None:
    def fail(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fail)

    with pytest.raises(ValueError, match="could not be resolved"):
        _validated_public_target("http://missing.example/page")


def test_public_target_reports_empty_answer(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="could not be resolved"):
        _validated_public_target("http://empty.example/page")


def test_public_target_rejects_invalid_resolved_address(monkeypatch) -> None:
    def junk(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("not-an-ip", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", junk)

    with pytest.raises(ValueError, match="invalid address"):
        _validated_public_target("http://junk.example/page")


def test_open_pinned_connection_selects_scheme_without_connecting() -> None:
    from dataclasses import replace as _replace

    from noah_code.tools.web_tools import _HttpTarget, _open_pinned_connection

    http_target = _HttpTarget(
        scheme="http", host="example.com", port=80, request_target="/", addresses=("93.184.216.34",)
    )
    https_target = _replace(http_target, scheme="https")

    assert isinstance(
        _open_pinned_connection(http_target, "93.184.216.34", 0.1), _PinnedHTTPConnection
    )
    assert isinstance(
        _open_pinned_connection(https_target, "93.184.216.34", 0.1), _PinnedHTTPSConnection
    )
