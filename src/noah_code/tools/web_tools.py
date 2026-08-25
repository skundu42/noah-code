"""Permission-gated outbound web tools."""

from __future__ import annotations

import asyncio
import html
import http.client
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import quote_plus, urljoin, urlsplit

from nooa import Skill

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine

_DEFAULT_SEARCH = "https://html.duckduckgo.com/html/?q={query}"
_MAX_BYTES = 400_000
_MAX_CHARS = 12_000
_TIMEOUT = 20.0
_USER_AGENT = "noah-code/0.2 (+https://github.com/skundu42/noah-code)"
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_RATE_LIMIT_STATUS = 429
_MAX_RATE_LIMIT_RETRIES = 2
_DEFAULT_BACKOFF_SECONDS = 0.5
_MAX_RETRY_AFTER_SECONDS = 5.0

# Module level so tests can stub time without real delays.
_sleep = time.sleep


class WebTransport(Protocol):
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        """Return ``(content_type, body)``."""


@dataclass(frozen=True)
class _HttpTarget:
    scheme: str
    host: str
    port: int
    request_target: str
    addresses: tuple[str, ...]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connect to an already-validated address while preserving the Host header."""

    def __init__(self, target: _HttpTarget, address: str, timeout: float) -> None:
        super().__init__(target.host, target.port, timeout=timeout)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to validated DNS while retaining TLS SNI checks."""

    def __init__(self, target: _HttpTarget, address: str, timeout: float) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(
            target.host,
            target.port,
            timeout=timeout,
            context=self._tls_context,
        )
        self._validated_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
        )
        try:
            self.sock = self._tls_context.wrap_socket(raw_socket, server_hostname=self.host)
        except BaseException:
            raw_socket.close()
            raise


def _open_pinned_connection(
    target: _HttpTarget, address: str, timeout: float
) -> http.client.HTTPConnection:
    connection_type = _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
    return connection_type(target, address, timeout)


def _request_pinned(
    target: _HttpTarget,
    address: str,
    *,
    timeout: float,
    max_bytes: int,
) -> tuple[int, str | None, str | None, str, bytes]:
    connection = _open_pinned_connection(target, address, timeout)
    try:
        connection.request(
            "GET",
            target.request_target,
            headers={
                "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
                "Connection": "close",
                "User-Agent": _USER_AGENT,
            },
        )
        response = connection.getresponse()
        return (
            response.status,
            response.getheader("Location"),
            response.getheader("Retry-After"),
            str(response.getheader("Content-Type") or "text/plain"),
            response.read(max_bytes + 1),
        )
    finally:
        connection.close()


class _PublicWebTransport:
    """Bounded public-web transport with redirect and DNS-rebinding defenses."""

    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        current_url = _require_http_url(url)
        redirects = 0
        rate_limit_retries = 0
        while True:
            target = _validated_public_target(current_url)
            payload: tuple[int, str | None, str | None, str, bytes] | None = None
            connection_error: OSError | None = None
            for address in target.addresses:
                try:
                    payload = _request_pinned(
                        target,
                        address,
                        timeout=timeout,
                        max_bytes=max_bytes,
                    )
                except OSError as exc:
                    connection_error = exc
                    continue
                break
            if payload is None:
                if connection_error is not None:
                    raise connection_error
                raise OSError(f"could not connect to public URL: {target.host}")

            status, location, retry_after, content_type, body = payload
            if status == _RATE_LIMIT_STATUS:
                if rate_limit_retries >= _MAX_RATE_LIMIT_RETRIES:
                    raise OSError(f"web rate limited by {target.host}")
                rate_limit_retries += 1
                _sleep(min(_retry_after_seconds(retry_after), _MAX_RETRY_AFTER_SECONDS))
                continue
            if status in _REDIRECT_STATUSES and location:
                if redirects >= _MAX_REDIRECTS:
                    raise ValueError(f"web redirect limit exceeded ({_MAX_REDIRECTS})")
                redirects += 1
                current_url = _require_http_url(urljoin(current_url, location))
                continue

            if len(body) > max_bytes:
                body = body[:max_bytes]
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
            try:
                return content_type, body.decode(charset, errors="replace")
            except LookupError:
                return content_type, body.decode("utf-8", errors="replace")


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._skip = tag in {"script", "style", "noscript"}
        if tag in {"p", "div", "h1", "h2", "h3", "li", "br", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag in {"p", "div", "h1", "h2", "h3", "li"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._chunks.append(data)

    def text(self) -> str:
        collapsed = re.sub(r"[ \t]+", " ", "".join(self._chunks))
        return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


class _SearchLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._href: str | None = None
        self._label: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_map = dict(attrs)
        href = attrs_map.get("href") or ""
        if href.startswith("http"):
            self._href = href
            self._label = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        label = " ".join(part.strip() for part in self._label if part.strip())
        if label:
            self.results.append((label, self._href))
        self._href = None
        self._label = []


class WebTools(Skill):
    """Fetch URLs and search the public web. Both actions are permission-gated."""

    def __init__(
        self,
        engine: PermissionEngine,
        approvals: ApprovalBroker,
        *,
        transport: WebTransport | None = None,
        search_url: str = _DEFAULT_SEARCH,
        timeout: float = _TIMEOUT,
        max_bytes: int = _MAX_BYTES,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._approvals = approvals
        self._transport = transport or _PublicWebTransport()
        self._search_url = search_url
        self._timeout = timeout
        self._max_bytes = max_bytes

    async def fetch(self, url: str) -> str:
        """Read a public HTTP(S) page and return readable text."""

        normalized = _require_http_url(url)
        await self._approvals.require(
            self._engine.decide(PermissionCategory.WEBFETCH, normalized, tool="web_fetch")
        )
        # Blocking network I/O must not freeze the agent event loop.
        content_type, body = await asyncio.to_thread(
            self._transport.fetch,
            normalized,
            timeout=self._timeout,
            max_bytes=self._max_bytes,
        )
        text = _to_text(content_type, body)
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n...(truncated)..."
        return text

    async def search(self, query: str) -> str:
        """Search the public web and return titles plus URLs."""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("search query is required")
        await self._approvals.require(
            self._engine.decide(PermissionCategory.WEBSEARCH, cleaned, tool="web_search")
        )
        url = self._search_url.format(query=quote_plus(cleaned))
        _content_type, body = await asyncio.to_thread(
            self._transport.fetch,
            url,
            timeout=self._timeout,
            max_bytes=self._max_bytes,
        )
        parser = _SearchLinks()
        parser.feed(body)
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for title, href in parser.results:
            if href in seen:
                continue
            seen.add(href)
            unique.append((title, href))
            if len(unique) >= 8:
                break
        if not unique:
            return f"No search results for {cleaned!r}."
        lines = [f"Web search: {cleaned}", ""]
        for index, (title, href) in enumerate(unique, start=1):
            lines.append(f"{index}. {title}\n   {href}")
        return "\n".join(lines)


def _retry_after_seconds(header: str | None) -> float:
    """Parse a numeric ``Retry-After`` value; fall back to the default backoff.

    HTTP-date forms are not honored (they would require a clock read); the
    default backoff keeps behavior bounded either way.
    """

    if not header:
        return _DEFAULT_BACKOFF_SECONDS
    try:
        return max(0.0, float(header.strip()))
    except ValueError:
        return _DEFAULT_BACKOFF_SECONDS


def _require_http_url(url: str) -> str:
    cleaned = url.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in cleaned):
        raise ValueError("url must not contain control characters")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url credentials are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("url has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("url has an invalid port")
    if "%" in parsed.hostname:
        raise ValueError("scoped IP addresses are not allowed")
    return parsed.geturl()


def _validated_public_target(url: str) -> _HttpTarget:
    parsed = urlsplit(_require_http_url(url))
    hostname = parsed.hostname or ""
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            host = hostname.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("url hostname is invalid") from exc
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            records = socket.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as exc:
            raise ValueError(f"url hostname could not be resolved: {host}") from exc
        addresses = tuple(dict.fromkeys(record[4][0] for record in records))
    else:
        host = address.compressed
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = (host,)

    if not addresses:
        raise ValueError(f"url hostname could not be resolved: {host}")
    try:
        parsed_addresses = tuple(ipaddress.ip_address(item) for item in addresses)
    except ValueError as exc:
        raise ValueError("url hostname resolved to an invalid address") from exc
    if any(not _is_public_unicast(address) for address in parsed_addresses):
        raise ValueError("url must resolve only to public network addresses")

    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    return _HttpTarget(
        scheme=parsed.scheme,
        host=host,
        port=port,
        request_target=request_target,
        addresses=tuple(address.compressed for address in parsed_addresses),
    )


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        not address.is_global
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address) and address in _NAT64_WELL_KNOWN_PREFIX:
        embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        return _is_public_unicast(embedded)
    return True


def _to_text(content_type: str, body: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLText()
        parser.feed(html.unescape(body))
        return parser.text()
    return body.strip()
