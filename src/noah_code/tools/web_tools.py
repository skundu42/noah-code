"""Permission-gated outbound web tools."""

from __future__ import annotations

import asyncio
import html
import re
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from nooa import Skill

from noah_code.approvals import ApprovalBroker
from noah_code.permissions import PermissionCategory, PermissionEngine

_DEFAULT_SEARCH = "https://html.duckduckgo.com/html/?q={query}"
_MAX_BYTES = 400_000
_MAX_CHARS = 12_000
_TIMEOUT = 20.0
_USER_AGENT = "noah-code/0.2 (+https://github.com/skundu42/noah-code)"


class WebTransport(Protocol):
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        """Return ``(content_type, body)``."""


class _UrlLibTransport:
    def fetch(self, url: str, *, timeout: float, max_bytes: int) -> tuple[str, str]:
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is permission-gated
            content_type = str(response.headers.get("Content-Type") or "text/plain")
            body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            body = body[:max_bytes]
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        return content_type, body.decode(charset, errors="replace")


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
        self._transport = transport or _UrlLibTransport()
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


def _require_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an http or https URL")
    return parsed.geturl()


def _to_text(content_type: str, body: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in {"text/html", "application/xhtml+xml"}:
        parser = _HTMLText()
        parser.feed(html.unescape(body))
        return parser.text()
    return body.strip()
