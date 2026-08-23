"""Deterministic record/replay transport around NOOA's UnifiedLLM clients.

Requests are keyed on a canonical hash of model + messages + tool schemas +
structured-output type + sampling kwargs, so a harness can rerun identical
turns without provider calls. Modes:

- ``record``  call through, persist every response
- ``replay``  serve from disk only; a miss raises ``CacheMissError``
- ``auto``    replay hits, otherwise call through and record

API keys and other credential kwargs never enter the cache key or payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

_UNSAFE_KWARG_PARTS = ("api_key", "token", "secret", "credential", "authorization")


class LLMCacheError(RuntimeError):
    """Record/replay transport failure."""


class CacheMissError(LLMCacheError):
    """Replay mode encountered a request with no cached response."""


def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in sorted(kwargs.items()):
        lowered = key.lower()
        if any(part in lowered for part in _UNSAFE_KWARG_PARTS):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = repr(value)
    return clean


def _tool_payloads(tools: list[Any] | None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for tool in tools or []:
        schema = tool.get_parameter_schema() if hasattr(tool, "get_parameter_schema") else None
        payloads.append(
            {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", ""),
                "parameters": schema,
            }
        )
    return payloads


def _output_model_identity(output_model: Any) -> str | None:
    if output_model is None:
        return None
    candidate = output_model if isinstance(output_model, type) else type(output_model)
    qualified_name = f"{candidate.__module__}.{candidate.__qualname__}"
    schema_factory = getattr(candidate, "model_json_schema", None)
    if not callable(schema_factory):
        return qualified_name
    try:
        schema = schema_factory()
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=repr)
    except Exception:  # noqa: BLE001 - third-party schema factories are optional
        return qualified_name
    schema_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{qualified_name}:{schema_hash}"


def request_key(
    model: str,
    messages: list[dict],
    tools: list[Any] | None,
    kwargs: dict[str, Any],
    *,
    output_model: Any = None,
) -> str:
    canonical = {
        "model": str(model),
        "messages": messages,
        "tools": _tool_payloads(tools),
        "output_model": _output_model_identity(output_model),
        "kwargs": _sanitize_kwargs(kwargs),
    }
    blob = json.dumps(canonical, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _content_to_payload(content: Any) -> dict[str, Any]:
    import pydantic

    if isinstance(content, pydantic.BaseModel):
        return {
            "kind": "pydantic",
            "module": type(content).__module__,
            "name": type(content).__name__,
            "data": content.model_dump(mode="json"),
        }
    return {"kind": "str", "data": content}


def _content_from_payload(payload: dict[str, Any], *, output_model: Any = None) -> Any:
    kind = payload.get("kind")
    if output_model is not None:
        validator = getattr(output_model, "model_validate", None)
        if kind != "pydantic" or not callable(validator):
            raise LLMCacheError("cached structured output is incompatible with output_model")
        return validator(payload["data"])
    if kind == "pydantic":
        # Never import a module named by a cache file. Besides being unsafe for
        # a user-configurable cache directory, dynamic and function-local model
        # classes cannot be reconstructed that way. The request's output model
        # is the only authoritative validator.
        raise LLMCacheError("cached structured output requires its output_model")
    return payload.get("data")


def response_to_payload(response: Any) -> dict[str, Any]:
    """Serialize an LLMResponse for the cache; raw provider payloads excluded."""

    return {
        "version": 1,
        "content": _content_to_payload(getattr(response, "content", "")),
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in (getattr(response, "tool_calls", None) or [])
        ],
        "finish_reason": getattr(response, "finish_reason", "stop"),
        "assistant_message": getattr(response, "assistant_message", {}) or {},
        "reasoning": getattr(response, "reasoning", None),
        "usage": getattr(response, "usage", None),
    }


def response_from_payload(payload: dict[str, Any], *, output_model: Any = None) -> Any:
    from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall

    return LLMResponse(
        raw_response=None,
        content=_content_from_payload(
            payload.get("content", {"kind": "str", "data": ""}),
            output_model=output_model,
        ),
        tool_calls=[
            ToolCall(id=item["id"], name=item["name"], arguments=item.get("arguments", ""))
            for item in payload.get("tool_calls", [])
        ],
        finish_reason=payload.get("finish_reason", "stop"),
        assistant_message=payload.get("assistant_message", {}) or {},
        reasoning=payload.get("reasoning"),
        usage=payload.get("usage"),
    )


class CachedLLM:
    """UnifiedLLM-shaped wrapper adding transparent record/replay."""

    def __init__(self, inner: Any, cache_dir: Path, mode: str) -> None:
        if mode not in {"record", "replay", "auto"}:
            raise ValueError("cache mode must be record, replay, or auto")
        self._inner = inner
        self._dir = cache_dir.expanduser().resolve()
        self._mode = mode
        self._lock = threading.Lock()
        self.hits = 0
        self.misses_recorded = 0
        self.replay_misses = 0

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def context_window(self) -> Any:
        return getattr(self._inner, "context_window", None)

    def _path_for(self, key: str) -> Path:
        bucket = key[:2]
        return self._dir / bucket / f"{key}.json"

    def _load(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _store(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{key}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.write("\n")
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    async def acall(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        key = request_key(self.model, messages, tools, kwargs, output_model=output_model)
        cached = self._load(key)
        if cached is not None and self._mode in {"replay", "auto"}:
            with self._lock:
                self.hits += 1
            return response_from_payload(cached, output_model=output_model)
        if self._mode == "replay":
            with self._lock:
                self.replay_misses += 1
            raise CacheMissError(f"no cached response for request {key[:12]} in {self._dir}")
        response = await self._inner.acall(messages, tools=tools, output_model=output_model, **kwargs)
        self._store(key, response_to_payload(response))
        with self._lock:
            self.misses_recorded += 1
        return response

    def call(self, messages: list[dict], tools=None, output_model=None, **kwargs) -> Any:
        key = request_key(self.model, messages, tools, kwargs, output_model=output_model)
        cached = self._load(key)
        if cached is not None and self._mode in {"replay", "auto"}:
            with self._lock:
                self.hits += 1
            return response_from_payload(cached, output_model=output_model)
        if self._mode == "replay":
            with self._lock:
                self.replay_misses += 1
            raise CacheMissError(f"no cached response for request {key[:12]} in {self._dir}")
        response = self._inner.call(messages, tools=tools, output_model=output_model, **kwargs)
        self._store(key, response_to_payload(response))
        with self._lock:
            self.misses_recorded += 1
        return response

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def get_model_info(self) -> Any:
        return self._inner.get_model_info()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses_recorded": self.misses_recorded,
                "replay_misses": self.replay_misses,
            }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_with_cache(client: Any, cache_dir: Path | str | None, mode: str | None) -> Any:
    """Identity when disabled; CachedLLM otherwise."""

    if not cache_dir or not mode or mode == "off":
        return client
    return CachedLLM(client, Path(cache_dir), mode)


_CACHE_MODE_ENV = "NOAH_CODE_LLM_CACHE"
_CACHE_DIR_ENV = "NOAH_CODE_LLM_CACHE_DIR"


def resolve_cache_settings(env: dict[str, str] | None = None) -> tuple[str | None, Path | None]:
    """Mode+dir from environment; invalid values disable rather than crash."""

    source = os.environ if env is None else env
    mode = (source.get(_CACHE_MODE_ENV) or "").strip().lower() or None
    directory = (source.get(_CACHE_DIR_ENV) or "").strip() or None
    if mode not in {None, "", "record", "replay", "auto"}:
        return None, None
    if bool(directory) != bool(mode):
        return None, None
    if mode is None or directory is None:
        return None, None
    return mode, Path(directory)


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(text: str, limit: int = 60) -> str:
    return _SLUG_RE.sub("-", text.strip())[:limit].strip("-") or "session"
