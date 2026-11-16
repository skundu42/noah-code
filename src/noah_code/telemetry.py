"""Privacy-first OpenTelemetry for Noah's agent lifecycle.

NOOA owns detailed OpenInference spans for model and tool execution. This
module adds Noah's orchestration boundary, standard GenAI metrics, structured
OTLP events, and a final exporter-side privacy/size guard.
"""

from __future__ import annotations

import contextlib
import io
import os
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from noah_code import __version__
from noah_code.redaction import redact_secrets

if TYPE_CHECKING:
    from noah_code.config import TracingConfig

_INSTRUMENTATION_NAME = "noah_code.agent"
_AGENT_NAME = "noah-code"

_CONTENT_ATTRIBUTES = frozenset(
    {
        "agent.args",
        "agent.kwargs",
        "agent.result",
        "error.message",
        "exception.message",
        "exception.stacktrace",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.tool.definitions",
        "input.value",
        "output.value",
        "tool.arguments",
        "tool.result",
        "tool_call.function.arguments",
        "tool_call.function.output",
    }
)
_CONTENT_PREFIXES = (
    "llm.input_messages.",
    "llm.output_messages.",
    "llm.prompt_template.variables.",
)
_RESOURCE_DENY_ATTRIBUTES = frozenset({"host.id", "host.name", "hostname"})
_RESOURCE_DENY_PREFIXES = ("git.",)
_PRIORITY_ATTRIBUTES = (
    "session.id",
    "gen_ai.conversation.id",
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.agent.name",
    "gen_ai.agent.id",
    "openinference.span.kind",
    "error.type",
    "tool.name",
    "gen_ai.tool.name",
    "gen_ai.tool.call.id",
    "noah.run.id",
)
_TOKEN_BUCKETS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def _is_content_attribute(name: str) -> bool:
    return name in _CONTENT_ATTRIBUTES or name.startswith(_CONTENT_PREFIXES)


def _bounded_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        cleaned = redact_secrets(value)
        return cleaned if len(cleaned) <= limit else cleaned[: max(limit - 1, 0)] + "…"
    if isinstance(value, tuple):
        return tuple(_bounded_value(item, limit) for item in value)
    if isinstance(value, list):
        return [_bounded_value(item, limit) for item in value]
    return value


def filtered_span_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    capture_content: bool,
    max_attributes: int,
    max_attribute_length: int,
) -> dict[str, Any]:
    """Apply the final content denylist and OTel attribute bounds."""

    if not attributes:
        return {}
    eligible = {
        str(key): _bounded_value(value, max_attribute_length)
        for key, value in attributes.items()
        if capture_content or not _is_content_attribute(str(key))
    }
    if len(eligible) <= max_attributes:
        return eligible
    ordered: dict[str, Any] = {}
    for key in _PRIORITY_ATTRIBUTES:
        if key in eligible:
            ordered[key] = eligible[key]
    for key, value in eligible.items():
        if len(ordered) >= max_attributes:
            break
        ordered.setdefault(key, value)
    return ordered


def filtered_resource_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    max_attributes: int,
    max_attribute_length: int,
) -> dict[str, Any]:
    """Bound resource metadata and remove host/repository identifiers."""

    if not attributes:
        return {}
    filtered = (
        (str(key), _bounded_value(value, max_attribute_length))
        for key, value in attributes.items()
        if str(key) not in _RESOURCE_DENY_ATTRIBUTES
        and not str(key).startswith(_RESOURCE_DENY_PREFIXES)
    )
    return dict(list(filtered)[:max_attributes])


class _FilteredReadableSpan:
    def __init__(
        self,
        span: Any,
        attributes: Mapping[str, Any],
        *,
        capture_content: bool,
        max_attributes: int,
        max_attribute_length: int,
    ) -> None:
        from opentelemetry.sdk.resources import Resource

        self._span = span
        self.name = _bounded_value(getattr(span, "name", ""), max_attribute_length)
        self.attributes = attributes
        self.events = tuple(
            _FilteredAttributeCarrier(
                event,
                capture_content=capture_content,
                max_attributes=max_attributes,
                max_attribute_length=max_attribute_length,
                bound_name=True,
            )
            for event in getattr(span, "events", ())
        )
        self.links = tuple(
            _FilteredAttributeCarrier(
                link,
                capture_content=capture_content,
                max_attributes=max_attributes,
                max_attribute_length=max_attribute_length,
            )
            for link in getattr(span, "links", ())
        )
        status = getattr(span, "status", None)
        self.status = _FilteredStatus(
            status,
            capture_content=capture_content,
            max_attribute_length=max_attribute_length,
        )
        resource = getattr(span, "resource", None)
        self.resource = Resource(
            attributes=filtered_resource_attributes(
                getattr(resource, "attributes", None),
                max_attributes=max_attributes,
                max_attribute_length=max_attribute_length,
            ),
            schema_url=getattr(resource, "schema_url", None),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class _FilteredAttributeCarrier:
    def __init__(
        self,
        value: Any,
        *,
        capture_content: bool,
        max_attributes: int,
        max_attribute_length: int,
        bound_name: bool = False,
    ) -> None:
        self._value = value
        self.attributes = filtered_span_attributes(
            getattr(value, "attributes", None),
            capture_content=capture_content,
            max_attributes=max_attributes,
            max_attribute_length=max_attribute_length,
        )
        if bound_name:
            self.name = _bounded_value(getattr(value, "name", ""), max_attribute_length)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)


class _FilteredStatus:
    def __init__(
        self,
        status: Any,
        *,
        capture_content: bool,
        max_attribute_length: int,
    ) -> None:
        self._status = status
        self.status_code = getattr(status, "status_code", None)
        description = getattr(status, "description", None)
        self.description = (
            _bounded_value(description, max_attribute_length) if capture_content else None
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._status, name)


class PrivacyFilteringSpanExporter:
    """Exporter adapter that keeps all trace destinations privacy-equivalent."""

    synchronous = False

    def __init__(
        self,
        inner: Any,
        *,
        capture_content: bool,
        max_attributes: int,
        max_attribute_length: int,
    ) -> None:
        self.inner = inner
        self.capture_content = capture_content
        self.max_attributes = max_attributes
        self.max_attribute_length = max_attribute_length

    def export(self, spans: Sequence[Any]) -> Any:
        filtered = tuple(
            _FilteredReadableSpan(
                span,
                filtered_span_attributes(
                    getattr(span, "attributes", None),
                    capture_content=self.capture_content,
                    max_attributes=self.max_attributes,
                    max_attribute_length=self.max_attribute_length,
                ),
                capture_content=self.capture_content,
                max_attributes=self.max_attributes,
                max_attribute_length=self.max_attribute_length,
            )
            for span in spans
        )
        return self.inner.export(filtered)

    def shutdown(self) -> Any:
        return self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        flush = getattr(self.inner, "force_flush", None)
        return bool(flush(timeout_millis)) if callable(flush) else True

    def describe(self) -> str:
        describe = getattr(self.inner, "describe", None)
        target = describe() if callable(describe) else type(self.inner).__name__
        content = "content=bounded" if self.capture_content else "content=off"
        return f"{target} ({content})"


def _signal_endpoint(base: str, signal: str) -> str:
    """Turn an OTLP/HTTP base endpoint into a signal-specific endpoint."""

    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    for known in ("traces", "metrics", "logs"):
        suffix = f"/v1/{known}"
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    path = f"{path}/v1/{signal}" if path else f"/v1/{signal}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _otlp_configured(config: TracingConfig) -> bool:
    return any(_signal_otlp_configured(config, signal) for signal in ("traces", "metrics", "logs"))


def _signal_otlp_configured(config: TracingConfig, signal: str) -> bool:
    common = config.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    return bool(common or os.getenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT"))


def _provider_name(model: str) -> str:
    prefix = model.partition("/")[0].lower().replace("_", ".")
    aliases = {
        "azure": "azure.ai.openai",
        "bedrock": "aws.bedrock",
        "gemini": "gcp.gen_ai",
        "vertex.ai": "gcp.vertex_ai",
    }
    return aliases.get(prefix, prefix or "unknown")


@dataclass
class TurnObservation:
    span: Any
    context_token: Any
    started_at: float
    mode: str
    recovery: bool


class AgentTelemetry:
    """Best-effort telemetry facade; failures never affect agent execution."""

    def __init__(self) -> None:
        self.enabled = False
        self.info = "disabled"
        self.session_id = ""
        self.model = ""
        self.capture_content = False
        self._tracer: Any = None
        self._logger: Any = None
        self._logger_provider: Any = None
        self._meter_provider: Any = None
        self._registered_exporters: list[Any] = []
        self._capture_max_attributes = 128
        self._capture_max_attribute_length = 4096
        self._llm_started: dict[tuple[str, int], deque[float]] = {}
        self._llm_finished: dict[tuple[str, int], deque[tuple[float, bool, str]]] = {}
        self._tool_started: dict[str, tuple[float, str]] = {}
        self._turn_duration: Any = None
        self._llm_duration: Any = None
        self._token_usage: Any = None
        self._cost: Any = None
        self._retries: Any = None
        self._tool_count: Any = None
        self._tool_duration: Any = None

    def start_turn(
        self,
        *,
        run_id: str | None,
        mode: str,
        recovery: bool,
    ) -> TurnObservation | None:
        if not self.enabled or self._tracer is None:
            return None
        try:
            from opentelemetry import context, trace
            from opentelemetry.trace import SpanKind

            attrs: dict[str, Any] = {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": _AGENT_NAME,
                "gen_ai.conversation.id": self.session_id,
                "noah.agent.mode": mode,
                "noah.agent.recovery": recovery,
            }
            if run_id:
                attrs["noah.run.id"] = run_id
            span = self._tracer.start_span(
                f"invoke_agent {_AGENT_NAME}",
                kind=SpanKind.INTERNAL,
                attributes=attrs,
            )
            token = context.attach(trace.set_span_in_context(span))
            observation = TurnObservation(span, token, time.perf_counter(), mode, recovery)
            self.event(
                "gen_ai.agent.invocation.started",
                {"gen_ai.operation.name": "invoke_agent", "noah.agent.mode": mode},
            )
            return observation
        except Exception:
            return None

    def end_turn(
        self,
        observation: TurnObservation | None,
        *,
        outcome: str,
        error_type: str = "",
    ) -> None:
        if observation is None:
            return
        duration = max(time.perf_counter() - observation.started_at, 0.0)
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": _AGENT_NAME,
            "noah.agent.mode": observation.mode,
            "noah.agent.outcome": outcome,
            "noah.agent.recovery": observation.recovery,
        }
        if error_type:
            attrs["error.type"] = error_type
        try:
            from opentelemetry.trace import Status, StatusCode

            observation.span.set_attribute("noah.agent.outcome", outcome)
            if error_type:
                observation.span.set_attribute("error.type", error_type)
                observation.span.set_status(Status(StatusCode.ERROR))
            elif outcome == "completed":
                observation.span.set_status(Status(StatusCode.OK))
            self.event("gen_ai.agent.invocation.completed", {**attrs, "duration": duration})
            if self._turn_duration is not None:
                self._turn_duration.record(duration, attrs)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                observation.span.end()
            with contextlib.suppress(Exception):
                from opentelemetry import context

                context.detach(observation.context_token)

    def llm_start(self, event: Any) -> None:
        key = self._event_key(event)
        self._llm_started.setdefault(key, deque()).append(time.perf_counter())
        self.event(
            "gen_ai.client.inference.started",
            {
                "gen_ai.operation.name": "chat",
                "noah.generation.id": key[0],
                "noah.generation.turn": key[1],
            },
        )

    def llm_end(self, event: Any) -> None:
        key = self._event_key(event)
        starts = self._llm_started.get(key)
        if not starts:
            return
        started = starts.popleft()
        if not starts:
            self._llm_started.pop(key, None)
        success = bool(getattr(event, "success", True))
        error_type = str(getattr(event, "exception_type", "") or "")
        self._llm_finished.setdefault(key, deque()).append(
            (max(time.perf_counter() - started, 0.0), success, error_type)
        )
        if not success:
            self._record_llm_completion(key, self.model, 0, 0, 0, 0, 0.0)

    def llm_complete(self, event: Any) -> None:
        key = self._event_key(event)
        usage = getattr(event, "usage", None)

        def _pick(*names: str, default: Any = 0) -> Any:
            for source in (usage, event):
                if source is None:
                    continue
                for name in names:
                    value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
                    if value is not None:
                        return value
            return default

        self._record_llm_completion(
            key,
            str(getattr(event, "model_name", "") or self.model),
            int(_pick("input_tokens", "prompt_tokens") or 0),
            int(_pick("output_tokens", "completion_tokens") or 0),
            int(_pick("cached_input_tokens", "cached_tokens") or 0),
            int(_pick("reasoning_tokens") or 0),
            float(_pick("cost_usd", "cost", default=0.0) or 0.0),
        )

    def retry(self, payload: Mapping[str, Any]) -> None:
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": str(payload.get("model", self.model)),
            "gen_ai.provider.name": _provider_name(str(payload.get("model", self.model))),
            "error.type": str(payload.get("error", "unknown")),
            "noah.retry.attempt": int(payload.get("attempt", 0) or 0),
            "noah.retry.fallback_index": int(payload.get("fallback_index", 0) or 0),
            "noah.retry.delay": float(payload.get("delay_seconds", 0.0) or 0.0),
        }
        try:
            if self._retries is not None:
                self._retries.add(1, attrs)
            self.event("gen_ai.client.inference.retry", attrs)
        except Exception:
            pass

    def tool_start(self, tool_name: str, call_id: str) -> None:
        if call_id:
            self._tool_started[call_id] = (time.perf_counter(), tool_name)
        self.event(
            "gen_ai.tool.execution.started",
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.call.id": call_id,
            },
        )

    def tool_finish(self, tool_name: str, call_id: str, status: str) -> None:
        started = self._tool_started.pop(call_id, None)
        if started is not None:
            duration = max(time.perf_counter() - started[0], 0.0)
            tool_name = started[1]
        else:
            duration = 0.0
        outcome = "error" if status in {"error", "failed", "fail"} else "ok"
        attrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "noah.tool.outcome": outcome,
        }
        try:
            if self._tool_count is not None:
                self._tool_count.add(1, attrs)
            if self._tool_duration is not None and started is not None:
                self._tool_duration.record(duration, attrs)
            self.event(
                "gen_ai.tool.execution.completed",
                {**attrs, "gen_ai.tool.call.id": call_id, "duration": duration},
            )
        except Exception:
            pass

    def event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        """Emit a trace event and an OTLP log with the current trace context."""

        if not self.enabled:
            return
        raw_attrs = dict(attributes or {})
        raw_attrs.setdefault("gen_ai.agent.name", _AGENT_NAME)
        raw_attrs.setdefault("gen_ai.conversation.id", self.session_id)
        attrs = filtered_span_attributes(
            raw_attrs,
            capture_content=self.capture_content,
            max_attributes=self._capture_max_attributes,
            max_attribute_length=self._capture_max_attribute_length,
        )
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span.is_recording():
                span.add_event(name, attributes=attrs)
        except Exception:
            pass
        if self._logger is None:
            return
        try:
            from opentelemetry import context
            from opentelemetry._logs import LogRecord, SeverityNumber

            is_error = bool(attrs.get("error.type"))
            self._logger.emit(
                LogRecord(
                    timestamp=time.time_ns(),
                    observed_timestamp=time.time_ns(),
                    context=context.get_current(),
                    severity_text="ERROR" if is_error else "INFO",
                    severity_number=SeverityNumber.ERROR if is_error else SeverityNumber.INFO,
                    body=name,
                    attributes=attrs,
                    event_name=name,
                )
            )
        except Exception:
            pass

    def shutdown(self, timeout_millis: int = 5_000) -> None:
        if self.session_id:
            _unregister_session_exporters(self.session_id, self._registered_exporters, timeout_millis)
        for provider in (self._logger_provider, self._meter_provider):
            if provider is None:
                continue
            with contextlib.suppress(Exception):
                provider.force_flush(timeout_millis)
            with contextlib.suppress(Exception):
                provider.shutdown()

    @staticmethod
    def _event_key(event: Any) -> tuple[str, int]:
        return (
            str(getattr(event, "generation_id", "") or ""),
            int(getattr(event, "turn_number", 0) or 0),
        )

    def _record_llm_completion(
        self,
        key: tuple[str, int],
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
        cost_usd: float,
    ) -> None:
        finished = self._llm_finished.get(key)
        if finished:
            duration, success, error_type = finished.popleft()
            if not finished:
                self._llm_finished.pop(key, None)
        else:
            duration, success, error_type = (0.0, True, "")
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": _provider_name(model),
            "gen_ai.request.model": model,
        }
        if not success:
            attrs["error.type"] = error_type or "unknown"
        try:
            if self._llm_duration is not None and duration:
                self._llm_duration.record(duration, attrs)
            if self._token_usage is not None:
                if input_tokens:
                    self._token_usage.record(input_tokens, {**attrs, "gen_ai.token.type": "input"})
                if output_tokens:
                    self._token_usage.record(
                        output_tokens, {**attrs, "gen_ai.token.type": "output"}
                    )
            if self._cost is not None and cost_usd > 0:
                self._cost.add(cost_usd, attrs)
            self.event(
                "gen_ai.client.inference.completed",
                {
                    **attrs,
                    "duration": duration,
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                    "gen_ai.usage.cache_read.input_tokens": cached_tokens,
                    "gen_ai.usage.reasoning.output_tokens": reasoning_tokens,
                    "noah.gen_ai.cost": cost_usd,
                },
            )
        except Exception:
            pass


_ROUTER_LOCK = threading.Lock()
_SESSION_EXPORTERS: dict[str, list[Any]] = {}
_ROUTER_EXPORTER: Any = None


class _SessionRoutingSpanExporter:
    """Route spans by session.id so concurrent sessions keep separate JSONL sinks."""

    synchronous = False

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        buckets: dict[str, list[Any]] = {}
        with _ROUTER_LOCK:
            snapshot = {sid: list(items) for sid, items in _SESSION_EXPORTERS.items()}
        if not snapshot:
            return SpanExportResult.SUCCESS
        fallback_sid = next(iter(snapshot)) if len(snapshot) == 1 else ""
        for span in spans:
            attrs = getattr(span, "attributes", None) or {}
            sid = str(attrs.get("session.id") or attrs.get("gen_ai.conversation.id") or "")
            if sid not in snapshot:
                sid = fallback_sid
            if sid:
                buckets.setdefault(sid, []).append(span)
        result = SpanExportResult.SUCCESS
        for sid, group in buckets.items():
            for exporter in snapshot.get(sid, ()):
                try:
                    exported = exporter.export(group)
                    if exported is not None and exported != SpanExportResult.SUCCESS:
                        result = exported
                except Exception:  # noqa: BLE001
                    result = SpanExportResult.FAILURE
        return result

    def shutdown(self) -> None:
        with _ROUTER_LOCK:
            all_exporters = [exp for items in _SESSION_EXPORTERS.values() for exp in items]
            _SESSION_EXPORTERS.clear()
        for exporter in all_exporters:
            with contextlib.suppress(Exception):
                exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        with _ROUTER_LOCK:
            all_exporters = [exp for items in _SESSION_EXPORTERS.values() for exp in items]
        ok = True
        for exporter in all_exporters:
            flush = getattr(exporter, "force_flush", None)
            if callable(flush):
                with contextlib.suppress(Exception):
                    ok = bool(flush(timeout_millis)) and ok
        return ok


def _unregister_session_exporters(
    session_id: str, expected: list[Any], timeout_millis: int
) -> None:
    with _ROUTER_LOCK:
        registered = _SESSION_EXPORTERS.get(session_id)
        if registered is not None and (not expected or registered == expected):
            exporters = _SESSION_EXPORTERS.pop(session_id)
        else:
            exporters = []
    for exporter in exporters:
        flush = getattr(exporter, "force_flush", None)
        if callable(flush):
            with contextlib.suppress(Exception):
                flush(timeout_millis)
        with contextlib.suppress(Exception):
            exporter.shutdown()


def setup_agent_telemetry(
    config: TracingConfig,
    *,
    session_id: str,
    model: str,
    jsonl_dir: str | None,
) -> AgentTelemetry:
    """Configure NOOA traces plus Noah logs/metrics. Never raises."""

    global _ROUTER_EXPORTER
    telemetry = AgentTelemetry()
    telemetry.session_id = session_id
    telemetry.model = model
    telemetry.capture_content = config.capture_content
    telemetry._capture_max_attributes = config.max_span_attributes
    telemetry._capture_max_attribute_length = config.max_attribute_length
    if not config.enabled:
        return telemetry

    try:
        from nooa.tracing import enable_tracing, exporters, set_session

        set_session(session_id)
        destinations: list[str] = []
        span_exporters: list[Any] = []
        if config.jsonl_enabled and jsonl_dir:
            local = exporters.jsonl(trace_dir=jsonl_dir)
            span_exporters.append(_filtered_exporter(local, config))
            destinations.append(f"jsonl={jsonl_dir}/{session_id}.jsonl")

        otlp_enabled = _otlp_configured(config)
        resource = filtered_resource_attributes(
            {
                "service.name": _AGENT_NAME,
                "service.version": __version__,
                "gen_ai.agent.name": _AGENT_NAME,
            },
            max_attributes=config.max_span_attributes,
            max_attribute_length=config.max_attribute_length,
        )
        if _signal_otlp_configured(config, "traces"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                trace_endpoint = (
                    _signal_endpoint(config.otlp_endpoint, "traces")
                    if config.otlp_endpoint
                    else None
                )
                remote = OTLPSpanExporter(endpoint=trace_endpoint)
                span_exporters.append(_filtered_exporter(remote, config))
                destinations.append(f"otlp={config.otlp_endpoint or 'environment'}")
            except Exception as exc:
                destinations.append(f"otlp-traces=unavailable({type(exc).__name__})")

        if span_exporters:
            with _ROUTER_LOCK:
                previous = _SESSION_EXPORTERS.pop(session_id, [])
                _SESSION_EXPORTERS[session_id] = span_exporters
                install_router = _ROUTER_EXPORTER is None
                if install_router:
                    _ROUTER_EXPORTER = _SessionRoutingSpanExporter()
                router = _ROUTER_EXPORTER
            for old in previous:
                with contextlib.suppress(Exception):
                    old.shutdown()
            telemetry._registered_exporters = span_exporters
            if install_router:
                with contextlib.redirect_stdout(io.StringIO()):
                    enable_tracing(exporters=[router], extra_resource_attrs=resource)
            # First-time NOOA setup installs a generated default session after
            # exporters are configured. Restore Noah's durable session route.
            set_session(session_id)

        from opentelemetry import trace

        telemetry._tracer = trace.get_tracer(_INSTRUMENTATION_NAME, __version__)
        content_status = "bounded" if config.capture_content else "off"
        telemetry.info = f"{'; '.join(destinations) or 'disabled'}; content={content_status}"
        if otlp_enabled:
            with contextlib.suppress(Exception):
                _setup_otlp_logs_and_metrics(telemetry, config, resource)
        telemetry.enabled = bool(
            span_exporters or telemetry._logger_provider or telemetry._meter_provider
        )
        return telemetry
    except Exception as exc:
        telemetry.info = f"unavailable ({type(exc).__name__})"
        return telemetry


def _filtered_exporter(inner: Any, config: TracingConfig) -> PrivacyFilteringSpanExporter:
    return PrivacyFilteringSpanExporter(
        inner,
        capture_content=config.capture_content,
        max_attributes=config.max_span_attributes,
        max_attribute_length=config.max_attribute_length,
    )


def _setup_otlp_logs_and_metrics(
    telemetry: AgentTelemetry,
    config: TracingConfig,
    resource_attributes: Mapping[str, Any],
) -> None:
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(dict(resource_attributes))
    if config.logs_enabled and _signal_otlp_configured(config, "logs"):
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        endpoint = _signal_endpoint(config.otlp_endpoint, "logs") if config.otlp_endpoint else None
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=endpoint),
                schedule_delay_millis=1000,
                max_queue_size=2048,
                max_export_batch_size=256,
                export_timeout_millis=config.export_timeout_millis,
            )
        )
        telemetry._logger_provider = logger_provider
        telemetry._logger = logger_provider.get_logger(_INSTRUMENTATION_NAME, __version__)

    if config.metrics_enabled and _signal_otlp_configured(config, "metrics"):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.metrics.view import (
            ExplicitBucketHistogramAggregation,
            View,
        )

        endpoint = (
            _signal_endpoint(config.otlp_endpoint, "metrics") if config.otlp_endpoint else None
        )
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint),
            export_interval_millis=config.metric_export_interval_millis,
            export_timeout_millis=config.export_timeout_millis,
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
            views=[
                View(
                    instrument_name="gen_ai.client.token.usage",
                    aggregation=ExplicitBucketHistogramAggregation(_TOKEN_BUCKETS),
                ),
                View(
                    instrument_name="gen_ai.client.operation.duration",
                    aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS),
                ),
                View(
                    instrument_name="gen_ai.invoke_agent.duration",
                    aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS),
                ),
                View(
                    instrument_name="noah.agent.tool.duration",
                    aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS),
                ),
            ],
        )
        meter = meter_provider.get_meter(_INSTRUMENTATION_NAME, __version__)
        telemetry._meter_provider = meter_provider
        telemetry._turn_duration = meter.create_histogram(
            "gen_ai.invoke_agent.duration", unit="s", description="Agent invocation duration."
        )
        telemetry._llm_duration = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s", description="Model operation duration."
        )
        telemetry._token_usage = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}", description="Input and output tokens used."
        )
        telemetry._cost = meter.create_counter(
            "noah.gen_ai.cost", unit="USD", description="Estimated model cost."
        )
        telemetry._retries = meter.create_counter(
            "noah.agent.retries", unit="{retry}", description="Model retry attempts."
        )
        telemetry._tool_count = meter.create_counter(
            "noah.agent.tool.executions", unit="{execution}", description="Tool executions."
        )
        telemetry._tool_duration = meter.create_histogram(
            "noah.agent.tool.duration", unit="s", description="Tool execution duration."
        )
