"""Privacy, cardinality, and signal tests for agent telemetry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from noah_code.config import TracingConfig
from noah_code.telemetry import (
    AgentTelemetry,
    PrivacyFilteringSpanExporter,
    _signal_endpoint,
    _signal_otlp_configured,
    filtered_resource_attributes,
    filtered_span_attributes,
)


class FakeInstrument:
    def __init__(self) -> None:
        self.measurements: list[tuple[float, dict[str, object]]] = []

    def record(self, value: float, attributes: dict[str, object]) -> None:
        self.measurements.append((value, attributes))

    def add(self, value: float, attributes: dict[str, object]) -> None:
        self.measurements.append((value, attributes))


def test_default_tracing_config_is_privacy_first() -> None:
    config = TracingConfig()

    assert config.enabled is True
    assert config.jsonl_enabled is True
    assert config.capture_content is False
    assert config.logs_enabled is True
    assert config.metrics_enabled is True
    assert config.otlp_endpoint is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "collector:4318",
        "ftp://collector.example",
        "http://collector.example:4318",
        "https://user:secret@example.com",
        "https://collector.example/v1/traces?token=secret",
    ],
)
def test_otlp_endpoint_rejects_unsafe_or_ambiguous_urls(endpoint: str) -> None:
    with pytest.raises(ValueError):
        TracingConfig(otlp_endpoint=endpoint)


def test_signal_endpoint_accepts_base_or_signal_url() -> None:
    assert _signal_endpoint("http://collector:4318", "traces") == (
        "http://collector:4318/v1/traces"
    )
    assert _signal_endpoint("https://otel.example/root/v1/traces", "metrics") == (
        "https://otel.example/root/v1/metrics"
    )


def test_signal_specific_environment_only_enables_that_exporter(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:4318/v1/metrics")
    config = TracingConfig(jsonl_enabled=False)

    assert _signal_otlp_configured(config, "metrics") is True
    assert _signal_otlp_configured(config, "traces") is False
    assert _signal_otlp_configured(config, "logs") is False


def test_resource_filter_removes_host_and_repository_identity() -> None:
    filtered = filtered_resource_attributes(
        {
            "service.name": "noah-code",
            "hostname": "private-host",
            "host.id": "private-id",
            "git.commit": "private-repository-state",
            "python.version": "3.13",
        },
        max_attributes=32,
        max_attribute_length=256,
    )

    assert filtered == {"service.name": "noah-code", "python.version": "3.13"}


def test_span_filter_removes_agent_content_and_preserves_operational_fields() -> None:
    attributes = {
        "session.id": "session-1",
        "gen_ai.operation.name": "invoke_agent",
        "input.value": "private prompt",
        "output.value": "private answer",
        "llm.input_messages.0.message.content": "private history",
        "tool_call.function.arguments": '{"password":"secret"}',
        "exception.stacktrace": "private path and data",
        "gen_ai.usage.input_tokens": 42,
    }

    filtered = filtered_span_attributes(
        attributes,
        capture_content=False,
        max_attributes=32,
        max_attribute_length=256,
    )

    assert filtered == {
        "session.id": "session-1",
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.usage.input_tokens": 42,
    }


def test_opt_in_content_is_still_bounded_and_priority_attributes_survive() -> None:
    attributes = {f"custom.{index}": "x" * 100 for index in range(40)}
    attributes["session.id"] = "session-1"
    attributes["input.value"] = "private prompt"

    filtered = filtered_span_attributes(
        attributes,
        capture_content=True,
        max_attributes=32,
        max_attribute_length=16,
    )

    assert len(filtered) == 32
    assert filtered["session.id"] == "session-1"
    assert all(not isinstance(value, str) or len(value) <= 16 for value in filtered.values())


def test_exporter_filters_without_mutating_original_span() -> None:
    class InnerExporter:
        def __init__(self) -> None:
            self.spans = ()
            self.closed = False

        def export(self, spans):
            self.spans = spans
            return "ok"

        def shutdown(self):
            self.closed = True

    original = SimpleNamespace(
        name="chat",
        attributes={"input.value": "secret", "safe": 1},
        events=(
            SimpleNamespace(
                name="exception",
                attributes={"exception.message": "secret failure", "error.type": "ValueError"},
            ),
        ),
        links=(SimpleNamespace(attributes={"input.value": "secret link", "safe": 2}),),
        status=SimpleNamespace(status_code="ERROR", description="secret failure"),
        resource=SimpleNamespace(
            attributes={"service.name": "noah-code", "hostname": "private-host"},
            schema_url=None,
        ),
    )
    inner = InnerExporter()
    exporter = PrivacyFilteringSpanExporter(
        inner,
        capture_content=False,
        max_attributes=32,
        max_attribute_length=256,
    )

    assert exporter.export((original,)) == "ok"
    assert inner.spans[0].attributes == {"safe": 1}
    assert inner.spans[0].events[0].attributes == {"error.type": "ValueError"}
    assert inner.spans[0].links[0].attributes == {"safe": 2}
    assert inner.spans[0].status.description is None
    assert dict(inner.spans[0].resource.attributes) == {"service.name": "noah-code"}
    assert original.attributes["input.value"] == "secret"
    exporter.shutdown()
    assert inner.closed is True


def test_llm_metrics_use_standard_names_and_low_cardinality_attributes() -> None:
    telemetry = AgentTelemetry()
    telemetry.enabled = True
    telemetry.session_id = "session-sensitive-cardinality"
    telemetry.model = "openai/gpt-test"
    telemetry._llm_duration = FakeInstrument()
    telemetry._token_usage = FakeInstrument()
    telemetry._cost = FakeInstrument()
    event = SimpleNamespace(generation_id="generation-1", turn_number=1)

    telemetry.llm_start(event)
    telemetry.llm_end(SimpleNamespace(**vars(event), success=True, exception_type=None))
    telemetry.llm_complete(
        SimpleNamespace(
            **vars(event),
            model_name="openai/gpt-test",
            prompt_tokens=100,
            completion_tokens=25,
            cached_tokens=40,
            reasoning_tokens=5,
            cost_usd=0.01,
        )
    )

    assert len(telemetry._llm_duration.measurements) == 1
    token_measurements = telemetry._token_usage.measurements
    assert [item[0] for item in token_measurements] == [100, 25]
    assert {item[1]["gen_ai.token.type"] for item in token_measurements} == {
        "input",
        "output",
    }
    for _value, attributes in [*token_measurements, *telemetry._cost.measurements]:
        assert "gen_ai.conversation.id" not in attributes
        assert "noah.run.id" not in attributes
        assert attributes["gen_ai.provider.name"] == "openai"
