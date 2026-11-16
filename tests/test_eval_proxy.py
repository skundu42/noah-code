"""Cost reservations must hold even across concurrent retries and failures."""

from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_spec = spec_from_file_location("eval_proxy", Path(__file__).parents[1] / "scripts/eval_proxy.py")
assert _spec is not None and _spec.loader is not None
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
EvalProxy = _module.EvalProxy


def payload():
    return {"model": "selected/model", "messages": [{"role": "user", "content": "Fix it"}]}


def test_reserves_full_output_and_enforces_provider_prices():
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6)
    request, record = proxy.reserve({**payload(), "max_tokens": 1, "stream": True})
    assert request["max_completion_tokens"] == 4096
    assert "max_tokens" not in request
    assert request["provider"] == {
        "require_parameters": True,
        "max_price": {"prompt": 1.0, "completion": 2.0, "request": 0},
    }
    assert request["stream_options"] == {"include_usage": True}
    assert record["reserved_usd"] > 4096 * 2e-6
    assert proxy.summary()["reported_cost_usd"] is None
    assert "test-secret" not in str(proxy.summary())


@pytest.mark.parametrize("changes", [
    {"model": "unmetered/model"}, {"plugins": [{"id": "web"}]},
    {"tools": [{"type": "web_search"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}]},
])
def test_rejects_unmetered_requests(changes):
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6)
    with pytest.raises(ValueError):
        proxy.reserve({**payload(), **changes})
    assert proxy.summary()["reserved_usd"] == 0


def test_concurrent_requests_cannot_overspend():
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6, budget_usd=.1)

    def attempt(_):
        try:
            proxy.reserve(payload())
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(32)))
    summary = proxy.summary()
    assert 0 < sum(results) < 32
    assert summary["reserved_usd"] <= .1
    assert len(summary["requests"]) == sum(results)


def test_usage_from_sse_with_keepalive_comments():
    record = {}
    EvalProxy._capture_usage(
        b': OPENROUTER PROCESSING\n\ndata: {"choices": []}\n\n'
        b'data: {"usage": {"cost": 0.001, "completion_tokens": 5}}\n\ndata: [DONE]\n', record,
    )
    assert record["usage"]["cost"] == .001


def test_equal_run_limits_do_not_share_or_reset_total_budget():
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6, budget_usd=.1,
                      provider="verified/endpoint")
    proxy.begin_run("noah", budget_usd=.05, max_requests=1)
    request, _ = proxy.reserve(payload())
    assert request["provider"]["only"] == ["verified/endpoint"]
    assert request["provider"]["allow_fallbacks"] is False
    with pytest.raises(ValueError, match="per-run"):
        proxy.reserve(payload())
    before = proxy.summary()["reserved_usd"]
    proxy.begin_run("opencode", budget_usd=.05, max_requests=1)
    proxy.reserve(payload())
    assert proxy.summary()["reserved_usd"] == pytest.approx(before * 2)
    assert [row["run"] for row in proxy.summary()["requests"]] == ["noah", "opencode"]


def test_accepts_null_tools_and_preserves_tail_usage_over_2mb():
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6)
    request, _ = proxy.reserve({**payload(), "tools": None})
    assert request["tools"] is None

    proxy_two = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6, max_output_tokens=16)
    with pytest.raises(ValueError, match="output cap"):
        EvalProxy("selected/model", "test-secret", 1e-6, 2e-6, max_output_tokens=True)  # type: ignore[arg-type]

    chunks = [b"data: " + b"x" * 600_000 + b"\n\n" for _ in range(4)]
    tail = b'data: {"usage": {"cost": 0.0042, "completion_tokens": 9}}\n\n'
    buf = bytearray()
    for chunk in [*chunks, tail]:
        if len(chunk) >= 2_000_000:
            buf = bytearray(chunk[-2_000_000:])
        else:
            excess = len(buf) + len(chunk) - 2_000_000
            if excess > 0:
                del buf[:excess]
            buf.extend(chunk)
    record: dict[str, object] = {}
    proxy_two._capture_usage(bytes(buf), record)
    assert record["usage"] == {"cost": 0.0042, "completion_tokens": 9}


def test_large_multi_file_body_cap_and_cli_override():
    assert _module._MAX_BODY == 4_000_000
    proxy = EvalProxy("selected/model", "test-secret", 1e-8, 2e-8, budget_usd=4.0)
    large_text = "x" * 600_000  # > old 256 KB limit, < 4 MB
    _, record = proxy.reserve(
        {"model": "selected/model", "messages": [{"role": "user", "content": large_text}]}
    )
    assert record["reserved_usd"] > 0

    tight = EvalProxy(
        "selected/model",
        "test-secret",
        1e-8,
        2e-8,
        max_body_bytes=4_096,
    )
    with pytest.raises(ValueError, match="size limit"):
        tight.reserve(
            {"model": "selected/model", "messages": [{"role": "user", "content": "x" * 5_000}]}
        )

    parsed = _module.build_arg_parser().parse_args(
        [
            "--model",
            "selected/model",
            "--input-price-per-token",
            "0.000001",
            "--output-price-per-token",
            "0.000002",
            "--max-body-bytes",
            "2000000",
        ]
    )
    assert parsed.max_body_bytes == 2_000_000


def test_reconciles_unused_reservation_on_http_200_and_keeps_worst_case_on_failures():
    proxy = EvalProxy("selected/model", "test-secret", 1e-6, 2e-6, budget_usd=0.08)
    proxy.begin_run("noah", budget_usd=0.06, max_requests=4)

    _, first = proxy.reserve(payload())
    initial = first["reserved_usd"]
    assert initial > 0.03

    # Without reconciliation, a second request would exceed the $0.06 per-run budget.
    with pytest.raises(ValueError, match="per-run"):
        proxy.reserve(payload())

    # HTTP 200 with verified usage + cost refunds unused conservative delta.
    first["usage"] = {"prompt_tokens": 500, "completion_tokens": 100, "cost": 0.001}
    with proxy._lock:
        proxy._reconcile_reservation(first, http_status=200)
    assert first["reserved_usd"] == pytest.approx(0.001)
    assert first["refunded_usd"] == pytest.approx(initial - 0.001)
    assert proxy.summary()["reserved_usd"] == pytest.approx(0.001)

    # Now subsequent multi-turn requests in the same run succeed cleanly.
    _, second = proxy.reserve(payload())
    second_initial = second["reserved_usd"]

    # Non-200 or unknown-cost responses must retain 100% of worst-case reservation.
    second["usage"] = {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0002}
    with proxy._lock:
        proxy._reconcile_reservation(second, http_status=500)
    assert second["reserved_usd"] == pytest.approx(second_initial)

    second["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
    with proxy._lock:
        proxy._reconcile_reservation(second, http_status=200)
    assert second["reserved_usd"] == pytest.approx(second_initial)
    assert proxy.summary()["reserved_usd"] == pytest.approx(0.001 + second_initial)
