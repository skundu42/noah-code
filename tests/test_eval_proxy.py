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
