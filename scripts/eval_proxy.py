"""Loopback-only, conservatively metered OpenRouter relay for live evaluations.

Each request reserves its worst-case token cost before dispatch. Failed,
interrupted, or unknown-cost requests retain their full worst-case reservation.
Completed HTTP 200 responses with valid usage and known provider cost reconcile
their unused conservative delta while enforcing metered token floors. Provider
routing enforces the same price ceilings. Only text chat/function tools are
accepted; paid server tools, media, fallback models and alternate routes are
rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

_MAX_BODY = 4_000_000
_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "response_format",
        "reasoning",
        "reasoning_effort",
        "frequency_penalty",
        "presence_penalty",
        "user",
        "store",
    }
)


class EvalProxy:
    def __init__(
        self,
        model: str,
        api_key: str,
        input_price_per_token: float,
        output_price_per_token: float,
        *,
        budget_usd: float = 4,
        max_output_tokens: int = 4096,
        max_body_bytes: int = _MAX_BODY,
        provider: str | None = None,
    ) -> None:
        prices = (input_price_per_token, output_price_per_token, budget_usd)
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in prices):
            raise ValueError("prices and budget must be finite and non-negative")
        if (
            not api_key
            or not model
            or not 0 < budget_usd <= 5
            or not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or not 1 <= max_output_tokens <= 8192
            or not isinstance(max_body_bytes, int)
            or isinstance(max_body_bytes, bool)
            or not 1_024 <= max_body_bytes <= 16_000_000
        ):
            raise ValueError("invalid evaluation model, key, budget, or output cap")
        self.model = model
        self.provider = provider
        self._api_key = api_key
        self.client_key = secrets.token_urlsafe(32)
        self._input_price = Decimal(str(input_price_per_token))
        self._output_price = Decimal(str(output_price_per_token))
        self._budget = Decimal(str(budget_usd))
        self._reserved = Decimal(0)
        self.max_output_tokens = max_output_tokens
        self.max_body_bytes = max_body_bytes
        self._lock = threading.Lock()
        self._requests: list[dict[str, Any]] = []
        self._record_reservations: dict[int, Decimal] = {}
        self._run: dict[str, Any] | None = None
        self._server: ThreadingHTTPServer | None = None
        self.base_url = ""

    def begin_run(self, label: str, *, budget_usd: float, max_requests: int = 12) -> None:
        """Allocate the same request and spend limits to each sequential agent run."""
        if not math.isfinite(budget_usd) or not 0 < budget_usd <= float(self._budget):
            raise ValueError("invalid per-run budget")
        if not label or not 1 <= max_requests <= 48:
            raise ValueError("invalid run label or request limit")
        with self._lock:
            self._run = {
                "label": label,
                "budget": Decimal(str(budget_usd)),
                "reserved": Decimal(0),
                "requests": 0,
                "max_requests": max_requests,
            }

    def reserve(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate and reserve atomically; useful independently of the HTTP relay."""
        if payload.get("model") != self.model or set(payload) - _FIELDS:
            raise ValueError("only the selected model and metered chat parameters are allowed")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages are required")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            content = message.get("content")
            if (
                content is not None
                and not isinstance(content, str)
                and (
                    not isinstance(content, list)
                    or any(
                        not isinstance(part, dict)
                        or part.get("type") != "text"
                        or not isinstance(part.get("text"), str)
                        for part in content
                    )
                )
            ):
                raise ValueError("only text content is allowed")
        tools = payload.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or any(
                not isinstance(tool, dict) or tool.get("type") != "function"
                for tool in tools
            )
        ):
            raise ValueError("paid server tools are not allowed")
        request = dict(payload)
        if "store" in request:
            if not isinstance(request["store"], bool):
                raise ValueError("store must be boolean")
            request["store"] = False
        request.pop("max_tokens", None)
        request["max_completion_tokens"] = self.max_output_tokens
        request["provider"] = {
            "require_parameters": True,
            "max_price": {
                "prompt": float(self._input_price * 1_000_000),
                "completion": float(self._output_price * 1_000_000),
                "request": 0,
            },
        }
        if self.provider:
            request["provider"]["only"] = [self.provider]
            request["provider"]["allow_fallbacks"] = False
        if request.get("stream"):
            request["stream_options"] = {"include_usage": True}
        # Serialized UTF-8 bytes overcount text tokens. Extra per-message/tool
        # allowance covers provider-added role/schema framing and reasoning fields.
        size = len(json.dumps(request, ensure_ascii=False).encode())
        if size > self.max_body_bytes:
            raise ValueError("request exceeds evaluation size limit")
        ceiling = (
            2 * size
            + 16_384
            + 1024 * (len(messages) + len(request.get("tools") or []))
        )
        reservation = ceiling * self._input_price + self.max_output_tokens * self._output_price
        # Reserve an extra 25% for rounding/account-dependent billing overhead.
        reservation *= Decimal("1.25")
        with self._lock:
            if self._reserved + reservation > self._budget:
                raise ValueError("evaluation request or cost reservation limit reached")
            if self._run is not None:
                if (
                    self._run["requests"] >= self._run["max_requests"]
                    or self._run["reserved"] + reservation > self._run["budget"]
                ):
                    raise ValueError("per-run evaluation budget reached")
                self._run["requests"] += 1
                self._run["reserved"] += reservation
            elif len(self._requests) >= 24:
                raise ValueError("evaluation request limit reached")
            index = len(self._requests)
            record: dict[str, Any] = {
                "index": index,
                "run": self._run["label"] if self._run else "",
                "reserved_usd": float(reservation),
                "input_token_ceiling": ceiling,
                "output_token_ceiling": self.max_output_tokens,
                "usage": None,
                "status": "reserved",
            }
            self._requests.append(record)
            self._record_reservations[index] = reservation
            self._reserved += reservation
        return request, record

    def _reconcile_reservation(self, record: dict[str, Any], *, http_status: int) -> None:
        """Refund unused conservative margin only on HTTP 200 with verified cost."""
        if http_status != 200:
            return
        usage = record.get("usage")
        if not isinstance(usage, dict):
            return
        raw_cost = usage.get("cost")
        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
            or not math.isfinite(float(raw_cost))
            or float(raw_cost) < 0
        ):
            return
        index = record.get("index")
        if not isinstance(index, int) or index not in self._record_reservations:
            return
        initial = self._record_reservations[index]
        settled = Decimal(str(raw_cost))
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        ):
            token_floor = (
                Decimal(prompt_tokens) * self._input_price
                + Decimal(completion_tokens) * self._output_price
            )
            settled = max(settled, token_floor)
        if settled < initial:
            refund = initial - settled
            self._reserved = max(Decimal(0), self._reserved - refund)
            if self._run is not None and self._run["label"] == record.get("run"):
                self._run["reserved"] = max(Decimal(0), self._run["reserved"] - refund)
            self._record_reservations[index] = settled
            record["initial_reserved_usd"] = float(initial)
            record["reserved_usd"] = float(settled)
            record["refunded_usd"] = float(refund)
        elif settled > initial:
            extra = settled - initial
            self._reserved += extra
            if self._run is not None and self._run["label"] == record.get("run"):
                self._run["reserved"] += extra
            self._record_reservations[index] = settled
            record["initial_reserved_usd"] = float(initial)
            record["reserved_usd"] = float(settled)
            record["refunded_usd"] = 0.0

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = [dict(row) for row in self._requests]
            costs = [
                row["usage"].get("cost") if isinstance(row["usage"], dict) else None for row in rows
            ]
            complete = all(isinstance(cost, (float, int)) for cost in costs)
            return {
                "model": self.model,
                "provider": self.provider,
                "input_price_per_token": float(self._input_price),
                "output_price_per_token": float(self._output_price),
                "budget_usd": float(self._budget),
                "reserved_usd": float(self._reserved),
                "max_body_bytes": self.max_body_bytes,
                "requests": rows,
                "reported_cost_usd": sum(costs) if complete else None,
                "cost_complete": complete,
            }

    def __enter__(self) -> EvalProxy:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def error(self, code: int, message: str) -> None:
                data = json.dumps(
                    {"error": {"message": message, "type": "evaluation_guard"}}
                ).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_POST(self) -> None:  # noqa: N802
                self.connection.settimeout(15)
                if self.path != "/v1/chat/completions" or self.headers.get("Origin"):
                    self.error(404, "unsupported evaluation endpoint")
                    return
                if not secrets.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {owner.client_key}"
                ):
                    self.error(401, "evaluation credential required")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if self.headers.get("Transfer-Encoding") or size <= 0:
                        raise ValueError("invalid request size")
                    if size > owner.max_body_bytes:
                        self.error(413, "request exceeds evaluation size limit")
                        return
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError("invalid request")
                    request, record = owner.reserve(payload)
                except (ValueError, OSError, TypeError) as exc:
                    self.error(400, str(exc))
                    return
                started = False
                try:
                    with (
                        httpx.Client(timeout=90, follow_redirects=False, trust_env=False) as client,
                        client.stream(
                            "POST",
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {owner._api_key}"},
                            json=request,
                        ) as response,
                    ):
                        with owner._lock:
                            record["http_status"] = response.status_code
                        self.send_response(response.status_code)
                        self.send_header(
                            "Content-Type",
                            response.headers.get("Content-Type", "application/json"),
                        )
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                        buffer = bytearray()
                        for chunk in response.iter_bytes():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            if len(chunk) >= 2_000_000:
                                buffer = bytearray(chunk[-2_000_000:])
                            else:
                                excess = len(buffer) + len(chunk) - 2_000_000
                                if excess > 0:
                                    del buffer[:excess]
                                buffer.extend(chunk)
                        with owner._lock:
                            owner._capture_usage(bytes(buffer), record)
                            owner._reconcile_reservation(
                                record, http_status=response.status_code
                            )
                            record["status"] = "finished"
                except (OSError, httpx.HTTPError):
                    with owner._lock:
                        record["status"] = "interrupted"
                    if not started:
                        self.error(502, "evaluation upstream request failed")
                finally:
                    self.close_connection = True

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        return self

    @staticmethod
    def _capture_usage(body: bytes, record: dict[str, Any]) -> None:
        candidates = [
            body,
            *[line[5:].strip() for line in body.splitlines() if line.startswith(b"data:")],
        ]
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("usage"), dict):
                record["usage"] = value["usage"]

    def __exit__(self, *_exc: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-price-per-token", type=float, required=True)
    parser.add_argument("--output-price-per-token", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, default=4.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-body-bytes", type=int, default=_MAX_BODY)
    parser.add_argument("--provider", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    with EvalProxy(
        args.model,
        api_key,
        args.input_price_per_token,
        args.output_price_per_token,
        budget_usd=args.budget_usd,
        max_output_tokens=args.max_output_tokens,
        max_body_bytes=args.max_body_bytes,
        provider=args.provider,
    ) as proxy:
        print(json.dumps({"base_url": proxy.base_url, "max_body_bytes": proxy.max_body_bytes}))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
