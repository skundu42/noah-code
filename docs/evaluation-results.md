# Noah and OpenCode v2 fixture comparison — 2026-09-26

In the final trial, round 4, Noah and OpenCode 2.0.18 both passed both fixture
acceptance suites and completed all four tasks with exit code 0. Earlier failures
remain recorded below. The fixes and harness correction address the reproduced
problems; these small trials do not establish a general coding-quality ranking.

## Round 4: final verification with corrected fixture permissions

Both agents used the same `openrouter/z-ai/glm-5.3-flash` model, `inceptron/fp8`
endpoint, prompts, baseline commits, and frozen acceptance tests. Limits remained
12 requests, 150 seconds per run, and 4,096 output tokens per request, with a $1.50
total conservative reservation cap ($0.375 per run). Noah's launch now uses the
existing explicit trusted-fixture rules without `--auto`; product permission
checks remain enabled. This configuration change means differences from earlier
rounds cannot be attributed solely to product improvements.

| Fixture | Agent | Independent acceptance | Process result | Time | Requests | Known provider cost |
|---|---|---|---|---:|---:|---:|
| Page ranges | Noah | Pass; all 5 tests | Completed, exit 0 | 80.047 s | 5 | $0.00285746 |
| Page ranges | OpenCode v2 | Pass; all 5 tests | Completed, exit 0 | 47.080 s | 6 | $0.00434829 |
| Recursive config merge | Noah | Pass; all 3 tests | Completed, exit 0 | 30.541 s | 6 | $0.00254775 |
| Recursive config merge | OpenCode v2 | Pass; all 3 tests | Completed, exit 0 | 36.241 s | 7 | $0.00383672 plus 1 unknown request |

Noah ran the exact requested unittest command successfully in both fixtures and
retained current passing verification records through completion. Its live traces
had no workspace argument/return-shape mistakes, multiline corruption, or empty
model responses. The new bounded empty-response recovery was validated by
deterministic tests; it was not naturally exercised in this final live sample.
OpenCode retried one HTTP 429 during config merge and still completed. That
request reported no usage; its full $0.014966925 reservation remains counted.
All source fixture hashes remained unchanged, and each worktree changed only its
implementation module.

Round 4 reserved **$0.327859225** across 24 requests, with **$0.01359022** in known
provider charges plus the unknown 429 charge. Across all four closed attempts,
conservative reservations total **$1.477142325**, below the user's $5 limit.
Known provider charges total **$0.06000354**, with seven requests lacking cost
records (six initial 404s and the final 429). Unknown charges are not treated as
zero and remain covered by their reservations. No further paid trial was run.

The evaluated source-tree SHA-256 was
`d8aeea477abddd47f77f9429e79152ecea499d2a97eada0987b342a40533f996`;
tracked diff SHA-256 was
`2530a8ff20b5e7b463d471888aca3edad78eeed4cfeb39f04e6e552257bda05e`.
The final product suite passed **1,388 tests** (1 deselected, 5 existing/upstream
warnings); Ruff, mypy over 69 source files, the package build, and whitespace
checks also passed. See [reliability.md](reliability.md) for the repaired contracts
and their limits.

## Round 3: after the first reliability repairs

Both agents again used `openrouter/z-ai/glm-5.3-flash` on `inceptron/fp8`, with the
same prompts and baseline commits, 150-second limit, 4,096 output tokens per
request, and 12 requests per run. The new conservative reservation cap was $1.50
total, equally divided at $0.375 per run. It was not exhausted.

| Fixture | Agent | Independently verified implementation | Process result | Time | Requests | Provider-reported cost |
|---|---|---|---|---:|---:|---:|
| Page ranges | Noah | Fail; original implementation, 5 tests ran | Empty model response | 57.052 s | 3 | $0.00289856 |
| Page ranges | OpenCode v2 | Pass; all 5 tests | Completed | 42.817 s | 6 | $0.00556682 |
| Recursive config merge | Noah | Pass; all 3 tests | Iteration limit | 31.314 s | 12 | $0.00559455 |
| Recursive config merge | OpenCode v2 | Pass; all 3 tests | Completed | 22.228 s | 6 | $0.00383890 |

Noah made no workspace argument/return-shape errors in these runs, and its
multiline edit remained intact. In page ranges, the third HTTP 200 response used
all 4,096 completion tokens, including 3,953 reported reasoning tokens, and
returned no executable tool call. Noah stopped with an explicit error before
editing; this was an output-budget failure, not a reservation-budget failure.

In config merge, Noah produced an implementation that passed the independent
scorer. Its first attempt to run the user-requested `python -m unittest discover
-s tests -p acceptance.py` was denied by Noah's intentional `--auto` interpreter
restriction. The harness had supplied a broad allow rule and `--auto`, not an
exact prior approval. The model then tried pytest, which failed to import the
fixture module, followed by blocked shell/REPL alternatives. It reached the
iteration limit without claiming completion. Passing the independent acceptance
tests and successfully finishing the agent task are distinct outcomes.

For round 4, the harness omitted Noah's `--auto` flag and relied on its existing explicit
trusted-fixture permission rules. The product's interpreter and secret guards
were not weakened. The later trial with that correction is not a pure comparison
of product changes: the verification permission configuration also changed.

Round 3 reserved **$0.391708325** and reported **$0.01789883** in provider charges
across 27 requests. Its source-tree SHA-256 was
`17f4635fe6d41e21e308195947afc49d0ce12119d08051e71c4edec904d993e4`;
tracked diff SHA-256 was
`7bf538e5fa8fdb2c7bdf166233bb8069bc1ca8513d1f9b3df3427c35e68de92e`.

## Round 2: matched model, before reliability repairs

Both agents used `openrouter/z-ai/glm-5.3-flash`, pinned to the same
`inceptron/fp8` endpoint. Published endpoint prices were $0.11 per million input
and $0.45 per million output tokens. Each run had a 150-second wall limit,
4,096 output-token ceiling per request, 12 requests including retries/auxiliary
calls, and $0.959879075 conservative reservation allocation. Prompts and baseline
commits were identical. Web tools and subagents were denied.

| Fixture | Agent | Independently verified result | Process time | Model requests | Provider-reported cost |
|---|---|---|---:|---:|---:|
| Page ranges | Noah | Fail; 5 tests ran, unchanged faulty implementation | 86.359 s | 12 | $0.00566797 |
| Page ranges | OpenCode v2 | Pass; all 5 tests | 43.484 s | 10 | $0.00845656 |
| Recursive config merge | Noah | Fail; syntax error prevented 3 tests loading | 25.445 s | 12 | $0.00483317 |
| Recursive config merge | OpenCode v2 | Pass; all 3 tests | 11.312 s | 6 | $0.00363315 |

Both Noah runs reached the 12-turn limit, with no failed model requests. In the
page-range run, the model treated string glob results as objects and repeatedly
guessed `apply_patch` argument shapes, then supplied mismatched edit text. In the
merge run it guessed list/replace signatures, eventually applied a malformed
replacement and left a syntax error. The traces exposed missing tool-contract
discovery through the REPL proxy. Subsequent deterministic replay also confirmed
a framework defect: the sandbox's async wrapper inserted indentation inside
multiline string literals. A correct 277-character preimage in the recorded
model code became 349 characters during execution. That explains part of the
page-range edit mismatch; the merge trace also contained malformed generated
code independently. The first repairs added safe tool help, preserved literal
contents, checked Python edit syntax before writing, and prevented completion
with failed or missing verification. Round 3 above records their follow-up
without deleting this earlier evidence. Both rounds used the same `--auto`
verification constraint described above.

OpenCode exited successfully for both tasks; neither agent timed out. Costs here
come from upstream usage records, including auxiliary requests. Noah's native
usage JSON reported zero cost for the custom model alias, so it was not used as
the billing authority.

## Initial attempt excluded from comparison

The first attempt exposed two harness defects. The cheapest endpoint excluded
Noah's `parallel_tool_calls` parameter and returned six HTTP 404 responses across
its two runs. OpenCode inherited a stale `PWD` and edited the harness's authored
source fixtures instead of its assigned worktrees. Tool logs identify only the
two fixture implementation writes; those files were restored byte-for-byte from
the immutable baseline repositories. Its acceptance claims were therefore not
valid worktree results.

| Fixture | Agent | Invalid-attempt outcome | Process time | Requests | Known provider cost |
|---|---|---|---:|---:|---:|
| Page ranges | Noah | Three routing 404s | 4.264 s | 3 | Unknown |
| Page ranges | OpenCode v2 | Edited wrong fixture location | 97.708 s | 9 | $0.00353278 |
| Recursive config merge | Noah | Three routing 404s | 3.633 s | 3 | Unknown |
| Recursive config merge | OpenCode v2 | Edited wrong fixture location | 23.384 s | 7 | $0.00239086 |

Before the valid run, endpoint discovery was changed to require the union of
both clients' tool parameters and pin the exact compatible endpoint. Subprocess
`PWD` was set to its actual worktree; a real, free loopback protocol smoke test
verified that both model contexts contained the intended fixture directory.

The initial attempt reserved $0.160483700. The second proxy's maximum allocation
was reduced to $3.839516300, so the combined allowed reservations never exceeded
$4, below the user's $5 authorization. Actual conservative reservations across
the first two closed attempts totaled **$0.757574775**. Including round 3, that
subtotal was **$1.149283100**. Provider-reported known charges through round 3
totaled **$0.04641332**, including $0.02259085 for round 2 and $0.01789883 for
round 3. Final four-attempt totals appear above. The six initial 404s did
not report usage or cost; their full reservations remain counted. These are not
claimed to be free. The bound assumes provider compliance with the documented
prices and output limits.

## Reproduction and scoring evidence

The exact prompts, fixture/source hashes, per-run results and failure details are
in [evaluation-results.json](evaluation-results.json). The complete local logs,
worktrees, requests and independently rescored outputs remain in:

- `/tmp/noah-comparison-live-20260926/`
- `/tmp/noah-comparison-live-20260926-r2/`
- `/tmp/noah-comparison-live-20260926-r3/`
- `/tmp/noah-comparison-live-20260926-r4/`

| Fixture | Immutable baseline commit | Fixture SHA-256 |
|---|---|---|
| Page ranges | `0334dc62576c02afb380f841fd3c6a417dc5bb50` | `7fc01f18a7079accaf48f2b833d79b0d4956b9980d551be7bebd35617957d428` |
| Recursive config merge | `0235ecba65176e3c11097b3681327cfa4d7e9aa5` | `2ca64d736d96a62ab0b7f97198d885ca3e5db26b172456e9fffb63808e797700` |

Noah ran the working source at commit
`8bf3018cacd34ac7f7a26d303f4d211955a4e7a3`, including uncommitted improvements.
The valid run's source-tree SHA-256 was
`343735c6a828898328b3ff94918caefb3b5e71918e3e10387dc6271c341a9c47`;
its tracked diff SHA-256 was
`4c394d37dd58b3ee9ca724e458a17e9f6e6d84f5b1dd2af63e8376bb48e9a71c`.
The initial attempt's distinct hashes are preserved in the JSON artifact.

After both attempts finished, scoring independently copied only the allowed
implementation module and acceptance bytes retrieved from the original Git
commit into a fresh evaluator directory. It ran Python with `-I`, imported the
trusted standard-library test runner before adding the scoring directory, and
required structured success plus the exact expected test count. Both source
fixtures matched their recorded hashes after the valid run. This confirms the
valid-run outcomes without additional model calls.

The current harness freezes all baselines and acceptance bytes before any model
runs, uses that stronger scorer, invalidates source-fixture hash changes, and
captures Git status/diff before scoring. The first two runs predate that final
scoring change and were independently rescored as described above; rounds 3 and 4 used
the stronger scorer directly. Worktrees,
XDG state and per-run configuration separate task state; they are **not an OS
sandbox**. The processes retain normal filesystem access, including their user
home and Noah's source import path. Use a container or separate OS account for
adversarial repositories or agents.

See [evaluation.md](evaluation.md) for commands and methodology. Protocol sources:
[OpenCode v2 models](https://opencode.ai/v2/docs/models),
[OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints),
and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
