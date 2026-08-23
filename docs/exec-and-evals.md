# Eval harness and automation interface

Noah Code exposes a scriptable, machine-readable execution surface for agent
harnesses, benchmark runners, and CI pipelines. Everything below works
without a TTY and without touching user configuration.

## `noah exec`: multi-turn driver

```bash
# Single prompt, human-readable output
noah exec "Find the failing test and fix it" /path/to/repo --auto

# Scripted multi-turn: first prompt from argv, follow-ups from stdin lines
printf 'Inspect the parser module\nFix the bug you found\nRun the tests\n' \
  | noah exec "Start by reading AGENTS.md" . --auto --output-format stream-json

# One final JSON document instead of an event stream
noah run "Explain auth" . --auto --output-format json   # via noah run too
```

- `--output-format text` (default): human lines on stdout.
- `--output-format json`: nothing on stdout until one final summary document.
- `--output-format stream-json`: one NDJSON event per line as they happen,
  ending with `turn_result` records and a final `result` summary.

### NDJSON event schema

Every line is `{"type": ..., "text": ..., ...meta}` where `type` mirrors
host events: `message`, `reasoning`, `tool_start`, `tool_finish`,
`shell_chunk`, `error`, `summary`, `status`, `stop`. Automation-specific
records: `approval_request`, `question_request`, `turn_result`, and `result`.

`turn_result` fields: `turn`, `exit_code`, `response` (last assistant
message), `stop`, `tool_calls`, `duration_seconds`, `usage_delta`.

The final `result` adds `session_id`, `model`, `mode`, all `turns`, the full
`events` transcript (json mode), provider `usage`, optional `budget` state,
and `llm_cache` hit/miss stats.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | All turns completed |
| 1 | Agent/provider failure during a turn |
| 2 | Configuration or usage error |
| 3 | Permission denied or approval rejected (including pre-tool hook vetoes) |
| 124 | Budget cap reached — tokens, cost, or wall clock |
| 130 | Interrupted |

## Scoped permission rules

Self-describing runs without config edits. The engine is last-match-wins, so
explicit denies outrank earlier allows:

```bash
noah exec "..." . --auto \
  --allow 'edit:*' \
  --allow 'bash:git status*' \
  --deny 'bash:git push*' \
  --deny 'read:**/.env*'
```

Format is `CATEGORY:PATTERN`; omitting `CATEGORY:` defaults to `*`.

## Budget caps

```bash
noah exec "..." . --auto --max-tokens 200000 --max-cost-usd 5 --time-limit 1800
```

Caps are enforced inside the model-call wrapper (tokens/wall-clock) and at
turn boundaries (provider-reported cost). A breach cancels the turn and exits
with code 124; `result.budget` reports the counters.

## Pre/post tool-use hooks

Declared in **user** configuration only (`~/.config/noah-code/config.toml`);
a cloned repository can never define hooks.

```toml
[[hooks.pre_tool]]
match = "execute_python"      # glob over tool name or permission category
command = "/opt/guards/log-tool.sh"
timeout_seconds = 5

[[hooks.post_tool]]
match = "ws_run"
command = "make lint-quiet || true"
```

Hooks receive `NOAH_HOOK_PHASE`, `NOAH_HOOK_TOOL`, `NOAH_HOOK_CATEGORY`, and
`NOAH_HOOK_TARGET` in their environment and run with the workspace as cwd.
A non-zero `pre_tool` exit vetoes the call and its stderr becomes the
model-visible rejection reason (counted under exit code 3). `post_tool`
failures are reported but never abort the turn.

## Unified-diff edit path

Models trained on patch formats can apply standard unified diffs atomically:

```
await self.ws.apply_unified_diff(diff_text)
```

Parsing verifies hunk context against current content (including `\ No
newline at end of file`); application reuses the transactional
`apply_patch` pipeline with authorization, TOCTOU checks, journaling, and
rollback.

## Worktree checkpoints

Automatic per-turn snapshots for clean eval resets:

```bash
noah exec "..." . --auto --checkpoint        # or [checkpoints] enabled=true in user config
/checkpoints                                  # list inside a session
noah checkpoints list [SESSION_ID]
noah checkpoints restore REF                 # index+worktree only; HEAD unchanged
```

Snapshots are commits under `refs/noah-code/checkpoints/<session>/NNNN`
built through a temporary index, so capture never disturbs HEAD, the user's
index, or untracked files. Restoring recovers tracked files; files created
after the checkpoint are left in place (diff against the base commit for
exact eval scoring).

## Record/replay transport

Deterministic regression runs without provider calls:

```bash
# Record once
NOAH_CODE_LLM_CACHE=record NOAH_CODE_LLM_CACHE_DIR=/tmp/eval-cache \
  noah exec "..." . --auto

# Replay forever; misses fail loudly
NOAH_CODE_LLM_CACHE=replay NOAH_CODE_LLM_CACHE_DIR=/tmp/eval-cache \
  noah exec "..." . --auto

# Or per invocation
noah exec "..." . --auto --llm-cache /tmp/eval-cache --llm-cache-mode auto
```

Requests are keyed on a canonical hash of model + messages + tool schemas +
sampling kwargs; API keys never enter the key or payload.

## Sampling controls

```bash
noah exec "..." . --temperature 0 --seed 42          # per run
noah exec "..." . --top-p 0.9 --seed 7
```

Unset values are omitted entirely so providers keep their defaults.

## Task-success benchmark (`noah bench`)

`noah bench` runs curated SWE-bench-Verified tasks through the real agent host and scores
resolved rate alongside token, turn, and cost efficiency.

```bash
# Run the shipped smoke suite (8 pytest-based tasks across flask/pylint/pytest/seaborn)
noah bench run swebench-verified-smoke --model openai/gpt-4o --budget-tokens 400000

# Run only the first three tasks of any suite
noah bench run path/to/suite.json --limit 3 --setup "pip install -e ."

# Build a custom suite from Verified instance ids (fetched once, then cached)
noah bench pull --ids sympy__sympy-22914,pytest-dev__pytest-7982 --out my-suite.json

# Report and compare runs
noah bench report .noah-code/bench-runs/<run-id>
noah bench compare <baseline-run-dir> <candidate-run-dir>
```

How one task executes:

1. The repo mirror is cloned once into `~/.cache/noah-code/bench/repos/`, then each task gets a
   worktree checked out at its `base_commit`.
2. The optional setup command (suite `environment_setup`, `--setup`, or
   `NOAH_CODE_BENCH_SETUP`, in that order) runs **outside** the agent sandbox with network
   access — dependency installs belong there.
3. The agent receives the problem statement only (never the gold patch or hints) with auto
   approvals and the configured budget caps.
4. The `test_patch` is applied and `FAIL_TO_PASS` plus `PASS_TO_PASS` suites run under
   `python -m pytest`; resolved requires both green.

Artifacts land in `.noah-code/bench-runs/<run-id>/`: per-task `trace.jsonl` event streams,
test logs, `result.json`, and a top-level `result.json` report. Failed tasks clean their
worktrees automatically; pass `--keep-worktrees` to inspect fixes.
