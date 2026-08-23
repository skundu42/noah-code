# Reliability and long-running sessions

Noah Code separates conversational persistence from host runtime persistence. The NOOA session
database stores conversation and snapshots; Noah's runtime database stores the state needed to
recover work safely: active runs, queued steering, file-operation intents, user interactions,
background jobs, external effects, usage, budgets, and a bounded operational event log.

## What survives a restart

A resumed session restores:

- conversation history, summaries, todos, mode, model, reasoning effort, and remembered session
  permissions;
- pending steering messages and attachment paths, in FIFO order;
- token, cost, and wall-clock budget counters;
- managed full-output artifacts and background-job logs;
- the persistent undo/redo journal and rolling Git checkpoints; and
- the original request and state of an interrupted run.

Resume the latest session for a repository or select an exact ID:

```bash
noah --continue .
noah sessions list .
noah --session SESSION_ID .
```

Inside the TUI, use `Ctrl+O`, `/continue`, or `/sessions SESSION_ID`.

## Startup recovery

At startup Noah performs recovery before accepting a new turn:

1. By default it acquires an exclusive lease for the canonical checkout. A second process must use
   a separate Git worktree, created with `/worktree create` or `noah worktree create`.
2. It restores the latest atomic host-state generation, including usage and budget counters.
3. It rolls back workspace-tool file operations whose durable intent was recorded but never
   committed.
4. It verifies process identity before terminating orphaned background process groups. Durable logs
   remain readable after restart.
5. It expires approval or question prompts that belonged to the old process.
6. It restores queued steering and discovers the newest non-terminal run.

Running, recovering, retrying, and process-waiting runs continue automatically. A run that was
waiting for user input stays paused; the next user message continues that run instead of creating a
new one. Completed, failed, cancelled, and explicitly rejected work is never auto-resumed.

When an agent returns `WAIT`, Noah requires at least one live managed job. The host sleeps without
spending model calls, wakes when a job reaches a terminal state or steering arrives, injects a
fresh process-status snapshot, and continues the same journaled turn.

## File and external-effect safety

Direct workspace writes and atomic multi-file patches record their pre-images in the runtime
database before changing the checkout. A crash before the commit record restores those pre-images
on the next launch. This protection complements the persistent edit journal used by `/undo` and
`/redo`.

Git checkpoints are enabled by default. Noah captures them at turn boundaries and before mutating
shell commands, stores them under `refs/noah-code/checkpoints/<session>/`, and keeps the newest 50
by default. Capturing uses a temporary Git index and does not move `HEAD` or disturb the user's
index. Inspect and restore them with:

```text
/checkpoints
```

```bash
noah checkpoints list
noah checkpoints restore REF .
```

Structured GitHub calls and MCP calls whose tool names are classified as mutating use a durable
effect ledger. A completed identical call returns its recorded result. GitHub PR creation and
comments inspect the remote state after an ambiguous interruption; classified MCP mutations refuse
automatic replay and require the remote system to be checked first.

Arbitrary shell commands cannot be made transactionally reversible. Noah marks turns that ran a
mutating shell command as potentially incomplete for `/undo`, and the pre-command Git checkpoint
is the recovery boundary. Commands may also modify untracked, ignored, generated, or external
state that Git cannot restore; inspect those effects explicitly after a crash.

## Provider resilience

Model requests have a deadline and retry only failures classified as transient, such as timeouts,
connection failures, rate limits, and supported 5xx responses. Noah uses exponential backoff with
jitter, respects numeric `Retry-After` values, and never retries authentication, invalid-request,
content-policy, or context-window errors through this path.

Fallback models are tried in configured order after the primary exhausts its attempts:

```toml
[reliability.retries]
max_attempts = 5
base_delay_seconds = 0.5
max_delay_seconds = 20
jitter_ratio = 0.2
request_timeout_seconds = 180
fallback_models = ["openrouter/anthropic/claude-sonnet-4", "openai/gpt-5-mini"]
```

A successful model response is not replayed by the retry wrapper. Tool execution begins only after
that response returns, and tool-side durability handles structured external mutations separately.

## Bounds and unattended-operation controls

The defaults favor long-running work while keeping growth finite:

| Control | Default | Purpose |
| --- | ---: | --- |
| `processes.max_jobs` | `8` | Maximum managed background jobs |
| `processes.max_runtime_seconds` | `86400` | Per-job runtime ceiling (24 hours) |
| `reliability.interaction_timeout_seconds` | `86400` | Approval/question timeout |
| `reliability.artifact_max_bytes` | `2000000000` | Full-output artifact quota per session |
| `reliability.session_max_bytes` | `5000000000` | Total session-storage ceiling |
| `reliability.max_runtime_events` | `20000` | Retained operational events |
| `checkpoints.max_per_session` | `50` | Rolling Git checkpoint count |

Optional `[budget]` limits cap cumulative model tokens, estimated/provider-reported cost, or session
wall-clock time. Counters persist across restarts:

```toml
[budget]
max_tokens = 500000
max_cost_usd = 25
max_seconds = 28800
```

When a session exceeds its storage quota, Noah refuses another model turn rather than allowing
unbounded growth. Delete sessions that are no longer needed with `noah sessions delete SESSION_ID`.

## Observability

Use `/health` during idle or active work. It reports runtime database and artifact sizes, pending
inbox items and interactions, live jobs, retained runtime events, and the current non-terminal run.
Use `/status` for workspace/session state, `/tokens` for model and cache usage, `F2` for activity,
`self.processes.status()`/`logs()` for managed jobs, and `/trace` for the session trace destination.

Tracing is enabled by default. Without an explicit `tracing.jsonl_dir`, JSONL traces are stored
inside the session directory so they follow the same ownership and storage limits.
