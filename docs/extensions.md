# Commands, skills, MCP, and tracing

## Custom slash commands

Add reusable Markdown prompts in either location:

- `~/.config/noah-code/commands/*.md` for user commands.
- `.noah-code/commands/*.md` for repository commands. Repository commands override user commands
  with the same non-built-in name. A repository cannot shadow Noah's built-in slash commands.

For example, `.noah-code/commands/fix.md`:

```markdown
---
description: Fix a bug and run focused tests
mode: build
---
Fix $ARGUMENTS. Inspect nearby tests, make the smallest coherent change, and run the focused
test.
```

Invoke it as `/fix the parser`. Commands support `$ARGUMENTS` and positional placeholders `$1`
through `$9`. Front matter may also select a mode or model — but only for trusted user commands:
`mode` and `model` front matter in repository commands is ignored, so a checkout cannot switch the
session out of plan mode or onto another model without confirmation.

## Subagents

The parent agent invokes nested NOOA agents with `self.task.run("explore", prompt)` or
`self.task.run("general", prompt)`, and fans out independent units concurrently with
`self.task.run_many([("explore", "..."), ("general", "...")])`. Each child gets isolated
in-memory session storage, its own permission-engine clone (so concurrent children never race
on mode), and the same approval broker. Read-only agents can fan out concurrently; agents that can
mutate the checkout share one mutation lane and cannot write over each other. Results are bounded:
oversized transcripts are condensed by the lightweight model (fallback: truncation with a recall
pointer) before they enter the parent's context. Tune with `subagent_result_max_chars` and
`max_concurrent_subagents` under `[efficiency]`. Add project or user markdown agents:

Use `self.task.collaborate(objective, assignments, lead="general")` when parallel findings need a
deliberate handoff. Contributors run under the same concurrency and mutation-lane rules, then the
lead receives their bounded reports and returns one synthesized result. Live queued, running,
completed, failed, and cancelled states are visible in the TUI work ledger (`F4`) and `/work`.

- `~/.config/noah-code/agents/*.md`
- `.noah-code/agents/*.md`

```markdown
---
description: Review a diff without editing
readonly: true
model: openrouter/anthropic/claude-sonnet-4
---
Review the assigned change. Cite files. Do not edit.
```

`/agents` lists discovered names. Project files override user files with the same stem, except that
the built-in `explore` and `general` agents are reserved and cannot be replaced by repository
content.

Repository command and agent files are limited to 64 KiB and must be ordinary, singly linked files
inside their expected `.noah-code` directory. Symlinks, hardlinks, linked directories, special
files, and oversized files are ignored. User-level files remain linkable because they are trusted
configuration owned by the user.

## Skills

Open the dedicated searchable picker with `/skills` or `Ctrl+G`. Selecting a document skill
inserts `$skill-name ` into the composer so you can add the task directly. Noah reads the standard
`SKILL.md` directory format used by Codex and Claude, including companion `scripts/`, `references/`,
and `assets/` folders.

Project skills are discovered before user skills from:

- `.agents/skills/`
- `.claude/skills/`
- `.noah-code/skills/`
- `./skills/`
- `~/.agents/skills/`
- `~/.claude/skills/`
- `~/.codex/skills/`
- `~/.config/noah-code/skills/`

Add an existing skill folder from the picker or the terminal:

```text
/skills add ~/path/to/my-skill
```

Noah validates the `SKILL.md` front matter, copies the whole directory to the shared
`~/.agents/skills/` root, and will not overwrite an existing skill. Skills that depend on a
vendor-specific runtime, binary, or remote tool still require that dependency to be installed.

Discovery does not grant access. Activate trusted skills from user configuration with patterns
such as:

```toml
enabled_skills = ["cmd.*"]
```

Explicit `$skill-name TASK` invocation is also gated by the `skill` permission category. Project
configuration cannot activate skills.

## MCP

MCP support is optional:

```bash
uv sync --extra mcp
```

Open `/mcp` to search configured servers, connect one, or add a server through the guided STDIO
and Streamable HTTP setup. Terminal equivalents are:

```text
/mcp add stdio filesystem npx -y @modelcontextprotocol/server-filesystem /path
/mcp add http remote https://example.com/mcp
/mcp connect filesystem
```

Noah reads the portable `{"mcpServers": {...}}` structure from `.mcp.json`,
`.noah-code/mcp.json`, and `~/.config/noah-code/mcp.json`. The guided setup writes to the user
file with mode `0600`; use environment variables or a manually edited config for secrets and auth
headers. STDIO, SSE, Streamable HTTP, custom headers, and OAuth fields are passed through to the
MCP runtime. Configured servers from trusted user configuration attach at session start, so their
tools are on the agent (`self.<server>.<tool>(...)`) for the first turn. Workspace `.mcp.json`
catalogs stay disconnected until you select a server in `/mcp` or run `/mcp connect NAME`.
Attachment is gated by the `mcp` permission category and asks by default; trusted user servers skip
that prompt at startup. Set `efficiency.lazy_mcp = true` in trusted user configuration to catalog
servers without connecting them until `/mcp connect`.

Mutating MCP calls use Noah's durable external-effect ledger. Repeating a call that completed
returns its recorded result. If Noah stopped after dispatch but before recording the response, it
refuses to replay the same mutation automatically and asks the agent to inspect the remote system
first. Read-only MCP calls are not ledgered.

## Tool hooks

Trusted user configuration can run bounded shell hooks before or after matching tool operations:

```toml
[hooks]
pre_tool = [
  { match = "ws_*", command = "./scripts/pre-tool-check", timeout_seconds = 10 }
]
post_tool = [
  { match = "*", command = "./scripts/audit-tool", timeout_seconds = 10 }
]
```

`match` uses shell-style glob patterns against the tool name. A pre-tool hook can veto execution by
returning a non-zero status. Post-tool hooks are owned background tasks: Noah drains them before
persisting or closing the session. Hooks are ignored in repository configuration because they run
local executables; define them only in `~/.config/noah-code/config.toml`.

## Tracing

Noah Code emits OpenTelemetry traces for the complete agent invocation and NOOA/OpenInference child
spans for model generations and tool execution. Local JSONL traces are enabled by default and go
into the active session directory unless `tracing.jsonl_dir` selects another trusted location.
Use `/trace` to inspect active destinations and `/health` to inspect bounded runtime-event and
artifact state. Trace files count toward the session-storage quota when they use the default path.

Set an OTLP/HTTP collector endpoint to additionally export batched traces, metrics, and structured
logs:

```toml
[tracing]
otlp_endpoint = "http://localhost:4318"
logs_enabled = true
metrics_enabled = true
capture_content = false
```

Install the `tracing` extra when Noah was installed without the standard installer. Configured
remote endpoints must use HTTPS; plain HTTP is accepted only for a loopback collector.

Or use standard OpenTelemetry environment configuration:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.com
export OTEL_EXPORTER_OTLP_HEADERS='authorization=Bearer%20...'
```

Operational telemetry includes agent/LLM/tool duration, token usage, cached and reasoning token
counts, estimated cost, retries, outcomes, model/provider names, and trace-correlated lifecycle
events. Metric labels deliberately exclude session IDs, run IDs, workspace paths, prompts, and tool
targets to avoid high-cardinality series.

Content capture is off by default. Prompt and response messages, reasoning, generated code, tool
arguments/results, exception messages, stack traces, and file contents are removed at the exporter
boundary. `capture_content = true` is intended only for controlled development environments; values
and span attribute counts remain bounded, and NOOA's secret scrubber still runs before export. For
production, keep content disabled and apply an allowlist/redaction processor in the collector as a
second boundary.

The emitted standard instruments include `gen_ai.invoke_agent.duration`,
`gen_ai.client.operation.duration`, and `gen_ai.client.token.usage`. Noah-specific instruments cover
estimated cost, retries, and tool execution counts/duration. Export is best-effort and batched;
collector failure never blocks or fails an agent turn, and providers are flushed during orderly
shutdown.
