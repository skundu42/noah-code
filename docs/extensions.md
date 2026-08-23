# Commands, skills, MCP, and tracing

## Custom slash commands

Add reusable Markdown prompts in either location:

- `~/.config/noah-code/commands/*.md` for user commands.
- `.noah-code/commands/*.md` for repository commands. Repository commands override user commands
  with the same name.

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
through `$9`. Front matter may also select a mode or model.

## Subagents

The parent agent invokes nested NOOA agents with `self.task.run("explore", prompt)` or
`self.task.run("general", prompt)`, and fans out independent units concurrently with
`self.task.run_many([("explore", "..."), ("general", "...")])`. Each child gets isolated
in-memory session storage, its own permission-engine clone (so concurrent children never race
on mode), and the same approval broker. Results are bounded: oversized transcripts are
condensed by the lightweight model (fallback: truncation with a recall pointer) before they
enter the parent's context. Tune with `subagent_result_max_chars` and
`max_concurrent_subagents` under `[efficiency]`. Add project or user markdown agents:

- `~/.config/noah-code/agents/*.md`
- `.noah-code/agents/*.md`

```markdown
---
description: Review a diff without editing
readonly: true
---
Review the assigned change. Cite files. Do not edit.
```

`/agents` lists discovered names. Project files override user files with the same stem.

## Skills

Open the dedicated searchable picker with `/skills` or `Ctrl+K`. Selecting a document skill
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

## Tracing

Noah Code integrates with NOOA tracing. When a local viewer is available, spans can be exported
to it. JSONL export can also be enabled in user configuration. Use `/trace` to inspect the active
destination.
