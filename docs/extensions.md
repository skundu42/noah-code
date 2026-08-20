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
`self.task.run("general", prompt)`. Each child gets isolated in-memory session storage and the
same permission broker. Add project or user markdown agents:

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
MCP runtime. Under the default `lazy_mcp` setting, servers are cataloged but not connected at
startup, keeping unused schemas and connection latency out of the session. Select a server in
`/mcp` or run `/mcp connect NAME` to attach it to the live agent. Attachment is gated by the `mcp`
permission category and asks by default. Set `efficiency.lazy_mcp = false` in trusted user
configuration only when eager attachment is desired.

## Tracing

Noah Code integrates with NOOA tracing. When a local viewer is available, spans can be exported
to it. JSONL export can also be enabled in user configuration. Use `/trace` to inspect the active
destination.
