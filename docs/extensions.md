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

## Skills

Skills are discovered from:

- `~/.config/noah-code/skills/`
- `.noah-code/skills/`
- `./skills/`

Discovery does not grant access. Activate trusted skills from user configuration with patterns
such as:

```toml
enabled_skills = ["cmd.*"]
```

Use `/skills` to inspect the resulting registry. Project configuration cannot activate skills.

## MCP

MCP support is optional:

```bash
uv sync --extra mcp
```

Configure servers in trusted user configuration or a compatible `.mcp.json` or
`.noah-code/mcp.json` file. Attaching a server is gated by the `mcp` permission category and asks
by default.

## Tracing

Noah Code integrates with NOOA tracing. When a local viewer is available, spans can be exported
to it. JSONL export can also be enabled in user configuration. Use `/trace` to inspect the active
destination.
