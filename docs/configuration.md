# Configuration, permissions, and updates

## Configuration

Configuration is merged in this order, with later layers taking precedence:

1. Built-in defaults.
2. User config at `~/.config/noah-code/config.toml`.
3. Project config at `.noah-code/config.toml`.
4. `NOAH_CODE_*` environment variables.
5. CLI flags.

Example user configuration:

```toml
model = "gpt-4o-mini"
lightweight_model = "gpt-4o-mini"
mode = "build"
max_iterations = 40
cell_timeout = 120
command_timeout = 60

[ui]
frontend = "tui"       # "tui" or "console"
markdown = true
stream_shell = true
show_reasoning = false

[summarization]
policy = "token_budget" # or "none"
preserve_recent = 10
target_chars = 4000

[tracing]
enabled = true
viewer = true
# jsonl_dir = "~/.local/share/noah-code/traces"

[updates]
auto_install = true
interval_hours = 24
```

Supported environment overrides include:

- `NOAH_CODE_MODEL`
- `NOAH_CODE_LIGHTWEIGHT_MODEL`
- `NOAH_CODE_AUTO`
- `NOAH_CODE_SESSION_DIR`
- `NOAH_CODE_MODE`
- `NOAH_CODE_UNSAFE_INPROCESS`
- `NOAH_CODE_AUTO_UPDATE`

Repository-controlled configuration cannot weaken the host trust boundary. Project config is
ignored for `auto_approve`, `enabled_skills`, `mcp`, `permission_rules`, `session_dir`, `tracing`,
`updates`, and `unsafe_inprocess_code_execution`. Put those settings in trusted user config, the
environment, or an explicit CLI flag.

A user-configured `permission_rules` array replaces the default rule array. Copy forward every
default you still want before adding overrides. Hard secret, destructive-shell, and plan-mode
gates remain enforced in code.

Inspect the resolved configuration with:

```bash
noah config show .
```

Model and provider configuration follows NOOA conventions, including its model registry,
environment variables, and configuration under `~/.config/nooa/`.

## Modes and permissions

| Mode | Behavior |
|------|----------|
| `build` | Reads are allowed; edits and shell commands follow permission rules and ask by default |
| `plan` | Reads are allowed; file edits and mutating shell commands are denied |

Switch modes with `--mode`, `/mode build`, or `/mode plan`. The active mode is stored with the
session.

Permission rules are evaluated in order, and the last matching rule wins. The default policy:

- Allows ordinary reads.
- Denies likely secrets, including `.env` variants, private keys, `.git` internals, and session
  databases. `.env.example` remains readable.
- Asks before workspace edits and shell commands.
- Denies `git push`, `git clean`, and `git reset --hard`.
- Keeps file tools inside the active workspace and asks before skill or MCP access.
- Denies plan-mode mutations regardless of broader allow rules.

`--auto` changes ask decisions to allow but never overrides an explicit deny. Compound shell
commands and mutating or unrecognized Git commands cannot be silently auto-approved.

## Installation and updates

The README executes the checked-in bootstrapper directly from GitHub. From a local clone, run the
same installer with:

```bash
sh install.sh
```

Installs created by the one-line command check PyPI at most once every 24 hours and install newer
Noah Code releases through uv. When an update is installed, Noah exits before starting the task
so it cannot mix old and new runtime modules. Rerun the command to continue on the new version.

```bash
# Check without changing the installation
noah update --check

# Update immediately
noah update
```

Disable automatic installation from trusted user configuration or the environment:

```toml
[updates]
auto_install = false
interval_hours = 24
```

```bash
export NOAH_CODE_AUTO_UPDATE=0
```
