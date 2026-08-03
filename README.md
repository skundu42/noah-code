# Noah Code

**Noah Code** (`noah-code`) is a terminal coding agent for understanding repositories, planning changes,
editing files, running commands, and carrying work across persistent sessions. It is built on
the [NVIDIA OO Agents (NOOA)](https://github.com/NVIDIA-NeMo/labs-OO-Agents) runtime and adds a
repository-aware tool layer, interactive approvals, undo/redo, and a full-screen terminal UI.

Install everything with one command—no Python, Homebrew, or system package setup required:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && ~/.local/bin/uv tool install --managed-python --python 3.12 --no-build --force 'noah-code[mcp,tracing]'
```

Open a new terminal and run `noah .` inside a repository.

## Features:

### Work with the repository

- Read files or focused line ranges and search with ripgrep.
- List files deterministically without handing a glob to the shell.
- Inspect Git status, diffs, and recent commits through narrow read-only helpers.
- Make anchored replacements or create and rewrite files.
- Run permission-gated shell commands with timeouts and streamed output.
- Load repository instructions from `AGENTS.md`, `CLAUDE.md`, and
  `.noah-code/instructions.md`.
- Track multi-step work with todos and compact older context when conversations grow.

### Stay in control

- Choose **build** mode for implementation or **plan** mode for read-only analysis.
- Approve an action once, remember an approval for the session, or reject it.
- Use ordered `allow`, `ask`, and `deny` rules, with explicit denies taking precedence over
  automatic approval.
- Undo and redo journaled file edits. Concurrent user changes are detected before restoration.
- Keep generated Python behind fail-closed native macOS/Linux sandboxes and a narrow host-side
  broker.

### Choose how to work

- Full-screen Textual TUI with live tool, CodeAct, and shell output.
- Classic line-oriented console with `--console`.
- One-shot, non-interactive execution with `noah run`.
- Persistent, workspace-scoped sessions that can be listed, resumed, switched, or deleted.
- Model selection through the NOOA unified model registry.
- Project and user slash commands, opt-in skills, optional MCP servers, and tracing.

## Supported platforms

- macOS on Apple Silicon or Intel.
- Linux on arm64 or x86_64 with Landlock and seccomp support.
- An LLM provider account such as OpenAI, Anthropic, or NVIDIA.

## Installation

Run the one-line installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && ~/.local/bin/uv tool install --managed-python --python 3.12 --no-build --force 'noah-code[mcp,tracing]'
```

It uses uv's standalone binary and automatically downloads an isolated Python runtime when one
is not already available. After opening a new terminal:

```bash
noah --version
noah .
```

For repository maintainers or mirrored installations, the checked-in bootstrapper performs the
same installation with platform checks and clearer diagnostics:

```bash
sh install.sh
```

The package installs `noah`, `noah-code`, and `nc` as equivalent entry points. Because `nc`
commonly refers to netcat, `noah` or `noah-code` is recommended.

Model and provider configuration follows NOOA conventions, including its model registry,
environment variables, and configuration under `~/.config/nooa/`. Do not place API keys in the
repository or in noah-code session metadata.

### Updates

Installs created by the one-line command check PyPI at most once every 24 hours and install newer
noah-code releases through uv. When an update is installed, noah exits before starting the task
so it cannot mix old and new runtime modules; rerun the command to continue on the new version.

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

## Quick start

Start the TUI in the current repository:

```bash
noah .
```

Then describe the desired end state in plain language, for example:

```text
Find the cause of the failing parser tests, implement the smallest safe fix, and run the
focused test file.
```

Other common forms:

```bash
# Open a different workspace
noah /path/to/repository

# Use the line-oriented console
noah --console .

# Run a single task and exit
noah run "Explain how authentication is wired" .

# Implement without stopping at ask decisions; deny rules still apply
noah run --auto "Fix the failing unit test" .

# Inspect and plan without editing
noah --mode plan .

# Resume work
noah --continue .
noah --session SESSION_ID .
```

Diagnostics and session management:

```bash
noah doctor .
noah config show .
noah sessions list .
noah sessions show SESSION_ID
noah sessions delete SESSION_ID
noah update --check
```

## Interactive interface

### TUI keybindings

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Send the current message |
| `Ctrl+P` | Open the command palette |
| `Ctrl+O` | Open the session picker |
| `Ctrl+N` | Start a new session |
| `Ctrl+C` | Cancel the active turn; press twice while idle to quit |
| `Ctrl+Q` | Quit |
| `F1` or `?` | Show help |

At an approval prompt, press `1` to approve once, `2` to remember the approval for the current
session, or `3`/`Esc` to reject it.

### Built-in slash commands

| Command | Purpose |
|---------|---------|
| `/help` | Show interactive help |
| `/mode` | Show or switch between `build` and `plan` |
| `/model` | Show or switch the active model |
| `/session`, `/sessions`, `/new`, `/continue` | Inspect, switch, create, and resume sessions |
| `/compact` | Summarize older conversation context |
| `/todos` | Show the agent's current task list |
| `/status`, `/diff` | Inspect the repository |
| `/undo`, `/redo` | Restore or reapply journaled file edits |
| `/skills` | Inspect discovered skills |
| `/trace` | Show the active tracing destination |
| `/exit` | End the session |

## Modes and permissions

| Mode | Behavior |
|------|----------|
| `build` | Reads are allowed; edits and shell commands follow permission rules and ask by default |
| `plan` | Reads are allowed; file edits and mutating shell commands are denied |

Switch modes with `--mode`, `/mode build`, or `/mode plan`. The active mode is stored with the
session.

Permission rules are evaluated in order and the last matching rule wins. The default policy is:

- allow ordinary reads;
- deny likely secrets, including `.env` variants, private keys, `.git` internals, and session
  databases (`.env.example` remains readable);
- ask before workspace edits and shell commands;
- deny `git push`, `git clean`, and `git reset --hard`;
- keep file tools inside the active workspace and ask before skill or MCP access;
- deny plan-mode mutations regardless of a broader allow rule.

`--auto` changes ask decisions to allow, but never overrides an explicit deny. Compound shell
commands and mutating or unrecognized Git commands cannot be silently auto-approved.

## Generated-code security

NOOA CodeAct lets a model express a turn as Python, so noah-code treats that interpreter as
untrusted. Generated cells always execute in a separate, brokered worker. On Linux the worker
uses:

- Landlock filesystem confinement;
- seccomp network isolation;
- memory, CPU, and execution-time limits;
- restricted imports;
- no direct workspace mount;
- a narrow broker exposing only permission-gated workspace, Git, message, mode, and todo
  operations.

On macOS the worker uses a native deny-by-default Sandbox profile that blocks repository file
contents and outbound networking, allows read access only to the managed Python runtime, and
applies CPU and execution-time limits. The workspace is never mounted directly; approved work
still crosses the same narrow parent broker. Linux additionally enforces an address-space memory
limit.

If those controls cannot be installed, execution fails closed. For trusted development tests
only, containment can be disabled with one of the following:

```bash
noah --unsafe-inprocess-code-execution .
NOAH_CODE_UNSAFE_INPROCESS=1 noah .
```

Or in trusted user configuration:

```toml
unsafe_inprocess_code_execution = true
```

This removes the generated-code security boundary. Do not use it for untrusted prompts,
repositories, or models.

## Sessions, context, and undo

Each session has a NOOA-backed SQLite database plus metadata for its workspace identity, model,
mode, title, remembered permission rules, todos, and edit journal. Session files are created with
private filesystem permissions and cannot be resumed against a different workspace accidentally.

Long conversations use token-budget summarization while preserving recent messages. You can
force compaction with `/compact`.

File edits made through the workspace tools record pre- and post-images with content hashes:

- `/undo` refuses to overwrite a file changed concurrently by the user;
- a failed multi-file restoration is rolled back;
- `/redo` survives process restarts;
- a turn that ran a mutating shell command is marked as not fully reversible because the shell
  may have changed files outside the journal.

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
therefore ignored for `auto_approve`, `enabled_skills`, `mcp`, `permission_rules`, `session_dir`,
`tracing`, `updates`, and `unsafe_inprocess_code_execution`. Put those settings in trusted user
config, the environment, or an explicit CLI flag. A user-configured `permission_rules` array
replaces the default rule array, so copy forward every default you still want before adding
overrides; hard secret, destructive-shell, and plan-mode gates remain enforced in code.

Inspect the fully resolved configuration with:

```bash
noah config show .
```

## Custom slash commands

Add reusable Markdown prompts in either location:

- `~/.config/noah-code/commands/*.md` for user commands;
- `.noah-code/commands/*.md` for repository commands, which override user commands with the
  same name.

For example, `.noah-code/commands/fix.md`:

```markdown
---
description: Fix a bug and run focused tests
mode: build
---
Fix $ARGUMENTS. Inspect nearby tests, make the smallest coherent change, and run the focused
test.
```

Invoke it as `/fix the parser`. Commands support `$ARGUMENTS` and positional placeholders
`$1` through `$9`; front matter may also select a mode or model.

## Skills and MCP

Skills are discovered from:

- `~/.config/noah-code/skills/`;
- `.noah-code/skills/`;
- `./skills/`.

Discovery does not grant access. Activate trusted skills from user configuration with patterns
such as:

```toml
enabled_skills = ["cmd.*"]
```

Use `/skills` to inspect the resulting registry. Project configuration cannot activate skills.

MCP support is optional:

```bash
uv sync --extra mcp
```

Configure servers in trusted user configuration or a compatible `.mcp.json`/
`.noah-code/mcp.json` file. Attaching a server is gated by the `mcp` permission category and asks
by default.

## Tracing

Noah Code integrates with NOOA tracing. When a local viewer is available, spans can be exported
to it; JSONL export can also be enabled in user configuration. Use `/trace` to inspect the active
destination.

## Development

Install the development environment and run the local checks:

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest tests
uv build
```

The default test suite is hermetic and does not require network access or provider keys.

## CI and releases

GitHub Actions runs the complete test suite on Python 3.12 and 3.13, plus platform smoke tests on
Linux and macOS for both arm64 and x86_64. Every pull request and push to `main` must pass lint,
tests, lockfile validation, and a package build.

A `v*` tag starts the release pipeline, which:

1. verifies that the tag, `pyproject.toml`, and package versions match;
2. reruns tests and builds the wheel and source distribution;
3. validates distribution metadata and generates SHA-256 checksums;
4. publishes to PyPI using short-lived OIDC credentials;
5. creates provenance attestations and a GitHub release with the artifacts.

Before the first release, configure a PyPI Trusted Publisher for package `noah-code`, workflow
`release.yml`, and GitHub environment `pypi`. Protect that environment with required reviewers.
Then update both version declarations, commit them, and push a matching tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Once PyPI publishes the new version, existing uv-tool installations discover it through the
built-in updater.

## License

Apache-2.0, as declared in the project metadata. NOOA remains separately licensed by its
upstream project.

## Credits

Built on [NVIDIA OO Agents (NOOA)](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Thanks to the
NVIDIA NeMo team and NOOA contributors for the agent runtime that powers Noah Code.
