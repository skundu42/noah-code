# Noah Code

[![PyPI](https://img.shields.io/pypi/v/noah-code.svg)](https://pypi.org/project/noah-code/)
[![CI](https://github.com/skundu42/noah-code/actions/workflows/ci.yml/badge.svg)](https://github.com/skundu42/noah-code/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/noah-code.svg)](https://pypi.org/project/noah-code/)

**Noah Code** (`noah-code`) is a terminal coding agent for understanding repositories, planning
changes, editing files, running commands, and carrying work across persistent sessions. It is
built on the [NVIDIA OO Agents (NOOA)](https://github.com/NVIDIA-NeMo/labs-OO-Agents) runtime.

## Install

Install Noah Code and its managed Python runtime with one command. No existing Python, Homebrew,
or system package setup is required.

```bash
curl -LsSf https://raw.githubusercontent.com/skundu42/noah-code/main/install.sh | sh
```
Open a new terminal, move into a repository, and run:

```bash
noah .
```

Noah Code supports macOS on Apple Silicon and Intel, plus Linux on arm64 and x86_64 with Landlock
and seccomp support. You also need an LLM provider account such as OpenAI, Anthropic, OpenRouter or NVIDIA.

## Features

- Read files, search with ripgrep, and inspect Git status, diffs, and history.
- Edit files with anchored replacements, full rewrites, and concurrent-change detection.
- Run permission-gated shell commands with timeouts and streamed output.
- Follow repository instructions from `AGENTS.md`, `CLAUDE.md`, and `.noah-code/instructions.md`.
- Switch between implementation-focused **build** mode and read-only **plan** mode.
- Approve actions once or for a session with ordered `allow`, `ask`, and `deny` rules.
- Undo and redo journaled file edits across process restarts.
- Work in an adaptive Atom One Dark cockpit, a classic console, or one-shot non-interactive mode.
- Type `/` for a live-filtering command and configuration reference; press Enter to send.
- Follow batched live tool output, then inspect compact execution records with `F2`.
- Resume workspace-scoped sessions with todos and automatic context compaction.
- Switch AI models between turns, with optional cross-repository defaults.
- Extend workflows with slash commands, opt-in skills, MCP servers, model selection, and tracing.

## Quick start

Start the TUI in the current repository and describe the desired end state in plain language:

```bash
noah .
```

```text
Find the cause of the failing parser tests, implement the smallest safe fix, and run the
focused test file.
```

Common commands:

```bash
# Open another workspace
noah /path/to/repository

# Use the line-oriented console
noah --console .

# Run one task and exit
noah run "Explain how authentication is wired" .

# Allow actions that would normally ask; explicit deny rules still apply
noah run --auto "Fix the failing unit test" .

# Inspect and plan without editing
noah --mode plan .

# Resume work
noah --continue .
noah --session SESSION_ID .
```

Check the installation or inspect resolved configuration with:

```bash
noah --version
noah doctor .
noah config show .
noah update --check
```

The package also installs `noah-code` and `nc` as equivalent entry points. Because `nc` commonly
refers to netcat, `noah` or `noah-code` is recommended. Keep provider API keys in the environment
or trusted NOOA user configuration, never in a repository or Noah Code session metadata.

Inside a session, `/model MODEL` switches the active model immediately and remembers it when that
session is resumed. It does not change other sessions. Use `/model --global MODEL` when the new
model should also become the default for future sessions in every repository.

The TUI keeps the conversation central and adds a session-and-plan rail on terminals at least
110 columns wide. Tool output streams in a bounded activity panel and compacts after completion,
keeping long runs responsive without deleting persisted session data. Press `F2` for activity
details or `F3` for paginated conversation history.

## Documentation

- [Interactive interface and sessions](docs/interactive-reference.md)
- [Configuration, modes, permissions, and updates](docs/configuration.md)
- [Generated-code security](docs/security.md)
- [Custom commands, skills, MCP, and tracing](docs/extensions.md)
- [Development, CI, and releases](docs/development.md)
- [Release notes](docs/releases/)

## Updates

Noah Code checks PyPI for new versions at most once every 24 hours and updates uv-managed
installations automatically. To check or update immediately:

```bash
noah update --check
noah update
```

## Development

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest tests
uv build
```

See the [development guide](docs/development.md) for platform checks and the release process.

## License

Apache-2.0, as declared in the project metadata. NOOA remains separately licensed by its upstream
project.

## Credits

Built on [NVIDIA OO Agents (NOOA)](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Thanks to the
NVIDIA NeMo team and NOOA contributors for the agent runtime that powers Noah Code.
