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
- Navigate definitions, implementations, references, symbols, hover types, and diagnostics through
  lazily launched language servers plus an mtime-cached repository map.
- Edit files with anchored replacements or atomic multi-file patches with exact preimages,
  rollback, immediate diagnostics, and concurrent-change detection.
- Run permission-gated shell commands with timeouts and streamed output, or own long-running
  servers and watchers as bounded background jobs with cursor-based logs.
- Follow repository instructions from `AGENTS.md`, `CLAUDE.md`, and `.noah-code/instructions.md`.
- Switch between implementation-focused **build** mode and read-only **plan** mode.
- Approve actions once or for a session with ordered `allow`, `ask`, and `deny` rules.
- Undo and redo journaled file edits across process restarts.
- Work in an adaptive themed cockpit, a classic console, or one-shot non-interactive mode.
- Type `/` for a live-filtering command and configuration reference; press Enter to send.
- Follow batched live tool output, then inspect compact execution records with `F2`.
- Review `/diff` in a keyboard-driven staged/worktree ledger with per-file patches, line deltas,
  validation state, changed symbols, explicit revert, and checkpoint undo.
- Resume workspace-scoped sessions with todos and automatic context compaction.
- Keep full oversized tool results privately while sending the model a focused, reopenable preview.
- Inspect token, prompt-cache, model-wait, and tool-output usage with `/tokens`; switch live
  `fast`, `balanced`, and `deep` budgets with `/efficiency`.
- Switch AI models between turns, with optional cross-repository defaults.
- Delegate research to nested **explore** and **general** subagents, or markdown agents in
  `.noah-code/agents/`.
- Fetch pages and search the web, ask structured questions mid-turn, and attach `@files` or images.
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
noah benchmark .
```

On the first TUI launch, Noah opens guided model setup automatically: choose the provider, paste
the key into the masked field, choose the model, then select its reasoning effort. Reopen the same
flow later with `/model`. Like OpenCode,
Noah saves provider credentials in `~/.local/share/noah-code/auth.json`; the directory is
owner-only and the file uses mode `0600`. API-key values are never written to Noah config,
repository, or session files. Set `XDG_DATA_HOME` to relocate the data directory.

Environment variables and the CLI remain available for scripts and headless use:

```bash
export OPENAI_API_KEY="..."        # or ANTHROPIC_API_KEY / OPENROUTER_API_KEY
noah providers list
noah providers add openai --model MODEL_NAME
noah .
```

For reasoning models, choose `default`, `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
Provider and model support varies; `default` omits the parameter. You can set it in the guided
`/model` flow, switch it live with `/reasoning high`, or launch with:

```bash
uv run noah --model openai/MODEL --reasoning-effort high .
```

Custom OpenAI-compatible gateways, vLLM, LM Studio, Ollama, Azure OpenAI, Bedrock, Gemini, Groq,
Mistral, xAI, DeepSeek, Together AI, and Perplexity are also supported. See the
[provider configuration guide](docs/configuration.md#bring-your-own-api-provider).

The package also installs `noah-code` and `nc` as equivalent entry points. Because `nc` commonly
refers to netcat, `noah` or `noah-code` is recommended. Keep provider API keys in Noah's private
auth file or environment, never in a repository or Noah Code session metadata.

Inside a session, bare `/model` opens guided provider, API-key, model, and reasoning setup. `/model MODEL`
switches the active model immediately and remembers it when that session is resumed. It does not
change other sessions. Use `/model --global MODEL` when the new model should also become the
default for future sessions in every repository.

The TUI keeps the conversation central and adds a session-and-plan rail on terminals at least
110 columns wide. Tool output streams in a bounded activity panel and compacts after completion,
keeping long runs responsive without deleting persisted session data. Press `F2` for activity
details or `F3` for paginated conversation history. Switch among Atom One Dark, Noah Ocean,
Graphite, and High Contrast with `/theme`.

The default `fast` profile uses compact NOOA trajectory rendering, bounded tool results, batched
repository inspection, cache-friendly turn-boundary context refresh, and MCP tools from trusted
user-configured servers on the first turn. A
configured `lightweight_model` handles coding-session compaction; deterministic titles avoid an
otherwise unnecessary model request. Run `noah benchmark .` for the deterministic offline fixture
and `/tokens` for real provider-reported usage in the current run.

## Documentation

- [Interactive interface and sessions](docs/interactive-reference.md)
- [Configuration, modes, permissions, and updates](docs/configuration.md)
- [Generated-code security](docs/security.md)
- [Custom commands, skills, MCP, and tracing](docs/extensions.md)
- [Development, CI, and releases](docs/development.md)
- [Release notes](docs/releases/)

## Updates

Noah Code checks PyPI for new versions at most once every 24 hours. New TUI sessions show a
temporary banner when an update is available and retain the version in the context rail. Updates
are explicit by default; to check or install immediately:

```bash
noah update --check
noah update
```

## Development

```bash
uv sync --extra dev --extra mcp --extra tracing
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
