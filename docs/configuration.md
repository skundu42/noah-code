# Configuration, permissions, and updates

## Configuration

Configuration is merged in this order, with later layers taking precedence:

1. Built-in defaults.
2. User config at `~/.config/noah-code/config.toml`.
3. Project config at `.noah-code/config.toml`.
4. `NOAH_CODE_*` environment variables.
5. CLI flags.

### First-run model setup

The first interactive `noah` launch asks for an AI model before starting the agent. Noah Code
saves that choice as the top-level `model` in `~/.config/noah-code/config.toml`, making it the
default for every repository. Enter a LiteLLM model name or an alias from the NOOA model registry.

An explicit `noah --model MODEL` on the first launch is saved without prompting. Project config,
`NOAH_CODE_MODEL`, and later `--model` flags still override the user default according to the
precedence above. Non-interactive commands such as `noah run`, `doctor`, and `config show` never
open the onboarding prompt.

Inside an interactive session, switch only the current session or replace the global default:

```text
/model
/model openai/MODEL_NAME
/model --global anthropic/MODEL_NAME
```

Bare `/model` opens a three-step TUI flow: search for a provider, enter its API key in a masked
field, and enter the model ID. Noah attempts to save that key in the operating system credential
store. If no secure backend is available, the key remains active only in the current Noah process
and the TUI says so. Keys are never written to Noah configuration or session metadata.

Model switches take effect between turns and are stored in the current session metadata, so a
resumed session continues with its most recently selected model. A session-only `/model MODEL`
does not change the user configuration or affect new sessions.

### Bring your own API provider

Enter `/model` for the common provider → API key → model flow. Run `noah providers list` or open
`/providers` in the TUI for advanced, secret-free setup. Noah supports
LiteLLM's provider routing and includes guided presets for OpenAI, Anthropic Claude, OpenRouter,
Google Gemini, Groq, Mistral, xAI, DeepSeek, Together AI, Perplexity, Azure OpenAI, Amazon
Bedrock, and local Ollama.

For scripting, export credentials before starting Noah and select a provider/model:

```bash
# OpenAI
export OPENAI_API_KEY="..."
noah providers add openai --model MODEL_NAME

# Anthropic Claude
export ANTHROPIC_API_KEY="..."
noah providers add anthropic --model MODEL_NAME

# OpenRouter
export OPENROUTER_API_KEY="..."
noah providers add openrouter --model PROVIDER/MODEL

# Google Gemini (GOOGLE_API_KEY is also accepted)
export GEMINI_API_KEY="..."
noah providers add gemini --model MODEL_NAME
```

Use `--no-set-default` to print a one-launch command without changing Noah's global default.
Inside Noah, `/providers use openrouter PROVIDER/MODEL` switches the current session and saves the
new global default.

For vLLM, LM Studio, a company gateway, or another OpenAI-compatible API, create a secret-free
NOOA model alias:

```bash
export COMPANY_LLM_API_KEY="..."
noah providers add custom \
  --alias company-llm \
  --model MODEL_ID \
  --base-url https://llm.example.com/v1 \
  --api-key-env COMPANY_LLM_API_KEY
```

The alias is stored in `~/.config/nooa/llm_config.yaml` with mode `0600`. Only the environment
variable's name is stored. For an unauthenticated local endpoint, omit `--api-key-env`. You can
then use the alias anywhere a model is accepted: `noah --model company-llm .` or
`/model company-llm`.

Example user configuration:

```toml
model = "gpt-4o-mini"
lightweight_model = "gpt-4o-mini"
mode = "build"
max_iterations = 40
cell_timeout = 120
command_timeout = 60

[ui]
theme = "atom-one-dark"
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

Inspect the resolved configuration from the CLI or inside an interactive session. `/config`
lists every nested path, while an optional path scopes the output. Values whose names look like
credentials are redacted.

```bash
noah config show .
```

```text
/config
/config summarization
/config updates.auto_install
```

Model and provider configuration follows NOOA conventions, including its model registry,
environment variables, and configuration under `~/.config/nooa/`. Provider strings not shown in
the guided list still pass through to LiteLLM, so additional supported services can be selected
with `--model PROVIDER/MODEL`.

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
