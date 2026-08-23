# Configuration, permissions, and updates

## Configuration

Configuration is merged in this order, with later layers taking precedence:

1. Built-in defaults.
2. User config at `~/.config/noah-code/config.toml`.
3. Project config at `.noah-code/config.toml`.
4. `NOAH_CODE_*` environment variables.
5. CLI flags.

### First-run model setup

The first TUI launch opens one guided provider → API key → model → reasoning flow before starting
the agent. Noah Code saves the selected model as the top-level `model` in
`~/.config/noah-code/config.toml`, making it the default for every repository, and stores provider
credentials separately in its private auth file. The classic `--console` frontend retains a
line-oriented model prompt for environments that cannot open the TUI.

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

Bare `/model` opens a guided TUI flow: search for a provider, enter its API key in a masked
field, enter the model ID, and select reasoning effort. Noah saves the credential in
`~/.local/share/noah-code/auth.json`, using the same provider-keyed record shape as OpenCode.
The containing directory uses mode `0700` and the file uses mode `0600`. If the file cannot be
written, the key remains active only in the current Noah process and the TUI says so. Keys are
never written to Noah configuration or session metadata. `XDG_DATA_HOME` relocates the data root.

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
reasoning_effort = "default" # default, none, minimal, low, medium, high, or xhigh
lightweight_model = "gpt-4o-mini"
mode = "build"
max_iterations = 40
cell_timeout = 120
command_timeout = 60
max_output_chars = 16000

[efficiency]
profile = "fast"          # "fast", "balanced", or "deep"
strategy = "lean"         # "standard" is the comparison fallback
deterministic_titles = true
lazy_mcp = false         # true catalogs servers without attaching them at start
max_output_lines = 250
max_search_results = 100
max_file_results = 500
tool_output_retention_hours = 24

[lsp]
enabled = true
timeout_seconds = 5
max_symbols = 300
# Trusted user config may override a language server command:
# servers.python = ["basedpyright-langserver", "--stdio"]

[processes]
max_jobs = 8
max_runtime_seconds = 3600
max_buffer_chars = 64000
stop_grace_seconds = 2

[ui]
theme = "atom-one-dark" # atom-one-dark, noah-ocean, graphite, or high-contrast
frontend = "tui"       # "tui" or "console"
markdown = true
stream_shell = true
show_reasoning = false

[summarization]
policy = "token_budget" # or "none"
trigger_ratio = 0.35
preserve_recent = 6
target_chars = 2500

[tracing]
enabled = true
viewer = true
# jsonl_dir = "~/.local/share/noah-code/traces"

[updates]
auto_install = false
interval_hours = 24
```

Supported environment overrides include:

- `NOAH_CODE_MODEL`
- `NOAH_CODE_LIGHTWEIGHT_MODEL`
- `NOAH_CODE_REASONING_EFFORT`
- `NOAH_CODE_AUTO`
- `NOAH_CODE_SESSION_DIR`
- `NOAH_CODE_MODE`
- `NOAH_CODE_EFFICIENCY`
- `NOAH_CODE_UNSAFE_INPROCESS`
- `NOAH_CODE_AUTO_UPDATE`

Repository-controlled configuration cannot weaken the host trust boundary. Project config is
ignored for `auto_approve`, `efficiency`, `enabled_skills`, `lsp`, `mcp`, `permission_rules`,
`processes`, `session_dir`, `tracing`, `updates`, and `unsafe_inprocess_code_execution`. Put those
settings in trusted user config, the environment, or an explicit CLI flag. Language-server
overrides are user-only because they launch local executables.

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

Reasoning effort is passed through NOOA to LiteLLM only when it is not `default`. Supported values
are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`, but each provider/model may support
only a subset. Change the current session or the cross-repository default with:

```text
/reasoning
/reasoning high
/reasoning --global low
```

For headless launches use `--reasoning-effort high`, or add
`--reasoning-effort high` to `noah providers add ...` when saving a global model default.

### Efficiency and model routing

`fast` is the default: 16,000 characters and 250 lines per model-facing tool result. Configured MCP
servers from trusted user configuration attach at start so their tools are in the prompt; set
`efficiency.lazy_mcp = true` to catalog them instead. The agent runs until it finishes; if a
provider reports a context overflow, Noah compacts eligible history and retries that step once.
`balanced` raises the preview to 24,000
characters/400 lines. `deep` permits legacy-sized 80,000-character previews. `max_iterations` is a
safety rail (default 40), not an efficiency-profile cap. Switch without restarting:

```text
/efficiency
/efficiency balanced
/efficiency deep
```

Oversized results are not discarded. Noah writes the exact output to a private cache file for the
configured retention period, returns a bounded head/tail preview, and gives the agent an output ID
for focused line-range retrieval. A truncated file preview is never returned as an editable Match
anchor.

Set `lightweight_model` to route compaction to a faster or cheaper model. If it is omitted, that
route follows live `/model` switches. Compaction starts at 35% of the active main model's context
window by default, preserves the six newest events, and writes a coding checkpoint covering the
objective, decisions, files, validation, blockers, and next steps.

## Modes and permissions

| Mode | Behavior |
|------|----------|
| `build` | Reads are allowed; edits and shell commands follow permission rules and ask by default |
| `plan` | Reads are allowed; file edits and mutating shell commands are denied. `self.plan.write` may pin `.noah-code/plan.md` |

Switch modes with `--mode`, `/mode build`, or `/mode plan`. The agent can propose a switch with
`self.plan.enter()` / `self.plan.exit_to_build()` after writing a plan. The active mode is stored with the
session.

Permission rules are evaluated in order, and the last matching rule wins. The default policy:

- Allows ordinary reads.
- Denies likely secrets, including `.env` variants, private keys, `.git` internals, and session
  databases. `.env.example` remains readable.
- Asks before workspace edits, shell commands, web fetches, web searches, and subagents.
- Allows the question tool so the agent can pause for a structured choice.
- Denies `git push`, `git clean`, `git reset --hard`, and mutating `gh pr`
  (`create`, `checkout`, `merge`, `close`, `ready`, `review`). Push and PR
  mutations go through `/pr` / `self.github` instead.
- Allows listing and viewing pull requests; asks before create, push, checkout,
  or comment.
- Keeps file tools inside the active workspace and asks before skill or MCP access.
- Denies plan-mode mutations regardless of broader allow rules. Plan mode may still run
  read-only subagents.

`--auto` changes routine ask decisions to allow but never overrides an explicit deny.
Elevated-risk commands such as file removal, downloads, and package installation still require
explicit approval. Compound shell commands and mutating or unrecognized Git commands cannot be
silently auto-approved.

## Installation and updates

The README executes the checked-in bootstrapper directly from GitHub. From a local clone, run the
same installer with:

```bash
sh install.sh
```

Installs created by the one-line command check PyPI at most once every 24 hours. If a release is
available, a new TUI session shows a temporary banner and keeps the version in the context rail.
Run `noah update` when ready. When an update is installed, Noah exits before starting the task so
it cannot mix old and new runtime modules. Rerun the command to continue on the new version.

```bash
# Check without changing the installation
noah update --check

# Update immediately
noah update
```

Updates are notification-only by default. Enable unattended installation from trusted user
configuration or the environment only when that behavior is desired:

```toml
[updates]
auto_install = true
interval_hours = 24
```

```bash
export NOAH_CODE_AUTO_UPDATE=1
```
