# Comparing agent behavior

`scripts/evaluate_agents.py` is a developer harness for matched coding fixtures.
It compares Noah from the current source checkout with **OpenCode 2.0.18**, using
the same upstream model, prompt, baseline Git commit, output-token ceiling and
acceptance command. The included page-range and recursive-merge fixtures are
small functional smoke tests, not evidence of general coding superiority.

## Preview without running models

```bash
uv run python scripts/evaluate_agents.py
```

Dry-run is the default. It reports the prompts, fixture hashes, model and maximum
reserved spend without launching agents, creating worktrees or contacting a
provider. Select one task with `--task page-ranges` or `--task config-merge`.

## Run a matched pair

Install the pinned comparison executable into a temporary directory. This does
not replace an existing global OpenCode installation:

```bash
npm install --prefix /tmp/noah-eval-opencode-2.0.18 --no-audit --no-fund @opencode/cli@2.0.18
uv run python scripts/evaluate_agents.py --live --task page-ranges \
  --opencode /tmp/noah-eval-opencode-2.0.18/node_modules/.bin/opencode \
  --model openrouter/z-ai/glm-5.3-flash \
  --output /tmp/noah-comparison-page-ranges
```

The harness requires an already configured OpenRouter credential, read from
Noah's credential store or the environment. It never logs the credential or
copies it into the agent subprocesses. Both agents receive a temporary local
proxy credential and run against the same exact upstream model.

The guarded proxy checks current public OpenRouter endpoint pricing and parameter
support, pins both clients to the same compatible endpoint, limits output tokens,
and reserves a conservative amount before forwarding each request. The default
total reservation ceiling is **$4**, leaving margin below a $5 authorization.
Reservations are never refunded for failed or uncertain requests. This bound
assumes the provider honors the documented token limits and prices; observed
usage and provider-reported cost are recorded separately. Additional paid tools,
image billing, request fees and unknown model pricing are rejected. The harness
disables web search, web fetch and subagents for these small single-agent tasks.
Each agent receives an equal share of the total reservation and a maximum of
12 model requests, including retries and auxiliary requests.

The fixture configuration explicitly allows local tool operations and denies web
tools and subagents. Noah runs without `--auto`: that option intentionally blocks
interpreter commands even when a broad rule allows them, which prevented the
fixture's requested `python -m unittest` command in rounds 2 and 3. The corrected
launch uses the existing trusted fixture rules; product permission checks,
including protected-secret denies, remain enabled. This harness configuration
is intended for the included curated fixtures, not untrusted repositories.

Each process has a 150-second wall deadline, and each acceptance check has a
30-second deadline. Logs are drained with a 256 KB retention limit per stream,
and on POSIX the harness kills owned process groups when they finish or time out. It uses
temporary Noah/NOOA configuration and OpenCode XDG directories; user settings,
credentials and the global OpenCode executable are left in place.

## Results and interpretation

The output directory contains a fixture baseline repository, separate detached
worktrees for each agent, per-agent state, patches and `results.json`. Results
include baseline commit and fixture hashes, the exact prompt and command, exit
status, elapsed time, acceptance output, explicitly reported usage and proxy
request reservations/costs. Missing usage is recorded as unknown, not zero.
All fixture baselines and acceptance bytes are frozen before model execution.
Scoring copies only the manifest's allowed implementation modules and frozen
tests into a fresh directory. Python runs with `-I`, loading trusted standard
library modules before adding this directory; success requires the expected
test count and a structured success record. Source-fixture hash changes invalidate
the run, and pre-score Git status/diff plus untracked filenames expose test edits.
Worktrees are retained for
inspection and are owned by the generated baseline repository, not the Noah
checkout.

This is task-state separation, not an OS security sandbox: the agent subprocesses
retain normal filesystem access, including the user's home directory and Noah's
source import path. Use an isolated container/account for adversarial workloads.

Compare acceptance success first, then latency and provider cost. A failed model
request or process timeout is an execution result, not evidence that the agent
cannot solve the task. The current source's commit and tracked diff hash are
recorded along with a source-tree hash including new source files, because local
uncommitted changes may be under evaluation. Repeat runs
and add representative repository tasks before drawing broader conclusions.

Noah's `NOAH_CODE_CONFIG` environment variable selects an alternative trusted
user configuration file for isolated runs. It also applies to settings saved by
that process. Omit it to use the normal `~/.config/noah-code/config.toml` path.

Protocol references: [OpenCode v2 run](https://opencode.ai/v2/docs/cli/commands),
[v2 model configuration](https://opencode.ai/v2/docs/models), and
[OpenRouter models](https://openrouter.ai/docs/api-reference/models/get-models).
