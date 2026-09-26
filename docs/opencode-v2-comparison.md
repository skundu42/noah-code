# Noah and OpenCode v2: gaps and implementation

Assessment dated 2026-09-26. Noah baseline: `8bf3018` (0.8.0); comparison
executable: OpenCode **2.0.18**, installed separately from the user's v1 CLI.
Feature claims below use v2 documentation, not v1 assumptions. Implementation
status refers to Noah 1.0.0.

Noah already used Git worktrees for separate top-level sessions. The initial
gap was that delegated child agents had in-memory conversations and shared
their parent's checkout. Noah also already had checkpoints, recovery,
permissions, MCP, LSP, skills, compaction, and coordinated teams.

## Largest gaps, in priority order

| Area | Baseline gap | Implemented here | Remaining difference |
| --- | --- | --- | --- |
| Immediate feedback and setup | Buffered provider responses; mostly manual model IDs | Real completion-stream preview; model discovery; first-run permission choices; Codex account sign-in | Generic Responses transport still buffers; broader account management and local-runtime discovery remain |
| Service and editor architecture | Host embedded in terminal process; no public service/ACP | Shared `AgentService` boundary, bearer-authenticated HTTP/SSE, ACP v1 stdio adapter | TUI still runs locally; no auto-discovered daemon, web/desktop client, or complete ACP surface |
| Child autonomy | Ephemeral child context; serialized shared writers | Durable child snapshots/status, follow-ups, independent cancellation, optional retained worktrees | Child execution stops with owner; no automatic merge or autonomous restart after a crash |
| Browser and extensions | General MCP and tool hooks only | Pinned isolated Playwright MCP preset; session/turn/worktree hooks | No embedded browser UI or versioned general-purpose plugin SDK/marketplace |
| Evidence | Feature comparison did not establish coding quality | Same-model fixture harness with identical prompts, baselines, acceptance tests and enforced budgets | Small smoke tasks cannot establish general superiority |

OpenCode's default clients share a background server owning sessions and tool
execution. That makes a service boundary the largest architectural investment,
because it enables several clients without duplicating the agent loop. Noah's
new explicit service supplies that boundary; automatic daemon lifecycle and
client attachment remain separate work. [OpenCode v2 CLI](https://opencode.ai/v2/docs/cli)

OpenCode's ACP implementation supports more session operations, model/mode
options, media inputs, and incremental conversational output. Noah supports
initialization, new/load, prompts, cancellation, permissions, and history, with
capabilities advertised conservatively. Noah's raw CodeAct generation is
provisional and cannot safely be sent as irrevocable ACP text chunks.
[OpenCode v2 ACP](https://opencode.ai/v2/docs/cli/acp)

OpenCode supports foreground/background child sessions and configurable agent
permissions. Persistence and independent controls therefore matter before
worktree isolation: isolation solves conflicting edits but does not make a child
resumable. [OpenCode v2 agents](https://opencode.ai/v2/docs/agents)

Provider account flows and local runtime discovery still need more breadth in
Noah. OpenCode documents device OAuth for Copilot, account management, and
discovery for Ollama, LM Studio, and vLLM. Noah's new picker covers API keys,
compatible endpoints, and Codex/ChatGPT account sign-in through the official
Codex CLI. The Codex adapter requires CLI 0.153.4 or newer and uses experimental
app-server fields; it does not supply the other providers' account flows.
[OpenCode v2 providers](https://opencode.ai/v2/docs/providers),
[account commands](https://opencode.ai/v2/docs/cli/commands)

Noah's observational hooks complement existing skills, tools, and MCP. They are
a smaller extension contract than OpenCode's loadable packages, reload behavior,
and separate server/TUI plugins. [OpenCode v2 plugins](https://opencode.ai/v2/docs/plugins)

## Verification and evaluation

See [service](service.md), [persistent children](persistent-tasks.md),
[browser](browser.md), and [hooks](lifecycle-hooks.md) for contracts and limits.
The [evaluation guide](evaluation.md) describes reproducible fixtures, pinned
versions, equal request/time/output/spend limits, immutable acceptance tests,
and recorded usage. Compare acceptance first; neither a feature checklist nor
a single successful task establishes overall agent quality.

The [recorded trials](evaluation-results.md) exposed concrete execution defects:
missing tool-contract discovery and sandbox corruption of multiline string
literals. The working tree now supplies bounded `self.tools.help()` references,
preserves those literals, checks Python syntax before edits, guards completion
with verification evidence, and bounds recovery from empty responses caused by
output-token exhaustion. In the final four-run trial, Noah and OpenCode each
passed both acceptance suites and completed both tasks with exit code 0. The
harness also corrected Noah's permission launch so its explicitly allowed
unittest command could run; product security checks were preserved. Earlier
failures, the configuration change, and all costs remain documented, so the
result is evidence for these targeted repairs rather than a general ranking.
