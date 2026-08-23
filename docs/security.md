# Generated-code security

NOOA CodeAct lets a model express a turn as Python, so Noah Code treats that interpreter as
untrusted. Generated cells always execute in a separate, brokered worker.

On Linux, the worker uses:

- Landlock filesystem confinement.
- seccomp network isolation.
- Memory, CPU, and execution-time limits.
- Restricted imports.
- No direct workspace mount.
- A narrow broker exposing only permission-gated workspace, Git, message, mode, and todo
  operations.

On macOS, the worker starts with Python's clean `spawn` process model rather than forking the
multithreaded TUI, then installs a native deny-by-default Sandbox profile. It blocks repository file
contents and outbound networking, permits reads only from the managed Python runtime, and applies
CPU and execution-time limits. The workspace is never mounted directly, and approved work still
crosses the same narrow parent broker. Linux additionally enforces an address-space memory limit.

Execution fails closed if these controls cannot be installed. For trusted development tests only,
containment can be disabled with either of the following:

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

## Host-side safety

The generated-code sandbox is only one layer. Host-owned tools also enforce:

- canonical workspace path checks and hard secret-file denials;
- ordered `allow`, `ask`, and `deny` rules that `--auto` cannot use to override explicit denials or
  elevated-risk approval;
- an exclusive checkout lease enabled by default, preventing two Noah processes from mutating the
  same worktree;
- serialized mutating subagents and atomic multi-file patches with exact pre-images;
- durable file intents that restore interrupted workspace-tool writes after a crash;
- public-unicast DNS pinning and redirect validation for web requests; and
- durable effect records for structured GitHub and mutating MCP operations.

Approval and structured-question waits have a configurable timeout and are marked interrupted
rather than silently reused after a process restart. Generic shell commands remain outside the
transactional file journal and may affect ignored files or remote systems; Noah checkpoints Git
state before mutating shell execution, marks the turn as not fully reversible, and reports that
limitation in `/status`.

See [Reliability and long-running sessions](reliability.md) for crash recovery and external-effect
replay behavior.
