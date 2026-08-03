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

On macOS, the worker uses a native deny-by-default Sandbox profile. It blocks repository file
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
