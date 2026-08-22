"""Layered configuration for Noah Code."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from noah_code.themes import ThemeName, get_theme

PermissionAction = Literal["allow", "ask", "deny"]
ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


class PermissionRule(BaseModel):
    """Ordered permission rule; last match wins."""

    category: str = "*"
    pattern: str = "*"
    action: PermissionAction = "ask"
    reason: str = ""


class TracingConfig(BaseModel):
    enabled: bool = True
    jsonl_dir: str | None = None
    viewer: bool = True


class SummarizationPolicy(BaseModel):
    policy: Literal["token_budget", "none"] = "token_budget"
    max_tokens: int | None = None
    trigger_ratio: float = Field(default=0.35, gt=0.05, lt=0.95)
    preserve_recent: int = Field(default=6, ge=2, le=50)
    target_chars: int = Field(default=2500, ge=500, le=20_000)


class EfficiencyConfig(BaseModel):
    """Token and latency controls for the coding harness."""

    profile: Literal["fast", "balanced", "deep"] = "fast"
    strategy: Literal["lean", "standard"] = "lean"
    deterministic_titles: bool = True
    lazy_mcp: bool = False
    max_output_lines: int = Field(default=250, ge=20, le=5000)
    max_search_results: int = Field(default=100, ge=10, le=1000)
    max_file_results: int = Field(default=500, ge=50, le=5000)
    tool_output_retention_hours: int = Field(default=24, ge=1, le=24 * 30)


class UIConfig(BaseModel):
    theme: ThemeName = "atom-one-dark"
    show_reasoning: bool = False
    markdown: bool = True
    stream_shell: bool = True
    frontend: Literal["tui", "console"] = "tui"


class UpdateConfig(BaseModel):
    auto_install: bool = False
    interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    check_timeout_seconds: float = Field(default=3.0, gt=0, le=30)


class LSPConfig(BaseModel):
    """Lazy local language-server settings."""

    enabled: bool = True
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    max_symbols: int = Field(default=300, ge=20, le=5000)
    servers: dict[str, list[str]] = Field(default_factory=dict)


class ProcessConfig(BaseModel):
    """Bounds for background commands owned by one Noah session."""

    max_jobs: int = Field(default=8, ge=1, le=32)
    max_runtime_seconds: float = Field(default=3600.0, gt=1, le=86_400)
    max_buffer_chars: int = Field(default=64_000, ge=4000, le=2_000_000)
    stop_grace_seconds: float = Field(default=2.0, gt=0, le=30)


class SamplingConfig(BaseModel):
    """Deterministic sampling controls forwarded to the provider client."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0)

    def overrides(self) -> dict[str, Any]:
        return {k: v for k in ("temperature", "top_p", "seed") if (v := getattr(self, k)) is not None}


class BudgetConfig(BaseModel):
    """Hard session caps; the first breach cancels the turn and exits."""

    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    max_seconds: float | None = Field(default=None, gt=0.0)


class HookSpec(BaseModel):
    """One shell hook bound to tool names via a glob pattern."""

    match: str = "*"
    command: str
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=600.0)


class HooksConfig(BaseModel):
    pre_tool: list[HookSpec] = Field(default_factory=list)
    post_tool: list[HookSpec] = Field(default_factory=list)


class CheckpointConfig(BaseModel):
    """Automatic git worktree snapshots captured at turn boundaries."""

    enabled: bool = False
    max_per_session: int = Field(default=50, ge=1, le=500)


class NoahCodeConfig(BaseModel):
    """Resolved configuration for a noah-code run."""

    model: str = "gpt-4o-mini"
    reasoning_effort: ReasoningEffort = "default"
    lightweight_model: str | None = None
    max_iterations: int = 40
    cell_timeout: float = 120.0
    command_timeout: float = 60.0
    summarization: SummarizationPolicy = Field(default_factory=SummarizationPolicy)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    session_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "noah-code" / "sessions"
    )
    permission_rules: list[PermissionRule] = Field(default_factory=list)
    auto_approve: bool = False
    enabled_skills: list[str] = Field(default_factory=list)
    mcp: dict[str, Any] = Field(default_factory=dict)
    ui: UIConfig = Field(default_factory=UIConfig)
    updates: UpdateConfig = Field(default_factory=UpdateConfig)
    lsp: LSPConfig = Field(default_factory=LSPConfig)
    processes: ProcessConfig = Field(default_factory=ProcessConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mode: Literal["build", "plan"] = "build"
    max_file_bytes: int = 512_000
    max_output_chars: int = Field(default=16_000, ge=1000, le=1_000_000)
    undo_blob_limit: int = 2_000_000
    unsafe_inprocess_code_execution: bool = False

    @field_validator("session_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return (
            Path(value).expanduser()
            if value is not None
            else Path.home() / ".local/share/noah-code/sessions"
        )


DEFAULT_PERMISSION_RULES: list[PermissionRule] = [
    PermissionRule(category="read", pattern="*", action="allow", reason="reads allowed"),
    PermissionRule(
        category="read",
        pattern="**/.env",
        action="deny",
        reason="secret env files denied",
    ),
    PermissionRule(
        category="read",
        pattern="**/.env.*",
        action="deny",
        reason="secret env files denied",
    ),
    PermissionRule(
        category="read",
        pattern="**/.env.example",
        action="allow",
        reason="example env files are safe",
    ),
    PermissionRule(
        category="read",
        pattern="**/*.pem",
        action="deny",
        reason="private keys denied",
    ),
    PermissionRule(
        category="read",
        pattern="**/*id_rsa*",
        action="deny",
        reason="private keys denied",
    ),
    PermissionRule(
        category="read",
        pattern="**/.git/**",
        action="deny",
        reason=".git internals denied",
    ),
    PermissionRule(
        category="read",
        pattern="**/noah-code/**/*.db",
        action="deny",
        reason="session databases denied",
    ),
    PermissionRule(category="edit", pattern="*", action="ask", reason="edits require approval"),
    PermissionRule(category="bash", pattern="*", action="ask", reason="shell requires approval"),
    PermissionRule(
        category="bash",
        pattern="git push*",
        action="deny",
        reason="push denied by default",
    ),
    PermissionRule(
        category="bash",
        pattern="git clean*",
        action="deny",
        reason="destructive git clean denied",
    ),
    PermissionRule(
        category="bash",
        pattern="git reset --hard*",
        action="deny",
        reason="destructive git reset denied",
    ),
    PermissionRule(
        category="external_directory",
        pattern="*",
        action="ask",
        reason="external paths require approval",
    ),
    PermissionRule(category="task", pattern="*", action="ask", reason="subagents require approval"),
    PermissionRule(category="skill", pattern="*", action="ask", reason="skills require approval"),
    PermissionRule(category="mcp", pattern="*", action="ask", reason="MCP requires approval"),
    PermissionRule(category="lsp", pattern="*", action="allow", reason="LSP read-only by default"),
    PermissionRule(
        category="webfetch",
        pattern="*",
        action="ask",
        reason="web fetches require approval",
    ),
    PermissionRule(
        category="websearch",
        pattern="*",
        action="ask",
        reason="web searches require approval",
    ),
    PermissionRule(
        category="question",
        pattern="*",
        action="allow",
        reason="asking the user is allowed",
    ),
]


def _user_config_path() -> Path:
    return Path.home() / ".config" / "noah-code" / "config.toml"


_TOP_LEVEL_MODEL_RE = re.compile(r"^(?P<indent>\s*)model\s*=.*$")
_TOP_LEVEL_REASONING_RE = re.compile(r"^(?P<indent>\s*)reasoning_effort\s*=.*$")


def user_default_model() -> str | None:
    """Return the explicitly configured cross-repository model, if any."""

    value = _load_toml(_user_config_path()).get("model")
    return value.strip() if isinstance(value, str) and value.strip() else None


def save_user_default_model(model: str) -> Path:
    """Persist a top-level model while preserving the rest of the user TOML file."""

    selected = model.strip()
    if not selected or any(character.isspace() for character in selected):
        raise ValueError("model must be a non-empty name without whitespace")

    path = _user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = path.read_text() if path.is_file() else ""
    lines = existing.splitlines(keepends=True)
    encoded = json.dumps(selected, ensure_ascii=False)
    replacement = f"model = {encoded}\n"

    first_table = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("[")),
        len(lines),
    )
    model_line = next(
        (
            index
            for index, line in enumerate(lines[:first_table])
            if _TOP_LEVEL_MODEL_RE.match(line)
        ),
        None,
    )
    if model_line is not None:
        lines[model_line] = replacement
    else:
        if first_table and not lines[first_table - 1].endswith(("\n", "\r")):
            lines[first_table - 1] += "\n"
        insertion = [replacement]
        if first_table < len(lines) and lines[first_table].lstrip().startswith("["):
            insertion.append("\n")
        lines[first_table:first_table] = insertion

    content = "".join(lines) or replacement
    descriptor, temporary_name = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def save_user_reasoning_effort(effort: str) -> Path:
    """Persist the cross-repository reasoning effort without disturbing other settings."""

    selected = effort.strip().lower()
    if selected not in REASONING_EFFORTS:
        raise ValueError(
            "reasoning effort must be default, none, minimal, low, medium, high, or xhigh"
        )

    path = _user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = path.read_text() if path.is_file() else ""
    lines = existing.splitlines(keepends=True)
    replacement = f"reasoning_effort = {json.dumps(selected)}\n"
    first_table = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("[")),
        len(lines),
    )
    effort_line = next(
        (
            index
            for index, line in enumerate(lines[:first_table])
            if _TOP_LEVEL_REASONING_RE.match(line)
        ),
        None,
    )
    if effort_line is not None:
        lines[effort_line] = replacement
    else:
        if first_table and not lines[first_table - 1].endswith(("\n", "\r")):
            lines[first_table - 1] += "\n"
        insertion = [replacement]
        if first_table < len(lines) and lines[first_table].lstrip().startswith("["):
            insertion.append("\n")
        lines[first_table:first_table] = insertion

    content = "".join(lines) or replacement
    descriptor, temporary_name = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def save_user_theme(theme: str) -> Path:
    """Persist the UI theme inside the user ``[ui]`` table."""

    selected = get_theme(theme.strip().lower()).name
    path = _user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = path.read_text() if path.is_file() else ""
    lines = existing.splitlines(keepends=True)
    replacement = f"theme = {json.dumps(selected)}\n"

    table_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "[ui]"),
        None,
    )
    if table_start is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(["[ui]\n", replacement])
    else:
        table_end = next(
            (
                index
                for index, line in enumerate(lines[table_start + 1 :], start=table_start + 1)
                if line.lstrip().startswith("[")
            ),
            len(lines),
        )
        theme_line = next(
            (
                index
                for index, line in enumerate(lines[table_start + 1 : table_end], start=table_start + 1)
                if re.match(r"^\s*theme\s*=", line)
            ),
            None,
        )
        if theme_line is None:
            lines.insert(table_end, replacement)
        else:
            lines[theme_line] = replacement

    descriptor, temporary_name = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write("".join(lines))
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def _project_config_path(workspace: Path) -> Path:
    return workspace / ".noah-code" / "config.toml"


# Repository-controlled configuration must not be able to weaken the host's
# trust boundary. These settings are accepted only from user config, the
# environment, or explicit CLI overrides.
_USER_ONLY_CONFIG_KEYS = frozenset(
    {
        "auto_approve",
        "budget",
        "efficiency",
        "enabled_skills",
        "hooks",
        "mcp",
        "lsp",
        "permission_rules",
        "processes",
        "session_dir",
        "tracing",
        "unsafe_inprocess_code_execution",
        "updates",
    }
)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return data if isinstance(data, dict) else {}


def _load_project_toml(path: Path) -> dict[str, Any]:
    data = _load_toml(path)
    return {key: value for key, value in data.items() if key not in _USER_ONLY_CONFIG_KEYS}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if model := os.environ.get("NOAH_CODE_MODEL"):
        out["model"] = model
    if light := os.environ.get("NOAH_CODE_LIGHTWEIGHT_MODEL"):
        out["lightweight_model"] = light
    if reasoning_effort := os.environ.get("NOAH_CODE_REASONING_EFFORT"):
        out["reasoning_effort"] = reasoning_effort.lower()
    if auto := os.environ.get("NOAH_CODE_AUTO"):
        out["auto_approve"] = auto.lower() in {"1", "true", "yes", "on"}
    if session_dir := os.environ.get("NOAH_CODE_SESSION_DIR"):
        out["session_dir"] = session_dir
    if mode := os.environ.get("NOAH_CODE_MODE"):
        out["mode"] = mode
    if unsafe := os.environ.get("NOAH_CODE_UNSAFE_INPROCESS"):
        out["unsafe_inprocess_code_execution"] = unsafe.lower() in {"1", "true", "yes", "on"}
    if auto_update := os.environ.get("NOAH_CODE_AUTO_UPDATE"):
        out["updates"] = {"auto_install": auto_update.lower() in {"1", "true", "yes", "on"}}
    if efficiency_profile := os.environ.get("NOAH_CODE_EFFICIENCY"):
        out["efficiency"] = {"profile": efficiency_profile.lower()}
    return out


def _normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    if "permission_rules" in data and isinstance(data["permission_rules"], list):
        data["permission_rules"] = [
            rule if isinstance(rule, PermissionRule) else PermissionRule.model_validate(rule)
            for rule in data["permission_rules"]
        ]
    if "summarization" in data and isinstance(data["summarization"], dict):
        data["summarization"] = SummarizationPolicy.model_validate(data["summarization"])
    if "tracing" in data and isinstance(data["tracing"], dict):
        data["tracing"] = TracingConfig.model_validate(data["tracing"])
    if "ui" in data and isinstance(data["ui"], dict):
        data["ui"] = UIConfig.model_validate(data["ui"])
    if "updates" in data and isinstance(data["updates"], dict):
        data["updates"] = UpdateConfig.model_validate(data["updates"])
    if "efficiency" in data and isinstance(data["efficiency"], dict):
        data["efficiency"] = EfficiencyConfig.model_validate(data["efficiency"])
    return data


def load_config(
    workspace: Path,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> NoahCodeConfig:
    """Load config with precedence: defaults < user < project < env < CLI."""
    merged: dict[str, Any] = {
        "permission_rules": [r.model_dump() for r in DEFAULT_PERMISSION_RULES],
    }
    merged = _deep_merge(merged, _load_toml(_user_config_path()))
    merged = _deep_merge(merged, _load_project_toml(_project_config_path(workspace)))
    merged = _deep_merge(merged, _env_overrides())
    if cli_overrides:
        cleaned = {k: v for k, v in cli_overrides.items() if v is not None}
        extra_rules = cleaned.pop("extra_permission_rules", None)
        merged = _deep_merge(merged, cleaned)
        if extra_rules:
            # CLI --allow/--deny rules append after file rules; the engine is
            # last-match-wins, so explicit denies outrank earlier allows.
            existing = list(merged.get("permission_rules") or [])
            merged["permission_rules"] = [*existing, *extra_rules]
    return NoahCodeConfig.model_validate(_normalize_raw(merged))


def config_sources(workspace: Path) -> dict[str, Path | None]:
    """Return config file locations for diagnostics."""
    user = _user_config_path()
    project = _project_config_path(workspace)
    return {
        "user": user if user.is_file() else None,
        "project": project if project.is_file() else None,
    }
