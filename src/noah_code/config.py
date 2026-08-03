"""Layered configuration for Noah Code."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


PermissionAction = Literal["allow", "ask", "deny"]


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
    preserve_recent: int = 10
    target_chars: int = 4000


class UIConfig(BaseModel):
    theme: Literal["atom-one-dark"] = "atom-one-dark"
    show_reasoning: bool = False
    markdown: bool = True
    stream_shell: bool = True
    frontend: Literal["tui", "console"] = "tui"


class UpdateConfig(BaseModel):
    auto_install: bool = True
    interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    check_timeout_seconds: float = Field(default=3.0, gt=0, le=30)


class NoahCodeConfig(BaseModel):
    """Resolved configuration for a noah-code run."""

    model: str = "gpt-4o-mini"
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
    mode: Literal["build", "plan"] = "build"
    max_file_bytes: int = 512_000
    max_output_chars: int = 80_000
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
]


def _user_config_path() -> Path:
    return Path.home() / ".config" / "noah-code" / "config.toml"


_TOP_LEVEL_MODEL_RE = re.compile(r"^(?P<indent>\s*)model\s*=.*$")


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
        (index for index, line in enumerate(lines[:first_table]) if _TOP_LEVEL_MODEL_RE.match(line)),
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


def _project_config_path(workspace: Path) -> Path:
    return workspace / ".noah-code" / "config.toml"


# Repository-controlled configuration must not be able to weaken the host's
# trust boundary. These settings are accepted only from user config, the
# environment, or explicit CLI overrides.
_USER_ONLY_CONFIG_KEYS = frozenset(
    {
        "auto_approve",
        "enabled_skills",
        "mcp",
        "permission_rules",
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
    if auto := os.environ.get("NOAH_CODE_AUTO"):
        out["auto_approve"] = auto.lower() in {"1", "true", "yes", "on"}
    if session_dir := os.environ.get("NOAH_CODE_SESSION_DIR"):
        out["session_dir"] = session_dir
    if mode := os.environ.get("NOAH_CODE_MODE"):
        out["mode"] = mode
    if unsafe := os.environ.get("NOAH_CODE_UNSAFE_INPROCESS"):
        out["unsafe_inprocess_code_execution"] = unsafe.lower() in {"1", "true", "yes", "on"}
    if auto_update := os.environ.get("NOAH_CODE_AUTO_UPDATE"):
        out["updates"] = {
            "auto_install": auto_update.lower() in {"1", "true", "yes", "on"}
        }
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
        merged = _deep_merge(merged, cleaned)
    return NoahCodeConfig.model_validate(_normalize_raw(merged))


def config_sources(workspace: Path) -> dict[str, Path | None]:
    """Return config file locations for diagnostics."""
    user = _user_config_path()
    project = _project_config_path(workspace)
    return {
        "user": user if user.is_file() else None,
        "project": project if project.is_file() else None,
    }
