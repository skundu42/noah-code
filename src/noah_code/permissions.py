"""Deterministic allow/ask/deny permission engine."""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

from noah_code.config import PermissionRule

PermissionAction = Literal["allow", "ask", "deny"]


class PermissionCategory(StrEnum):
    READ = "read"
    EDIT = "edit"
    BASH = "bash"
    EXTERNAL_DIRECTORY = "external_directory"
    TASK = "task"
    SKILL = "skill"
    MCP = "mcp"
    LSP = "lsp"
    WEBFETCH = "webfetch"
    WEBSEARCH = "websearch"
    QUESTION = "question"
    GITHUB = "github"


# Patterns that are always denied regardless of mode / auto.
_ALWAYS_DENY_BASH = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\brm\s+(-[^\s]*\s+)*-?[rR]?[fF]?[rR]?[fF]?\s+(/|\.|~|\*)",
        r"\brm\s+.*\s+(-[^\s]*r|-rf|-fr)\b",
        r"\b(mkfs|dd\s+if=|/dev/sd|/dev/disk|shred\b|wipefs\b)\b",
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;:",  # fork bomb
        r"\bgit\s+push\b",
        r"\bgit\s+clean\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgh\s+pr\s+(create|checkout|merge|close|ready|review)\b",
        r"\bgit\s+filter-branch\b",
        r"\bexport\s+-p\b",
        r"\bcat\s+.*\.pem\b",
        r"\bcat\s+.*id_rsa\b",
        r"\bchmod\s+-R\s+777\b",
        r"\bcurl\b.*\|\s*(ba)?sh\b",
        r"\bwget\b.*\|\s*(ba)?sh\b",
        r"\bsudo\b",
        r"\bchmod\b.+\s+/",
        r"\bchown\b.+\s+/",
    )
)

# Mutating patterns that always require ask (even if a broad allow matched earlier
# via auto) unless already an explicit session allow - handled by forcing ask
# when compound/uncertain; these bump deny-adjacent risk to ask minimum.
_ALWAYS_ASK_BASH = (
    re.compile(r"\brm\b"),
    re.compile(r"\bmv\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\bdocker\s+(rm|rmi|system\s+prune)\b"),
    re.compile(r"\bnpm\s+(publish|unpublish)\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\buv\s+pip\s+install\b"),
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
)

_MUTATING_GIT = re.compile(
    r"\bgit\s+(commit|add|push|pull|fetch|rebase|merge|reset|clean|checkout|stash|tag|remote)\b"
)
_READ_ONLY_PREFIXES = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "git rev-parse",
    "rg ",
    "grep ",
    "egrep ",
    "fgrep ",
    "find ",
    "ls ",
    "pwd",
    "head ",
    "tail ",
    "wc ",
    "file ",
    "stat ",
    "test ",
    "pytest --collect-only",
    "python -m pytest --collect-only",
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({"branch", "diff", "log", "rev-parse", "show", "status"})
_FIND_MUTATING_FLAGS = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fls",
        "-fprint",
        "-fprint0",
        "-fprintf",
    }
)

_SECRET_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_SECRET_ALLOW = {".env.example", ".env.sample", ".env.template"}


@dataclass(frozen=True)
class PermissionDecision:
    category: str
    target: str
    action: PermissionAction
    matching_rule: PermissionRule | None
    reason: str
    remember_pattern: str
    tool: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    @property
    def denied(self) -> bool:
        return self.action == "deny"

    @property
    def needs_ask(self) -> bool:
        return self.action == "ask"


def is_secret_path(path: str | Path) -> bool:
    # Case-insensitive by design: default macOS/Windows filesystems are
    # case-insensitive, so CERT.PEM is the same file as cert.pem there, and on
    # case-sensitive filesystems a false positive only costs one approval.
    p = Path(path)
    name = p.name.lower()
    if name in _SECRET_ALLOW:
        return False
    if name in _SECRET_BASENAMES:
        return True
    if name.startswith(".env."):
        return True
    if any(name.endswith(suf) for suf in _SECRET_SUFFIXES):
        return True
    if "id_rsa" in name or "id_ed25519" in name:
        return True
    parts = p.parts
    if ".git" in parts:
        return True
    return name.endswith(".db") and "noah-code" in str(p).lower()


def _match_rule(rule: PermissionRule, category: str, target: str) -> bool:
    cat_ok = rule.category in {"*", category}
    if not cat_ok:
        return False
    return fnmatch.fnmatch(target, rule.pattern)


class PermissionEngine:
    """Ordered wildcard rules; last matching rule wins."""

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        *,
        mode: Literal["build", "plan"] = "build",
        auto_approve: bool = False,
    ) -> None:
        self.rules: list[PermissionRule] = list(rules or [])
        self.mode = mode
        self.auto_approve = auto_approve
        self._session_rules: list[PermissionRule] = []

    def add_session_rule(self, rule: PermissionRule) -> None:
        self._session_rules.append(rule)

    def snapshot_session_rules(self) -> list[dict]:
        return [r.model_dump() for r in self._session_rules]

    def load_session_rules(self, raw: list[dict] | None) -> None:
        self._session_rules = [PermissionRule.model_validate(r) for r in (raw or [])]

    def decide(self, category: str, target: str, *, tool: str = "") -> PermissionDecision:
        decision = self._decide(category, target)
        return replace(decision, tool=tool) if tool else decision

    def _decide(self, category: str, target: str) -> PermissionDecision:
        normalized = target.strip() or "*"
        # Hard denies for secrets on read/edit.
        if (
            category in {PermissionCategory.READ, PermissionCategory.EDIT}
            and is_secret_path(normalized)
        ):
            return PermissionDecision(
                category=category,
                target=normalized,
                action="deny",
                matching_rule=None,
                reason="secret or credential path denied",
                remember_pattern=normalized,
            )

        if category == PermissionCategory.BASH:
            hard = self._hard_bash_deny(normalized)
            if hard is not None:
                return hard

        if self.mode == "plan":
            plan = self._plan_mode_gate(category, normalized)
            if plan is not None:
                return plan

        matching: PermissionRule | None = None
        for rule in [*self.rules, *self._session_rules]:
            if _match_rule(rule, category, normalized):
                matching = rule

        if matching is None:
            action: PermissionAction = "ask"
            reason = "no matching rule; default ask"
        else:
            action = matching.action
            reason = matching.reason or f"matched {matching.pattern}"

        if category == PermissionCategory.BASH and action == "allow":
            elevated = self._elevated_bash_ask(normalized)
            if elevated is not None:
                action = elevated.action
                reason = elevated.reason
                matching = elevated.matching_rule

        if action == "ask" and self.auto_approve:
            # --auto never overrides explicit deny; only ask → allow.
            action = "allow"
            reason = f"{reason} (auto-approved)"

        return PermissionDecision(
            category=category,
            target=normalized,
            action=action,
            matching_rule=matching,
            reason=reason,
            remember_pattern=self._remember_pattern(category, normalized),
        )

    def _remember_pattern(self, category: str, target: str) -> str:
        _ = category
        return target

    def _hard_bash_deny(self, command: str) -> PermissionDecision | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []

        for index, token in enumerate(tokens):
            if is_secret_path(token) or is_secret_path(Path(token).name):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="secret or credential path denied",
                    remember_pattern=command,
                )
            program = Path(token).name.lower()
            if program == "printenv":
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="credential dump denied",
                    remember_pattern=command,
                )
            if program == "env" and not any("=" in part for part in tokens[index + 1 :]):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="credential dump denied",
                    remember_pattern=command,
                )
            if program != "git":
                continue
            remaining = tokens[index + 1 :]
            if any(part in {"push", "clean", "filter-branch"} for part in remaining) or (
                "reset" in remaining and "--hard" in remaining
            ):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="destructive git command denied",
                    remember_pattern=command,
                )
            if self.auto_approve and not self.is_readonly_command(command):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="mutating or unrecognized git commands are not auto-approved",
                    remember_pattern=command,
                )

        if self.auto_approve and _is_env_dump(command, tokens):
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="environment dump denied under auto-approval",
                remember_pattern=command,
            )

        for pat in _ALWAYS_DENY_BASH:
            if pat.search(command):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="destructive or secret-exposing command denied",
                    remember_pattern=command,
                )
        lowered = command.lower()
        if "aws_secret" in lowered or "private key" in lowered:
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="credential dump denied",
                remember_pattern=command,
            )
        return None

    def _elevated_bash_ask(self, command: str) -> PermissionDecision | None:
        """Force ask for risky commands unless a session allow rule already matched."""
        session_allowed = any(
            r.action == "allow" and _match_rule(r, PermissionCategory.BASH, command)
            for r in self._session_rules
        )
        if session_allowed:
            return None
        for pat in _ALWAYS_ASK_BASH:
            if pat.search(command):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="ask",
                    matching_rule=None,
                    reason="elevated-risk shell command requires approval",
                    remember_pattern=self._remember_pattern(PermissionCategory.BASH, command),
                )
        return None

    def _plan_mode_gate(self, category: str, target: str) -> PermissionDecision | None:
        if category == PermissionCategory.EDIT:
            return PermissionDecision(
                category=category,
                target=target,
                action="deny",
                matching_rule=None,
                reason="plan mode forbids file edits",
                remember_pattern=target,
            )
        if category == PermissionCategory.GITHUB:
            operation = target.split(":", 1)[0].strip().lower()
            if operation in {"create", "push", "checkout", "comment"}:
                return PermissionDecision(
                    category=category,
                    target=target,
                    action="deny",
                    matching_rule=None,
                    reason="plan mode forbids GitHub mutations",
                    remember_pattern=target,
                )
        if category == PermissionCategory.BASH:
            if not self.is_readonly_command(target):
                return PermissionDecision(
                    category=category,
                    target=target,
                    action="deny",
                    matching_rule=None,
                    reason="plan mode forbids mutating shell commands",
                    remember_pattern=target,
                )
            if _command_has_external_path(target):
                return PermissionDecision(
                    category=category,
                    target=target,
                    action="deny",
                    matching_rule=None,
                    reason="plan mode forbids filesystem access outside the workspace",
                    remember_pattern=target,
                )
        return None

    @staticmethod
    def is_readonly_command(command: str) -> bool:
        cmd = command.strip()
        if not cmd:
            return True
        if _is_compound(cmd):
            return False
        lowered = cmd.lower()
        if _MUTATING_GIT.search(lowered):
            return False
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return False
        if not tokens:
            return True
        program = Path(tokens[0]).name.lower()
        if program == "git":
            return len(tokens) > 1 and tokens[1].lower() in _READ_ONLY_GIT_SUBCOMMANDS
        if program == "pwd":
            return len(tokens) == 1
        if program == "find":
            return not any(_is_mutating_find_flag(token) for token in tokens[1:])
        return any(lowered == p.strip() or lowered.startswith(p) for p in _READ_ONLY_PREFIXES[6:])

    @staticmethod
    def is_uncertain_shell(command: str) -> bool:
        if _is_compound(command):
            return True
        try:
            tokens = shlex.split(command)
        except ValueError:
            return True
        if not tokens:
            return False
        program = Path(tokens[0]).name.lower()
        return program in {"sh", "bash", "zsh", "ksh", "dash"} and any(
            token in {"-c", "-lc"} for token in tokens[1:]
        )


def _is_compound(command: str) -> bool:
    """Detect pipes, chains, redirects, substitutions, heredocs."""
    # Rough conservative scan - not a full shell parser.
    specials = ("|", "&&", "||", ";", "&", "`", "$(", "${", ">", "<", "<<", "\n")
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            for sp in specials:
                if command.startswith(sp, i):
                    return True
        i += 1
    try:
        shlex.split(command)
    except ValueError:
        return True
    return False


def _is_mutating_find_flag(token: str) -> bool:
    name = token.split("=", 1)[0]
    return name in _FIND_MUTATING_FLAGS or name.startswith(("-exec", "-ok", "-fprint", "-fprintf"))


def _command_has_external_path(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return True
    for token in tokens:
        pieces = token.split("=", 1)[1:] if token.startswith("-") and "=" in token else [token]
        if token.startswith("-") and "=" not in token:
            continue
        for piece in pieces:
            # Fail closed on expansion syntax: $HOME, ${HOME}, and `cmd`
            # resolve outside the workspace at execution time, so a plan-mode
            # readonly allow must never treat them as internal paths.
            if "$" in piece or "`" in piece:
                return True
            if piece.startswith(("~", "/")) or piece == "..":
                return True
            if ".." in Path(piece).parts:
                return True
    return False


_ENV_DUMP_PROGRAMS = frozenset(
    {"set", "declare", "local", "typeset", "readonly", "export"}
)
_AUTO_ENV_INTERPRETERS = frozenset({"python", "python3", "node", "perl", "ruby", "php"})


def _is_env_dump(command: str, tokens: list[str]) -> bool:
    """Detect commands whose primary effect is printing environment state.

    Only consulted under ``--auto``, where there is no user to approve an
    ask; printenv/bare-env are handled separately by the always-on deny.
    """
    if not tokens:
        return False

    program = Path(tokens[0]).name.lower()
    rest = tokens[1:]
    if any(fnmatch.fnmatch(part, "/proc/*/environ") for part in rest):
        return True
    if program in _ENV_DUMP_PROGRAMS:
        # Bare or flag-only invocations list variables; assignments do not.
        return not rest or all(part.startswith("-") for part in rest)
    if program in _AUTO_ENV_INTERPRETERS:
        for index in range(len(rest) - 1):
            if rest[index] not in {"-c", "-e"}:
                continue
            code = rest[index + 1].lower()
            if re.search(r"\bos\s*\.\s*environ\b", code) or re.search(
                r"\bprocess\s*\.\s*env\b", code
            ):
                return True
    return False
