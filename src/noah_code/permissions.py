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
    re.compile(r"\bfind\b[^|;&]*\s+-(?:delete|execdir|exec)\b"),
)

_MUTATING_GIT = re.compile(
    r"\bgit\s+(commit|add|push|pull|fetch|rebase|merge|reset|clean|checkout|stash|tag|remote)\b"
)
_READ_ONLY_PROGRAMS = frozenset(
    {
        "grep",
        "egrep",
        "fgrep",
        "find",
        "ls",
        "pwd",
        "head",
        "tail",
        "wc",
        "file",
        "stat",
        "test",
    }
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({"branch", "diff", "log", "rev-parse", "show", "status"})
_GIT_READ_UNSAFE_FLAGS = frozenset({"--ext-diff", "--output", "--textconv"})
_GIT_PATCH_SUBCOMMANDS = frozenset({"diff", "log", "show"})
# Flags that replace the patch body with a non-patch summary, so the command
# can no longer dump committed file contents.
_GIT_NON_PATCH_OUTPUT_FLAGS = frozenset(
    {
        "--dirstat",
        "--exit-code",
        "--format",
        "--name-only",
        "--name-status",
        "--no-patch",
        "--numstat",
        "--oneline",
        "--pretty",
        "--quiet",
        "--raw",
        "--shortlog",
        "--shortstat",
        "--stat",
        "--summary",
    }
)
_GIT_BRANCH_MUTATING_LONG_FLAGS = frozenset(
    {
        "--copy",
        "--create-reflog",
        "--delete",
        "--edit-description",
        "--force",
        "--move",
        "--no-track",
        "--recurse-submodules",
        "--set-upstream-to",
        "--track",
        "--unset-upstream",
    }
)
_GIT_BRANCH_LIST_LONG_FLAGS = frozenset(
    {
        "--abbrev",
        "--all",
        "--color",
        "--column",
        "--contains",
        "--format",
        "--ignore-case",
        "--list",
        "--merged",
        "--no-color",
        "--no-column",
        "--no-contains",
        "--no-merged",
        "--omit-empty",
        "--points-at",
        "--quiet",
        "--remotes",
        "--show-current",
        "--sort",
        "--verbose",
    }
)
_GIT_BRANCH_LIST_ACTION_LONG_FLAGS = frozenset(
    {
        "--all",
        "--contains",
        "--list",
        "--merged",
        "--no-contains",
        "--no-merged",
        "--points-at",
        "--remotes",
        "--show-current",
    }
)
_GIT_BRANCH_VALUE_LONG_FLAGS = frozenset({"--format", "--sort"})
_GIT_BRANCH_MUTATING_SHORT_FLAGS = frozenset("dDmMcCftu")
_GIT_BRANCH_LIST_SHORT_FLAGS = frozenset("ailrvq")
_GIT_BRANCH_LIST_ACTION_SHORT_FLAGS = frozenset("alr")
_RG_UNSAFE_LONG_FLAGS = frozenset({"--hostname-bin", "--pre", "--search-zip"})
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
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pgpass",
    ".pypirc",
    "credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore")
_SECRET_ALLOW = {".env.example", ".env.sample", ".env.template"}
# Credential stores whose file names are too generic to deny on their own,
# matched as (directory name, file name) on the final two path components.
_SECRET_DIR_FILES = {
    (".docker", "config.json"),
    (".kube", "config"),
}
_GLOB_META = frozenset("*?[")
_DOT_SECRET_GLOB_PROBES = (
    ".env",
    ".env.local",
    ".env.production",
    ".env.secret",
    ".git",
)
_BASENAME_SECRET_GLOB_PROBES = tuple(sorted(_SECRET_BASENAMES))
_SUFFIX_SECRET_GLOB_PROBES = tuple(f"private{suf}" for suf in _SECRET_SUFFIXES)


@dataclass(frozen=True)
class PermissionDecision:
    category: str
    target: str
    action: PermissionAction
    matching_rule: PermissionRule | None
    reason: str
    remember_pattern: str
    tool: str = ""
    # True when the elevated-risk floor downgraded this decision to ask. Hosts
    # reject such decisions outright in non-interactive --auto mode.
    elevated_floor: bool = False

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    @property
    def denied(self) -> bool:
        return self.action == "deny"


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
    parts = tuple(part.lower() for part in p.parts)
    if ".git" in parts:
        return True
    if len(parts) >= 2 and (parts[-2], parts[-1]) in _SECRET_DIR_FILES:
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
        if category in {PermissionCategory.READ, PermissionCategory.EDIT} and is_secret_path(
            normalized
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

        # The built-in balanced policy asks for arbitrary shell execution, but
        # does not interrupt users for commands the parser can prove are
        # read-only and confined to the workspace. A remembered session rule
        # remains authoritative, including an explicit ask. Read-only git
        # commands that still dump unscoped patch output are excluded: they
        # can expose committed secrets without naming a path.
        if (
            category == PermissionCategory.BASH
            and action == "ask"
            and matching is not None
            and matching.reason == "non-read-only shell commands require approval"
            and not any(rule is matching for rule in self._session_rules)
            and self.is_readonly_command(normalized)
            and not _command_has_external_path(normalized)
            and not _git_unscoped_patch_output(normalized)
        ):
            action = "allow"
            reason = "read-only workspace shell command allowed"

        if action == "ask" and self.auto_approve:
            # --auto never overrides explicit deny; only ask → allow.
            action = "allow"
            reason = f"{reason} (auto-approved)"

        elevated: PermissionDecision | None = None
        if category == PermissionCategory.BASH and action == "allow":
            # Apply the elevated-risk floor *after* ordinary auto-approval. This
            # keeps --auto useful for routine asks without silently approving
            # commands that are explicitly documented as always requiring a
            # human confirmation. An exact session allow remains authoritative.
            elevated = self._elevated_bash_ask(normalized)
            if elevated is not None:
                action = elevated.action
                reason = elevated.reason
                matching = elevated.matching_rule

        return PermissionDecision(
            category=category,
            target=normalized,
            action=action,
            matching_rule=matching,
            reason=reason,
            remember_pattern=self._remember_pattern(category, normalized),
            elevated_floor=elevated is not None,
        )

    def _remember_pattern(self, category: str, target: str) -> str:
        _ = category
        return target

    def _hard_bash_deny(self, command: str) -> PermissionDecision | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        normalized_tokens = " ".join(tokens)
        effective_tokens = _effective_shell_tokens(tokens)
        effective_program = _program_name(effective_tokens[0]) if effective_tokens else ""

        if effective_program in {"env", "printenv"}:
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="credential dump denied",
                remember_pattern=command,
            )

        for index, token in enumerate(tokens):
            if _is_secret_shell_token(token):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="deny",
                    matching_rule=None,
                    reason="secret or credential path denied",
                    remember_pattern=command,
                )
            program = Path(token).name.lower()
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

        if self.auto_approve and _is_env_dump(tokens):
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="environment dump denied under auto-approval",
                remember_pattern=command,
            )

        if self.auto_approve and _contains_executing_interpreter(tokens):
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="interpreter, eval, and indirect execution commands cannot be auto-approved",
                remember_pattern=command,
            )

        if self.auto_approve and _has_ambiguous_shell_expansion(command):
            return PermissionDecision(
                category=PermissionCategory.BASH,
                target=command,
                action="deny",
                matching_rule=None,
                reason="ambiguous shell expansion cannot be auto-approved",
                remember_pattern=command,
            )

        for pat in _ALWAYS_DENY_BASH:
            # Scan both the source and shlex-normalized tokens. Shell quote
            # fragments such as ``r''m`` execute as ``rm`` but deliberately do
            # not contain the raw substring matched by the policy regex.
            if pat.search(command) or (normalized_tokens and pat.search(normalized_tokens)):
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
        try:
            normalized_tokens = " ".join(shlex.split(command))
        except ValueError:
            normalized_tokens = ""
        for pat in _ALWAYS_ASK_BASH:
            if pat.search(command) or (normalized_tokens and pat.search(normalized_tokens)):
                return PermissionDecision(
                    category=PermissionCategory.BASH,
                    target=command,
                    action="ask",
                    matching_rule=None,
                    reason="elevated-risk shell command requires approval",
                    remember_pattern=self._remember_pattern(PermissionCategory.BASH, command),
                    elevated_floor=True,
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
        # Only the literal, unqualified executable names below are trusted.
        # Reducing ``./git`` or ``bin/rg`` to a basename would let a repository
        # replace a supposedly read-only utility with arbitrary executable code.
        source_program = cmd.split(None, 1)[0]
        program = tokens[0]
        if source_program != program or program not in _READ_ONLY_PROGRAMS | {"git", "rg"}:
            return False
        if program == "git":
            return _is_readonly_git(tokens[1:])
        if program == "rg":
            return _is_readonly_rg(tokens[1:])
        if program == "pwd":
            return len(tokens) == 1
        if program == "find":
            return not any(_is_mutating_find_flag(token) for token in tokens[1:])
        return program in _READ_ONLY_PROGRAMS

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
    if _has_ambiguous_shell_expansion(command):
        return True
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


def _has_ambiguous_shell_expansion(command: str) -> bool:
    """Return true when the shell can derive arguments hidden from ``shlex``.

    Expansion is safe to approve only after it has happened, but executing a
    command to discover its expansion introduces a TOCTOU boundary. Auto mode
    therefore fails closed on variable/command/ANSI-C expansion and on brace
    forms that the shell expands into multiple arguments. Single-quoted and
    backslash-escaped literals remain ordinary arguments.
    """

    in_single = False
    in_double = False
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\" and not in_single:
            index += 2
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            index += 1
            continue
        if not in_single and char in {"$", "`"}:
            return True
        if not in_single and not in_double and char == "{":
            closing = command.find("}", index + 1)
            if closing >= 0:
                body = command[index + 1 : closing]
                if "," in body or ".." in body:
                    return True
        index += 1
    return False


def _is_mutating_find_flag(token: str) -> bool:
    name = token.split("=", 1)[0]
    return name in _FIND_MUTATING_FLAGS or name.startswith(("-exec", "-ok", "-fprint", "-fprintf"))


# Short flag cluster carrying a joined value, e.g. ``-f/etc/passwd`` or
# ``-F~/log``: the characters after the flag letter are a path candidate.
_SHORT_FLAG_JOINED_VALUE = re.compile(r"^-[a-zA-Z].{1,}")
# Windows drive prefix (``C:`` alone or ``C:\...``/``C:/...``) — its colon
# does not introduce a git ``<rev>:<path>`` object path.
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:(?:$|[\\/])")


def _git_object_path_candidate(value: str) -> str | None:
    """Return the path part of a git ``<rev>:<path>`` object token, if any.

    URLs (``scheme://``) and Windows drive paths are excluded so ordinary
    arguments are not misread as object syntax.
    """
    if ":" not in value or "://" in value or _WINDOWS_DRIVE_PATH.match(value):
        return None
    return value.rsplit(":", 1)[1]


def _is_secret_shell_token(token: str) -> bool:
    """Detect direct or secret-specific glob path arguments.

    The permission engine cannot expand model-provided globs safely without a
    workspace/cwd and a new TOCTOU window. Instead, recognize patterns that are
    specific enough to target Noah's denied filename families while leaving
    broad, ordinary source globs such as ``src/*.py`` untouched. Flag-joined
    values (``-f.env``) and git ``<rev>:<path>`` object syntax (``HEAD:.env``)
    are scanned as path candidates as well.
    """

    if token.startswith("-") and "=" in token:
        values = token.split("=", 1)[1:]
    elif _SHORT_FLAG_JOINED_VALUE.match(token):
        values = [token[2:]]
    else:
        values = [token]
    candidates = list(values)
    for value in values:
        git_object_path = _git_object_path_candidate(value)
        if git_object_path is not None:
            candidates.append(git_object_path)
    for candidate in candidates:
        if is_secret_path(candidate):
            return True
        for component in Path(candidate).parts:
            if _glob_component_may_match_secret(component):
                return True
    return False


def _glob_component_may_match_secret(component: str) -> bool:
    pattern = component.lower()
    if not any(marker in pattern for marker in _GLOB_META):
        return False
    literal_runs = _glob_literal_runs(pattern)
    specificity = _glob_pattern_specificity(pattern)

    # Every .env.* name except the three exact templates is denied. A wildcard
    # in this namespace can therefore never be proven to select only an allow.
    if pattern.startswith(".env."):
        return True

    if specificity >= 2 and any(
        fnmatch.fnmatchcase(probe, pattern) for probe in _DOT_SECRET_GLOB_PROBES
    ):
        return True

    for probe in _BASENAME_SECRET_GLOB_PROBES:
        if not fnmatch.fnmatchcase(probe, pattern):
            continue
        stem = probe.rsplit(".", 1)[0]
        distinctive = [fragment for run in literal_runs for fragment in run.split(".")]
        if any(len(fragment) >= 2 and fragment in stem for fragment in distinctive):
            return True

    return specificity >= 2 and any(
        fnmatch.fnmatchcase(probe, pattern) for probe in _SUFFIX_SECRET_GLOB_PROBES
    )


def _glob_literal_runs(pattern: str) -> list[str]:
    """Return literal runs, retaining classes that name one distinct character."""

    runs: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char in {"*", "?"}:
            if current:
                runs.append("".join(current))
                current = []
            index += 1
            continue
        if char == "[":
            closing = pattern.find("]", index + 1)
            if closing < 0:
                current.append(char)
                index += 1
                continue
            members = pattern[index + 1 : closing]
            if members and not members.startswith(("!", "^")) and len(set(members)) == 1:
                current.append(members[0])
            elif current:
                runs.append("".join(current))
                current = []
            index = closing + 1
            continue
        current.append(char)
        index += 1
    if current:
        runs.append("".join(current))
    return runs


def _glob_pattern_specificity(pattern: str) -> int:
    """Count literal and character-class constraints in a shell glob."""

    specificity = 0
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char in {"*", "?"}:
            index += 1
            continue
        if char == "[":
            closing = pattern.find("]", index + 1)
            if closing >= 0:
                specificity += 1
                index = closing + 1
                continue
        specificity += 1
        index += 1
    return specificity


def _git_unscoped_patch_output(command: str) -> bool:
    """Detect read-only git commands that still dump full patch contents.

    ``git show``/``git log``/``git diff`` print complete diffs by default (or
    with ``-p``/``--patch``), exposing committed secrets without naming a path
    token. Unless the output is a non-patch summary or the command is scoped
    to an explicit path, the read-only auto-allow bump must not apply. The
    read-only classification itself is unchanged.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if len(tokens) < 2 or tokens[0] != "git":
        return False
    if tokens[1].lower() not in _GIT_PATCH_SUBCOMMANDS:
        return False
    patch_requested = False
    patch_suppressed = False
    args = tokens[2:]
    for index, token in enumerate(args):
        if token == "--":
            # Only an explicit path limiter after ``--`` scopes the output.
            if args[index + 1 :]:
                return False
            break
        lowered = token.lower()
        name = lowered.split("=", 1)[0]
        if lowered == "-p" or name == "--patch":
            patch_requested = True
            continue
        if lowered == "-s" or name in _GIT_NON_PATCH_OUTPUT_FLAGS:
            patch_suppressed = True
            continue
        if not token.startswith("-") and (
            "/" in token or ":" in token or token.startswith((".", "~"))
        ):
            # A path-like argument (pathspec or ``<rev>:<path>`` object)
            # scopes the output without needing ``--``.
            return False
    return patch_requested or not patch_suppressed


def _is_readonly_git(args: list[str]) -> bool:
    if not args:
        return False
    subcommand = args[0].lower()
    if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
        return False
    subcommand_args = args[1:]
    if subcommand == "branch":
        return _is_readonly_git_branch(subcommand_args)
    options_done = False
    for token in subcommand_args:
        if token == "--":
            options_done = True
            continue
        if options_done:
            continue
        name = token.lower().split("=", 1)[0]
        if name in _GIT_READ_UNSAFE_FLAGS or (
            name.startswith("--")
            and any(unsafe.startswith(name) for unsafe in _GIT_READ_UNSAFE_FLAGS)
        ):
            return False
    return True


def _is_readonly_git_branch(args: list[str]) -> bool:
    """Recognize branch listing/query forms without admitting ref mutations."""

    if not args:
        return True
    listing = False
    options_done = False
    expects_value = False
    for token in args:
        lowered = token.lower()
        if expects_value:
            expects_value = False
            continue
        if options_done:
            if not listing:
                return False
            continue
        if lowered == "--":
            options_done = True
            continue
        if lowered.startswith("--"):
            name = lowered.split("=", 1)[0]
            if name in _GIT_BRANCH_MUTATING_LONG_FLAGS:
                return False
            if name not in _GIT_BRANCH_LIST_LONG_FLAGS:
                return False
            if name in _GIT_BRANCH_LIST_ACTION_LONG_FLAGS:
                listing = True
            if "=" not in lowered and name in _GIT_BRANCH_VALUE_LONG_FLAGS:
                expects_value = True
            continue
        if lowered.startswith("-") and lowered != "-":
            flags = lowered[1:]
            if any(flag in _GIT_BRANCH_MUTATING_SHORT_FLAGS for flag in flags):
                return False
            if not flags or any(flag not in _GIT_BRANCH_LIST_SHORT_FLAGS for flag in flags):
                return False
            if any(flag in _GIT_BRANCH_LIST_ACTION_SHORT_FLAGS for flag in flags):
                listing = True
            continue
        # A bare branch name creates a ref unless a listing/query option has
        # selected the non-mutating synopsis.
        if not listing:
            return False
    return not expects_value


def _is_readonly_rg(args: list[str]) -> bool:
    options_done = False
    for token in args:
        if token == "--":
            options_done = True
            continue
        if options_done:
            continue
        lowered = token.lower()
        name = lowered.split("=", 1)[0]
        if name in _RG_UNSAFE_LONG_FLAGS:
            return False
        if lowered.startswith("-") and not lowered.startswith("--") and "z" in lowered[1:]:
            # -z/--search-zip launches external decompressors.
            return False
    return True


def _command_has_external_path(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return True
    for token in tokens:
        if token.startswith("-") and "=" in token:
            pieces = token.split("=", 1)[1:]
        elif _SHORT_FLAG_JOINED_VALUE.match(token):
            # Short flag cluster with a joined value: -f/etc/passwd, -F~/log.
            pieces = [token[2:]]
        elif token.startswith("-"):
            continue
        else:
            pieces = [token]
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


_ENV_DUMP_PROGRAMS = frozenset({"set", "declare", "local", "typeset", "readonly", "export"})
_AUTO_ENV_INTERPRETERS = frozenset({"python", "python3", "node", "perl", "ruby", "php"})
_AUTO_INTERPRETER_NAME = re.compile(
    r"(?:pythonw?|pypy)(?:\d+(?:\.\d+)*)?t?|node(?:js)?|perl|ruby|php|(?:ba|z|k|da)?sh"
)
_INERT_INTERPRETER_FLAGS = frozenset({"-v", "-V", "-VV", "--version"})
_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\+?=.*", re.DOTALL)
_ENV_OPTIONS_WITH_VALUE = frozenset({"-u", "--unset", "-C", "--chdir"})
_ENV_SPLIT_OPTIONS = frozenset({"-S", "--split-string"})
_AUTO_INDIRECT_EXECUTORS = frozenset(
    {"exec", "nohup", "nice", "timeout", "gtimeout", "time", "xargs"}
)


def _program_name(token: str) -> str:
    return Path(token).name.lower().removesuffix(".exe")


def _effective_shell_tokens(tokens: list[str]) -> list[str]:
    """Resolve the command position through simple shell builtins/wrappers."""

    index = 0
    while index < len(tokens) and _SHELL_ASSIGNMENT.fullmatch(tokens[index]):
        index += 1

    while index < len(tokens):
        wrapper_start = index
        program = _program_name(tokens[index])
        if program == "command":
            index += 1
            while index < len(tokens) and tokens[index].startswith("-"):
                option = tokens[index]
                if option in {"-v", "-V"}:
                    # Query-only forms inspect a name without invoking it.
                    return tokens[wrapper_start:]
                index += 1
                if option == "--":
                    break
            continue
        if program == "builtin":
            index += 1
            if index < len(tokens) and tokens[index] == "--":
                index += 1
                continue
            if index < len(tokens) and tokens[index].startswith("-"):
                # ``builtin -p/-a/-s`` query builtin metadata; it does not run
                # the following name as a command.
                return tokens[wrapper_start:]
            continue
        if program == "env":
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if _SHELL_ASSIGNMENT.fullmatch(token):
                    index += 1
                    continue
                option_name = token.split("=", 1)[0]
                if option_name in _ENV_SPLIT_OPTIONS:
                    if "=" in token:
                        split_value = token.split("=", 1)[1]
                        remainder = tokens[index + 1 :]
                    elif index + 1 < len(tokens):
                        split_value = tokens[index + 1]
                        remainder = tokens[index + 2 :]
                    else:
                        return tokens[wrapper_start:]
                    try:
                        split_tokens = shlex.split(split_value)
                    except ValueError:
                        return tokens[wrapper_start:]
                    return _effective_shell_tokens([*split_tokens, *remainder])
                if option_name in _ENV_OPTIONS_WITH_VALUE and "=" not in token:
                    index += 2
                    continue
                if token == "--":
                    index += 1
                    break
                if token.startswith("-"):
                    index += 1
                    continue
                break
            if index >= len(tokens):
                # ``env`` with only options/assignments prints the environment.
                return tokens[wrapper_start:]
            continue
        break
    return tokens[index:]


def _contains_executing_interpreter(tokens: list[str]) -> bool:
    """Detect interpreters that could execute model- or repository-owned code.

    The effective-command pass prevents argument text such as ``echo python``
    from being mistaken for execution while retaining wrapper coverage.
    Version-only invocations remain a useful inert operation under auto.
    """

    effective = _effective_shell_tokens(tokens)
    if not effective:
        return False
    program = _program_name(effective[0])
    if effective[0] == "." or program in {"eval", "source"}:
        return True
    if program in _AUTO_INDIRECT_EXECUTORS:
        return True

    if _AUTO_INTERPRETER_NAME.fullmatch(program) is not None:
        remaining = effective[1:]
        return not (len(remaining) == 1 and remaining[0] in _INERT_INTERPRETER_FLAGS)

    if program == "uv" and "run" in effective[1:]:
        run_index = effective.index("run", 1)
        run_args = effective[run_index + 1 :]
        if any(arg in {"-m", "--module", "-s", "--script", "--gui-script"} for arg in run_args):
            return True
        for index, token in enumerate(run_args):
            if _AUTO_INTERPRETER_NAME.fullmatch(_program_name(token)) is None:
                continue
            remaining = run_args[index + 1 :]
            return not (len(remaining) == 1 and remaining[0] in _INERT_INTERPRETER_FLAGS)
    return False


def _is_env_dump(tokens: list[str]) -> bool:
    """Detect commands whose primary effect is printing environment state.

    Only consulted under ``--auto``, where there is no user to approve an
    ask; printenv/bare-env are handled separately by the always-on deny.
    """
    effective = _effective_shell_tokens(tokens)
    if not effective:
        return False

    program = _program_name(effective[0])
    rest = effective[1:]
    if any(fnmatch.fnmatch(part, "/proc/*/environ") for part in rest):
        return True
    if program in {"env", "printenv"}:
        return True
    if program in _ENV_DUMP_PROGRAMS:
        if "-p" in rest:
            return True
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
