"""Permission engine tests."""

from __future__ import annotations

from noah_code.config import DEFAULT_PERMISSION_RULES, PermissionRule
from noah_code.permissions import PermissionEngine, is_secret_path


def test_last_matching_rule_wins() -> None:
    rules = [
        PermissionRule(category="edit", pattern="*", action="deny", reason="deny all"),
        PermissionRule(category="edit", pattern="src/*", action="allow", reason="allow src"),
    ]
    engine = PermissionEngine(rules, auto_approve=False)
    assert engine.decide("edit", "src/a.py").action == "allow"
    assert engine.decide("edit", "other/a.py").action == "deny"


def test_auto_cannot_override_deny() -> None:
    rules = [PermissionRule(category="bash", pattern="*", action="deny", reason="no")]
    engine = PermissionEngine(rules, auto_approve=True)
    d = engine.decide("bash", "echo hi")
    assert d.action == "deny"


def test_auto_allows_ask() -> None:
    rules = [PermissionRule(category="edit", pattern="*", action="ask", reason="ask")]
    engine = PermissionEngine(rules, auto_approve=True)
    assert engine.decide("edit", "a.py").action == "allow"


def test_secret_paths() -> None:
    assert is_secret_path(".env")
    assert is_secret_path(".env.local")
    assert is_secret_path("keys/id_rsa")
    assert is_secret_path("cert.pem")
    assert not is_secret_path(".env.example")


def test_default_secret_deny_and_example_allow() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    assert engine.decide("read", ".env").action == "deny"
    assert engine.decide("read", ".env.secret").action == "deny"
    assert engine.decide("read", ".env.example").action == "allow"


def test_plan_mode_denies_edit_and_mutating_bash() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan")
    assert engine.decide("edit", "a.py").action == "deny"
    assert engine.decide("bash", "rm file.txt").action == "deny"
    assert engine.decide("bash", "git status").action == "ask"
    assert engine.decide("read", "a.py").action == "allow"


def test_destructive_and_env_dump_denied() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "printenv").action == "deny"
    assert engine.decide("bash", "git push origin main").action == "deny"
    assert engine.decide("bash", "git clean -fd").action == "deny"
    assert engine.decide("bash", "git -c core.pager=cat push origin main").action == "deny"


def test_read_commands_are_not_implicitly_allowed_outside_workspace() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    assert engine.decide("bash", "grep -R password /tmp").action == "ask"


def test_build_edit_asks_by_default() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="build", auto_approve=False)
    assert engine.decide("edit", "src/x.py").action == "ask"


def test_session_remember_keeps_exact_bash_command() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    decision = engine.decide("bash", "sh -c echo hi")
    assert decision.remember_pattern == "sh -c echo hi"
    engine.add_session_rule(
        PermissionRule(
            category="bash",
            pattern=decision.remember_pattern,
            action="allow",
            reason="remembered for session",
        )
    )
    assert engine.decide("bash", "sh -c echo hi").action == "allow"
    assert engine.decide("bash", 'sh -c "find . -delete"').action != "allow"
    assert engine.decide("bash", "sh -c 'cat credentials.json'").action != "allow"


def test_find_delete_is_not_readonly_and_denied_in_plan_auto() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    assert engine.is_readonly_command("find .") is True
    assert engine.is_readonly_command("find . -delete") is False
    assert engine.is_readonly_command("find . -exec rm {} +") is False
    plan = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    assert plan.decide("bash", "find . -delete").action == "deny"
    assert plan.decide("bash", "find . -exec rm {} +").action == "deny"


def test_plan_mode_denies_readonly_shell_outside_workspace() -> None:
    plan = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    assert plan.decide("bash", "head README.md").action == "allow"
    assert plan.decide("bash", "head /etc/passwd").action == "deny"
    assert plan.decide("bash", "ls ~/.ssh").action == "deny"
    assert plan.decide("bash", "grep -R password /etc").action == "deny"


def test_bash_denies_secret_path_arguments() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "cat credentials.json").action == "deny"
    assert engine.decide("bash", "od -c foo/credentials.json").action == "deny"
    assert engine.decide("bash", "cat .env.example").action == "allow"


def test_env_dump_deny_is_case_insensitive_and_ignores_dotenv_names() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "printenv").action == "deny"
    assert engine.decide("bash", "ENV").action == "deny"
    assert engine.decide("bash", "Env").action == "deny"
    assert engine.decide("bash", "echo hello.env").action == "allow"


def test_background_ampersand_is_uncertain_shell() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    assert engine.is_uncertain_shell("sleep 1 &") is True
    assert engine.is_uncertain_shell("pwd") is False


def test_permission_pattern_does_not_match_foreign_basenames() -> None:
    engine = PermissionEngine(
        [
            PermissionRule(category="read", pattern="*", action="deny", reason="deny all"),
            PermissionRule(category="read", pattern="a.py", action="allow", reason="exact file"),
        ]
    )
    assert engine.decide("read", "a.py").action == "allow"
    assert engine.decide("read", "deep/a.py").action == "deny"
