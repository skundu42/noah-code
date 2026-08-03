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
