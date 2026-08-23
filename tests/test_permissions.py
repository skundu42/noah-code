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


def test_auto_does_not_override_elevated_bash_ask() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    for command in (
        "rm generated.txt",
        "mv old.txt new.txt",
        "curl https://example.com/archive.tar.gz",
        "pip install example-package",
    ):
        decision = engine.decide("bash", command)
        assert decision.action == "ask", command
        assert "elevated-risk" in decision.reason


def test_exact_session_allow_still_authorizes_elevated_bash() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    command = "rm generated.txt"
    engine.add_session_rule(
        PermissionRule(
            category="bash",
            pattern=command,
            action="allow",
            reason="explicit session approval",
        )
    )

    assert engine.decide("bash", command).action == "allow"


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


def test_secret_paths_match_case_insensitively() -> None:
    assert is_secret_path("CERT.PEM")
    assert is_secret_path("Server.KEY")
    assert is_secret_path("ID_RSA")
    assert is_secret_path(".ENV")
    assert is_secret_path(".Env.Local")
    assert is_secret_path("CREDENTIALS.JSON")
    assert is_secret_path("Service-Account.JSON")
    assert is_secret_path("deploy/ID_ED25519")
    assert is_secret_path("repo/.GIT/config")
    assert is_secret_path("repo/.Git/HEAD")
    assert not is_secret_path(".ENV.EXAMPLE")
    assert not is_secret_path("README.MD")


def test_default_read_rule_denies_uppercase_secret_names() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    assert engine.decide("read", "CERT.PEM").action == "deny"
    assert engine.decide("read", "ID_RSA").action == "deny"
    assert engine.decide("edit", ".ENV").action == "deny"
    assert engine.decide("bash", "cat CERT.PEM").action == "deny"


def test_plan_mode_rejects_variable_expansion_paths() -> None:
    plan = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    assert plan.decide("bash", "rg pattern $HOME/secrets").action == "deny"
    assert plan.decide("bash", "rg pattern ${HOME}/x").action == "deny"
    assert plan.decide("bash", "head `pwd`/../etc/passwd").action == "deny"
    assert plan.decide("bash", "--flag=$HOME/x rg p .").action == "deny"
    assert plan.decide("bash", "rg pattern ./src").action == "allow"


def test_auto_denies_environment_dumps() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "set").action == "deny"
    assert engine.decide("bash", "declare -p").action == "deny"
    assert engine.decide("bash", "export").action == "deny"
    assert engine.decide("bash", "export -p").action == "deny"
    assert engine.decide("bash", "readonly").action == "deny"
    assert engine.decide("bash", "cat /proc/self/environ").action == "deny"
    assert engine.decide("bash", "cat /proc/1/environ").action == "deny"
    assert engine.decide("bash", "python -c 'import os; print(os.environ)'").action == "deny"
    assert engine.decide("bash", "node -e 'console.log(process.env)'").action == "deny"


def test_auto_still_allows_var_assignments_and_shell_options() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "export FOO=bar").action == "allow"
    assert engine.decide("bash", "declare x=1").action == "allow"
    assert engine.decide("bash", "python -c 'print(1+1)'").action == "allow"
    assert engine.decide("bash", "cat /proc/cpuinfo").action == "allow"


def test_auto_env_dump_detection_ignores_arguments_and_plain_words() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)
    assert engine.decide("bash", "echo export").action == "allow"
    assert engine.decide("bash", "rg readonly").action == "allow"
    assert engine.decide("bash", "python -c \"print('environment')\"").action == "allow"


def test_non_auto_env_builtins_still_ask() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    decision = engine.decide("bash", "set")
    assert decision.action == "ask"


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
    assert engine.decide("bash", "gh pr create --title x").action == "deny"
    assert engine.decide("bash", "gh pr checkout 12").action == "deny"


def test_quote_fragments_cannot_hide_dangerous_programs() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    assert engine.decide("bash", "r''m -rf /").action == "deny"
    assert engine.decide("bash", 's""udo whoami').action == "deny"
    assert engine.decide("bash", "g''h pr create --title x").action == "deny"
    assert engine.decide("bash", "r''m generated.txt").action == "ask"


def test_bash_denies_secret_specific_shell_globs() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    for command in (
        "cat .e*",
        "c''at .e*",
        "cat '.e*'",
        "head .env.*",
        'head ".ENV.*"',
        "cat credentials.*",
        "cat deploy/ID_*",
        "cat deploy/id*",
        "cat keys/*.pem",
        "cat keys/*.p?m",
        "cat keys/*.p??",
        "cat .g*/config",
        "cat [.]g*/config",
        "cat .GIT/config",
    ):
        decision = engine.decide("bash", command)
        assert decision.action == "deny", command
        assert "secret" in decision.reason


def test_bash_secret_glob_detection_preserves_benign_globs() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=True)

    for command in (
        "cat src/*.py",
        "cat 'src/*.py'",
        "cat docs/**/*.md",
        "head tests/test_*.py",
        "ls build/*",
        "cat reports/*.txt",
        "cat *.json",
        "cat .env.example",
    ):
        assert engine.decide("bash", command).action == "allow", command


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


def test_git_branch_mutations_are_not_readonly() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    destructive = (
        "git branch feature-x",
        "git branch -d feature-x",
        "git branch -D feature-x",
        "git branch -m main renamed",
        "git branch -M main renamed",
        "git branch --delete feature-x",
        "git branch --move main renamed",
        "git branch --edit-description main",
        "git branch --set-upstream-to=origin/main main",
        "git branch -q feature-x",
        "git branch -v feature-x",
        "git branch --color feature-x",
        "git branch --format='%(refname:short)' feature-x",
    )

    for command in destructive:
        assert engine.is_readonly_command(command) is False, command

    plan = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    for command in destructive:
        assert plan.decide("bash", command).action == "deny", command


def test_git_branch_listing_forms_remain_readonly() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)

    for command in (
        "git branch",
        "git branch -a",
        "git branch -l 'feature/*'",
        "git branch -rvv",
        "git branch --list 'feature/*'",
        "git branch --contains HEAD",
        "git branch --show-current",
        "git branch --format='%(refname:short)'",
        "git branch --format '%(refname:short)'",
    ):
        assert engine.is_readonly_command(command) is True, command


def test_readonly_git_rejects_flags_that_write_or_execute_helpers() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)

    for command in (
        "git diff --output=changes.patch",
        "git diff --out=changes.patch",
        "git diff --ext-diff",
        "git diff --ext",
        "git show --textconv HEAD:file.txt",
        "git show --textc HEAD:file.txt",
    ):
        assert engine.is_readonly_command(command) is False, command

    assert engine.is_readonly_command("git diff --no-ext-diff -- src") is True
    assert engine.is_readonly_command("git diff -p -- src") is True
    assert engine.is_readonly_command("git diff -- --output=tracked-name") is True


def test_ripgrep_preprocessors_and_external_decompressors_are_not_readonly() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES)
    unsafe = (
        "rg --pre=rm needle .",
        "rg --pre touch needle .",
        "rg --search-zip needle .",
        "rg -z needle .",
        "rg -zn needle .",
    )

    for command in unsafe:
        assert engine.is_readonly_command(command) is False, command

    plan = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=True)
    for command in unsafe:
        assert plan.decide("bash", command).action == "deny", command

    assert engine.is_readonly_command("rg -n --hidden needle src") is True
    assert engine.is_readonly_command("rg --files src") is True
    assert engine.is_readonly_command("rg -- --pre source.txt") is True


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


def test_web_and_question_defaults() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, auto_approve=False)
    assert engine.decide("webfetch", "https://example.com").action == "ask"
    assert engine.decide("websearch", "asyncio").action == "ask"
    assert engine.decide("question", "Approach").action == "allow"
    assert engine.decide("task", "explore").action == "ask"
    assert engine.decide("github", "list").action == "allow"
    assert engine.decide("github", "view").action == "allow"
    assert engine.decide("github", "create").action == "ask"
    assert engine.decide("github", "push").action == "ask"


def test_plan_mode_allows_questions_and_web_asks() -> None:
    engine = PermissionEngine(DEFAULT_PERMISSION_RULES, mode="plan", auto_approve=False)
    assert engine.decide("question", "Approach").action == "allow"
    assert engine.decide("webfetch", "https://example.com").action == "ask"
    assert engine.decide("edit", "a.py").action == "deny"
    assert engine.decide("github", "list").action == "allow"
    assert engine.decide("github", "create").action == "deny"


def test_permission_pattern_does_not_match_foreign_basenames() -> None:
    engine = PermissionEngine(
        [
            PermissionRule(category="read", pattern="*", action="deny", reason="deny all"),
            PermissionRule(category="read", pattern="a.py", action="allow", reason="exact file"),
        ]
    )
    assert engine.decide("read", "a.py").action == "allow"
    assert engine.decide("read", "deep/a.py").action == "deny"
