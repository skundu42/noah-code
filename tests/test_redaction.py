"""Secret-safe user-visible error formatting."""

from __future__ import annotations

from noah_code.redaction import safe_error_message


def test_safe_error_message_redacts_common_secret_forms() -> None:
    raw = (
        "Authorization: Bearer opaque-credential-123 "
        "password='two words secret' apiSecretKey=custom-secret "
        "https://operator:password@example.test/path"
    )

    result = safe_error_message(raw)

    for secret in (
        "opaque-credential-123",
        "two words secret",
        "custom-secret",
        "operator:password",
    ):
        assert secret not in result
    assert result.count("***") >= 4


def test_safe_error_message_preserves_benign_text_and_bounds_output() -> None:
    assert safe_error_message("provider temporarily unavailable") == (
        "provider temporarily unavailable"
    )
    assert len(safe_error_message("x" * 1000, limit=80)) == 80


def test_safe_error_message_redacts_generic_and_prefixed_environment_keys() -> None:
    raw = (
        "token=SUPERSECRET123456 secret: OTHERSECRET123456 "
        "OPENAI_API_KEY=OPENAISECRET123456 "
        "X_API_KEY: XPROVIDERSECRET123456 "
        "ANTHROPIC_AUTH_TOKEN='ANTHROPICSECRET123456' "
        "AWS_SECRET_ACCESS_KEY=AWSSECRET123456"
    )

    result = safe_error_message(raw)

    for secret in (
        "SUPERSECRET123456",
        "OTHERSECRET123456",
        "OPENAISECRET123456",
        "XPROVIDERSECRET123456",
        "ANTHROPICSECRET123456",
        "AWSSECRET123456",
    ):
        assert secret not in result
    assert result.count("***") == 6


def test_safe_error_message_does_not_redact_non_secret_key_suffixes() -> None:
    assert safe_error_message("tokenizer=enabled secretary=available") == (
        "tokenizer=enabled secretary=available"
    )
