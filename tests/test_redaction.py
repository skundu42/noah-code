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
