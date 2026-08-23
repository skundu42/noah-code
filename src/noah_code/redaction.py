"""Bounded secret scrubbing for user-visible error text."""

from __future__ import annotations

import re

_KEYED_SECRET = re.compile(
    r"(?P<prefix>\b(?:authorization|proxy[-_ ]?authorization|api[-_ ]?key|apiKey|"
    r"access[-_ ]?token|accessToken|refresh[-_ ]?token|refreshToken|auth[-_ ]?token|"
    r"client[-_ ]?secret|clientSecret|password|passwd|private[-_ ]?key|"
    r"api[-_ ]?secret[-_ ]?key|secret[-_ ]?key|session[-_ ]?cookie|"
    r"set[-_ ]?cookie|cookie)"
    r"[\"']?\s*[:=]\s*)"
    r"(?:(?P<quote>[\"'])(?P<quoted>.*?)(?P=quote)|"
    r"(?P<bearer>Bearer\s+[^\s,;}]+)|(?P<bare>[^\s,;}]+))",
    re.IGNORECASE,
)
_BEARER_SECRET = re.compile(
    r"(?P<prefix>\bBearer\s+)(?P<secret>[A-Za-z0-9_.\-/+=]{8,})",
    re.IGNORECASE,
)
_PREFIXED_SECRET = re.compile(
    r"(?P<prefix>\b(?:sk-(?:ant-)?|nvapi-|gh[pousr]_|glpat-|xox[bporas]-))"
    r"(?P<secret>[A-Za-z0-9_.\-/+=]{8,})",
    re.IGNORECASE,
)
_AWS_ACCESS_KEY = re.compile(r"(?P<secret>(?:AKIA|ASIA|AIDA|AROA)[A-Z0-9]{16})")
_GOOGLE_API_KEY = re.compile(r"(?P<secret>AIza[A-Za-z0-9_-]{35})")
_PRIVATE_KEY = re.compile(
    r"(?P<secret>-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    r".*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_URL_CREDENTIALS = re.compile(
    r"(?P<scheme>https?://)(?P<userinfo>[^/@\s]+)@",
    re.IGNORECASE,
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _redact_keyed_secret(match: re.Match[str]) -> str:
    quote = match.group("quote") or ""
    return f"{match.group('prefix')}{quote}***{quote}"


def _redact_named_secret(match: re.Match[str]) -> str:
    prefix = match.groupdict().get("prefix", "")
    return f"{prefix}***"


def safe_error_message(error: BaseException | str, *, limit: int = 700) -> str:
    """Return single-line error text with common credential forms removed."""

    if limit < 2:
        raise ValueError("limit must be at least 2")
    if isinstance(error, BaseException):
        message = str(error).strip() or type(error).__name__
    else:
        message = error.strip()
    if not message:
        return ""
    message = _ANSI_ESCAPE.sub("", message)
    message = re.sub(r"\s+", " ", message)
    message = _URL_CREDENTIALS.sub(r"\g<scheme>***@", message)
    message = _KEYED_SECRET.sub(_redact_keyed_secret, message)
    message = _BEARER_SECRET.sub(_redact_named_secret, message)
    message = _PREFIXED_SECRET.sub(_redact_named_secret, message)
    message = _AWS_ACCESS_KEY.sub(_redact_named_secret, message)
    message = _GOOGLE_API_KEY.sub(_redact_named_secret, message)
    message = _PRIVATE_KEY.sub(_redact_named_secret, message)
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return message
