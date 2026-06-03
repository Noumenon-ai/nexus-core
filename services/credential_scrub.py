from __future__ import annotations

import re
from typing import Any, Optional

_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("stripe_or_openai", re.compile(r"\bsk_(?:live|test)?_?[A-Za-z0-9]{16,}\b")),
    ("stripe_publishable", re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.=]{16,}", re.IGNORECASE)),
    ("password_kv", re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE)),
    ("api_key_kv", re.compile(r"\bapi[_-]?key\s*[:=]\s*\S+", re.IGNORECASE)),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("generic_secret_kv", re.compile(r"\b(?:secret|access[_-]?key)\s*[:=]\s*\S+", re.IGNORECASE)),
]


def find_credential_matches(text: Optional[str]) -> list[str]:
    """Return list of pattern names that matched in `text`. Empty list = clean."""
    if not text or not isinstance(text, str):
        return []
    return [name for name, pat in _CREDENTIAL_PATTERNS if pat.search(text)]


def contains_credential(text: Optional[str]) -> bool:
    return bool(find_credential_matches(text))


def scrub_messages(
    messages: list[dict[str, Any]],
    *,
    content_key: str = "content",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition messages into (surviving, dropped) by credential presence.

    Messages without `content_key`, with None content, or with empty content
    are kept (no credential to scrub).
    """
    surviving: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for msg in messages:
        text = msg.get(content_key)
        if isinstance(text, str) and contains_credential(text):
            dropped.append(msg)
        else:
            surviving.append(msg)
    return surviving, dropped
