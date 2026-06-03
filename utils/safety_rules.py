from __future__ import annotations

import re
from dataclasses import dataclass


SENSITIVE_ACTION_TYPES = {
    'outbound_message',
    'delete_memory',
    'delete_reminder',
    'delete_task',
    'delete_email_cache',
    'financial_action',
    'legal_action',
    'third_party_contact',
    'external_state_change',
}

SECRET_PATTERNS = [
    re.compile(r'api[_\s-]?key', re.IGNORECASE),
    re.compile(r'bearer\s+[a-z0-9._-]+', re.IGNORECASE),
    re.compile(r'password', re.IGNORECASE),
    re.compile(r'token', re.IGNORECASE),
    re.compile(r'secret', re.IGNORECASE),
]


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    sensitive: bool
    reason: str



def is_sensitive_action(action_type: str) -> bool:
    return action_type in SENSITIVE_ACTION_TYPES



def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)
