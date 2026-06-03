"""Multi-account Gmail token storage (H2-043 FIX B).

Pre-H2-043 the nexus-email MCP server assumed exactly one Gmail account per
Telegram user, with tokens at ``<token_dir>/<user_id>.json``. The user has a
second Gmail account (ExampleBrand) they wanted reachable from NEXUS, so this module
introduces labeled accounts:

    <token_dir>/<account_label>/<user_id>.json

The default label is ``primary``. Tokens that already exist at the legacy
path (``<token_dir>/<user_id>.json``) are treated as the user's ``primary``
account and migrated forward on first read so the rest of the codebase only
has to think about the labeled scheme.

Pure module — no MCP / FastMCP imports — so both ``mcp_servers/nexus_email``
and ``mcp_servers/nexus_self`` can use it without coupling the two servers
together.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_ACCOUNT_LABEL = 'primary'

_DEFAULT_TOKEN_DIR = Path(
    '/opt/nexus-core/.data/gmail_tokens'
)

# Account labels are used as filesystem directory names. Restrict to a safe
# alphabet so a malicious label can't escape the token_dir via path traversal
# or collide with reserved Windows names. We keep it permissive enough for
# real-world labels ("primary", "shop", "work_2025") without surprising users.
_LABEL_PATTERN = re.compile(r'^[a-z][a-z0-9_-]{0,31}$')


def token_dir() -> Path:
    """Resolve the on-disk root holding all Gmail tokens. Honors the
    NEXUS_GMAIL_TOKEN_DIR env var so tests can point at a temp directory."""
    raw = os.environ.get('NEXUS_GMAIL_TOKEN_DIR')
    return Path(raw) if raw else _DEFAULT_TOKEN_DIR


def is_valid_label(label: str) -> bool:
    """Reject empty / overlong / path-traversal-shaped labels at the
    boundary so downstream code can trust the on-disk layout."""
    return bool(label and _LABEL_PATTERN.match(label))


def account_token_path(user_id: str, account: str = DEFAULT_ACCOUNT_LABEL) -> Path:
    """Return the canonical token path for (user_id, account). Does NOT
    create the file; the caller is responsible for that during OAuth."""
    return token_dir() / account / f'{user_id}.json'


def _legacy_token_path(user_id: str) -> Path:
    """Pre-H2-043 layout: <token_dir>/<user_id>.json with no account dir."""
    return token_dir() / f'{user_id}.json'


def resolve_token_path(user_id: str, account: str = DEFAULT_ACCOUNT_LABEL) -> Path | None:
    """Return the path to load credentials from, or None when nothing exists.

    Side effect: when ``account == "primary"`` and a legacy token file exists
    at ``<token_dir>/<user_id>.json``, this function migrates it forward to
    the canonical labeled path on first call. Migration is idempotent — if
    the canonical path is already present we leave the legacy file alone
    (the canonical wins). The migrate-forward path uses ``os.replace`` so
    no window exists where both copies are live.
    """
    canonical = account_token_path(user_id, account)
    if canonical.exists():
        return canonical
    if account == DEFAULT_ACCOUNT_LABEL:
        legacy = _legacy_token_path(user_id)
        if legacy.exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(canonical.parent, 0o700)
            except OSError:
                pass
            os.replace(legacy, canonical)
            try:
                os.chmod(canonical, 0o600)
            except OSError:
                pass
            return canonical
    return None


def list_accounts_for_user(user_id: str) -> list[str]:
    """Enumerate every account label that has a token file for ``user_id``.

    Includes the legacy unscoped token (counted as ``primary``) when the
    canonical primary path is absent. Result is sorted with ``primary``
    first so the natural-language summary in nexus-self can lead with
    the most-used account.
    """
    base = token_dir()
    found: set[str] = set()
    if base.exists():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            label = child.name
            if not is_valid_label(label):
                continue
            if (child / f'{user_id}.json').exists():
                found.add(label)
    if DEFAULT_ACCOUNT_LABEL not in found and _legacy_token_path(user_id).exists():
        found.add(DEFAULT_ACCOUNT_LABEL)

    def _sort_key(label: str) -> tuple[int, str]:
        # primary first, then alphabetical
        return (0 if label == DEFAULT_ACCOUNT_LABEL else 1, label)

    return sorted(found, key=_sort_key)
