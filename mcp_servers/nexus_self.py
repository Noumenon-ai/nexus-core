"""nexus-self MCP server — cross-source capability introspection (H2-043 FIX A).

When the user asks "what Gmails do you have access to?", NEXUS pre-H2-043 only
saw one of two surfaces: either the nexus-email OAuth token (single account)
OR the dashboard quick_links (browser-clickable bookmarks). The answer was
silo-shaped and almost always wrong because the ExampleBrand Gmail lived in one
surface while the primary Gmail lived in the other.

This server exposes ``describe_my_access(domain)`` which queries both
surfaces and returns a structured capability map plus a natural-language
summary. The brain_router system prompt instructs the LLM to call this tool
first whenever the user asks about capabilities, access, or "what can you do
with X".

Domains supported:
  - email      → Gmail accounts (via services.gmail_accounts) + dashboard
                  quick_links with category="email"
  - messaging  → quick_links with category="messaging" + the static
                  nexus-messaging surface (WhatsApp Web)
  - drive      → quick_links with category="drive"
  - docs       → quick_links with category="docs"
  - all / None → every domain, plus an "other" bucket for uncategorized links

Tokens / DB state are read-only here. Nothing in this server mutates anything.
"""
from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from services.gmail_accounts import (
    list_accounts_for_user,
    resolve_token_path,
)


mcp = FastMCP("nexus-self")


# Domain → quick_link category mapping. Domain names align with the spec
# the user wrote; category names align with dashboard.storage._VALID_CATEGORIES.
# Both happen to match for the categories that exist; we map explicitly so
# the coupling is obvious if either side renames in the future.
_DOMAIN_TO_QUICK_LINK_CATEGORY = {
    'email': 'email',
    'messaging': 'messaging',
    'drive': 'drive',
    'docs': 'docs',
}

# Domains we know how to summarize in describe_my_access. Anything else →
# graceful invalid_domain error.
_KNOWN_DOMAINS = {*_DOMAIN_TO_QUICK_LINK_CATEGORY.keys(), 'all'}


# ---------- internals ----------

def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get('NEXUS_MCP_DEFAULT_USER_ID')
    return env_uid.strip() if env_uid else None


def _list_quick_links_safe(category: str | None = None) -> list[dict[str, Any]]:
    """Read dashboard.storage quick_links. Returns [] on any failure
    (DB missing, schema drift, etc.) — this is an introspection tool and
    must not crash the caller when a downstream surface is unavailable."""
    try:
        from dashboard.storage import list_quick_links  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        all_links = list_quick_links() or []
    except Exception:
        return []
    if category is None:
        return all_links
    return [link for link in all_links if (link.get('category') or '').strip() == category]


def _email_capability(user_id: str) -> dict[str, Any]:
    """API-authenticated Gmail accounts + browser-link email surfaces."""
    api_labels = list_accounts_for_user(user_id)
    api_accounts = [
        {'label': label,
         'token_path': str(resolve_token_path(user_id, label) or '<unknown>')}
        for label in api_labels
    ]
    browser_links = _list_quick_links_safe('email')

    api_names = ', '.join(f'"{a["label"]}"' for a in api_accounts) or '(none)'
    link_labels = ', '.join(f'"{l["label"]}"' for l in browser_links) or '(none)'
    if not api_accounts and not browser_links:
        summary = ('I have no Gmail API access configured and no email quick_links '
                   'in the dashboard. Run scripts/add_gmail_account.py <label> to add '
                   'API access.')
    else:
        summary_parts: list[str] = []
        if api_accounts:
            summary_parts.append(
                f'I have API access to Gmail accounts: {api_names}. I can read, '
                f'search, draft, and send mail through these.'
            )
        if browser_links:
            summary_parts.append(
                f'I also see browser-clickable email links from your dashboard: '
                f'{link_labels}. These open in a browser; I cannot read or send '
                f'mail through them.'
            )
        summary_parts.append(
            'To add another Gmail account, run '
            '`scripts/add_gmail_account.py <label>` on the host and sign in '
            'with that account in the browser, then ask me about accounts '
            'again — the new token becomes visible to me immediately.'
        )
        summary = ' '.join(summary_parts)

    return {
        'domain': 'email',
        'api_authenticated': api_accounts,
        'browser_links': browser_links,
        'summary_text': summary,
    }


def _quick_link_only_capability(domain: str, *,
                                singular: str,
                                api_note: str) -> dict[str, Any]:
    """Generic capability shape for domains whose API surface NEXUS doesn't
    introspect from this server (calendar/drive/docs/messaging). Returns
    dashboard quick_links and a summary that explicitly says API state isn't
    enumerated here."""
    category = _DOMAIN_TO_QUICK_LINK_CATEGORY.get(domain)
    browser_links = _list_quick_links_safe(category)
    if browser_links:
        labels = ', '.join(f'"{l["label"]}"' for l in browser_links)
        summary = (f'{api_note} Browser-clickable {singular} links from your '
                   f'dashboard: {labels}.')
    else:
        summary = (f'{api_note} No {singular} quick_links in the dashboard either.')
    return {
        'domain': domain,
        'api_authenticated': [],
        'browser_links': browser_links,
        'summary_text': summary,
    }


# ---------- tools ----------

@mcp.tool()
def describe_my_access(domain: str | None = None, user_id: str | None = None) -> dict:
    """Cross-source capability introspection.

    Returns what NEXUS has access to for ``domain``, combining API-
    authenticated state with dashboard browser-link bookmarks. Call this
    tool first when the user asks about capabilities, access, or "what
    can you do with X" — the answer was historically wrong because the
    bot only saw one surface at a time.

    domain values:
      - "email"     → Gmail accounts + dashboard email links
      - "messaging" → dashboard messaging links + WhatsApp Web (browser)
      - "drive"     → dashboard drive links
      - "docs"      → dashboard docs links
      - "all"/None  → map covering every domain
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return {'ok': False, 'reason': 'no_user_id',
                'detail': 'pass user_id or set NEXUS_MCP_DEFAULT_USER_ID'}

    requested = (domain or 'all').strip().lower()
    if requested not in _KNOWN_DOMAINS:
        return {'ok': False, 'reason': 'invalid_domain',
                'detail': f'domain {requested!r} not in {sorted(_KNOWN_DOMAINS)}'}

    if requested == 'all':
        domains_payload = {
            'email': _email_capability(uid),
            'messaging': _quick_link_only_capability(
                'messaging',
                singular='messaging',
                api_note=('I can drive WhatsApp Web through the nexus-messaging '
                          'browser tools (no native API auth).'),
            ),
            'drive': _quick_link_only_capability(
                'drive',
                singular='drive',
                api_note=('I do not have a Google Drive API integration in this '
                          'Nexus build.'),
            ),
            'docs': _quick_link_only_capability(
                'docs',
                singular='docs',
                api_note=('I do not have a docs API integration.'),
            ),
        }
        # Cross-domain summary leads with the bit users ask about most: email.
        return {
            'ok': True,
            'user_id': uid,
            'domain': 'all',
            'domains': domains_payload,
            'summary_text': domains_payload['email']['summary_text'],
        }

    if requested == 'email':
        return {'ok': True, 'user_id': uid, **_email_capability(uid)}

    notes = {
        'messaging': ('I can drive WhatsApp Web through the nexus-messaging '
                      'browser tools (no native API auth).'),
        'drive': ('I do not have a Google Drive API integration in this Nexus build.'),
        'docs': ('I do not have a docs API integration.'),
    }
    return {'ok': True, 'user_id': uid,
            **_quick_link_only_capability(requested,
                                          singular=requested,
                                          api_note=notes[requested])}


def main() -> None:
    mcp.run()


if __name__ == '__main__':
    main()
