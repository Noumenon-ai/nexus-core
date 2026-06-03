"""nexus-contacts MCP server — Phase 2.5b Server 5.

Google Contacts (People API) read + write + groups + vCard import.

Auth: reads OAuth token from NEXUS_GOOGLE_TOKEN_DIR/{user_id}.json — the same
file nexus-calendar uses. The token must include the `auth/contacts` scope
(granted by the existing Google integration OAuth flow).

Configuration:
  NEXUS_GOOGLE_TOKEN_DIR      directory of {user_id}.json files (defaults
                               to /opt/nexus-core/.data/google_tokens)
  NEXUS_MCP_DEFAULT_USER_ID   user UUID for tools without explicit user_id

Tools that touch calendar/email for recent_interactions reuse nexus-calendar
and nexus-email pluming so the credential surface stays simple.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from services.file_system_tools import FileSystemSecurityError, _validate_path


mcp = FastMCP("nexus-contacts")

# SHARED Google OAuth scope set — see mcp_servers.nexus_calendar for rationale.
# Every MCP server that touches .data/google_tokens/<uid>.json passes the SAME
# superset so refresh() doesn't narrow the persisted scopes.
_GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/tasks",
]
_DEFAULT_TOKEN_DIR = Path(
    "/opt/nexus-core/.data/google_tokens"
)

_CONTACT_READ_MASK = (
    "names,emailAddresses,phoneNumbers,addresses,organizations,"
    "memberships,biographies,birthdays,urls,nicknames,events"
)


# ---------- internals ----------

def _token_dir() -> Path:
    raw = os.environ.get("NEXUS_GOOGLE_TOKEN_DIR")
    return Path(raw) if raw else _DEFAULT_TOKEN_DIR


def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get("NEXUS_MCP_DEFAULT_USER_ID")
    return env_uid.strip() if env_uid else None


def _get_people(user_id: str | None) -> Any | dict:
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id",
                "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    token_path = _token_dir() / f"{uid}.json"
    if not token_path.exists():
        return {"ok": False, "reason": "no_google_credentials",
                "detail": f"token file missing: {token_path}"}
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        return {"ok": False, "reason": "google_libs_not_installed", "detail": str(exc)}
    try:
        # scopes=None preserves the file's persisted scope set across refreshes
        # — see nexus_calendar._get_calendar for the rationale.
        creds = Credentials.from_authorized_user_file(str(token_path))
    except Exception as exc:
        return {"ok": False, "reason": "credentials_load_failed", "detail": str(exc)[:300]}
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "reason": "token_refresh_failed", "detail": str(exc)[:300]}
    try:
        return build("people", "v1", credentials=creds, cache_discovery=False)
    except Exception as exc:
        return {"ok": False, "reason": "client_build_failed", "detail": str(exc)[:300]}


def _serialize_contact(person: dict) -> dict:
    names = person.get("names") or []
    emails = person.get("emailAddresses") or []
    phones = person.get("phoneNumbers") or []
    addresses = person.get("addresses") or []
    orgs = person.get("organizations") or []
    memberships = person.get("memberships") or []
    bios = person.get("biographies") or []
    birthdays = person.get("birthdays") or []
    urls = person.get("urls") or []
    nicknames = person.get("nicknames") or []
    return {
        "id": person.get("resourceName"),
        "etag": person.get("etag"),
        "display_name": names[0].get("displayName") if names else None,
        "given_name": names[0].get("givenName") if names else None,
        "family_name": names[0].get("familyName") if names else None,
        "emails": [{"value": e.get("value"), "type": e.get("type")} for e in emails],
        "phones": [{"value": p.get("value"), "type": p.get("type")} for p in phones],
        "addresses": [{"formatted": a.get("formattedValue"), "type": a.get("type")} for a in addresses],
        "organizations": [{"name": o.get("name"), "title": o.get("title")} for o in orgs],
        "nicknames": [n.get("value") for n in nicknames],
        "groups": [m.get("contactGroupMembership", {}).get("contactGroupId")
                   for m in memberships if m.get("contactGroupMembership")],
        "notes": bios[0].get("value") if bios else None,
        "urls": [{"value": u.get("value"), "type": u.get("type")} for u in urls],
        "birthday": birthdays[0].get("date") if birthdays else None,
    }


def _resolve_group_id(svc, group_name: str) -> str | None:
    if group_name.startswith("contactGroups/"):
        return group_name
    resp = svc.contactGroups().list(pageSize=200).execute()
    for grp in resp.get("contactGroups") or []:
        if (grp.get("formattedName") or "").lower() == group_name.lower() \
                or (grp.get("name") or "").lower() == group_name.lower():
            return grp.get("resourceName")
    return None


# ---------- read tools ----------

@mcp.tool()
def find_contact(query: str, limit: int = 25, user_id: str | None = None) -> dict:
    """Search contacts by name/email/phone/company substring (Google People searchContacts)."""
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "reason": "empty_query", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": ""}
    limit = max(1, min(limit, 100))
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        resp = svc.people().searchContacts(
            query=query.strip(), readMask=_CONTACT_READ_MASK, pageSize=limit,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    results = resp.get("results") or []
    contacts = [_serialize_contact(r.get("person") or {}) for r in results]
    return {"ok": True, "query": query, "count": len(contacts), "contacts": contacts}


@mcp.tool()
def get_contact_details(contact_id: str, user_id: str | None = None) -> dict:
    """Full contact record by resourceName (e.g. 'people/c1234567890123')."""
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        person = svc.people().get(resourceName=contact_id, personFields=_CONTACT_READ_MASK).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact": _serialize_contact(person)}


@mcp.tool()
def recent_interactions(contact_id: str, days: int = 30, user_id: str | None = None) -> dict:
    """Recent calendar events + Gmail messages involving this contact's email addresses.

    Uses nexus-calendar's token (same as people) for events and nexus-email's
    token for messages. Returns an aggregate summary; caller LLM can interpret.
    """
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    try:
        days = int(days)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_days", "detail": ""}
    if not 1 <= days <= 365:
        return {"ok": False, "reason": "days_out_of_range", "detail": "1..365"}

    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        person = svc.people().get(resourceName=contact_id, personFields=_CONTACT_READ_MASK).execute()
    except Exception as exc:
        return {"ok": False, "reason": "contact_fetch_failed", "detail": str(exc)[:300]}
    serialized = _serialize_contact(person)
    emails = [e["value"] for e in serialized["emails"] if e.get("value")]

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)
    events_summary: list[dict] = []
    emails_summary: list[dict] = []

    # Calendar events
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        token_path = _token_dir() / f"{_resolve_user_id(user_id)}.json"
        cal_creds = Credentials.from_authorized_user_file(str(token_path))  # scopes from file (don't narrow)
        if cal_creds.expired and cal_creds.refresh_token:
            cal_creds.refresh(Request())
        cal_svc = build("calendar", "v3", credentials=cal_creds, cache_discovery=False)
        ev_resp = cal_svc.events().list(
            calendarId="primary",
            timeMin=window_start.isoformat().replace("+00:00", "Z"),
            timeMax=now.isoformat().replace("+00:00", "Z"),
            singleEvents=True, orderBy="startTime", maxResults=200,
        ).execute()
        for ev in ev_resp.get("items") or []:
            attendee_emails = {(a.get("email") or "").lower() for a in (ev.get("attendees") or [])}
            if any(e.lower() in attendee_emails for e in emails):
                start = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
                events_summary.append({"id": ev.get("id"), "summary": ev.get("summary"), "start": start})
    except Exception as exc:
        events_summary = [{"error": f"calendar_fetch_failed: {str(exc)[:200]}"}]

    # Gmail messages
    try:
        gmail_token_dir = Path(os.environ.get("NEXUS_GMAIL_TOKEN_DIR",
                                              "/opt/nexus-core/.data/gmail_tokens"))
        gmail_token = gmail_token_dir / f"{_resolve_user_id(user_id)}.json"
        if gmail_token.exists() and emails:
            from google.oauth2.credentials import Credentials as GCreds
            from google.auth.transport.requests import Request as GReq
            from googleapiclient.discovery import build as gbuild
            gm_creds = GCreds.from_authorized_user_file(str(gmail_token))  # scopes from file
            if gm_creds.expired and gm_creds.refresh_token:
                gm_creds.refresh(GReq())
            gm_svc = gbuild("gmail", "v1", credentials=gm_creds, cache_discovery=False)
            q_parts = [f"(from:{e} OR to:{e})" for e in emails]
            q = "(" + " OR ".join(q_parts) + f") newer_than:{days}d"
            resp = gm_svc.users().messages().list(userId="me", q=q, maxResults=20).execute()
            for item in resp.get("messages") or []:
                msg = gm_svc.users().messages().get(
                    userId="me", id=item["id"], format="metadata",
                    metadataHeaders=["Subject", "From", "Date", "To"],
                ).execute()
                hdrs = {h["name"]: h["value"] for h in (msg.get("payload") or {}).get("headers", [])}
                emails_summary.append({
                    "id": msg.get("id"), "subject": hdrs.get("Subject"),
                    "from": hdrs.get("From"), "date": hdrs.get("Date"),
                })
    except Exception as exc:
        emails_summary = [{"error": f"gmail_fetch_failed: {str(exc)[:200]}"}]

    return {
        "ok": True,
        "contact_id": contact_id,
        "contact_emails": emails,
        "window_days": days,
        "calendar_event_count": len([e for e in events_summary if "error" not in e]),
        "calendar_events": events_summary,
        "gmail_message_count": len([m for m in emails_summary if "error" not in m]),
        "gmail_messages": emails_summary,
    }


@mcp.tool()
def list_groups(user_id: str | None = None) -> dict:
    """All contact groups (Google labels: family, work, custom, etc.)."""
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        resp = svc.contactGroups().list(pageSize=200).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    groups = []
    for g in resp.get("contactGroups") or []:
        groups.append({
            "id": g.get("resourceName"),
            "name": g.get("formattedName") or g.get("name"),
            "type": g.get("groupType"),
            "member_count": g.get("memberCount"),
        })
    return {"ok": True, "count": len(groups), "groups": groups}


@mcp.tool()
def list_contacts_in_group(group_name: str, limit: int = 200, user_id: str | None = None) -> dict:
    """Contacts within a group (resolved by name or resourceName)."""
    if not isinstance(group_name, str) or not group_name.strip():
        return {"ok": False, "reason": "empty_group_name", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": ""}
    limit = max(1, min(limit, 1000))
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    group_id = _resolve_group_id(svc, group_name)
    if group_id is None:
        return {"ok": False, "reason": "group_not_found", "detail": group_name}
    try:
        grp = svc.contactGroups().get(resourceName=group_id, maxMembers=limit).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    member_resource_names = grp.get("memberResourceNames") or []
    if not member_resource_names:
        return {"ok": True, "group_id": group_id, "count": 0, "contacts": []}
    try:
        people_resp = svc.people().getBatchGet(
            resourceNames=member_resource_names, personFields=_CONTACT_READ_MASK,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "batch_fetch_failed", "detail": str(exc)[:300]}
    contacts = [_serialize_contact(r.get("person") or {}) for r in (people_resp.get("responses") or [])]
    return {"ok": True, "group_id": group_id, "group_name": group_name,
            "count": len(contacts), "contacts": contacts}


@mcp.tool()
def birthday_reminders(days_ahead: int = 30, user_id: str | None = None) -> dict:
    """Contacts with birthdays in the next N days (Google People birthday field)."""
    try:
        days_ahead = int(days_ahead)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_days_ahead", "detail": ""}
    if not 1 <= days_ahead <= 366:
        return {"ok": False, "reason": "days_ahead_out_of_range", "detail": "1..366"}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    upcoming: list[dict] = []
    today = datetime.now(timezone.utc).date()
    page_token = None
    try:
        while True:
            kw = {"resourceName": "people/me", "pageSize": 1000, "personFields": "names,birthdays"}
            if page_token:
                kw["pageToken"] = page_token
            resp = svc.people().connections().list(**kw).execute()
            for conn in resp.get("connections") or []:
                bdays = conn.get("birthdays") or []
                if not bdays:
                    continue
                date_part = bdays[0].get("date") or {}
                m = date_part.get("month")
                d = date_part.get("day")
                if not m or not d:
                    continue
                year_for_compare = today.year
                try:
                    candidate = today.replace(year=year_for_compare, month=m, day=d)
                except ValueError:
                    continue
                if candidate < today:
                    try:
                        candidate = candidate.replace(year=year_for_compare + 1)
                    except ValueError:
                        continue
                delta = (candidate - today).days
                if delta <= days_ahead:
                    names = conn.get("names") or []
                    upcoming.append({
                        "id": conn.get("resourceName"),
                        "name": names[0].get("displayName") if names else None,
                        "birthday_date": f"{m:02d}-{d:02d}",
                        "days_until": delta,
                    })
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    upcoming.sort(key=lambda r: r["days_until"])
    return {"ok": True, "days_ahead": days_ahead, "count": len(upcoming), "birthdays": upcoming}


# ---------- write tools ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Create contact?"})
def create_contact(name: str, email: str | None = None, phone: str | None = None,
                   notes: str | None = None, groups: list[str] | None = None,
                   user_id: str | None = None) -> dict:
    """Create a new Google contact. groups = list of group names (resolved server-side)."""
    if not isinstance(name, str) or not name.strip():
        return {"ok": False, "reason": "empty_name", "detail": ""}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    body: dict[str, Any] = {"names": [{"unstructuredName": name.strip()}]}
    if email:
        body["emailAddresses"] = [{"value": email}]
    if phone:
        body["phoneNumbers"] = [{"value": phone}]
    if notes:
        body["biographies"] = [{"value": notes, "contentType": "TEXT_PLAIN"}]
    if groups:
        resolved = []
        for g in groups:
            gid = _resolve_group_id(svc, g)
            if gid:
                resolved.append({"contactGroupMembership": {"contactGroupId": gid.split("/")[-1]}})
        if resolved:
            body["memberships"] = resolved
    try:
        created = svc.people().createContact(body=body).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact": _serialize_contact(created)}


_UPDATABLE_FIELDS = {
    "names": "names",
    "emails": "emailAddresses",
    "phones": "phoneNumbers",
    "notes": "biographies",
    "addresses": "addresses",
    "organizations": "organizations",
}


@mcp.tool(meta={"approval_required": True, "approval_template": "Update contact?"})
def update_contact(contact_id: str, fields: dict, user_id: str | None = None) -> dict:
    """Modify a contact. fields keys: names (str), emails (list[str]), phones (list[str]), notes (str).

    Only the keys provided are updated; etag from the existing record is sent
    back to avoid clobbering concurrent edits.
    """
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    if not isinstance(fields, dict) or not fields:
        return {"ok": False, "reason": "empty_fields", "detail": ""}
    extra = set(fields) - set(_UPDATABLE_FIELDS)
    if extra:
        return {"ok": False, "reason": "unknown_fields", "detail": f"unknown: {sorted(extra)}"}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        existing = svc.people().get(resourceName=contact_id, personFields=_CONTACT_READ_MASK).execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}
    body: dict[str, Any] = {"etag": existing.get("etag")}
    update_mask_parts: list[str] = []
    if "names" in fields:
        body["names"] = [{"unstructuredName": fields["names"]}]
        update_mask_parts.append("names")
    if "emails" in fields:
        emails = fields["emails"]
        if not isinstance(emails, list):
            return {"ok": False, "reason": "invalid_emails", "detail": "must be list of strings"}
        body["emailAddresses"] = [{"value": e} for e in emails]
        update_mask_parts.append("emailAddresses")
    if "phones" in fields:
        phones = fields["phones"]
        if not isinstance(phones, list):
            return {"ok": False, "reason": "invalid_phones", "detail": "must be list of strings"}
        body["phoneNumbers"] = [{"value": p} for p in phones]
        update_mask_parts.append("phoneNumbers")
    if "notes" in fields:
        body["biographies"] = [{"value": fields["notes"], "contentType": "TEXT_PLAIN"}]
        update_mask_parts.append("biographies")
    if "addresses" in fields:
        addrs = fields["addresses"]
        if not isinstance(addrs, list):
            return {"ok": False, "reason": "invalid_addresses", "detail": "must be list of strings"}
        body["addresses"] = [{"formattedValue": a} for a in addrs]
        update_mask_parts.append("addresses")
    if "organizations" in fields:
        orgs = fields["organizations"]
        if not isinstance(orgs, list):
            return {"ok": False, "reason": "invalid_organizations", "detail": "must be list of strings"}
        body["organizations"] = [{"name": o} for o in orgs]
        update_mask_parts.append("organizations")
    try:
        updated = svc.people().updateContact(
            resourceName=contact_id, updatePersonFields=",".join(update_mask_parts), body=body,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact": _serialize_contact(updated)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Delete contact?"})
def delete_contact(contact_id: str, user_id: str | None = None) -> dict:
    """DELETE a Google contact by resourceName."""
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    try:
        svc.people().deleteContact(resourceName=contact_id).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact_id": contact_id}


@mcp.tool(meta={"approval_required": True, "approval_template": "Add contact to group?"})
def add_to_group(contact_id: str, group_name: str, user_id: str | None = None) -> dict:
    """Add a contact to a group. Creates the group if it doesn't exist."""
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    if not isinstance(group_name, str) or not group_name.strip():
        return {"ok": False, "reason": "empty_group_name", "detail": ""}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    group_id = _resolve_group_id(svc, group_name)
    if group_id is None:
        try:
            created = svc.contactGroups().create(body={"contactGroup": {"name": group_name}}).execute()
            group_id = created.get("resourceName")
        except Exception as exc:
            return {"ok": False, "reason": "group_create_failed", "detail": str(exc)[:300]}
    try:
        svc.contactGroups().members().modify(
            resourceName=group_id, body={"resourceNamesToAdd": [contact_id]},
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact_id": contact_id, "group_id": group_id, "group_name": group_name}


@mcp.tool(meta={"approval_required": True, "approval_template": "Remove contact from group?"})
def remove_from_group(contact_id: str, group_name: str, user_id: str | None = None) -> dict:
    """Remove a contact from a group."""
    if not isinstance(contact_id, str) or not contact_id.strip():
        return {"ok": False, "reason": "empty_contact_id", "detail": ""}
    if not isinstance(group_name, str) or not group_name.strip():
        return {"ok": False, "reason": "empty_group_name", "detail": ""}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    group_id = _resolve_group_id(svc, group_name)
    if group_id is None:
        return {"ok": False, "reason": "group_not_found", "detail": group_name}
    try:
        svc.contactGroups().members().modify(
            resourceName=group_id, body={"resourceNamesToRemove": [contact_id]},
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "contact_id": contact_id, "group_id": group_id}


# ---------- vCard import ----------

def _parse_vcard(text: str) -> list[dict]:
    """Minimal vCard 3.0/4.0 parser. Returns a list of {name,email,phone,org,bday} dicts."""
    out: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("BEGIN:VCARD"):
            current = {"emails": [], "phones": [], "name": None, "org": None, "bday": None, "notes": None}
            continue
        if line.upper().startswith("END:VCARD"):
            if current and (current["name"] or current["emails"] or current["phones"]):
                out.append(current)
            current = None
            continue
        if current is None:
            continue
        if ":" not in line:
            continue
        prop, value = line.split(":", 1)
        prop_upper = prop.split(";", 1)[0].upper()
        if prop_upper == "FN":
            current["name"] = value
        elif prop_upper == "N" and not current["name"]:
            parts = value.split(";")
            family = parts[0] if parts else ""
            given = parts[1] if len(parts) > 1 else ""
            current["name"] = f"{given} {family}".strip() or family or given
        elif prop_upper == "EMAIL":
            if value:
                current["emails"].append(value)
        elif prop_upper == "TEL":
            if value:
                current["phones"].append(value)
        elif prop_upper == "ORG":
            current["org"] = value.split(";", 1)[0]
        elif prop_upper == "BDAY":
            current["bday"] = value
        elif prop_upper == "NOTE":
            current["notes"] = value
    return out


@mcp.tool(meta={"approval_required": True, "approval_template": "Import vCard contacts?"})
def import_vcard(vcard_path: str, user_id: str | None = None) -> dict:
    """Bulk-import contacts from a .vcf file inside ALLOWED_ROOTS.

    Parses VCARD 3.0/4.0 (FN, N, EMAIL, TEL, ORG, BDAY, NOTE), creates each
    contact via People.createContact, returns per-record success/failure.
    """
    try:
        resolved = _validate_path(vcard_path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "vcard_path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    if resolved.suffix.lower() not in {".vcf", ".vcard"}:
        return {"ok": False, "reason": "wrong_extension", "detail": resolved.suffix}
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "reason": "read_failed", "detail": str(exc)[:200]}
    parsed = _parse_vcard(text)
    if not parsed:
        return {"ok": True, "imported_count": 0, "skipped_count": 0, "results": [],
                "note": "vcard file contained no parseable VCARD entries"}
    svc = _get_people(user_id)
    if isinstance(svc, dict):
        return svc
    results: list[dict] = []
    for entry in parsed:
        body: dict[str, Any] = {}
        if entry["name"]:
            body["names"] = [{"unstructuredName": entry["name"]}]
        if entry["emails"]:
            body["emailAddresses"] = [{"value": e} for e in entry["emails"]]
        if entry["phones"]:
            body["phoneNumbers"] = [{"value": p} for p in entry["phones"]]
        if entry["org"]:
            body["organizations"] = [{"name": entry["org"]}]
        if entry["notes"]:
            body["biographies"] = [{"value": entry["notes"], "contentType": "TEXT_PLAIN"}]
        if not body:
            results.append({"name": entry["name"], "ok": False, "reason": "no_fields_to_save"})
            continue
        try:
            created = svc.people().createContact(body=body).execute()
            results.append({"name": entry["name"], "ok": True, "id": created.get("resourceName")})
        except Exception as exc:
            results.append({"name": entry["name"], "ok": False, "reason": "api_error",
                            "detail": str(exc)[:200]})
    return {
        "ok": True,
        "vcard_path": str(resolved),
        "parsed_count": len(parsed),
        "imported_count": sum(1 for r in results if r.get("ok")),
        "skipped_count": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
