"""nexus-email MCP server — Phase 2.5b Server 3.

Gmail read/search/draft/send/label/archive/delete + attachment extraction.

Auth: reads OAuth token from NEXUS_GMAIL_TOKEN_DIR/<account>/<user_id>.json
(default .data/gmail_tokens) — H2-043 multi-account layout. Pre-H2-043 tokens
at the legacy unscoped path <token_dir>/<user_id>.json are treated as the
``primary`` account and migrated forward on first read. Requires the 4
scopes: gmail.modify, gmail.compose, gmail.send, gmail.labels.

Configuration:
  NEXUS_GMAIL_TOKEN_DIR     directory holding <account>/<user_id>.json files
                              (default /opt/nexus-core/.data/gmail_tokens)
  NEXUS_MCP_DEFAULT_USER_ID  user UUID for tools without explicit user_id

Every tool returns a dict; failures return {'ok': False, 'reason': str, 'detail': str}.
Tools requiring approval in the spec still execute on call — Claude CLI gates them
at the user-permission layer (same pattern as nexus-filesystem).
"""
from __future__ import annotations

import base64
import email
import os
import re
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from services.file_system_tools import FileSystemSecurityError, _validate_path
from services.gmail_accounts import (
    DEFAULT_ACCOUNT_LABEL,
    is_valid_label,
    list_accounts_for_user,
    resolve_token_path,
)
from mcp_servers.nexus_filesystem import _validate_write_target


mcp = FastMCP("nexus-email")

_GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.labels",
]
_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024  # 25 MB — same as Gmail attachment cap


# ---------- internals ----------

def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get("NEXUS_MCP_DEFAULT_USER_ID")
    return env_uid.strip() if env_uid else None


def _get_gmail(user_id: str | None, account: str = DEFAULT_ACCOUNT_LABEL) -> Any | dict:
    """Build a Gmail service object for ``(user_id, account)``, or return an
    error-shaped dict. H2-043: ``account`` selects which labeled token file
    to load; ``"primary"`` covers the pre-multi-account path."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id",
                "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    label = (account or DEFAULT_ACCOUNT_LABEL).strip()
    if not is_valid_label(label):
        return {"ok": False, "reason": "invalid_account_label",
                "detail": f"label {label!r} must match [a-z][a-z0-9_-]{{0,31}} "
                          f"(used as a directory name on disk)"}
    token_path = resolve_token_path(uid, label)
    if token_path is None:
        return {"ok": False, "reason": "no_gmail_credentials",
                "detail": f"no token for user {uid} account {label!r}. "
                          f"Run scripts/add_gmail_account.py {label}"}
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        return {"ok": False, "reason": "google_libs_not_installed", "detail": str(exc)}
    try:
        # scopes=None preserves the file's persisted scope set across refreshes
        # — passing a subset here would cause to_json() to write back the
        # narrower scope set, breaking other servers that share the same token.
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
        return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as exc:
        return {"ok": False, "reason": "client_build_failed", "detail": str(exc)[:300]}


def _headers_dict(message: dict) -> dict[str, str]:
    return {h["name"]: h["value"] for h in (message.get("payload") or {}).get("headers", []) or []}


def _walk_parts(payload: dict, want_mime: str) -> str | None:
    """Recursively find the first part of want_mime and return decoded body text."""
    if not payload:
        return None
    if payload.get("mimeType") == want_mime and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    for sub in payload.get("parts") or []:
        out = _walk_parts(sub, want_mime)
        if out is not None:
            return out
    return None


def _message_body(message: dict) -> dict:
    payload = message.get("payload") or {}
    text = _walk_parts(payload, "text/plain")
    html = _walk_parts(payload, "text/html") if text is None else None
    return {"text": text, "html": html}


def _serialize_summary(message: dict) -> dict:
    """Compact metadata summary for list-style returns."""
    headers = _headers_dict(message)
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "label_ids": message.get("labelIds", []),
        "snippet": message.get("snippet"),
        "subject": headers.get("Subject"),
        "from": headers.get("From"),
        "to": headers.get("To"),
        "date": headers.get("Date"),
    }


def _serialize_full(message: dict) -> dict:
    headers = _headers_dict(message)
    body = _message_body(message)
    attachments = []
    payload = message.get("payload") or {}
    def _walk(part):
        if not part:
            return
        filename = part.get("filename")
        if filename:
            attachments.append({
                "filename": filename,
                "mime_type": part.get("mimeType"),
                "attachment_id": (part.get("body") or {}).get("attachmentId"),
                "size_bytes": (part.get("body") or {}).get("size"),
            })
        for sub in part.get("parts") or []:
            _walk(sub)
    _walk(payload)
    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "label_ids": message.get("labelIds", []),
        "snippet": message.get("snippet"),
        "headers": headers,
        "subject": headers.get("Subject"),
        "from": headers.get("From"),
        "to": headers.get("To"),
        "cc": headers.get("Cc"),
        "date": headers.get("Date"),
        "body_text": body["text"],
        "body_html": body["html"],
        "attachments": attachments,
    }


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_email(s: str) -> bool:
    return isinstance(s, str) and bool(_EMAIL_RE.match(s.strip()))


def _b64url_message(mime_msg) -> dict:
    return {"raw": base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("ascii")}


# ---------- read tools ----------

@mcp.tool()
def list_unread(folder: str = "INBOX", limit: int = 20, user_id: str | None = None, account: str = "primary") -> dict:
    """Recent unread messages. folder is a Gmail label name (INBOX, IMPORTANT, etc.)."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": repr(limit)}
    limit = max(1, min(limit, 200))
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    label = (folder or "INBOX").strip().upper()
    try:
        resp = svc.users().messages().list(
            userId="me", q="is:unread", labelIds=[label], maxResults=limit,
        ).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    items = resp.get("messages", []) or []
    summaries = []
    for it in items:
        msg = svc.users().messages().get(
            userId="me", id=it["id"], format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        summaries.append(_serialize_summary(msg))
    return {"ok": True, "folder": label, "count": len(summaries), "messages": summaries}


@mcp.tool()
def search_email(query: str, folder: str | None = None, date_range: list[str] | None = None,
                 limit: int = 25, user_id: str | None = None, account: str = "primary") -> dict:
    """Gmail-style query (from:, subject:, has:attachment). Optional folder + ISO date_range."""
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "reason": "empty_query", "detail": ""}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": ""}
    limit = max(1, min(limit, 200))
    parts = [query.strip()]
    if date_range:
        if not isinstance(date_range, list) or len(date_range) != 2:
            return {"ok": False, "reason": "invalid_date_range", "detail": "must be [start, end] ISO dates"}
        try:
            start_date = datetime.fromisoformat(date_range[0]).date() if date_range[0] else None
            end_date = datetime.fromisoformat(date_range[1]).date() if date_range[1] else None
        except ValueError as exc:
            return {"ok": False, "reason": "invalid_date_range", "detail": str(exc)}
        if start_date:
            parts.append(f"after:{start_date.strftime('%Y/%m/%d')}")
        if end_date:
            parts.append(f"before:{end_date.strftime('%Y/%m/%d')}")
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    kwargs = {"userId": "me", "q": " ".join(parts), "maxResults": limit}
    if folder:
        kwargs["labelIds"] = [folder.strip().upper()]
    try:
        resp = svc.users().messages().list(**kwargs).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    summaries = []
    for it in resp.get("messages", []) or []:
        msg = svc.users().messages().get(
            userId="me", id=it["id"], format="metadata",
            metadataHeaders=["Subject", "From", "To", "Date"],
        ).execute()
        summaries.append(_serialize_summary(msg))
    return {"ok": True, "query": " ".join(parts), "count": len(summaries), "messages": summaries}


@mcp.tool()
def read_email(message_id: str, user_id: str | None = None, account: str = "primary") -> dict:
    """Full message content + headers + attachment list (no bytes)."""
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "message": _serialize_full(msg)}


@mcp.tool()
def summarize_inbox(hours: int = 24, max_messages: int = 30, user_id: str | None = None, account: str = "primary") -> dict:
    """Recent message summaries (subject/sender/snippet) from the last N hours. The CALLER (LLM) summarizes."""
    try:
        hours = int(hours)
        max_messages = int(max_messages)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_argument", "detail": ""}
    if hours < 1 or hours > 24 * 30:
        return {"ok": False, "reason": "hours_out_of_range", "detail": "1..720"}
    max_messages = max(1, min(max_messages, 200))
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    q = f"newer_than:{max(1, hours)}h"
    try:
        resp = svc.users().messages().list(userId="me", q=q, maxResults=max_messages).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    summaries = []
    for it in resp.get("messages", []) or []:
        msg = svc.users().messages().get(
            userId="me", id=it["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        summaries.append(_serialize_summary(msg))
    return {"ok": True, "hours": hours, "count": len(summaries), "messages": summaries}


@mcp.tool()
def find_emails_about(topic: str, limit: int = 25, user_id: str | None = None, account: str = "primary") -> dict:
    """Substring search across recent mail. Falls back to Gmail full-text search."""
    if not isinstance(topic, str) or not topic.strip():
        return {"ok": False, "reason": "empty_topic", "detail": ""}
    return search_email(query=topic.strip(), limit=limit, user_id=user_id, account=account)


# ---------- draft + send ----------

@mcp.tool()
def draft_email(to: list[str], subject: str, body: str,
                in_reply_to: str | None = None, cc: list[str] | None = None,
                bcc: list[str] | None = None, user_id: str | None = None, account: str = "primary") -> dict:
    """Create a Gmail draft. Returns draft_id."""
    if not isinstance(to, list) or not to:
        return {"ok": False, "reason": "empty_to", "detail": "to must be a non-empty list"}
    for addr in to + list(cc or []) + list(bcc or []):
        if not _is_email(addr):
            return {"ok": False, "reason": "invalid_email_address", "detail": addr}
    if not isinstance(subject, str):
        return {"ok": False, "reason": "non_string_subject", "detail": ""}
    if not isinstance(body, str):
        return {"ok": False, "reason": "non_string_body", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    mime = MIMEText(body, "plain", "utf-8")
    mime["To"] = ", ".join(to)
    if cc:
        mime["Cc"] = ", ".join(cc)
    if bcc:
        mime["Bcc"] = ", ".join(bcc)
    mime["Subject"] = subject
    if in_reply_to:
        mime["In-Reply-To"] = in_reply_to
        mime["References"] = in_reply_to
    try:
        draft = svc.users().drafts().create(userId="me", body={"message": _b64url_message(mime)}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "draft_id": draft.get("id"), "message_id": (draft.get("message") or {}).get("id")}


@mcp.tool()
def list_drafts(limit: int = 25, user_id: str | None = None, account: str = "primary") -> dict:
    """Pending drafts (id + subject + to)."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_limit", "detail": ""}
    limit = max(1, min(limit, 200))
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        resp = svc.users().drafts().list(userId="me", maxResults=limit).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    drafts = []
    for d in resp.get("drafts", []) or []:
        msg = svc.users().messages().get(
            userId="me", id=d["message"]["id"], format="metadata",
            metadataHeaders=["Subject", "To"],
        ).execute()
        headers = _headers_dict(msg)
        drafts.append({"draft_id": d["id"], "message_id": d["message"]["id"],
                       "subject": headers.get("Subject"), "to": headers.get("To")})
    return {"ok": True, "count": len(drafts), "drafts": drafts}


@mcp.tool(meta={"approval_required": True, "approval_template": "Send email?"})
def send_email(draft_id: str, user_id: str | None = None, account: str = "primary") -> dict:
    """SEND a saved draft. Approval-equivalent — Claude CLI gates."""
    if not isinstance(draft_id, str) or not draft_id.strip():
        return {"ok": False, "reason": "empty_draft_id", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        sent = svc.users().drafts().send(userId="me", body={"id": draft_id}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "draft_id": draft_id, "message_id": sent.get("id"), "thread_id": sent.get("threadId")}


@mcp.tool(meta={"approval_required": True, "approval_template": "Forward email?"})
def forward_email(message_id: str, to: list[str], note: str | None = None,
                  user_id: str | None = None, account: str = "primary") -> dict:
    """Create a forwarded draft (does NOT auto-send). Use send_email to dispatch."""
    if not isinstance(to, list) or not to or not all(_is_email(a) for a in to):
        return {"ok": False, "reason": "invalid_to", "detail": "to must be list of emails"}
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        orig = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}
    headers = _headers_dict(orig)
    orig_body = _message_body(orig)
    orig_text = orig_body["text"] or ""
    fwd_subject = headers.get("Subject", "")
    if not fwd_subject.lower().startswith("fwd:"):
        fwd_subject = f"Fwd: {fwd_subject}"
    quoted = (
        f"---------- Forwarded message ---------\n"
        f"From: {headers.get('From', '')}\n"
        f"Date: {headers.get('Date', '')}\n"
        f"Subject: {headers.get('Subject', '')}\n"
        f"To: {headers.get('To', '')}\n\n"
        f"{orig_text}"
    )
    body = (note + "\n\n" + quoted) if note else quoted
    mime = MIMEText(body, "plain", "utf-8")
    mime["To"] = ", ".join(to)
    mime["Subject"] = fwd_subject
    try:
        draft = svc.users().drafts().create(userId="me",
            body={"message": _b64url_message(mime)}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "action": "forward_draft_created", "draft_id": draft.get("id"),
            "original_message_id": message_id, "to": to}


@mcp.tool(meta={"approval_required": True, "approval_template": "Reply to email?"})
def reply_to_email(message_id: str, body: str, reply_all: bool = False,
                   user_id: str | None = None, account: str = "primary") -> dict:
    """Create a reply draft (does NOT auto-send). Use send_email to dispatch."""
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    if not isinstance(body, str):
        return {"ok": False, "reason": "non_string_body", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        orig = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}
    headers = _headers_dict(orig)
    from_addr = headers.get("From", "")
    to_addrs = [from_addr] if from_addr else []
    cc_addrs: list[str] = []
    if reply_all and headers.get("Cc"):
        cc_addrs = [a.strip() for a in headers["Cc"].split(",") if a.strip()]
    subject = headers.get("Subject", "")
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    msg_id_header = headers.get("Message-ID") or headers.get("Message-Id")
    mime = MIMEText(body, "plain", "utf-8")
    mime["To"] = ", ".join(to_addrs)
    if cc_addrs:
        mime["Cc"] = ", ".join(cc_addrs)
    mime["Subject"] = subject
    if msg_id_header:
        mime["In-Reply-To"] = msg_id_header
        mime["References"] = msg_id_header
    try:
        draft = svc.users().drafts().create(userId="me",
            body={"message": _b64url_message(mime), "threadId": orig.get("threadId")}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "action": "reply_draft_created", "draft_id": draft.get("id"),
            "thread_id": orig.get("threadId"), "to": to_addrs, "cc": cc_addrs}


# ---------- modify ----------

def _batch_modify(svc, ids: list[str], *, add: list[str] = None, remove: list[str] = None) -> dict:
    if not ids:
        return {"ok": False, "reason": "empty_ids", "detail": ""}
    body: dict[str, Any] = {"ids": ids}
    if add:
        body["addLabelIds"] = add
    if remove:
        body["removeLabelIds"] = remove
    try:
        svc.users().messages().batchModify(userId="me", body=body).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "modified_count": len(ids)}


@mcp.tool()
def mark_read(message_ids: list[str], user_id: str | None = None, account: str = "primary") -> dict:
    """Remove the UNREAD label from one or more messages."""
    if not isinstance(message_ids, list) or not message_ids:
        return {"ok": False, "reason": "empty_message_ids", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    return _batch_modify(svc, message_ids, remove=["UNREAD"])


@mcp.tool(meta={"approval_required": True, "approval_template": "Archive email(s)?"})
def archive(message_ids: list[str], user_id: str | None = None, account: str = "primary") -> dict:
    """Archive: remove from INBOX (keeps the message)."""
    if not isinstance(message_ids, list) or not message_ids:
        return {"ok": False, "reason": "empty_message_ids", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    return _batch_modify(svc, message_ids, remove=["INBOX"])


@mcp.tool(meta={"approval_required": True, "approval_template": "Delete email(s)?"})
def delete(message_ids: list[str], user_id: str | None = None, account: str = "primary") -> dict:
    """Move messages to Trash (recoverable 30 days). NOT permanent delete."""
    if not isinstance(message_ids, list) or not message_ids:
        return {"ok": False, "reason": "empty_message_ids", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    trashed: list[str] = []
    for mid in message_ids:
        try:
            svc.users().messages().trash(userId="me", id=mid).execute()
            trashed.append(mid)
        except Exception as exc:
            return {"ok": False, "reason": "api_error", "detail": str(exc)[:300],
                    "trashed_before_failure": trashed, "failed_at": mid}
    return {"ok": True, "trashed_count": len(trashed), "trashed_ids": trashed}


def _resolve_label_id(svc, label_name: str) -> str | None:
    if label_name.upper() in {"INBOX", "SENT", "DRAFT", "SPAM", "TRASH",
                              "STARRED", "IMPORTANT", "UNREAD"}:
        return label_name.upper()
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    for lbl in labels:
        if lbl["name"].lower() == label_name.lower():
            return lbl["id"]
    return None


@mcp.tool(meta={"approval_required": True, "approval_template": "Add label to email?"})
def add_label(message_id: str, label: str, user_id: str | None = None, account: str = "primary") -> dict:
    """Apply a Gmail label to a message. Creates the label if it doesn't exist."""
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    if not isinstance(label, str) or not label.strip():
        return {"ok": False, "reason": "empty_label", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    label_id = _resolve_label_id(svc, label)
    if label_id is None:
        try:
            new_label = svc.users().labels().create(userId="me", body={"name": label}).execute()
            label_id = new_label["id"]
        except Exception as exc:
            return {"ok": False, "reason": "label_create_failed", "detail": str(exc)[:300]}
    try:
        svc.users().messages().modify(userId="me", id=message_id,
            body={"addLabelIds": [label_id]}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "message_id": message_id, "label": label, "label_id": label_id}


@mcp.tool(meta={"approval_required": True, "approval_template": "Move email to folder?"})
def move_to_folder(message_id: str, folder: str, user_id: str | None = None, account: str = "primary") -> dict:
    """Move = apply label + remove INBOX. Gmail has no real folders; labels stand in."""
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    if not isinstance(folder, str) or not folder.strip():
        return {"ok": False, "reason": "empty_folder", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    label_id = _resolve_label_id(svc, folder)
    if label_id is None:
        try:
            new_label = svc.users().labels().create(userId="me", body={"name": folder}).execute()
            label_id = new_label["id"]
        except Exception as exc:
            return {"ok": False, "reason": "label_create_failed", "detail": str(exc)[:300]}
    try:
        svc.users().messages().modify(userId="me", id=message_id,
            body={"addLabelIds": [label_id], "removeLabelIds": ["INBOX"]}).execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    return {"ok": True, "message_id": message_id, "folder": folder, "label_id": label_id}


# ---------- attachments ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Extract email attachments to disk?"})
def extract_attachments(message_id: str, save_to_path: str, user_id: str | None = None, account: str = "primary") -> dict:
    """Save all attachments of a message into save_to_path (must be an existing dir inside ALLOWED_ROOTS)."""
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    try:
        out_dir = _validate_path(save_to_path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "save_path_denied", "detail": str(exc)}
    if not out_dir.is_dir():
        return {"ok": False, "reason": "save_path_not_a_directory", "detail": str(out_dir)}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:
        return {"ok": False, "reason": "fetch_failed", "detail": str(exc)[:300]}

    saved: list[dict] = []
    def _walk(part):
        if not part:
            return
        filename = (part.get("filename") or "").strip()
        body = part.get("body") or {}
        att_id = body.get("attachmentId")
        if filename and att_id:
            try:
                att = svc.users().messages().attachments().get(
                    userId="me", messageId=message_id, id=att_id,
                ).execute()
                data = base64.urlsafe_b64decode(att["data"])
            except Exception as exc:
                saved.append({"filename": filename, "ok": False, "detail": str(exc)[:200]})
                return
            if len(data) > _MAX_ATTACHMENT_BYTES:
                saved.append({"filename": filename, "ok": False, "detail": f"too_large {len(data)}"})
                return
            # Sanitize filename: no slashes, leading dot OK only for known data names.
            safe_name = filename.replace("/", "_").replace("\\", "_").lstrip(".")
            if not safe_name:
                safe_name = f"attachment_{att_id[:12]}"
            target = out_dir / safe_name
            try:
                # Reuse path validator on the COMPOSED final path to enforce denylist
                _ = _validate_write_target(str(target))
            except FileSystemSecurityError as exc:
                saved.append({"filename": filename, "ok": False, "detail": f"denied: {exc}"})
                return
            target.write_bytes(data)
            saved.append({"filename": safe_name, "ok": True,
                          "path": str(target), "size_bytes": len(data)})
        for sub in part.get("parts") or []:
            _walk(sub)
    _walk(msg.get("payload"))
    return {"ok": True, "message_id": message_id, "save_dir": str(out_dir),
            "saved_count": sum(1 for s in saved if s.get("ok")), "results": saved}


# ---------- action items (raw extract — LLM does the reasoning) ----------

@mcp.tool()
def parse_email_for_action_items(message_id: str, user_id: str | None = None, account: str = "primary") -> dict:
    """Return the email body + key headers so the calling LLM can extract action items.

    The MCP server intentionally does NOT do its own LLM call — it provides
    the raw content and lets the calling Claude do the reasoning.
    """
    if not isinstance(message_id, str) or not message_id.strip():
        return {"ok": False, "reason": "empty_message_id", "detail": ""}
    svc = _get_gmail(user_id, account)
    if isinstance(svc, dict):
        return svc
    try:
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:
        return {"ok": False, "reason": "api_error", "detail": str(exc)[:300]}
    full = _serialize_full(msg)
    return {
        "ok": True,
        "message_id": message_id,
        "subject": full["subject"],
        "from": full["from"],
        "to": full["to"],
        "date": full["date"],
        "body_text": full["body_text"],
        "body_html_truncated": (full["body_html"] or "")[:5000],
        "attachment_count": len(full["attachments"]),
        "note": "Caller LLM should extract action items from body_text.",
    }


# ---------- multi-account introspection ----------

@mcp.tool()
def list_gmail_accounts(user_id: str | None = None) -> dict:
    """H2-043: enumerate every Gmail account NEXUS has an OAuth token for, for
    this user. Returns labels sorted with ``primary`` first. Each entry also
    carries the on-disk token path so the caller can verify what's loaded.

    The pre-multi-account ``.data/gmail_tokens/<user_id>.json`` layout is
    surfaced as account ``primary`` and migrated forward on first read."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id",
                "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    labels = list_accounts_for_user(uid)
    accounts = [
        {"label": label,
         "token_path": str(resolve_token_path(uid, label) or '<unknown>')}
        for label in labels
    ]
    return {"ok": True, "user_id": uid, "count": len(accounts),
            "accounts": accounts}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
