"""Adversarial tests for nexus-email MCP server (Phase 2.5b Server 3).

Mocks `_get_gmail()` to return a stub service whose method calls record args
and return canned payloads — no real API traffic in tests.
"""
from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_USER_ID = "email-user-1234"


@pytest.fixture(autouse=True)
def default_user(monkeypatch):
    monkeypatch.setenv("NEXUS_MCP_DEFAULT_USER_ID", _USER_ID)
    yield


from mcp_servers.nexus_email import (
    add_label,
    archive,
    delete,
    draft_email,
    extract_attachments,
    find_emails_about,
    forward_email,
    list_drafts,
    list_unread,
    mark_read,
    move_to_folder,
    parse_email_for_action_items,
    read_email,
    reply_to_email,
    search_email,
    send_email,
    summarize_inbox,
)


def _build_stub(
    list_resp: dict | None = None,
    get_resp: dict | None = None,
    drafts_create: dict | None = None,
    drafts_list: dict | None = None,
    drafts_send: dict | None = None,
    labels_list: dict | None = None,
    labels_create: dict | None = None,
    modify_resp: dict | None = None,
    attach_get: dict | None = None,
    trash_resp: dict | None = None,
) -> MagicMock:
    svc = MagicMock()
    users = svc.users.return_value

    msgs = users.messages.return_value
    msgs.list.return_value.execute.return_value = list_resp or {"messages": []}
    msgs.get.return_value.execute.return_value = get_resp or {"id": "x"}
    msgs.modify.return_value.execute.return_value = {}
    msgs.batchModify.return_value.execute.return_value = {}
    msgs.trash.return_value.execute.return_value = trash_resp or {}
    msgs.attachments.return_value.get.return_value.execute.return_value = attach_get or {}

    drafts = users.drafts.return_value
    drafts.create.return_value.execute.return_value = drafts_create or {"id": "draft1", "message": {"id": "msg1"}}
    drafts.list.return_value.execute.return_value = drafts_list or {"drafts": []}
    drafts.send.return_value.execute.return_value = drafts_send or {"id": "msgsent", "threadId": "thr"}

    labels = users.labels.return_value
    labels.list.return_value.execute.return_value = labels_list or {"labels": []}
    labels.create.return_value.execute.return_value = labels_create or {"id": "Label_42", "name": "x"}
    return svc


def _allow_tmp_workspace(monkeypatch, tmp_path: Path) -> Path:
    workspace = tmp_path.resolve()
    monkeypatch.setattr("services.file_system_tools.ALLOWED_ROOTS", (workspace,))
    monkeypatch.setattr("mcp_servers.nexus_filesystem.ALLOWED_ROOTS", (workspace,))
    return workspace


# ---------- credential paths ----------

def test_no_user_id_returns_no_user_id(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    r = list_unread()
    assert r["ok"] is False
    assert r["reason"] == "no_user_id"


def test_missing_credentials_returns_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_GMAIL_TOKEN_DIR", str(tmp_path))
    r = list_unread()
    assert r["ok"] is False
    assert r["reason"] == "no_gmail_credentials"


# ---------- list_unread / search / read ----------

def test_list_unread_non_integer_limit_rejected():
    r = list_unread(limit="abc")
    assert r["ok"] is False
    assert r["reason"] == "non_integer_limit"


def test_list_unread_happy():
    stub = _build_stub(
        list_resp={"messages": [{"id": "m1"}, {"id": "m2"}]},
        get_resp={"id": "m1", "labelIds": ["UNREAD"],
                  "payload": {"headers": [{"name": "Subject", "value": "hi"},
                                          {"name": "From", "value": "a@b.com"}]}},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = list_unread(limit=10)
    assert r["ok"] is True
    assert r["folder"] == "INBOX"
    assert r["count"] == 2


def test_search_email_empty_query_rejected():
    r = search_email(query="")
    assert r["ok"] is False
    assert r["reason"] == "empty_query"


def test_search_email_invalid_date_range_rejected():
    r = search_email(query="foo", date_range=["not-a-date", "also-bad"])
    assert r["ok"] is False
    assert r["reason"] == "invalid_date_range"


def test_search_email_date_range_appended_to_query():
    stub = _build_stub(
        list_resp={"messages": [{"id": "m1"}]},
        get_resp={"id": "m1", "payload": {"headers": [{"name": "Subject", "value": "x"}]}},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = search_email(query="from:boss", date_range=["2026-05-01", "2026-05-12"])
    assert r["ok"] is True
    sent_q = stub.users.return_value.messages.return_value.list.call_args.kwargs["q"]
    assert "from:boss" in sent_q
    assert "after:2026/05/01" in sent_q
    assert "before:2026/05/12" in sent_q


def test_read_email_empty_id_rejected():
    r = read_email(message_id="")
    assert r["ok"] is False


def test_read_email_returns_full_payload():
    body_text = base64.urlsafe_b64encode(b"Hello world!").decode("ascii")
    stub = _build_stub(get_resp={
        "id": "m1", "threadId": "t1", "labelIds": ["INBOX"],
        "snippet": "Hello",
        "payload": {
            "headers": [{"name": "Subject", "value": "Hi"}, {"name": "From", "value": "a@b.com"}],
            "mimeType": "text/plain",
            "body": {"data": body_text},
        },
    })
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = read_email(message_id="m1")
    assert r["ok"] is True
    assert r["message"]["subject"] == "Hi"
    assert r["message"]["body_text"] == "Hello world!"


# ---------- summarize_inbox / find_emails_about ----------

def test_summarize_inbox_invalid_hours_rejected():
    r = summarize_inbox(hours=0)
    assert r["ok"] is False
    assert r["reason"] == "hours_out_of_range"


def test_summarize_inbox_uses_newer_than_query():
    stub = _build_stub(list_resp={"messages": []})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = summarize_inbox(hours=12)
    assert r["ok"] is True
    sent_q = stub.users.return_value.messages.return_value.list.call_args.kwargs["q"]
    assert "newer_than:12h" in sent_q


def test_find_emails_about_delegates_to_search():
    stub = _build_stub(list_resp={"messages": []})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = find_emails_about(topic="renewal")
    assert r["ok"] is True
    assert "renewal" in stub.users.return_value.messages.return_value.list.call_args.kwargs["q"]


def test_find_emails_about_empty_topic_rejected():
    r = find_emails_about(topic="   ")
    assert r["ok"] is False


# ---------- draft / send / forward / reply ----------

def test_draft_email_empty_to_rejected():
    r = draft_email(to=[], subject="s", body="b")
    assert r["ok"] is False
    assert r["reason"] == "empty_to"


def test_draft_email_invalid_address_rejected():
    r = draft_email(to=["not-an-email"], subject="s", body="b")
    assert r["ok"] is False
    assert r["reason"] == "invalid_email_address"


def test_draft_email_happy():
    stub = _build_stub(drafts_create={"id": "d1", "message": {"id": "m1"}})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = draft_email(to=["alice@example.com"], subject="Hi", body="hello",
                        cc=["bob@example.com"])
    assert r["ok"] is True
    assert r["draft_id"] == "d1"
    sent = stub.users.return_value.drafts.return_value.create.call_args.kwargs["body"]
    raw = base64.urlsafe_b64decode(sent["message"]["raw"] + "==").decode("utf-8", errors="replace")
    assert "To: alice@example.com" in raw
    assert "Cc: bob@example.com" in raw
    assert "Subject: Hi" in raw


def test_list_drafts_happy():
    stub = _build_stub(
        drafts_list={"drafts": [{"id": "d1", "message": {"id": "m1"}}]},
        get_resp={"id": "m1", "payload": {"headers": [
            {"name": "Subject", "value": "Pending"},
            {"name": "To", "value": "x@y.com"}]}},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = list_drafts()
    assert r["ok"] is True
    assert r["drafts"][0]["subject"] == "Pending"


def test_send_email_empty_draft_id_rejected():
    r = send_email(draft_id="")
    assert r["ok"] is False


def test_send_email_happy():
    stub = _build_stub(drafts_send={"id": "msgsent", "threadId": "thr"})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = send_email(draft_id="d1")
    assert r["ok"] is True
    assert r["message_id"] == "msgsent"
    body = stub.users.return_value.drafts.return_value.send.call_args.kwargs["body"]
    assert body == {"id": "d1"}


def test_forward_email_creates_draft_with_quoted_body():
    body_text = base64.urlsafe_b64encode(b"Original body content").decode("ascii")
    stub = _build_stub(
        get_resp={"id": "m1", "threadId": "t1",
                  "payload": {
                      "headers": [{"name": "Subject", "value": "Hi"},
                                  {"name": "From", "value": "x@y.com"},
                                  {"name": "Date", "value": "Mon"},
                                  {"name": "To", "value": "me@me.com"}],
                      "mimeType": "text/plain",
                      "body": {"data": body_text}}},
        drafts_create={"id": "d_fwd"},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = forward_email(message_id="m1", to=["new@dest.com"], note="See below")
    assert r["ok"] is True
    raw = base64.urlsafe_b64decode(
        stub.users.return_value.drafts.return_value.create.call_args.kwargs["body"]["message"]["raw"] + "==",
    ).decode("utf-8", errors="replace")
    assert "Subject: Fwd: Hi" in raw
    # MIMEText base64-encodes the body — decode the body section to inspect content.
    headers_end = raw.find("\n\n")
    body_b64 = raw[headers_end + 2:].replace("\n", "").rstrip()
    decoded_body = base64.b64decode(body_b64 + "==").decode("utf-8", errors="replace")
    assert "See below" in decoded_body
    assert "Original body content" in decoded_body


def test_forward_email_invalid_to_rejected():
    r = forward_email(message_id="m1", to=["not-an-email"])
    assert r["ok"] is False
    assert r["reason"] == "invalid_to"


def test_reply_to_email_sets_in_reply_to_header():
    stub = _build_stub(
        get_resp={"id": "m1", "threadId": "t1",
                  "payload": {"headers": [
                      {"name": "Subject", "value": "Re-check this"},
                      {"name": "From", "value": "x@y.com"},
                      {"name": "Message-ID", "value": "<orig@msg.id>"}]}},
        drafts_create={"id": "d_reply"},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = reply_to_email(message_id="m1", body="thanks")
    assert r["ok"] is True
    sent_body = stub.users.return_value.drafts.return_value.create.call_args.kwargs["body"]
    assert sent_body["threadId"] == "t1"
    raw = base64.urlsafe_b64decode(sent_body["message"]["raw"] + "==").decode("utf-8", errors="replace")
    assert "Subject: Re: Re-check this" in raw
    assert "In-Reply-To: <orig@msg.id>" in raw


def test_reply_to_email_reply_all_includes_cc():
    stub = _build_stub(
        get_resp={"id": "m1", "threadId": "t1",
                  "payload": {"headers": [
                      {"name": "Subject", "value": "Group"},
                      {"name": "From", "value": "x@y.com"},
                      {"name": "Cc", "value": "a@b.com, b@c.com"}]}},
        drafts_create={"id": "d_rall"},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = reply_to_email(message_id="m1", body="all", reply_all=True)
    assert r["ok"] is True
    assert set(r["cc"]) == {"a@b.com", "b@c.com"}


# ---------- modify ----------

def test_mark_read_empty_rejected():
    r = mark_read(message_ids=[])
    assert r["ok"] is False


def test_mark_read_calls_batchmodify_remove_unread():
    stub = _build_stub()
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = mark_read(message_ids=["m1", "m2"])
    assert r["ok"] is True
    sent = stub.users.return_value.messages.return_value.batchModify.call_args.kwargs["body"]
    assert sent["removeLabelIds"] == ["UNREAD"]
    assert sent["ids"] == ["m1", "m2"]


def test_archive_removes_inbox_label():
    stub = _build_stub()
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = archive(message_ids=["m1"])
    assert r["ok"] is True
    sent = stub.users.return_value.messages.return_value.batchModify.call_args.kwargs["body"]
    assert sent["removeLabelIds"] == ["INBOX"]


def test_delete_calls_trash_per_message():
    stub = _build_stub()
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = delete(message_ids=["m1", "m2"])
    assert r["ok"] is True
    assert r["trashed_count"] == 2


def test_add_label_resolves_existing_label():
    stub = _build_stub(labels_list={"labels": [{"id": "Label_99", "name": "Tenants"}]})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = add_label(message_id="m1", label="tenants")
    assert r["ok"] is True
    assert r["label_id"] == "Label_99"


def test_add_label_creates_when_missing():
    stub = _build_stub(
        labels_list={"labels": []},
        labels_create={"id": "Label_NEW", "name": "Tenants"},
    )
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = add_label(message_id="m1", label="Tenants")
    assert r["ok"] is True
    assert r["label_id"] == "Label_NEW"


def test_move_to_folder_adds_label_and_removes_inbox():
    stub = _build_stub(labels_list={"labels": [{"id": "Label_5", "name": "Archive2026"}]})
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = move_to_folder(message_id="m1", folder="Archive2026")
    assert r["ok"] is True
    body = stub.users.return_value.messages.return_value.modify.call_args.kwargs["body"]
    assert body["addLabelIds"] == ["Label_5"]
    assert body["removeLabelIds"] == ["INBOX"]


# ---------- attachments ----------

def test_extract_attachments_outside_root_rejected():
    r = extract_attachments(message_id="m1", save_to_path="/etc")
    assert r["ok"] is False
    assert r["reason"] == "save_path_denied"


def test_extract_attachments_happy(tmp_path, monkeypatch):
    out_dir = _allow_tmp_workspace(monkeypatch, tmp_path) / "_mcp_email_attach"
    out_dir.mkdir()
    try:
        att_bytes = b"hello attachment bytes"
        att_b64 = base64.urlsafe_b64encode(att_bytes).decode("ascii")
        stub = _build_stub(
            get_resp={
                "id": "m1",
                "payload": {
                    "headers": [],
                    "parts": [
                        {"filename": "note.txt", "body": {"attachmentId": "att1", "size": 22},
                         "mimeType": "text/plain"},
                    ],
                },
            },
            attach_get={"data": att_b64},
        )
        with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
            r = extract_attachments(message_id="m1", save_to_path=str(out_dir))
        assert r["ok"] is True
        assert r["saved_count"] == 1
        saved_path = Path(r["results"][0]["path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == att_bytes
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------- action items ----------

def test_parse_email_for_action_items_returns_raw_body():
    body_text = base64.urlsafe_b64encode(b"Please review the proposal by Friday.").decode("ascii")
    stub = _build_stub(get_resp={
        "id": "m1",
        "payload": {
            "headers": [{"name": "Subject", "value": "Proposal review"},
                        {"name": "From", "value": "boss@example.com"}],
            "mimeType": "text/plain",
            "body": {"data": body_text},
        },
    })
    with patch("mcp_servers.nexus_email._get_gmail", return_value=stub):
        r = parse_email_for_action_items(message_id="m1")
    assert r["ok"] is True
    assert r["subject"] == "Proposal review"
    assert "review the proposal" in r["body_text"]
