"""Adversarial tests for nexus-contacts MCP server (Phase 2.5b Server 5)."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_USER_ID = "contacts-user-7777"


@pytest.fixture(autouse=True)
def default_user(monkeypatch):
    monkeypatch.setenv("NEXUS_MCP_DEFAULT_USER_ID", _USER_ID)
    yield


from mcp_servers.nexus_contacts import (
    _parse_vcard,
    add_to_group,
    birthday_reminders,
    create_contact,
    delete_contact,
    find_contact,
    get_contact_details,
    import_vcard,
    list_contacts_in_group,
    list_groups,
    recent_interactions,
    remove_from_group,
    update_contact,
)


def _build_stub(
    search: dict | None = None,
    get: dict | None = None,
    create: dict | None = None,
    update: dict | None = None,
    groups_list: dict | None = None,
    group_create: dict | None = None,
    group_get: dict | None = None,
    batch_get: dict | None = None,
    connections_list: dict | None = None,
) -> MagicMock:
    svc = MagicMock()
    people = svc.people.return_value
    people.searchContacts.return_value.execute.return_value = search or {"results": []}
    people.get.return_value.execute.return_value = get or {"resourceName": "people/x"}
    people.createContact.return_value.execute.return_value = create or {"resourceName": "people/new"}
    people.updateContact.return_value.execute.return_value = update or {"resourceName": "people/x", "etag": "newetag"}
    people.deleteContact.return_value.execute.return_value = {}
    people.getBatchGet.return_value.execute.return_value = batch_get or {"responses": []}
    people.connections.return_value.list.return_value.execute.return_value = connections_list or {"connections": []}
    grps = svc.contactGroups.return_value
    grps.list.return_value.execute.return_value = groups_list or {"contactGroups": []}
    grps.create.return_value.execute.return_value = group_create or {"resourceName": "contactGroups/new"}
    grps.get.return_value.execute.return_value = group_get or {}
    grps.members.return_value.modify.return_value.execute.return_value = {}
    return svc


# ---------- credential paths ----------

def test_no_user_id_rejected(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_DEFAULT_USER_ID", raising=False)
    r = list_groups()
    assert r["ok"] is False
    assert r["reason"] == "no_user_id"


def test_missing_credentials_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUS_GOOGLE_TOKEN_DIR", str(tmp_path))
    r = list_groups()
    assert r["ok"] is False
    assert r["reason"] == "no_google_credentials"


# ---------- find_contact ----------

def test_find_contact_empty_query_rejected():
    r = find_contact(query="")
    assert r["ok"] is False


def test_find_contact_happy():
    stub = _build_stub(search={
        "results": [
            {"person": {
                "resourceName": "people/c1",
                "names": [{"displayName": "Sam T", "givenName": "Sam", "familyName": "S"}],
                "emailAddresses": [{"value": "sam@example.com", "type": "work"}],
                "phoneNumbers": [],
            }},
        ],
    })
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = find_contact(query="sam")
    assert r["ok"] is True
    assert r["count"] == 1
    assert r["contacts"][0]["display_name"] == "Sam T"
    assert r["contacts"][0]["emails"][0]["value"] == "sam@example.com"


# ---------- get_contact_details ----------

def test_get_contact_details_empty_id_rejected():
    r = get_contact_details(contact_id="")
    assert r["ok"] is False


def test_get_contact_details_happy():
    stub = _build_stub(get={
        "resourceName": "people/c1",
        "names": [{"displayName": "Bob"}],
        "biographies": [{"value": "Note text"}],
        "memberships": [{"contactGroupMembership": {"contactGroupId": "myContacts"}}],
    })
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = get_contact_details(contact_id="people/c1")
    assert r["ok"] is True
    assert r["contact"]["display_name"] == "Bob"
    assert r["contact"]["notes"] == "Note text"
    assert "myContacts" in r["contact"]["groups"]


# ---------- recent_interactions ----------

def test_recent_interactions_days_out_of_range_rejected():
    r = recent_interactions(contact_id="people/c1", days=0)
    assert r["ok"] is False
    assert r["reason"] == "days_out_of_range"


def test_recent_interactions_empty_id_rejected():
    r = recent_interactions(contact_id="")
    assert r["ok"] is False


# ---------- groups ----------

def test_list_groups_happy():
    stub = _build_stub(groups_list={
        "contactGroups": [
            {"resourceName": "contactGroups/family", "formattedName": "Family", "groupType": "USER_CONTACT_GROUP", "memberCount": 5},
            {"resourceName": "contactGroups/work", "name": "Work", "groupType": "USER_CONTACT_GROUP", "memberCount": 12},
        ],
    })
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = list_groups()
    assert r["ok"] is True
    assert r["count"] == 2
    names = {g["name"] for g in r["groups"]}
    assert "Family" in names and "Work" in names


def test_list_contacts_in_group_group_not_found_rejected():
    stub = _build_stub(groups_list={"contactGroups": []})
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = list_contacts_in_group(group_name="nonexistent")
    assert r["ok"] is False
    assert r["reason"] == "group_not_found"


def test_list_contacts_in_group_happy():
    stub = _build_stub(
        groups_list={"contactGroups": [
            {"resourceName": "contactGroups/family", "formattedName": "Family"},
        ]},
        group_get={
            "resourceName": "contactGroups/family",
            "memberResourceNames": ["people/a", "people/b"],
        },
        batch_get={"responses": [
            {"person": {"resourceName": "people/a", "names": [{"displayName": "A"}]}},
            {"person": {"resourceName": "people/b", "names": [{"displayName": "B"}]}},
        ]},
    )
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = list_contacts_in_group(group_name="Family")
    assert r["ok"] is True
    assert r["count"] == 2


def test_add_to_group_creates_missing_group():
    stub = _build_stub(
        groups_list={"contactGroups": []},
        group_create={"resourceName": "contactGroups/Custom_New"},
    )
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = add_to_group(contact_id="people/c1", group_name="ProjectXYZ")
    assert r["ok"] is True
    assert r["group_id"] == "contactGroups/Custom_New"


def test_remove_from_group_missing_rejected():
    stub = _build_stub(groups_list={"contactGroups": []})
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = remove_from_group(contact_id="people/c1", group_name="nope")
    assert r["ok"] is False
    assert r["reason"] == "group_not_found"


# ---------- birthday_reminders ----------

def test_birthday_reminders_out_of_range_rejected():
    r = birthday_reminders(days_ahead=400)
    assert r["ok"] is False


def test_birthday_reminders_filters_upcoming():
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone.utc).date()
    upcoming = today + timedelta(days=5)
    far = today + timedelta(days=100)
    stub = _build_stub(connections_list={
        "connections": [
            {"resourceName": "people/up", "names": [{"displayName": "Up"}],
             "birthdays": [{"date": {"month": upcoming.month, "day": upcoming.day}}]},
            {"resourceName": "people/far", "names": [{"displayName": "Far"}],
             "birthdays": [{"date": {"month": far.month, "day": far.day}}]},
            {"resourceName": "people/nobday", "names": [{"displayName": "NoB"}], "birthdays": []},
        ],
    })
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = birthday_reminders(days_ahead=30)
    assert r["ok"] is True
    names = {b["name"] for b in r["birthdays"]}
    assert "Up" in names
    assert "Far" not in names
    assert "NoB" not in names


# ---------- create / update / delete ----------

def test_create_contact_empty_name_rejected():
    r = create_contact(name="")
    assert r["ok"] is False


def test_create_contact_happy():
    stub = _build_stub(
        groups_list={"contactGroups": [{"resourceName": "contactGroups/family", "formattedName": "Family"}]},
        create={"resourceName": "people/created",
                "names": [{"displayName": "New Person"}],
                "emailAddresses": [{"value": "new@example.com"}]},
    )
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = create_contact(name="New Person", email="new@example.com",
                           phone="+15551234567", notes="hi", groups=["Family"])
    assert r["ok"] is True
    body = stub.people.return_value.createContact.call_args.kwargs["body"]
    assert body["names"][0]["unstructuredName"] == "New Person"
    assert body["emailAddresses"][0]["value"] == "new@example.com"
    assert body["memberships"][0]["contactGroupMembership"]["contactGroupId"] == "family"


def test_update_contact_unknown_fields_rejected():
    r = update_contact(contact_id="people/x", fields={"foo": "bar"})
    assert r["ok"] is False
    assert r["reason"] == "unknown_fields"


def test_update_contact_invalid_emails_type_rejected():
    stub = _build_stub(get={"resourceName": "people/x", "etag": "e"})
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = update_contact(contact_id="people/x", fields={"emails": "not_a_list"})
    assert r["ok"] is False
    assert r["reason"] == "invalid_emails"


def test_update_contact_happy():
    stub = _build_stub(
        get={"resourceName": "people/x", "etag": "e1", "names": [{"displayName": "Old"}]},
        update={"resourceName": "people/x", "etag": "e2", "names": [{"displayName": "New"}]},
    )
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = update_contact(contact_id="people/x", fields={"names": "New"})
    assert r["ok"] is True
    sent = stub.people.return_value.updateContact.call_args.kwargs
    assert sent["resourceName"] == "people/x"
    assert sent["updatePersonFields"] == "names"
    assert sent["body"]["etag"] == "e1"


def test_delete_contact_happy():
    stub = _build_stub()
    with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
        r = delete_contact(contact_id="people/doomed")
    assert r["ok"] is True
    assert r["contact_id"] == "people/doomed"


def test_delete_contact_empty_id_rejected():
    r = delete_contact(contact_id="")
    assert r["ok"] is False


# ---------- vCard ----------

VCARD_SAMPLE = """\
BEGIN:VCARD
VERSION:3.0
FN:Alice Example
N:Example;Alice;;;
EMAIL;TYPE=work:alice@example.com
TEL;TYPE=cell:+15551234567
ORG:Acme
BDAY:1990-05-12
NOTE:Important contact
END:VCARD

BEGIN:VCARD
VERSION:3.0
FN:Bob Builder
EMAIL:bob@builders.com
END:VCARD

BEGIN:VCARD
VERSION:3.0
END:VCARD
"""


def _allow_tmp_workspace(monkeypatch, tmp_path: Path) -> Path:
    workspace = tmp_path.resolve()
    monkeypatch.setattr("services.file_system_tools.ALLOWED_ROOTS", (workspace,))
    return workspace


def test_parse_vcard_extracts_fields():
    parsed = _parse_vcard(VCARD_SAMPLE)
    assert len(parsed) == 2  # third VCARD has no usable fields
    alice = parsed[0]
    assert alice["name"] == "Alice Example"
    assert alice["emails"] == ["alice@example.com"]
    assert alice["phones"] == ["+15551234567"]
    assert alice["org"] == "Acme"
    assert alice["bday"] == "1990-05-12"
    assert alice["notes"] == "Important contact"


def test_import_vcard_path_outside_root_rejected():
    r = import_vcard(vcard_path="/etc/passwd")
    assert r["ok"] is False
    assert r["reason"] == "vcard_path_denied"


def test_import_vcard_wrong_extension_rejected(tmp_path, monkeypatch):
    workspace = _allow_tmp_workspace(monkeypatch, tmp_path)
    bad = workspace / "_mcp_test_bad.txt"
    bad.write_text("BEGIN:VCARD\nFN:X\nEND:VCARD")
    try:
        r = import_vcard(vcard_path=str(bad))
        assert r["ok"] is False
        assert r["reason"] == "wrong_extension"
    finally:
        bad.unlink(missing_ok=True)


def test_import_vcard_happy(tmp_path, monkeypatch):
    workspace = _allow_tmp_workspace(monkeypatch, tmp_path)
    vcf = workspace / "_mcp_test_contacts.vcf"
    vcf.write_text(VCARD_SAMPLE, encoding="utf-8")
    try:
        stub = _build_stub(create={"resourceName": "people/created"})
        with patch("mcp_servers.nexus_contacts._get_people", return_value=stub):
            r = import_vcard(vcard_path=str(vcf))
        assert r["ok"] is True
        assert r["parsed_count"] == 2
        assert r["imported_count"] == 2
        assert stub.people.return_value.createContact.call_count == 2
    finally:
        vcf.unlink(missing_ok=True)
