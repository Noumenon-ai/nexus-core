"""Unit tests for services.destructive_intent_classifier.

These cover the pure classifier in isolation. The end-to-end approval-gate
flow (dispatcher prelude + callback re-fire) lives in test_gate_flow.py.
"""
from __future__ import annotations

import pytest

from services.destructive_intent_classifier import (
    DESTRUCTIVE_TOOL_REGISTRY,
    classify,
    render_default_preview,
)


# ---------- read-only prompts ----------

@pytest.mark.parametrize("prompt", [
    "show me my rental units",
    "list my LLCs",
    "what's my portfolio summary",
    "how many open maintenance requests do I have",
    "find emails about taxes",
    "summarize my inbox",
    "tell me about my upcoming renewals",
    "check the status",
    "where is the lease template stored",
])
def test_read_only_prompts_not_destructive(prompt):
    intent = classify(prompt)
    assert intent.is_destructive is False, f"false positive: {prompt!r}"


# ---------- specific-tool destructive prompts ----------

@pytest.mark.parametrize("prompt,expected_tool", [
    ("delete rental unit 1",            "delete_unit"),
    ("remove unit 3 from my rentals",   "delete_unit"),
    ("record payment of 1200 for unit 1", "add_payment"),
    ("log maintenance request for the leaky faucet", "add_maintenance"),
    ("delete the LLC named Acme",       "delete_llc"),
    ("delete business item 5",          "delete_business_item"),
    ("delete the gmail quick link",     "delete_quick_link"),
    ("send email to alex@example.com",  "send_email"),
    ("forward the latest invoice",      "forward_email"),
    ("reply to the bank email",         "reply_to_email"),
    ("delete file /tmp/junk.txt",       "delete_file"),
    ("move file foo.pdf to archive/",   "move_file"),
    ("write a haiku to /tmp/haiku.txt", "write_text_file"),
    ("append a note to my logs file",   "append_to_file"),
    ("cancel my 3pm meeting",           "cancel_event"),
    ("book a haircut for Tuesday",      "book_appointment"),
    ("send a whatsapp message to Maria","send_message"),
    ("delete contact John Doe",         "delete_contact"),
    ("forget what I told you about diet","forget"),
    ("delete the note about taxes",     "delete_note"),
    ("update the note about Q3 goals",  "update_note"),
    ("merge those two PDFs",            "merge_pdfs"),
    ("split the lease PDF",             "split_pdf"),
    ("sign the contract PDF",           "add_signature_to_pdf"),
    ("fill out the application form",   "fill_pdf_form"),
])
def test_destructive_prompts_match_tool(prompt, expected_tool):
    intent = classify(prompt)
    assert intent.is_destructive is True, f"false negative: {prompt!r}"
    assert expected_tool in intent.matched_tools, (
        f"prompt {prompt!r} matched {intent.matched_tools!r}, expected {expected_tool}"
    )
    assert intent.confidence >= 0.6
    assert intent.suggested_approval_template != ""


# ---------- ambiguous-as-destructive (conservative bias) ----------

@pytest.mark.parametrize("prompt", [
    "update something",        # generic update verb → destructive
    "send this",               # generic send verb
    "edit the thing",          # generic edit
])
def test_generic_destructive_verb_flags(prompt):
    intent = classify(prompt)
    assert intent.is_destructive is True


def test_read_dominated_prompt_downgrades():
    # Prompt mentions "delete" but heavily talks about *showing* things.
    intent = classify("show me what to delete and what to keep — list everything")
    # Two read verbs (show, list) vs one destructive (delete) ⇒ downgrade.
    assert intent.is_destructive is False


# ---------- edge cases ----------

def test_empty_prompt_not_destructive():
    assert classify("").is_destructive is False
    assert classify("   ").is_destructive is False


# ---------- reminders are ungated (owner directive 2026-06-02) ----------

@pytest.mark.parametrize("prompt", [
    "cancel the 8am reminder",
    "cancel my 3pm reminder",
    "delete reminder about laundry",
    "delete all my reminders",
    "update reminder 3 body to call mom",
    "change my reminder to 9am",
    "remove the reminder for the dentist",
])
def test_reminder_operations_are_not_destructive(prompt):
    """No approval gate on reminders — editing/cancelling the user's own
    reminder is low-risk and must act immediately, not arm an Approve/Cancel
    prompt."""
    intent = classify(prompt)
    assert intent.is_destructive is False, (
        f"reminder op {prompt!r} should be ungated, got {intent.matched_verbs!r}"
    )


def test_reminder_plus_other_destructive_target_still_gates():
    """The reminder fast path must NOT launder a compound prompt that also
    targets a real destructive tool. 'delete the reminder and the file'
    still matches delete_file and gates."""
    intent = classify("delete the reminder and the file /tmp/x.txt")
    assert intent.is_destructive is True
    assert "delete_file" in intent.matched_tools


def test_approval_callback_strings_bypass_classifier():
    intent = classify("approval:approve:abc123")
    assert intent.is_destructive is False
    intent = classify("approval:cancel:xyz")
    assert intent.is_destructive is False


def test_case_insensitive():
    intent = classify("DELETE UNIT 1")
    assert intent.is_destructive is True


# ---------- preview rendering ----------

def test_render_preview_truncates_long_prompt():
    long_prompt = "delete " + ("very long blah " * 50)
    intent = classify(long_prompt)
    preview = render_default_preview(long_prompt, intent)
    assert preview.endswith("…”") or len(preview) <= 400
    assert "delete" in preview.lower()


def test_render_preview_includes_template_and_prompt():
    intent = classify("delete rental unit 1")
    preview = render_default_preview("delete rental unit 1", intent)
    assert "delete rental unit" in preview.lower()
    assert "you asked" in preview.lower()


# ---------- registry sanity ----------

def test_registry_keys_are_unique_strings():
    keys = list(DESTRUCTIVE_TOOL_REGISTRY.keys())
    assert len(keys) == len(set(keys))
    for k in keys:
        assert isinstance(k, str) and k


def test_registry_templates_all_nonempty():
    for tool, (actions, targets, template) in DESTRUCTIVE_TOOL_REGISTRY.items():
        assert actions, f"{tool} has no action tokens"
        # targets can be empty (e.g. forget) — only validated by presence in
        # the tuple shape, not non-emptiness
        assert isinstance(targets, tuple)
        assert template, f"{tool} has empty template"
