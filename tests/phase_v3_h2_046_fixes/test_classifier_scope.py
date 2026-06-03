"""H2-046 — embedded-document-block stripping before classification.

Three production bugs share a root cause: classifiers matched against the
entire prompt, including the multi-kilobyte PDF body folded in by
telegram_bot._handle_document. Lease PDFs contain "personal" / "financial"
(trip sensitive route) and "signature" / "contract" (trip destructive
add_signature_to_pdf rule). After H2-046 the classifier sees a placeholder
in place of the embedded block.
"""
from __future__ import annotations

import pytest

from services.prompt_content_filters import strip_embedded_content
from services.intent_classifier import Provider, Path, classify as classify_intent
from services.destructive_intent_classifier import classify as classify_destructive


# ---------------------------------------------------------------------------
# strip_embedded_content — primitive
# ---------------------------------------------------------------------------


def test_strip_returns_unchanged_when_no_markers():
    out = strip_embedded_content('Just a plain sentence about leases and signatures.')
    assert out == 'Just a plain sentence about leases and signatures.'


def test_strip_removes_pdf_block():
    """The exact delimiter pair used by telegram_bot._handle_document."""
    raw = (
        "The user uploaded a PDF.\n\n"
        "--- BEGIN PDF TEXT ---\n"
        "This Lease Agreement contains a signature line for tenant.\n"
        "Personal financial information must remain confidential.\n"
        "--- END PDF TEXT ---\n\n"
        "Give a concise summary."
    )
    out = strip_embedded_content(raw)
    assert 'signature' not in out.lower()
    assert 'personal' not in out.lower()
    assert 'financial' not in out.lower()
    assert 'confidential' not in out.lower()
    assert 'embedded content removed' in out
    assert 'Give a concise summary.' in out
    assert 'The user uploaded a PDF.' in out


def test_strip_handles_audio_transcript_markers():
    raw = "Summarize this:\n--- BEGIN AUDIO TRANSCRIPT ---\nhello world\n--- END AUDIO TRANSCRIPT ---"
    out = strip_embedded_content(raw)
    assert 'hello world' not in out
    assert 'Summarize this:' in out


def test_strip_handles_multiple_blocks_in_one_prompt():
    raw = (
        "Two attachments.\n"
        "--- BEGIN PDF TEXT ---\nFirst PDF body\n--- END PDF TEXT ---\n"
        "And another:\n"
        "--- BEGIN ATTACHMENT ---\nattachment body\n--- END ATTACHMENT ---"
    )
    out = strip_embedded_content(raw)
    assert 'First PDF body' not in out
    assert 'attachment body' not in out


def test_strip_passes_through_empty_input():
    assert strip_embedded_content('') == ''
    assert strip_embedded_content(None) is None


def test_strip_is_case_insensitive_on_markers():
    raw = "--- begin pdf text ---\nsecret stuff\n--- end pdf text ---"
    out = strip_embedded_content(raw)
    assert 'secret stuff' not in out


# ---------------------------------------------------------------------------
# intent_classifier — sensitive route no longer fires on PDF body
# ---------------------------------------------------------------------------


def test_intent_classify_lease_pdf_does_not_route_to_ollama():
    """The bug-1 repro: lease PDF body contains sensitive words; classifier
    must look at the user's instruction text (the surrounding "give a
    summary" prompt), not the embedded PDF body."""
    prompt = (
        "The user uploaded a PDF named 'lease.pdf'. "
        "User did not include a caption — give a concise summary.\n\n"
        "--- BEGIN PDF TEXT ---\n"
        "PERSONAL AND CONFIDENTIAL — Tenant's financial obligations\n"
        "and bank account details are described herein.\n"
        "--- END PDF TEXT ---"
    )
    decision = classify_intent(prompt)
    assert decision.provider is not Provider.OLLAMA, (
        f"PDF body keywords leaked into classification: {decision.matched_keywords}"
    )


def test_intent_classify_user_actually_says_personal_still_routes_sensitive():
    """Don't regress real sensitive-content routing: when the user
    explicitly says it in their own message, route to OLLAMA as before."""
    decision = classify_intent("Tell me about my personal medical history.")
    assert decision.provider is Provider.OLLAMA
    assert 'personal' in decision.matched_keywords or 'medical' in decision.matched_keywords


# ---------------------------------------------------------------------------
# destructive_intent_classifier — PDF body no longer trips signature rule
# ---------------------------------------------------------------------------


def test_destructive_classify_lease_pdf_summary_not_flagged():
    """Bug-2 repro: PDF body contains 'signature' AND 'contract' — pre-H2-046
    that fired add_signature_to_pdf and gate-armed an approval prompt. After
    the fix the embedded body is stripped first; the user's instruction
    text alone shouldn't flag."""
    prompt = (
        "The user uploaded a PDF named 'lease.pdf'. "
        "User did not include a caption — give a concise summary of the document.\n\n"
        "--- BEGIN PDF TEXT ---\n"
        "RESIDENTIAL LEASE AGREEMENT\n"
        "By signing below, the tenant agrees to the contract terms.\n"
        "Signature: ________  Date: ________\n"
        "--- END PDF TEXT ---"
    )
    intent = classify_destructive(prompt)
    assert intent.is_destructive is False, (
        f"PDF body tripped destructive gate: tools={intent.matched_tools} "
        f"template={intent.suggested_approval_template!r}"
    )


def test_destructive_classify_explicit_sign_request_still_fires():
    """The H2-039 contract — explicit "sign this contract" must still
    surface an approval prompt — must NOT regress."""
    intent = classify_destructive("Sign the contract attached")
    assert intent.is_destructive is True
    # We don't pin the exact template — just confirm the gate engaged.


# ---------------------------------------------------------------------------
# destructive_intent_classifier — numbered multi-intent detection
# ---------------------------------------------------------------------------


def test_destructive_classify_numbered_multi_intent_refuses_to_bundle():
    """The other bug-2 repro: a numbered multi-action message used to
    collapse to a single 'This may modify or send data' approval — now
    returns the multi-intent template asking the user to split it up."""
    intent = classify_destructive(
        "1. I'll send the 3 leases 2. Do it 3. I want to wire 6 accounts"
    )
    assert intent.is_destructive is True
    assert 'Multiple actions' in intent.suggested_approval_template
    assert intent.matched_verbs == ('multi_intent',)


def test_destructive_classify_numbered_two_items_min_threshold():
    """Threshold is ≥2 numbered items. A single numbered item shouldn't
    flip the multi-intent path."""
    intent = classify_destructive("1. Just send this one email")
    # Single numbered → falls through to normal classification. send_email
    # might still fire as a normal destructive match — that's fine, what
    # we're asserting is it's NOT mis-tagged as multi_intent.
    assert intent.matched_verbs != ('multi_intent',)


def test_destructive_classify_numbered_non_sequential_still_fires():
    """The check requires distinct numbers (not all the same), so
    "1. a 1. b" wouldn't trip it. "1. a 3. b" should."""
    intent = classify_destructive("1. delete a file 3. send an email")
    assert intent.is_destructive is True
    assert intent.matched_verbs == ('multi_intent',)


def test_destructive_classify_numbered_pattern_inside_pdf_does_not_fire():
    """A numbered list inside the PDF body shouldn't surface as the user's
    multi-intent. The PDF body is stripped before classification."""
    intent = classify_destructive(
        "Summarize this document.\n\n"
        "--- BEGIN PDF TEXT ---\n"
        "Lease terms:\n"
        "1. Rent is due monthly.\n"
        "2. Late fees apply after the 5th.\n"
        "3. Security deposit is refundable.\n"
        "--- END PDF TEXT ---"
    )
    assert intent.is_destructive is False, (
        f"PDF-internal numbered list tripped multi-intent: {intent}"
    )


def test_destructive_classify_currency_amounts_dont_false_positive():
    """Numbers like '$1,000' or 'I have 1 cat and 2 dogs' should NOT trip
    the multi-intent detector (it requires `digit.` or `digit)` followed
    by whitespace then a word)."""
    intent = classify_destructive("I have 1 cat and 2 dogs at home")
    assert intent.matched_verbs != ('multi_intent',)
