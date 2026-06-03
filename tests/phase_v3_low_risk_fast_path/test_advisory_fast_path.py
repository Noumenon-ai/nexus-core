"""Low-risk fast path for the destructive-intent gate (2026-05-31).

Questions / advice / analysis / opinions / document review / brainstorming /
planning ask the assistant to *think*, not *act* — they must never trip an
approval prompt, even when they mention a destructive verb. Only an actual
operation (sending / deleting / modifying / scheduling / writing) gates.

Safety net: an advisory-framed message that ALSO carries an explicit go-ahead
command ("…and send it", "just delete it", "do it now") still gates.
"""
from __future__ import annotations

import pytest

from services.destructive_intent_classifier import classify


# ---------- fast path: think, don't act → never gate ----------

@pytest.mark.parametrize("prompt", [
    # opinions / questions
    "what do you think about changing my pricing strategy?",
    "what's your opinion on whether to send the renewal notice?",
    "wdyt about moving the tenant to a new unit?",
    # advice (even when naming a destructive verb)
    "should I update my resume?",
    "should I delete this old account?",
    "should I cancel the vendor contract?",
    "is it worth scheduling a meeting about this?",
    "what would you do about the late tenant?",
    # analysis / comparison
    "pros and cons of switching banks?",
    "analyze this contract for me",
    "compare these two lease options",
    # document review
    "can you review this lease and tell me what stands out?",
    "look over my draft and give me feedback",
    # explanation
    "explain how the lease renewal works",
    "help me understand this clause",
    # brainstorming
    "help me brainstorm names for the new unit",
    "give me some ideas for the listing",
    # planning
    "help me plan the renovation",
    "let's think through the eviction timeline",
])
def test_advisory_prompts_take_fast_path(prompt):
    decision = classify(prompt)
    assert decision.is_destructive is False, prompt
    assert "advisory_fast_path" in decision.matched_verbs


# ---------- real operations: still gate ----------

@pytest.mark.parametrize("prompt", [
    "delete the old log file",
    "send the email to the tenant",
    "schedule a meeting for 3pm tomorrow",
    "update unit 3 lease_end to december",
    "forget what I told you about the vendor",
    "cancel the 2pm appointment",
    "move the file to archive",
])
def test_real_operations_still_gate(prompt):
    assert classify(prompt).is_destructive is True, prompt


# ---------- advisory framing + explicit command: still gate ----------

@pytest.mark.parametrize("prompt", [
    "review the lease and send it now",
    "what do you think — just delete it",
    "should I... actually go ahead and cancel the meeting",
    "look this over and then send it to the tenant",
])
def test_advisory_with_explicit_command_still_gates(prompt):
    assert classify(prompt).is_destructive is True, prompt
