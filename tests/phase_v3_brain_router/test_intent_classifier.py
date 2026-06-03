"""Tests for services.intent_classifier — Phase 2 routing."""
from __future__ import annotations

import pytest

from services.intent_classifier import Path, Provider, classify


@pytest.mark.parametrize("text", [
    "hello",
    "how are you",
    "tell me a joke",
    "what's the meaning of life",
    "thanks",
])
def test_conversational_routes_to_claude(text):
    decision = classify(text)
    assert decision.provider is Provider.CLAUDE
    assert decision.path is Path.CONVERSATIONAL


@pytest.mark.parametrize("text", [
    "remind me at 3pm to call the bank",
    "what's on my calendar tomorrow",
    "schedule a meeting friday",
    "create a task to renew domain",
    "show me my email inbox",
    "lookup contact for sam",
    "search the web for python tutorials",
    "what's the weather tomorrow",
    "wikipedia article on relativity",
    "any news today",
    "am I free at 3pm",
    "delete that calendar event",
    # Phase 4 file system + terminal vocabulary additions (2026-05-11)
    "Read the file at /home/user/.bashrc",
    "open the contents of /home/user/notes.txt",
    "view the file at ~/projects/readme.md",
    "show me the files in my Documents folder",
    "list the files in the directory",
    "save this output to /home/user/nexus_workspace/scratch.txt",
    "write a note to /home/user/nexus_workspace/idea.txt",
    "run df -h in the terminal",
    "execute the command uptime",
    # Phase 2.5b followup additions (2026-05-12)
    "Summarize my unread emails",
    "Draft a reply to John about the lease",
    "Forward this email to sam@example.com",
    "Compose a message to the tenants group",
    "Show me archived messages from last week",
    "Save the attachments from yesterday's email",
])
def test_tool_likely_routes_to_claude_tool_path(text):
    decision = classify(text)
    assert decision.provider is Provider.CLAUDE
    assert decision.path is Path.TOOL_LIKELY
    assert len(decision.matched_keywords) > 0


@pytest.mark.parametrize("text", [
    "refactor this python code",
    "what's wrong with this function",
    "explain this stack trace",
    "git commit message for this change",
    "write a regex for email validation",
])
def test_code_routes_to_codex(text):
    decision = classify(text)
    assert decision.provider is Provider.CODEX


@pytest.mark.parametrize("text", [
    "this is private and confidential",
    "my medical diagnosis history",
    "personal journal entry",
])
def test_sensitive_routes_to_ollama(text):
    decision = classify(text)
    assert decision.provider is Provider.OLLAMA


def test_empty_string_defaults_to_claude_conversational():
    decision = classify("")
    assert decision.provider is Provider.CLAUDE
    assert decision.path is Path.CONVERSATIONAL


def test_whitespace_only_defaults_to_claude_conversational():
    decision = classify("   \n\t  ")
    assert decision.provider is Provider.CLAUDE
    assert decision.path is Path.CONVERSATIONAL
