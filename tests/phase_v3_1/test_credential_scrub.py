from __future__ import annotations

import pytest

from services.credential_scrub import (
    contains_credential,
    find_credential_matches,
    scrub_messages,
)

# Secret-shaped fixtures are assembled from fragments at runtime so no literal
# API-key pattern is committed to source (which would trip GitHub push
# protection). The scrubber under test still receives the full shapes.
_SK_LIVE = "sk_" + "live_abcd1234EFGH5678ijklmnop"
_SK_LIVE_SHORT = "sk_" + "live_abcd1234EFGH5678ijklm"
_SK_TEST = "sk_" + "test_abcd1234EFGH5678ijklmnop"
_GKEY_1 = "AIza" + "SyBdef1234567890abcdefghij_klmnoPQRST"
_GKEY_2 = "AIza" + "SyDdAbCdEfGhIjKlMnOpQrStUvWxYz01234"
_TG_TOKEN = "123456789:" + "AAEhBP0av28Q-XzrtqnMMx_jL5LYxF7c8aQ"


@pytest.mark.parametrize(
    "text",
    [
        f"my key is {_SK_LIVE}",
        f"openai {_SK_TEST} is leaked",
        "Bearer eyJabc123.eyJpayload.signature_xyz_long_enough",
        "Bearer abcdef0123456789ABCDEF",
        "password=hunter2",
        "PASSWORD: correct_horse_battery_staple",
        f"api_key={_GKEY_1}",
        "API-KEY: longvaluehere1234567890",
        f"use {_GKEY_2}",
        "AKIA" + "IOSFODNN7EXAMPLE",
        f"telegram bot {_TG_TOKEN}",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYiLCJuYW1lIjoiSm9obiJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "secret=superlongsecretvalue123",
        "access_key: ALONGENOUGHACCESSKEYVALUE123",
    ],
)
def test_contains_credential_true_for_obvious_secrets(text: str):
    assert contains_credential(text), f"failed to detect secret in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "remind me at 3 PM to call mom",
        "what's the weather today",
        "I love passwords as a concept but won't share one",
        "the api key concept is interesting but not its value",
        "bearer market sentiment is up",
        "skim through the report",
        "sk is too short",
        "password without equals or colon",
        "jwt is a token format",
        "tell me about Stripe in general",
    ],
)
def test_contains_credential_false_for_ordinary_text(text: str):
    assert not contains_credential(text), f"false positive on: {text!r}"


def test_find_credential_matches_returns_pattern_names():
    matches = find_credential_matches(f"password=hunter2 and {_SK_LIVE_SHORT}")
    assert "password_kv" in matches
    assert "stripe_or_openai" in matches


def test_find_credential_matches_empty_for_clean_text():
    assert find_credential_matches("hello there") == []


def test_scrub_messages_partitions_into_surviving_and_dropped():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Bearer eyJabc123.eyJpayload.signature_xyz_long_enough"},
        {"role": "user", "content": "remind me at 3"},
        {"role": "user", "content": "password=hunter2"},
    ]
    surviving, dropped = scrub_messages(messages)
    assert [m["content"] for m in surviving] == ["hi", "remind me at 3"]
    assert len(dropped) == 2


def test_scrub_messages_handles_empty_list():
    surviving, dropped = scrub_messages([])
    assert surviving == []
    assert dropped == []


def test_scrub_messages_handles_none_or_missing_content():
    messages = [
        {"role": "user", "content": None},
        {"role": "user"},
        {"role": "user", "content": ""},
    ]
    surviving, dropped = scrub_messages(messages)
    assert len(surviving) == 3
    assert dropped == []


def test_scrub_messages_custom_content_key():
    messages = [
        {"text": "ok"},
        {"text": "password=secret"},
    ]
    surviving, dropped = scrub_messages(messages, content_key="text")
    assert [m["text"] for m in surviving] == ["ok"]
    assert [m["text"] for m in dropped] == ["password=secret"]


def test_contains_credential_handles_none_input():
    assert contains_credential(None) is False  # type: ignore[arg-type]
    assert contains_credential("") is False
