"""MAXFIX repro: dangerous workflow commands must arm the approval gate."""
from __future__ import annotations

import pytest

from services.destructive_intent_classifier import classify


_DANGEROUS_COMMANDS = [
    "send lease renewal now",
    "send late fee notice",
    "approve vendor quote",
    "authorize repair",
    "accept handyman estimate",
    "pay invoice",
    "close tenant case",
    "mark issue resolved",
    "tell tenant they are responsible",
    "tell HOA they are at fault",
    "send legal notice",
    "delete this tenant",
    "delete this property",
    "remove this reminder",
    "forward tenant info to vendor",
    "approve the quote",
    "approve repair estimate",
    "approve handyman estimate",
    "authorize vendor",
    "authorize handyman",
    "accept repair estimate",
    "close the case",
    "close maintenance case",
    "mark case resolved",
    "dismiss tenant issue",
    "charge late fee",
    "mark rent unpaid",
    "mark rent paid",
    "send tenant phone to vendor",
    "share tenant contact",
]




@pytest.mark.parametrize(
    ("prompt", "expected_tool"),
    [
        ("approve vendor quote", "approve_vendor_action"),
        ("close the case", "close_case"),
        ("tell tenant they are responsible", "assign_fault"),
        ("charge late fee", "update_payment_status"),
        ("send tenant phone to vendor", "share_tenant_information"),
    ],
)
def test_new_mapping_entries_catch_representative_failures(
    prompt: str, expected_tool: str,
) -> None:
    intent = classify(prompt)
    assert expected_tool in intent.matched_tools

