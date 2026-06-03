"""Step 4 — flat CAPABILITIES map + is_available + available_list."""
from __future__ import annotations

import pytest

from services import capability_registry as cap_reg


def test_capabilities_map_matches_spec():
    # NEXUS_ARCHITECTURE_REFACTOR.md step 4 exactly.
    expected_wired = {
        'email_read', 'email_send',
        'reminder_create', 'reminder_list', 'reminder_update',
        'rental_read', 'business_read',
        'knowledge_read', 'knowledge_write',
        'square_read',
        'voice_input',
    }
    expected_unwired = {
        'messaging_whatsapp', 'messaging_sms',
        'trading', 'smart_home', 'car', 'osint',
        'voice_output',
    }
    actual_wired = {k for k, v in cap_reg.CAPABILITIES.items() if v}
    actual_unwired = {k for k, v in cap_reg.CAPABILITIES.items() if not v}
    assert actual_wired == expected_wired
    assert actual_unwired == expected_unwired


def test_is_available_for_known_wired_capability():
    assert cap_reg.is_available('email_read') is True
    assert cap_reg.is_available('reminder_create') is True


def test_is_available_for_known_unwired_capability():
    assert cap_reg.is_available('messaging_whatsapp') is False
    assert cap_reg.is_available('trading') is False


def test_is_available_for_unknown_capability_returns_false():
    # Default deny — never lie about an unknown capability.
    assert cap_reg.is_available('teleport') is False
    assert cap_reg.is_available('') is False


def test_available_list_is_sorted_and_only_wired():
    result = cap_reg.available_list()
    assert result == sorted(result), 'available_list must be sorted for cache stability'
    # No unwired capability appears
    assert 'messaging_whatsapp' not in result
    assert 'trading' not in result
    # Sample wired capabilities are present
    assert 'email_read' in result
    assert 'reminder_create' in result


def test_set_capability_runtime_toggle_and_restore():
    original = cap_reg.is_available('trading')
    try:
        cap_reg.set_capability('trading', True)
        assert cap_reg.is_available('trading') is True
        assert 'trading' in cap_reg.available_list()
        cap_reg.set_capability('trading', False)
        assert cap_reg.is_available('trading') is False
        assert 'trading' not in cap_reg.available_list()
    finally:
        cap_reg.set_capability('trading', original)


def test_set_capability_can_register_new_name():
    try:
        cap_reg.set_capability('experimental_x', True)
        assert cap_reg.is_available('experimental_x') is True
        assert 'experimental_x' in cap_reg.available_list()
    finally:
        cap_reg.CAPABILITIES.pop('experimental_x', None)


def test_flat_map_does_not_overlap_richer_capability_status():
    # The flat map is a sibling of CapabilityRegistry; both can coexist.
    # Confirm both surfaces still load without side effects.
    from services.capability_registry import CapabilityRegistry  # noqa: F401
    assert callable(cap_reg.is_available)
    assert callable(cap_reg.available_list)
    assert callable(cap_reg.set_capability)
