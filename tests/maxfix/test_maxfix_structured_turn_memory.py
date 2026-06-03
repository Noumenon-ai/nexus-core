from __future__ import annotations

from datetime import timedelta
import logging
from types import SimpleNamespace

from services.conversational_recovery import ConversationalRecoveryLayer


def _turn(turn_id: str, content: str):
    return SimpleNamespace(turn_id=turn_id, content=content)


def test_time_only_followup_keeps_wagner_in_pending_draft():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(text='tell Wagner tomorrow')
    second = layer.recover(
        text='at 10am',
        context={'recovery_state': first.context_updates},
    )

    assert first.context_updates['pending_draft']['recipient'] == 'Wagner'
    assert second.resolved_slots['recipient'] == 'Wagner'
    assert second.context_updates['context_source'] == 'pending_draft'
    assert second.context_updates['used_context'] is True


def test_unrelated_rental_chain_clears_old_contact_draft():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(text='tell wife tomorrow')
    rentals = layer.recover(
        text='did u update my 3 rentals m',
        context={'recovery_state': first.context_updates},
    )
    followup = layer.recover(
        text='at 10am',
        context={'recovery_state': rentals.context_updates},
    )

    assert 'pending_draft' not in rentals.context_updates
    assert 'recipient' not in followup.resolved_slots
    assert followup.context_updates.get('used_context') is not True


def test_other_one_only_auto_resolves_with_exactly_two_candidates():
    layer = ConversationalRecoveryLayer()

    resolved = layer.recover(
        text='send her the update no the other one',
        context={'recovery_state': {
            'last_action_kind': 'outbound_message',
            'last_recipient': 'Sarah',
            'recipient_candidates': ['Sarah', 'Wagner'],
            'confidence': 0.9,
        }},
    )
    unresolved = layer.recover(
        text='send her the update no the other one',
        context={'recovery_state': {
            'last_action_kind': 'outbound_message',
            'last_recipient': 'Sarah',
            'recipient_candidates': ['Sarah', 'Wagner', 'Acme Corp'],
            'confidence': 0.9,
        }},
    )

    assert resolved.resolved_slots['recipient'] == 'Wagner'
    assert resolved.outcome == 'auto_resolve'
    assert unresolved.outcome == 'hard_clarify'
    assert unresolved.missing_slot == 'recipient'


def test_stale_pending_draft_expires():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(text='tell Wagner tomorrow')
    expired = dict(first.context_updates)
    expired['pending_draft'] = dict(expired['pending_draft'])
    expired['pending_draft']['expires_at'] = (
        layer.recover.__globals__['_utc_now']() - timedelta(minutes=5)
    ).isoformat()

    result = layer.recover(
        text='at 10am',
        context={'recovery_state': expired},
    )

    assert 'recipient' not in result.resolved_slots
    assert result.context_updates.get('used_context') is not True
    assert 'pending_draft' not in result.context_updates


def test_explicit_recipient_overrides_old_context():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(
        text='tell Wagner friday',
        context={'recovery_state': {
            'last_action_kind': 'outbound_message',
            'last_recipient': 'Acme Corp',
            'last_message_body': 'i will do it tomorrow',
            'pending_draft': {
                'draft_kind': 'outbound_message',
                'action_kind': 'outbound_message',
                'recipient': 'Acme Corp',
                'message_body': 'i will do it tomorrow',
                'confidence': 0.9,
                'expires_at': (layer.recover.__globals__['_utc_now']() + timedelta(minutes=10)).isoformat(),
            },
        }},
    )

    assert result.resolved_slots['recipient'] == 'Wagner'
    assert result.context_updates['last_recipient'] == 'Wagner'
    assert result.context_updates['last_recipient_source'] == 'explicit'
    assert result.context_updates['pending_draft']['recipient'] == 'Wagner'


def test_risky_send_with_weak_pronoun_context_clarifies():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(
        text='send her the update',
        context={'recovery_state': {
            'last_action_kind': 'outbound_message',
            'last_recipient': 'Wagner',
            'last_recipient_source': 'global',
            'confidence': 0.4,
        }},
    )

    assert result.outcome == 'hard_clarify'
    assert result.missing_slot == 'recipient'
    assert result.context_updates.get('weak_context_blocked') is True


def test_read_only_followup_can_use_recent_context_more_freely():
    layer = ConversationalRecoveryLayer()

    result = layer.recover(
        text='did he reply?',
        recent_turns=[_turn('turn-1', 'check if Wagner replied')],
    )

    assert result.outcome == 'auto_resolve'
    assert result.recovered_text == 'check if Wagner replied'
    assert result.resolved_slots['recipient'] == 'Wagner'
    assert result.context_updates['context_source'] == 'recent_turn'


def test_recovery_logs_context_source_and_confidence(caplog):
    layer = ConversationalRecoveryLayer()

    first = layer.recover(text='tell Wagner tomorrow')
    with caplog.at_level(logging.INFO, logger='services.conversational_recovery'):
        second = layer.recover(
            text='at 10am',
            context={'recovery_state': first.context_updates},
        )

    assert second.context_updates['used_context'] is True
    record = next(
        record
        for record in caplog.records
        if record.message == 'recovery_context_applied'
    )
    assert record.context_source == 'pending_draft'
    assert record.confidence >= 0.8
