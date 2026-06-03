from __future__ import annotations

from services.conversational_recovery import ConversationalRecoveryLayer


def _unique_acme_alias(query: str) -> dict[str, object] | None:
    if query.strip().casefold() != 'acme':
        return None
    return {
        'ok': True,
        'match': 'unique',
        'alias_used': 'Acme Corp',
        'contact': {'aliases': ['Acme Corp']},
    }


def test_time_only_followup_preserves_explicit_wagner_recipient():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(
        text='Remind me tomorrow to send Wagner a message.',
    )
    second = layer.recover(
        text='at 10am',
        context={'recovery_state': first.context_updates},
    )

    assert first.context_updates['pending_draft']['recipient'] == 'Wagner'
    assert second.resolved_slots['recipient'] == 'Wagner'
    assert 'Acme' not in second.recovered_text
    assert 'Wagner' in second.recovered_text
    assert 'tomorrow' in second.recovered_text
    assert '10 AM' in second.recovered_text


def test_old_acme_context_cannot_override_new_wagner_reminder_chain():
    layer = ConversationalRecoveryLayer()

    state = {
        'last_action_kind': 'outbound_message',
        'last_recipient': 'Acme Corp',
        'last_message_body': 'i will do it tomorrow',
        'pending_draft': {
            'draft_kind': 'outbound_message',
            'action_kind': 'outbound_message',
            'recipient': 'Acme Corp',
            'message_body': 'i will do it tomorrow',
            'confidence': 0.88,
            'source_turns': ['tell acme ill do it tmrrw'],
        },
    }

    first = layer.recover(
        text='Remind me tomorrow to send a message to Wagner about the water leak.',
        context={'recovery_state': state},
    )
    second = layer.recover(
        text='at 10am',
        context={'recovery_state': first.context_updates},
    )

    assert first.context_updates['pending_draft']['recipient'] == 'Wagner'
    assert second.resolved_slots['recipient'] == 'Wagner'
    assert 'Acme' not in second.recovered_text
    assert 'water leak' in second.recovered_text


def test_explicit_recipient_correction_switches_active_draft_only():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(
        text='Remind me tomorrow to send a message to Wagner about the water leak.',
    )
    second = layer.recover(
        text='No not Wagner, Acme',
        context={'recovery_state': first.context_updates},
        resolve_contact_alias=_unique_acme_alias,
    )

    assert second.resolved_slots['recipient'] == 'Acme Corp'
    assert 'recipient' in second.corrections_applied
    assert 'Acme Corp' in second.recovered_text
    assert 'Wagner' not in second.recovered_text
    assert second.context_updates['pending_draft']['recipient'] == 'Acme Corp'


def test_pronoun_continuation_resolves_to_current_wagner_chain():
    layer = ConversationalRecoveryLayer()

    first = layer.recover(
        text='Send Wagner a reminder.',
    )
    second = layer.recover(
        text='Tell him tomorrow at 10.',
        context={'recovery_state': first.context_updates},
    )

    assert first.context_updates['last_recipient'] == 'Wagner'
    assert second.resolved_slots['recipient'] == 'Wagner'
    assert 'Acme' not in second.recovered_text


def test_unrelated_chains_do_not_bleed_into_new_wagner_reminder():
    layer = ConversationalRecoveryLayer()

    wife = layer.recover(
        text='tell acme ill do it tmrrw',
        resolve_contact_alias=_unique_acme_alias,
    )
    rentals = layer.recover(
        text='did u update my 3 rentals m',
        context={'recovery_state': wife.context_updates},
    )
    wagner = layer.recover(
        text='Remind me tomorrow to send a message to Wagner about the water leak.',
        context={'recovery_state': rentals.context_updates},
    )
    followup = layer.recover(
        text='at 10am',
        context={'recovery_state': wagner.context_updates},
    )

    assert followup.resolved_slots['recipient'] == 'Wagner'
    assert 'Acme' not in followup.recovered_text
    assert 'Wagner' in followup.recovered_text
