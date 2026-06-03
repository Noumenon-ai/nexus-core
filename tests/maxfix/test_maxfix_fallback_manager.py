from __future__ import annotations

import pytest

from services.capability_registry import CapabilityStatus
from services.fallback_manager import FallbackContext, FallbackManager


_OLD_TIMEOUT_TEXT = (
    "Approved workflow could not continue because the provider/router timed out."
)


def _context(
    *,
    route: str = '',
    stage: str = 'post_approval',
    provider: str = 'brain_router',
    root_reason: str = 'timeout',
    raw_text: str = 'test prompt',
    recovered_text: str = 'test prompt',
    recovery_metadata: dict | None = None,
    capability: CapabilityStatus | None = None,
    capability_name: str = '',
    delivery_state=None,
    details: dict | None = None,
) -> FallbackContext:
    return FallbackContext(
        route=route,
        stage=stage,
        provider=provider,
        root_reason=root_reason,
        raw_text=raw_text,
        recovered_text=recovered_text,
        recovery_metadata=recovery_metadata or {},
        capability=capability,
        capability_name=capability_name,
        delivery_state=delivery_state,
        details=details or {},
    )


def _assert_log_fields(decision, *, route: str, stage: str, root_reason: str) -> None:
    assert decision.log_fields['fallback_type'] == decision.fallback_type
    assert decision.log_fields['route'] == route
    assert decision.log_fields['stage'] == stage
    assert decision.log_fields['root_reason'] == root_reason
    assert decision.log_fields['safe_action_taken'] == decision.safe_action_taken
    assert decision.log_fields['unsafe_action_blocked'] == decision.unsafe_action_blocked


@pytest.mark.asyncio
async def test_unresolved_recipient_fallback_returns_specific_clarification():
    manager = FallbackManager()

    decision = await manager.decide_post_approval_timeout(
        context=_context(
            route='contact_send',
            recovery_metadata={
                'recipient_negated_unresolved': True,
                'negated_recipient_label': 'Acme Corp',
                'negated_recipient_reason': 'other_one',
            },
        )
    )

    assert decision.handled is True
    assert 'not Acme Corp' in decision.user_text
    assert 'who should I send it to?' in decision.user_text
    assert decision.fallback_type == 'clarification'
    assert decision.structured_failure is not None
    assert decision.structured_failure.root_reason == 'unresolved_recipient'
    _assert_log_fields(decision, route='contact_send', stage='post_approval', root_reason='unresolved_recipient')


@pytest.mark.asyncio
async def test_provider_timeout_with_internal_reminder_payload_reports_partial_success():
    manager = FallbackManager()

    async def internal_reminder_payload():
        return {
            'created': True,
            'reminder_id': 'r-1',
            'time_label': 'Friday morning',
            'target_name': 'Mike',
            'issue_summary': 'urgent water damage',
            'unit_reference': 'Unit 204',
        }

    decision = await manager.decide_post_approval_timeout(
        context=_context(
            route='follow_up',
            raw_text='follow up Friday morning with Mike about urgent water damage in Unit 204',
            details={'has_rich_reminder_context': True},
        ),
        internal_reminder_fallback=internal_reminder_payload,
    )

    assert decision.handled is True
    assert 'created the safe reminder part' in decision.user_text.lower()
    assert 'no message was sent' in decision.user_text.lower()
    assert decision.fallback_type == 'internal_reminder'
    assert decision.safe_action_taken == 'internal_reminder_created'
    _assert_log_fields(decision, route='follow_up', stage='post_approval', root_reason='timeout')


@pytest.mark.asyncio
async def test_provider_timeout_with_contact_reminder_payload_reports_scheduled_contact_reminder():
    manager = FallbackManager()

    async def contact_reminder_payload():
        return {
            'kind': 'contact_reminder',
            'created': True,
            'channel': 'whatsapp',
            'contact_label': 'Sam/wife',
            'time_label': 'tomorrow at noon',
            'body': 'buy diapers',
            'reminder_id': 'cr-1',
        }

    decision = await manager.decide_post_approval_timeout(
        context=_context(
            raw_text='Remind my wife on WhatsApp tomorrow at noon to buy diapers.',
            details={'has_contact_reminder_intent': True},
        ),
        contact_reminder_fallback=contact_reminder_payload,
    )

    assert decision.handled is True
    assert 'scheduled a WhatsApp reminder' in decision.user_text
    assert 'No message was sent now' in decision.user_text
    assert decision.route == 'contact_reminder'
    assert decision.fallback_type == 'contact_reminder'
    assert decision.safe_action_taken == 'contact_reminder_created'
    _assert_log_fields(decision, route='contact_reminder', stage='post_approval', root_reason='timeout')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('payload', 'expected_text', 'expected_safe_action'),
    [
        (
            {'kind': 'task_cleanup', 'task_id': 't-1', 'title': 'Complete NEXUS audit'},
            'Done. I marked "Complete NEXUS audit" as completed and removed it from your active tasks.',
            'task_completed',
        ),
        (
            {'kind': 'reminder_cleanup', 'reminder_id': 'r-1', 'title': 'NEXUS audit reminder'},
            'Done. I removed the active reminder "NEXUS audit reminder".',
            'reminder_removed',
        ),
    ],
)
async def test_cleanup_fallbacks_report_completed_or_removed_items(payload, expected_text, expected_safe_action):
    manager = FallbackManager()

    async def cleanup_payload():
        return payload

    decision = await manager.decide_post_approval_timeout(
        context=_context(
            raw_text='I already did nexus audit remove it',
        ),
        cleanup_fallback=cleanup_payload,
    )

    assert decision.handled is True
    assert decision.user_text == expected_text
    assert decision.route == 'cleanup'
    assert decision.fallback_type == 'cleanup'
    assert decision.safe_action_taken == expected_safe_action
    _assert_log_fields(decision, route='cleanup', stage='post_approval', root_reason='timeout')


@pytest.mark.asyncio
async def test_digest_cleanup_fallback_uses_recent_digest_context_safely():
    manager = FallbackManager()

    async def cleanup_payload():
        return {
            'kind': 'digest_cleanup_ambiguous',
            'matches': [
                {'label': 'Complete NEXUS audit', 'descriptor': 'task'},
                {'label': 'Call bank', 'descriptor': 'reminder'},
            ],
        }

    decision = await manager.decide_post_approval_timeout(
        context=_context(raw_text='I already did it, remove it'),
        cleanup_fallback=cleanup_payload,
    )

    assert decision.handled is True
    assert 'recent digest' in decision.user_text
    assert 'Which one should I remove?' in decision.user_text
    assert '1. Complete NEXUS audit - task' in decision.user_text
    assert decision.fallback_type == 'cleanup'
    _assert_log_fields(decision, route='cleanup', stage='post_approval', root_reason='selector_failed')


def test_capability_unavailable_returns_specific_runtime_reason():
    manager = FallbackManager()
    capability = CapabilityStatus(
        name='whatsapp_send',
        state='auth_required',
        reason='bridge waiting for QR',
        service='whatsapp-bridge.service',
        tool_name='dispatch_whatsapp_message',
        safe_to_attempt=False,
        manual_fix='Scan the WhatsApp QR.',
        details={},
    )

    decision = manager.decide_capability(
        context=_context(
            route='contact_send',
            stage='prelude',
            provider='local',
            root_reason='auth_required',
            capability=capability,
            capability_name='whatsapp_send',
        )
    )

    assert decision.handled is True
    assert 'bridge still needs pairing' in decision.user_text
    assert decision.fallback_type == 'capability_unavailable'
    assert decision.safe_action_taken == 'capability_reported'
    _assert_log_fields(decision, route='contact_send', stage='prelude', root_reason='auth_required')


@pytest.mark.parametrize(
    ('delivery_state', 'expected_text', 'expected_type', 'expected_safe_action'),
    [
        (
            {
                'ok': True,
                'platform': 'whatsapp',
                'delivery': {'status': 2, 'status_text': 'server_ack'},
            },
            'WhatsApp accepted it. Delivery is still pending.',
            'delivery_pending',
            'delivery_status_reported',
        ),
        (
            {
                'ok': True,
                'platform': 'whatsapp',
                'delivery': {'status': 0, 'status_text': 'error'},
            },
            'WhatsApp reported a send failure.',
            'delivery_failed',
            'delivery_failure_reported',
        ),
    ],
)
def test_delivery_fallbacks_report_pending_and_failed_truthfully(
    delivery_state,
    expected_text,
    expected_type,
    expected_safe_action,
):
    manager = FallbackManager()

    decision = manager.decide_delivery_state(
        context=_context(
            route='contact_send',
            stage='delivery',
            provider='whatsapp',
            root_reason='delivery_state',
            delivery_state=delivery_state,
        )
    )

    assert decision.handled is True
    assert decision.user_text == expected_text
    assert decision.fallback_type == expected_type
    assert decision.safe_action_taken == expected_safe_action
    assert 'sent successfully' not in decision.user_text.lower()
    _assert_log_fields(decision, route='contact_send', stage='delivery', root_reason=decision.structured_failure.root_reason)


@pytest.mark.asyncio
async def test_unknown_timeout_fallback_returns_structured_safe_failure_not_generic_timeout():
    manager = FallbackManager()

    decision = await manager.decide_post_approval_timeout(
        context=_context(
            route='contact_send',
            raw_text='Send the lease renewal to unit 204 now.',
            details={'has_outbound_message_intent': True},
        ),
    )

    assert decision.handled is False
    assert decision.fallback_type == 'none'
    assert decision.structured_failure is not None
    assert decision.structured_failure.root_reason == 'timeout'
    assert decision.user_text != _OLD_TIMEOUT_TEXT
    assert 'send path' in decision.user_text.lower()
    assert 'did not send anything' in decision.user_text.lower()
    _assert_log_fields(decision, route='contact_send', stage='post_approval', root_reason='timeout')
