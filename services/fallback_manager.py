from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from services.capability_registry import CapabilityStatus
from services.delivery_truth import (
    DeliveryTruth,
    normalize_dispatch_delivery,
    normalize_whatsapp_delivery,
)
from services.human_confirmation_style import HumanConfirmationStyle
from services.structured_failure import StructuredFailure, render_user_facing_failure


AsyncFallbackExecutor = Callable[[], Awaitable[dict[str, Any] | None]]
PROVIDER_UNAVAILABLE_USER_TEXT = (
    'Having trouble connecting right now. Please try again in a moment.'
)


@dataclass(slots=True)
class FallbackContext:
    route: str
    stage: str
    provider: str
    root_reason: str
    raw_text: str = ''
    recovered_text: str = ''
    recovery_metadata: dict[str, Any] = field(default_factory=dict)
    capability: CapabilityStatus | None = None
    capability_name: str = ''
    approval_state: dict[str, Any] = field(default_factory=dict)
    pending_draft: dict[str, Any] | None = None
    delivery_state: DeliveryTruth | dict[str, Any] | str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FallbackDecision:
    handled: bool
    user_text: str
    safe_action_taken: str
    unsafe_action_blocked: str
    route: str
    fallback_type: str
    structured_failure: StructuredFailure | None = None
    log_fields: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] | None = None


class FallbackManager:
    def __init__(
        self,
        *,
        human_confirmation_style: HumanConfirmationStyle | None = None,
    ) -> None:
        self.human_confirmation_style = (
            human_confirmation_style or HumanConfirmationStyle()
        )

    async def decide_post_approval_timeout(
        self,
        *,
        context: FallbackContext,
        contact_reminder_fallback: AsyncFallbackExecutor | None = None,
        cleanup_fallback: AsyncFallbackExecutor | None = None,
        internal_reminder_fallback: AsyncFallbackExecutor | None = None,
    ) -> FallbackDecision:
        payload: dict[str, Any] | None = None
        recovery = dict(context.recovery_metadata or {})
        if recovery.get('recipient_negated_unresolved'):
            payload = {
                'kind': 'send_recipient_clarification',
                'recipient_label': str(recovery.get('negated_recipient_label') or '').strip(),
                'reason': str(recovery.get('negated_recipient_reason') or '').strip(),
            }
        else:
            for executor in (
                contact_reminder_fallback,
                cleanup_fallback,
                internal_reminder_fallback,
            ):
                if executor is None:
                    continue
                payload = await executor()
                if payload is not None:
                    break

        return self._decision_from_payload(context=context, payload=payload)

    def normalize_provider_failure(
        self,
        *,
        context: FallbackContext,
        provider_text: str,
        is_provider_failure_text: bool,
        local_time_text: str | None,
        audit_guidance: str | None,
        vague_clarification: str | None,
        post_approval_resume: bool,
    ) -> tuple[str | None, str | None, StructuredFailure | None]:
        if not is_provider_failure_text:
            return None, None, None

        if context.details.get('is_time_request') and local_time_text:
            decision = self._make_decision(
                handled=True,
                user_text=local_time_text,
                route='time_lookup',
                fallback_type='local',
                root_reason='provider_unavailable',
                safe_action_taken='local_time_lookup',
                unsafe_action_blocked='none',
                stage=context.stage,
                provider=context.provider,
                technical_reason='provider_failure_text',
            )
            return decision.user_text, 'local', decision.structured_failure

        if context.details.get('is_audit_request') and audit_guidance:
            decision = self._make_decision(
                handled=True,
                user_text=audit_guidance,
                route='self_audit',
                fallback_type='audit',
                root_reason='provider_unavailable',
                safe_action_taken='audit_guidance',
                unsafe_action_blocked='none',
                stage=context.stage,
                provider=context.provider,
                technical_reason='provider_failure_text',
            )
            return decision.user_text, 'audit', decision.structured_failure

        route = context.route or self._infer_route(context, payload=None)
        if post_approval_resume:
            decision = self._make_decision(
                handled=False,
                user_text=render_user_facing_failure(
                    self._build_structured_failure(
                        route=route,
                        stage='post_approval',
                        provider=context.provider,
                        fallback='none',
                        root_reason='provider_unavailable',
                        safe_action_taken='none',
                        unsafe_action_blocked=self._infer_unsafe_action_blocked(route),
                        technical_reason='provider_failure_text',
                    )
                ),
                route=route,
                fallback_type='none',
                root_reason='provider_unavailable',
                safe_action_taken='none',
                unsafe_action_blocked=self._infer_unsafe_action_blocked(route),
                stage='post_approval',
                provider=context.provider,
                technical_reason='provider_failure_text',
            )
            return decision.user_text, 'approved_workflow', decision.structured_failure

        if vague_clarification:
            decision = self._make_decision(
                handled=True,
                user_text=vague_clarification,
                route=route,
                fallback_type='clarification',
                root_reason='provider_unavailable',
                safe_action_taken='clarification_requested',
                unsafe_action_blocked=self._infer_unsafe_action_blocked(route),
                stage=context.stage,
                provider=context.provider,
                technical_reason='provider_failure_text',
            )
            return decision.user_text, 'clarification', decision.structured_failure

        decision = self._make_decision(
            handled=False,
            user_text=PROVIDER_UNAVAILABLE_USER_TEXT,
            route=route,
            fallback_type='none',
            root_reason='provider_unavailable',
            safe_action_taken='none',
            unsafe_action_blocked=self._infer_unsafe_action_blocked(route),
            stage=context.stage,
            provider=context.provider,
            technical_reason='provider_failure_text',
        )
        return decision.user_text, 'retry_safe', decision.structured_failure

    def decide_capability(self, *, context: FallbackContext) -> FallbackDecision:
        capability = context.capability
        if capability is None:
            return self._make_decision(
                handled=False,
                user_text="I can't confirm that capability right now.",
                route=context.route or 'capability_check',
                fallback_type='capability_unavailable',
                root_reason='unknown',
                safe_action_taken='capability_reported',
                unsafe_action_blocked='approved_action',
                stage=context.stage,
                provider=context.provider,
            )

        name = context.capability_name or capability.name
        subject = str(context.details.get('subject') or 'your rental records').strip()
        if name == 'rentals_read':
            text = self._render_rental_status_capability_text(subject, capability=capability)
            fallback_type = 'capability_available' if capability.state == 'available' else 'capability_unavailable'
            return self._make_decision(
                handled=True,
                user_text=text,
                route=context.route or 'rental_status',
                fallback_type=fallback_type,
                root_reason=capability.state,
                safe_action_taken='capability_reported',
                unsafe_action_blocked='none',
                stage=context.stage,
                provider=context.provider,
                metadata={'capability': capability.to_dict()},
            )

        if name.startswith('calendar_'):
            return self._make_decision(
                handled=True,
                user_text=self._render_calendar_capability_text(capability),
                route=context.route or name,
                fallback_type='capability_unavailable',
                root_reason=capability.state,
                safe_action_taken='capability_reported',
                unsafe_action_blocked='approved_action',
                stage=context.stage,
                provider=context.provider,
                metadata={'capability': capability.to_dict()},
            )

        if name == 'whatsapp_send':
            return self._make_decision(
                handled=True,
                user_text=self._render_whatsapp_capability_text(capability),
                route=context.route or 'contact_send',
                fallback_type='capability_unavailable',
                root_reason=capability.state,
                safe_action_taken='capability_reported',
                unsafe_action_blocked='send_message',
                stage=context.stage,
                provider=context.provider,
                metadata={'capability': capability.to_dict()},
            )

        return self._make_decision(
            handled=True,
            user_text=(
                f"I can't confirm that right now because {name.replace('_', ' ')} "
                f'is {capability.state.replace("_", " ")}.'
            ),
            route=context.route or 'capability_check',
            fallback_type='capability_unavailable',
            root_reason=capability.state,
            safe_action_taken='capability_reported',
            unsafe_action_blocked='approved_action',
            stage=context.stage,
            provider=context.provider,
            metadata={'capability': capability.to_dict()},
        )

    def decide_delivery_state(self, *, context: FallbackContext) -> FallbackDecision:
        truth = self._coerce_delivery_truth(context.delivery_state)
        if truth.state in {'queued', 'sent_to_server'}:
            fallback_type = 'delivery_pending'
            safe_action_taken = 'delivery_status_reported'
        elif truth.state == 'failed':
            fallback_type = 'delivery_failed'
            safe_action_taken = 'delivery_failure_reported'
        elif truth.state == 'unknown':
            fallback_type = 'delivery_unknown'
            safe_action_taken = 'delivery_status_reported'
        else:
            fallback_type = 'delivery_status'
            safe_action_taken = 'delivery_status_reported'
        return self._make_decision(
            handled=True,
            user_text=truth.user_text,
            route=context.route or 'delivery_status',
            fallback_type=fallback_type,
            root_reason=truth.state,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked='none',
            stage=context.stage,
            provider=context.provider,
            metadata={'delivery_state': truth.to_dict()},
        )

    def _decision_from_payload(
        self,
        *,
        context: FallbackContext,
        payload: dict[str, Any] | None,
    ) -> FallbackDecision:
        route = self._infer_route(context, payload=payload)
        fallback_type = self._infer_fallback_type(payload)
        root_reason = self._infer_root_reason(payload, default=context.root_reason or 'timeout')
        safe_action_taken = self._infer_safe_action_taken(payload)
        unsafe_action_blocked = self._infer_unsafe_action_blocked(route)
        structured_failure = self._build_structured_failure(
            route=route,
            stage=context.stage,
            provider=context.provider,
            fallback=fallback_type,
            root_reason=root_reason,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked=unsafe_action_blocked,
            technical_reason='asyncio.TimeoutError',
        )
        if payload is not None:
            text = self._render_payload_text(payload)
            return self._make_decision(
                handled=True,
                user_text=render_user_facing_failure(
                    structured_failure,
                    default_text=text,
                ),
                route=route,
                fallback_type=fallback_type,
                root_reason=root_reason,
                safe_action_taken=safe_action_taken,
                unsafe_action_blocked=unsafe_action_blocked,
                stage=context.stage,
                provider=context.provider,
                technical_reason='asyncio.TimeoutError',
                payload=payload,
                metadata={
                    'structured_failure': structured_failure.to_metadata(),
                },
            )
        return self._make_decision(
            handled=False,
            user_text=render_user_facing_failure(structured_failure),
            route=route,
            fallback_type=fallback_type,
            root_reason=root_reason,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked=unsafe_action_blocked,
            stage=context.stage,
            provider=context.provider,
            technical_reason='asyncio.TimeoutError',
            metadata={
                'structured_failure': structured_failure.to_metadata(),
            },
        )

    def _make_decision(
        self,
        *,
        handled: bool,
        user_text: str,
        route: str,
        fallback_type: str,
        root_reason: str,
        safe_action_taken: str,
        unsafe_action_blocked: str,
        stage: str,
        provider: str,
        technical_reason: str | None = None,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FallbackDecision:
        structured_failure = self._build_structured_failure(
            route=route,
            stage=stage,
            provider=provider,
            fallback=fallback_type,
            root_reason=root_reason,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked=unsafe_action_blocked,
            technical_reason=technical_reason,
        )
        log_fields = structured_failure.to_metadata()
        log_fields['fallback_type'] = fallback_type
        return FallbackDecision(
            handled=handled,
            user_text=user_text,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked=unsafe_action_blocked,
            route=route,
            fallback_type=fallback_type,
            structured_failure=structured_failure,
            log_fields=log_fields,
            metadata=dict(metadata or {}),
            payload=payload,
        )

    def _build_structured_failure(
        self,
        *,
        route: str,
        stage: str,
        provider: str,
        fallback: str,
        root_reason: str,
        safe_action_taken: str,
        unsafe_action_blocked: str,
        technical_reason: str | None = None,
    ) -> StructuredFailure:
        return StructuredFailure(
            route=route,
            stage=stage,
            provider=provider,
            fallback=fallback,
            root_reason=root_reason,
            safe_action_taken=safe_action_taken,
            unsafe_action_blocked=unsafe_action_blocked,
            technical_reason=technical_reason,
        )

    def _infer_route(
        self,
        context: FallbackContext,
        payload: dict[str, Any] | None,
    ) -> str:
        if context.route:
            return context.route
        kind = str((payload or {}).get('kind') or '').strip()
        if kind in {'contact_reminder', 'contact_missing', 'contact_ambiguous'}:
            return 'contact_reminder'
        if kind in {
            'task_cleanup',
            'reminder_cleanup',
            'cleanup_ambiguous',
            'digest_cleanup_ambiguous',
            'digest_cleanup_missing',
            'cleanup_not_found',
            'cleanup_failed',
        }:
            return 'cleanup'
        if kind == 'send_recipient_clarification':
            return 'contact_send'
        if context.details.get('has_contact_reminder_intent'):
            return 'contact_reminder'
        if context.details.get('has_outbound_message_intent'):
            return 'contact_send'
        if context.details.get('has_rich_reminder_context'):
            return 'follow_up'
        if context.stage == 'post_approval':
            return 'approved_workflow'
        return 'tool_loop'

    def _infer_fallback_type(self, payload: dict[str, Any] | None) -> str:
        if payload is None:
            return 'none'
        kind = str(payload.get('kind') or '').strip()
        if kind == 'contact_reminder':
            return 'contact_reminder'
        if kind in {'contact_missing', 'contact_ambiguous', 'send_recipient_clarification'}:
            return 'clarification'
        if kind in {
            'task_cleanup',
            'reminder_cleanup',
            'cleanup_ambiguous',
            'digest_cleanup_ambiguous',
            'digest_cleanup_missing',
            'cleanup_not_found',
            'cleanup_failed',
        }:
            return 'cleanup'
        if 'target_name' in payload:
            return 'internal_reminder'
        return kind or 'none'

    def _infer_root_reason(self, payload: dict[str, Any] | None, *, default: str) -> str:
        if payload is None:
            return default
        kind = str(payload.get('kind') or '').strip()
        if kind == 'send_recipient_clarification':
            return 'unresolved_recipient'
        if kind == 'contact_missing':
            return 'contact_missing'
        if kind == 'contact_ambiguous':
            return 'ambiguous_contact'
        if kind in {'cleanup_ambiguous', 'digest_cleanup_ambiguous'}:
            return 'selector_failed'
        if kind in {'cleanup_not_found', 'digest_cleanup_missing'}:
            return 'target_not_found'
        if kind == 'cleanup_failed':
            return 'local_update_failed'
        return default

    def _infer_safe_action_taken(self, payload: dict[str, Any] | None) -> str:
        if payload is None:
            return 'none'
        kind = str(payload.get('kind') or '').strip()
        if kind == 'contact_reminder':
            return 'contact_reminder_created' if payload.get('created') else 'contact_reminder_kept'
        if kind in {
            'send_recipient_clarification',
            'contact_missing',
            'contact_ambiguous',
            'cleanup_ambiguous',
            'digest_cleanup_ambiguous',
            'digest_cleanup_missing',
        }:
            return 'clarification_requested'
        if kind == 'cleanup_not_found':
            return 'no_change_reported'
        if kind == 'cleanup_failed':
            return 'local_update_failed'
        if kind == 'task_cleanup':
            return 'task_completed'
        if kind == 'reminder_cleanup':
            return 'reminder_removed'
        if 'target_name' in payload:
            return 'internal_reminder_created' if payload.get('created') else 'internal_reminder_kept'
        return 'none'

    def _infer_unsafe_action_blocked(self, route: str) -> str:
        if route in {'contact_send', 'contact_reminder', 'follow_up'}:
            return 'send_message'
        if route == 'cleanup':
            return 'none'
        return 'approved_action'

    def _render_payload_text(self, payload: dict[str, Any]) -> str:
        styled = self.human_confirmation_style.render_partial_success(
            fallback_result=payload,
        )
        if styled:
            return styled

        kind = str(payload.get('kind') or '').strip()
        if kind == 'send_recipient_clarification':
            recipient_label = str(payload.get('recipient_label') or '').strip()
            if payload.get('reason') == 'other_one' and recipient_label:
                return (
                    f'I can’t send this yet — you said not {recipient_label}, '
                    'but I don’t know who “the other one” is.'
                )
            if recipient_label:
                return f'You said not {recipient_label} — who should I send it to?'
            return 'I can’t send this yet because I still don’t know who should receive it.'
        if kind == 'contact_reminder':
            channel = 'WhatsApp' if payload.get('channel') == 'whatsapp' else 'SMS'
            return (
                f'Approved. I scheduled a {channel} reminder for '
                f"{payload['contact_label']} {payload['time_label']}: "
                f'"{payload["body"]}."\n\n'
                'No message was sent now. It will be sent at the scheduled time.'
            )
        if kind == 'contact_missing':
            channel = 'WhatsApp' if payload.get('channel') == 'whatsapp' else 'SMS'
            return (
                f'I need you to add or confirm the {channel} contact for '
                f'"{payload["alias_label"]}" before I can schedule this.'
            )
        if kind == 'contact_ambiguous':
            channel = 'WhatsApp' if payload.get('channel') == 'whatsapp' else 'SMS'
            return (
                f'I need you to confirm which {channel} contact you mean by '
                f'"{payload["alias_label"]}" before I can schedule this.'
            )
        if kind == 'task_cleanup':
            return (
                f'Done. I marked "{payload["title"]}" as completed and removed it '
                'from your active tasks.'
            )
        if kind == 'reminder_cleanup':
            return f'Done. I removed the active reminder "{payload["title"]}".'
        if kind == 'cleanup_ambiguous':
            lines = [
                f'I found multiple matches for "{payload["query"]}". Which one should I remove?'
            ]
            for index, match in enumerate(payload.get('matches') or [], start=1):
                lines.append(f'{index}. {match["label"]} - {match["descriptor"]}')
            return '\n'.join(lines)
        if kind == 'digest_cleanup_ambiguous':
            lines = [
                'I couldn\'t safely resolve what "it" means from the recent digest. '
                'Which one should I remove?'
            ]
            for index, match in enumerate(payload.get('matches') or [], start=1):
                lines.append(f'{index}. {match["label"]} - {match["descriptor"]}')
            return '\n'.join(lines)
        if kind == 'digest_cleanup_missing':
            return (
                'I couldn\'t safely resolve what "it" means from a recent digest. '
                'Which task or reminder should I remove?'
            )
        if kind == 'cleanup_not_found':
            return (
                'I couldn\'t find an active task/reminder matching '
                f'"{payload["query"]}".'
            )
        if kind == 'cleanup_failed':
            return (
                f'I found "{payload["label"]}" but could not update it locally '
                'after approval.'
            )

        created = bool(payload.get('created'))
        verb = 'created one' if created else 'kept one'
        return (
            f"Approved. I {verb} {payload['time_label']} follow-up reminder for "
            f"{payload['target_name']} about {payload['issue_summary']} in "
            f"{payload['unit_reference']}.\n\n"
            'I did not send a message yet because the final recipient/action '
            'needs clarification.'
        )

    def _render_rental_status_capability_text(
        self,
        subject: str,
        *,
        capability: CapabilityStatus,
    ) -> str:
        confirmation = self.human_confirmation_style.render_natural_confirmation(
            recovered_intent=f'check rental update status for {subject}',
            confidence=0.84,
            risk_level='low',
            resolved_slots={
                'action_kind': 'rental_status_check',
                'rental_subject': subject,
            },
        )
        if capability.state == 'available':
            record_count = int(capability.details.get('record_count') or 0)
            if record_count <= 0:
                return (
                    f'{confirmation} I can access the rentals store, but I found no rental records yet.'
                    if confirmation
                    else 'I can access the rentals store, but I found no rental records yet.'
                )
            return (
                f'{confirmation} I can access the rentals dashboard, but this chat path still '
                "can't verify whether those records were updated yet."
                if confirmation
                else "I can access the rentals dashboard, but this chat path still can't verify "
                'whether those records were updated yet.'
            )
        if capability.state == 'not_wired':
            return (
                "The rentals dashboard exists, but this Telegram path isn't wired to read it yet. "
                'I need a rentals-read tool before I can verify rental records from chat.'
            )
        if capability.state == 'service_down':
            return (
                f"{confirmation} I couldn't reach the rentals dashboard just now."
                if confirmation
                else "I couldn't reach the rentals dashboard just now."
            )
        if confirmation:
            return (
                f"{confirmation} I can't verify that right now because rentals access is "
                f'{capability.state.replace("_", " ")}.'
            )
        return (
            f"I understand you're asking whether {subject} were updated. I can't verify "
            f"that right now because rentals access is {capability.state.replace('_', ' ')}."
        )

    def _render_calendar_capability_text(self, capability: CapabilityStatus) -> str:
        if capability.state == 'auth_required':
            return 'Calendar access needs Google auth before I can do that. Reconnect Google and try again.'
        if capability.state == 'not_wired':
            return "Calendar access exists in this runtime, but this Telegram path isn't wired for that calendar action yet."
        if capability.state == 'service_down':
            return "I couldn't reach the calendar path right now. Try again after Google Calendar is healthy."
        if capability.state == 'unavailable':
            return 'Google Calendar is disabled in this runtime right now.'
        return 'I need a clearer calendar path before I can do that.'

    def _render_whatsapp_capability_text(self, capability: CapabilityStatus) -> str:
        if capability.state == 'auth_required':
            return "WhatsApp isn't ready yet because the bridge still needs pairing. Scan the QR in the dashboard first."
        if capability.state == 'service_down':
            return 'WhatsApp send is unavailable right now because the bridge is down. Start or restart whatsapp-bridge.service first.'
        if capability.state == 'not_wired':
            return "This runtime doesn't have a WhatsApp send path wired for Telegram yet."
        return "I can't confirm that WhatsApp send is ready right now."

    def _coerce_delivery_truth(
        self,
        delivery_state: DeliveryTruth | dict[str, Any] | str | None,
    ) -> DeliveryTruth:
        if isinstance(delivery_state, DeliveryTruth):
            return delivery_state
        if isinstance(delivery_state, dict):
            if 'state' in delivery_state and 'user_text' in delivery_state:
                return DeliveryTruth(
                    state=str(delivery_state.get('state') or 'unknown'),
                    user_text=str(delivery_state.get('user_text') or ''),
                    ack_status=delivery_state.get('ack_status'),
                    ack_text=delivery_state.get('ack_text'),
                    message_id=delivery_state.get('message_id'),
                    platform=delivery_state.get('platform'),
                )
            if 'delivery' in delivery_state or delivery_state.get('platform') == 'whatsapp':
                return normalize_dispatch_delivery(delivery_state)
            if 'status' in delivery_state or 'status_text' in delivery_state:
                return normalize_whatsapp_delivery(delivery_state)
        if isinstance(delivery_state, str):
            mapping = {
                'queued': DeliveryTruth(state='queued', user_text='Queued.'),
                'sent_to_server': DeliveryTruth(
                    state='sent_to_server',
                    user_text='WhatsApp accepted it. Delivery is still pending.',
                ),
                'delivered': DeliveryTruth(state='delivered', user_text='Delivered.'),
                'read': DeliveryTruth(state='read', user_text='Read.'),
                'failed': DeliveryTruth(
                    state='failed',
                    user_text='WhatsApp reported a send failure.',
                ),
                'unknown': DeliveryTruth(
                    state='unknown',
                    user_text='I sent it to WhatsApp, but delivery status is unknown.',
                ),
            }
            return mapping.get(
                delivery_state,
                DeliveryTruth(
                    state='unknown',
                    user_text='I sent it to WhatsApp, but delivery status is unknown.',
                ),
            )
        return DeliveryTruth(
            state='unknown',
            user_text='I sent it to WhatsApp, but delivery status is unknown.',
        )
