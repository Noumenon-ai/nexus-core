from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(slots=True)
class HumanConfirmationStyleInput:
    recovered_intent: str = ''
    confidence: float = 0.0
    risk_level: str = 'low'
    missing_slot: str | None = None
    selected_clarification_option: str | None = None
    resolved_slots: dict[str, Any] = field(default_factory=dict)
    fallback_result: dict[str, Any] | None = None
    existing_text: str | None = None


class HumanConfirmationStyle:
    def render_natural_confirmation(
        self,
        *,
        recovered_intent: str,
        confidence: float,
        risk_level: str,
        resolved_slots: dict[str, Any] | None = None,
        selected_clarification_option: str | None = None,
    ) -> str | None:
        payload = HumanConfirmationStyleInput(
            recovered_intent=recovered_intent,
            confidence=confidence,
            risk_level=risk_level,
            resolved_slots=dict(resolved_slots or {}),
            selected_clarification_option=selected_clarification_option,
        )
        action_kind = str(payload.resolved_slots.get('action_kind') or '').strip()
        if action_kind == 'rental_status_check' or payload.selected_clarification_option == 'rental_status_check':
            subject = str(payload.resolved_slots.get('rental_subject') or 'your rental records').strip()
            return (
                f"You mean checking whether {subject} were updated. "
                "I'll check what I can see."
            )
        return None

    def render_specific_clarification(
        self,
        *,
        recovered_intent: str,
        confidence: float,
        risk_level: str,
        missing_slot: str | None,
        resolved_slots: dict[str, Any] | None = None,
        selected_clarification_option: str | None = None,
        fallback_result: dict[str, Any] | None = None,
        existing_text: str | None = None,
    ) -> str | None:
        payload = HumanConfirmationStyleInput(
            recovered_intent=recovered_intent,
            confidence=confidence,
            risk_level=risk_level,
            missing_slot=missing_slot,
            selected_clarification_option=selected_clarification_option,
            resolved_slots=dict(resolved_slots or {}),
            fallback_result=dict(fallback_result or {}) or None,
            existing_text=existing_text,
        )

        if payload.missing_slot == 'follow_up_target_and_topic':
            return 'Who should I follow up with, and what is it about?'

        if payload.fallback_result and payload.fallback_result.get('kind') == 'send_recipient_clarification':
            return self._render_send_recipient_clarification(payload)

        if payload.missing_slot == 'recipient':
            if payload.resolved_slots.get('negated_recipient_label'):
                return self._render_send_recipient_clarification(payload)
            if payload.existing_text and payload.existing_text.startswith('You mean '):
                return payload.existing_text
            lowered = payload.recovered_intent.casefold()
            if 'follow up' in lowered:
                return 'Who should I follow up with, and what is it about?'
            if any(token in lowered for token in ('send ', 'tell ', 'message ', 'text ', 'ask ')):
                if "i'll check" in lowered or 'i will check' in lowered or 'check it' in lowered:
                    return "Who should I tell I'll check it for?"
                if ' update ' in f' {lowered} ':
                    return 'Who should I send the update to?'
                return 'Who should I send it to?'

        if payload.missing_slot == 'message_body':
            recipient = str(payload.resolved_slots.get('recipient') or '').strip()
            if recipient:
                return f'What should I send {recipient}?'
            return 'What should I send?'

        if payload.missing_slot == 'recipient_and_message_body':
            return 'Who should I message, and what should I say?'

        return payload.existing_text

    def render_partial_success(
        self,
        *,
        fallback_result: dict[str, Any],
        recovered_intent: str = '',
        confidence: float = 0.0,
        risk_level: str = 'low',
        resolved_slots: dict[str, Any] | None = None,
    ) -> str | None:
        payload = HumanConfirmationStyleInput(
            recovered_intent=recovered_intent,
            confidence=confidence,
            risk_level=risk_level,
            resolved_slots=dict(resolved_slots or {}),
            fallback_result=dict(fallback_result or {}),
        )
        if not payload.fallback_result:
            return None

        if payload.fallback_result.get('kind') == 'send_recipient_clarification':
            return self._render_send_recipient_clarification(payload)

        if 'target_name' not in payload.fallback_result:
            return None

        detail = self._render_safe_reminder_detail(payload.fallback_result)
        if not detail:
            return None
        verb = 'created' if payload.fallback_result.get('created') else 'kept'
        return (
            f"I couldn't finish the send path, but I {verb} the safe reminder part: "
            f'{detail}. No message was sent.'
        )

    def render_clarification_follow_up(
        self,
        *,
        question: str,
        options: list[str],
    ) -> str:
        labels = [str(option).strip() for option in options if str(option).strip()]
        if len(labels) == 2:
            return f'I meant {labels[0]} or {labels[1]}. Which one do you want?'
        if len(labels) >= 3:
            return (
                'I meant '
                + ', '.join(labels[:-1])
                + f', or {labels[-1]}. Which one do you want?'
            )
        if labels:
            return f'I meant {labels[0]}. Is that the one you want?'
        return f'I still need one detail: {question}'

    def render_stale_clarification(self) -> str:
        return 'That earlier clarification is stale. Tell me the full request again.'

    def render_birthday_memory_confirmation(self, *, date_label: str) -> str:
        return (
            f"Got it — tomorrow's your birthday, {date_label}. "
            "Want me to remember that for next year? "
            "I can wish you tomorrow without saving it permanently."
        )

    def render_birthday_wish_reply(self) -> str:
        return (
            "Haha got it — I won't treat it like a celebration. "
            "I'll still wish you happy birthday tomorrow."
        )

    def render_memory_confirmation(self, *, subject: str) -> str:
        return (
            f"Got it — {subject}. "
            "If you want me to remember that permanently, say so explicitly."
        )

    def render_role_contamination_guard(self) -> str:
        return 'I ignored the transcript label. Reply in plain words.'

    def render_no_silent_reply(self, *, user_text: str) -> str:
        lowered = user_text.casefold()
        if 'birthday' in lowered or 'bday' in lowered:
            return 'Got it — I can keep this light. Tell me what you want me to do with it.'
        return "I'm here. Tell me what you want me to do with that."

    def compress_reply(self, *, text: str) -> str:
        cleaned = (text or '').strip()
        if not cleaned:
            return ''
        cleaned = re.sub(
            r'^(?:user|assistant|nexus|system|the ai)\s*:\s*',
            '',
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace('💕', '').replace('❤️', '').strip()
        spouse_match = re.match(
            r'^(?:your|my)\s+(wife|husband|partner)\s+is\s+([^.?!]+)\.\s+'
            r'anything you want me to do for (her|him)\??$',
            cleaned,
            flags=re.IGNORECASE,
        )
        if spouse_match is not None:
            name = spouse_match.group(2).strip().title()
            pronoun = spouse_match.group(3).strip().lower()
            verb = 'tell her' if pronoun == 'her' else 'tell him'
            return f'Got it — {name}. What should I {verb}?'
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _render_send_recipient_clarification(self, payload: HumanConfirmationStyleInput) -> str:
        fallback = payload.fallback_result or {}
        recipient_label = str(
            fallback.get('recipient_label')
            or payload.resolved_slots.get('negated_recipient_label')
            or ''
        ).strip()
        reason = str(
            fallback.get('reason')
            or payload.resolved_slots.get('negated_recipient_reason')
            or ''
        ).strip()
        if reason == 'other_one' and recipient_label:
            return (
                f"You said not {recipient_label} — who should I send it to?"
            )
        if recipient_label:
            return f"You said not {recipient_label} — who should I send it to?"
        return "I can't send this yet because I still don't know who should receive it."

    def _render_safe_reminder_detail(self, fallback: dict[str, Any]) -> str | None:
        time_label = str(fallback.get('time_label') or '').strip()
        target_name = str(fallback.get('target_name') or '').strip()
        issue_summary = str(fallback.get('issue_summary') or '').strip()
        unit_reference = str(fallback.get('unit_reference') or '').strip()
        if not time_label or not target_name or not issue_summary or not unit_reference:
            return None
        return (
            f'{time_label} follow-up for {target_name} '
            f'about {issue_summary} in {unit_reference}'
        )
