from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Callable, Literal


RiskLevel = Literal['low', 'medium', 'high']
RecoveryOutcome = Literal['auto_resolve', 'soft_confirm', 'hard_clarify']

logger = logging.getLogger(__name__)

_WEEKDAY_NAMES = (
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
    'today',
    'tomorrow',
    'tonight',
)
_OUTBOUND_VERBS = ('tell', 'send', 'message', 'text', 'ask', 'whatsapp')
_HIGH_RISK_HINTS = (
    'delete',
    'remove',
    'cancel',
    'disconnect',
    'write',
    'run',
    'terminal',
    'command',
    'pay',
    'charge',
    'record payment',
)
_MEDIUM_RISK_HINTS = (
    'remind me',
    'reminder',
    'follow up',
    'schedule',
    'task',
)
_GENERIC_OBJECT_PATTERNS = (
    'the thing',
    'that thing',
    'this thing',
    'it',
    'that',
    'this',
)
_RECIPIENT_NEGATION_RE = re.compile(
    r'\b(?:nah|no|wait no|actually no)?\s*(?:not her|not him|not them|not that one|the other one)\b',
    re.IGNORECASE,
)
_INLINE_TIME_CORRECTION_RE = re.compile(
    r'\b(?:wait\s+)?(?:actually\s+)?no\s+'
    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight)\b',
    re.IGNORECASE,
)
_CORRECTION_PREFIX_RE = re.compile(
    r'^(?:wait\s+)?(?:actually\s+)?(?:no\b|nah\b|nope\b)(?:\s+not)?\s*',
    re.IGNORECASE,
)
_CORRECTION_ONLY_PREFIX_RE = re.compile(
    r'^(?:wait\s+)?(?:actually\s+)?(?:no\b|nah\b|instead\b)',
    re.IGNORECASE,
)
_SHORTHAND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bu\b', re.IGNORECASE), 'you'),
    (re.compile(r'\bur\b', re.IGNORECASE), 'your'),
    (re.compile(r'\bmsg\b', re.IGNORECASE), 'message'),
    (re.compile(r'\bpls\b|\bplz\b', re.IGNORECASE), 'please'),
    (re.compile(r'\btmrrw\b|\btmrw\b|\btmr\b', re.IGNORECASE), 'tomorrow'),
    (re.compile(r'\bill\b', re.IGNORECASE), 'i will'),
    (re.compile(r'\bnah\b', re.IGNORECASE), 'no'),
)
_STOP_NAME_TOKENS = {
    'a',
    'an',
    'me',
    'i',
    'i’ll',
    "i'll",
    'i',
    'will',
    'my',
    'to',
    'our',
    'if',
    'about',
    'the',
    'that',
    'this',
    'tomorrow',
    'today',
    'tonight',
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
    'please',
    'no',
    'not',
    'wait',
    'actually',
    'message',
    'reminder',
    'update',
    'reply',
    'replied',
    'someone',
    'somebody',
    'anyone',
    'anybody',
    'person',
    'her',
    'him',
    'he',
    'she',
    'them',
    'tmrrw',
    'tmrw',
    'your',
}
_TIME_ONLY_CONNECTOR_TOKENS = {
    'at',
    'around',
    'by',
    'for',
    'on',
    'tomorrow',
    'today',
    'tonight',
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
    'am',
    'pm',
}
_TIME_TOKEN_RE = re.compile(r'^\d{1,2}(?::\d{2})?(?:am|pm)?$', re.IGNORECASE)
_RECOVERY_STATE_TTL = timedelta(hours=2)
_PENDING_DRAFT_TTL = timedelta(minutes=30)
_CONTEXT_SOURCE_PRIORITY = {
    'global': 1,
    'recent_turn': 2,
    'pending_draft': 3,
}
_CONTACT_CONTEXT_KEYS = (
    'pending_draft',
    'last_recipient',
    'last_recipient_source',
    'last_contact_id',
    'last_channel',
    'last_message_body',
    'last_message_body_source',
    'last_task',
    'recipient_candidates',
    'recipient_alias_query',
    'last_ambiguity',
    'last_confirmation_options',
    'negated_recipient_label',
    'negated_recipient_reason',
    'recipient_negated_unresolved',
)


@dataclass(slots=True)
class RecoveryResult:
    raw_text: str
    normalized_text: str
    recovered_text: str
    confidence: float
    risk_level: RiskLevel
    outcome: RecoveryOutcome = 'auto_resolve'
    clarification_text: str | None = None
    missing_slot: str | None = None
    selected_clarification_option: str | None = None
    resolved_slots: dict[str, Any] = field(default_factory=dict)
    corrections_applied: list[str] = field(default_factory=list)
    suppress_vague_clarification: bool = False
    context_updates: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_clarification(self) -> bool:
        return self.outcome != 'auto_resolve'


def build_recovery_prompt_block(result: RecoveryResult) -> str | None:
    if (
        result.raw_text == result.recovered_text
        and not result.resolved_slots
        and not result.corrections_applied
    ):
        return None

    lines = [
        '## Conversational recovery',
        f'Original user wording: {result.raw_text}',
        f'Use this recovered interpretation as the current user intent: {result.recovered_text}',
    ]
    if result.resolved_slots:
        slots_text = ', '.join(
            f'{key}={value}'
            for key, value in sorted(result.resolved_slots.items())
            if value not in (None, '', [], {})
        )
        if slots_text:
            lines.append(f'Resolved slots: {slots_text}')
    if result.corrections_applied:
        lines.append(
            'Latest explicit corrections override earlier wording for: '
            + ', '.join(result.corrections_applied)
        )
    return '\n'.join(lines)


class ConversationalRecoveryLayer:
    def recover(
        self,
        *,
        text: str,
        context: dict[str, Any] | None = None,
        recent_turns: list[Any] | None = None,
        resolve_contact_alias: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> RecoveryResult:
        raw_text = ' '.join((text or '').strip().split())
        normalized_text = _normalize_text(raw_text)
        risk_level = _classify_risk(normalized_text)
        result = RecoveryResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            recovered_text=normalized_text,
            confidence=0.55 if normalized_text else 0.0,
            risk_level=risk_level,
        )

        state = _merge_state_from_context_and_turns(context=context or {}, recent_turns=recent_turns or [])
        _reset_incompatible_context_for_turn(result, state=state)
        result.context_updates.update({
            'last_raw_text': raw_text,
            'last_normalized_text': normalized_text,
        })

        rental_result = _recover_rental_status(result, state=state)
        if rental_result is not None:
            _finalize_context_updates(rental_result, state)
            return rental_result

        _apply_explicit_recipient_capture(result)
        _apply_contact_resolution(
            result,
            state=state,
            resolve_contact_alias=resolve_contact_alias,
        )
        _apply_pending_draft_continuity(
            result,
            state=state,
            resolve_contact_alias=resolve_contact_alias,
        )
        _apply_pronoun_resolution(result, state=state)
        _apply_correction_memory(result, state=state)
        _apply_inline_time_correction(result)
        _apply_generic_object_resolution(result, state=state)
        _canonicalize_outbound_message(result, state=state)
        _apply_negated_recipient_block(result, state=state)

        if result.outcome == 'auto_resolve' and result.risk_level == 'high':
            recipient = str(result.resolved_slots.get('recipient') or '').strip()
            has_outbound_intent = _looks_like_outbound_message(result.normalized_text)
            if has_outbound_intent and _contains_recipient_pronoun(result.normalized_text.casefold()) and not recipient:
                result.outcome = 'hard_clarify'
                result.confidence = min(result.confidence, 0.3)
                body = _extract_outbound_message_body(result.normalized_text, '')
                if body and body.casefold().startswith('check'):
                    result.outcome = 'auto_resolve'
                    result.confidence = 0.55
                else:
                    result.missing_slot = 'recipient'
                    result.clarification_text = 'Who should I message?'

        _capture_turn_state(result)
        _capture_pending_draft(result, state=state)
        _finalize_context_updates(result, state)
        if result.context_updates.get('used_context'):
            logger.info(
                'recovery_context_applied',
                extra={
                    'context_source': result.context_updates.get('context_source'),
                    'confidence': result.context_updates.get('confidence', result.confidence),
                    'risk_level': result.context_updates.get('risk_level', result.risk_level),
                },
            )
        return result


def _normalize_text(text: str) -> str:
    normalized = ' '.join((text or '').strip().split())
    for pattern, replacement in _SHORTHAND_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r'[?!.]{2,}', '.', normalized)
    normalized = re.sub(r'\s+([,.;!?])', r'\1', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    normalized = re.sub(r'\b([A-Za-z])$', '', normalized).strip()
    normalized = re.sub(r'\bm$', '', normalized, flags=re.IGNORECASE).strip()
    return normalized


def _classify_risk(text: str) -> RiskLevel:
    lowered = text.casefold()
    if any(f'{verb} ' in lowered for verb in _OUTBOUND_VERBS):
        return 'high'
    if any(hint in lowered for hint in _HIGH_RISK_HINTS):
        return 'high'
    if any(hint in lowered for hint in _MEDIUM_RISK_HINTS):
        return 'medium'
    return 'low'


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _future_iso(delta: timedelta) -> str:
    return (_utc_now() + delta).isoformat()


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_expired(value: Any, *, now: datetime | None = None) -> bool:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return bool(value)
    return (now or _utc_now()) > parsed


def _recover_rental_status(result: RecoveryResult, *, state: dict[str, Any]) -> RecoveryResult | None:
    lowered = result.normalized_text.casefold()
    subject = _recover_rental_status_subject(lowered, state=state)
    if subject is None:
        return None
    result.recovered_text = f'check rental update status for {subject}'
    result.resolved_slots.update({
        'action_kind': 'rental_status_check',
        'rental_subject': subject,
    })
    result.confidence = 0.84
    result.suppress_vague_clarification = True
    result.context_updates.update({
        'last_action_kind': 'rental_status_check',
        'last_rental_subject': subject,
        'last_task': subject,
        'pending_menu_kind': 'rental_status',
    })
    if lowered in {'1', '1.', 'option 1'}:
        result.selected_clarification_option = 'rental_status_check'
    return result


def _recover_rental_status_subject(lowered: str, *, state: dict[str, Any]) -> str | None:
    if lowered in {'1', '1.', 'option 1'}:
        if state.get('last_action_kind') == 'rental_status_check':
            subject = str(state.get('last_rental_subject') or '').strip()
            return subject or None
        return None

    patterns = (
        r'^(?:did you update|did you do)\s+(?:my|the)?\s*(?:(\d+)\s+)?rentals?\b',
        r'^were\s+(?:my|the)?\s*(?:(\d+)\s+)?rentals?\s+updated\b',
        r'^check\s+(?:if|whether)\s+(?:my|the)?\s*(?:(\d+)\s+)?rentals?\s+were\s+updated\b',
    )
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match is None:
            continue
        count = match.group(1)
        return f'your {count} rental records' if count else 'your rental records'
    return None


def _apply_explicit_recipient_capture(result: RecoveryResult) -> None:
    candidate = _extract_explicit_contact_candidate(result.normalized_text)
    if not candidate:
        return
    if candidate.casefold() in _STOP_NAME_TOKENS:
        return
    result.resolved_slots.setdefault('recipient', candidate)
    result.context_updates['last_recipient'] = candidate
    result.context_updates['last_recipient_source'] = 'explicit'
    result.context_updates['recipient_explicit'] = True
    result.confidence = max(result.confidence, 0.72 if result.risk_level == 'high' else 0.66)


def _apply_pending_draft_continuity(
    result: RecoveryResult,
    *,
    state: dict[str, Any],
    resolve_contact_alias: Callable[[str], dict[str, Any] | None] | None,
) -> None:
    draft = _coerce_pending_draft(state.get('pending_draft'))
    if not draft:
        return
    draft_action_kind = str(draft.get('action_kind') or '').strip()
    state_action_kind = str(state.get('last_action_kind') or '').strip()
    if draft_action_kind and state_action_kind and draft_action_kind != state_action_kind:
        return

    if _is_time_only_followup(result.normalized_text):
        updated = _merge_time_only_followup_into_draft(draft, result.normalized_text)
        _hydrate_result_from_draft(
            result,
            draft=updated,
            confidence=max(result.confidence, 0.86),
        )
        result.suppress_vague_clarification = True
        if updated.get('date') != draft.get('date'):
            result.corrections_applied.append('date')
        if updated.get('time') != draft.get('time'):
            result.corrections_applied.append('time')
        result.context_updates['pending_draft'] = updated
        _record_context_usage(
            result,
            source='pending_draft',
            confidence=max(float(updated.get('confidence') or 0.0), result.confidence),
        )
        return

    replacement = _extract_recipient_correction_candidate(
        result.normalized_text,
        current_recipient=str(draft.get('recipient') or '').strip(),
    )
    if replacement:
        canonical = _resolve_alias_or_keep(replacement, resolve_contact_alias)
        updated = dict(draft)
        updated['recipient'] = canonical
        updated['confidence'] = max(float(updated.get('confidence') or 0.0), 0.84)
        updated['corrections'] = list(updated.get('corrections') or []) + ['recipient']
        updated['source_turns'] = _append_source_turn(updated.get('source_turns'), result.raw_text)
        updated['expires_at'] = _future_iso(_PENDING_DRAFT_TTL)
        _hydrate_result_from_draft(
            result,
            draft=updated,
            confidence=max(result.confidence, 0.84),
        )
        result.corrections_applied.append('recipient')
        result.context_updates['last_recipient'] = canonical
        result.context_updates['pending_draft'] = updated
        _record_context_usage(
            result,
            source='pending_draft',
            confidence=max(float(updated.get('confidence') or 0.0), result.confidence),
        )


def _coerce_pending_draft(payload: Any, *, now: datetime | None = None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if _is_expired(payload.get('expires_at'), now=now):
        return None
    recipient = str(payload.get('recipient') or '').strip()
    draft_kind = str(payload.get('draft_kind') or '').strip()
    if not recipient or not draft_kind:
        return None
    draft = dict(payload)
    draft['source_turn_ids'] = [
        str(item).strip()
        for item in draft.get('source_turn_ids') or []
        if str(item).strip()
    ][-6:]
    draft['source_turns'] = [
        str(item).strip()
        for item in draft.get('source_turns') or []
        if str(item).strip()
    ][-4:]
    return draft


def _is_time_only_followup(text: str) -> bool:
    lowered = _CORRECTION_PREFIX_RE.sub('', (text or '').strip().casefold())
    if not lowered:
        return False
    if _extract_explicit_contact_candidate(lowered):
        return False
    if any(f'{verb} ' in lowered for verb in _OUTBOUND_VERBS):
        return False
    cleaned = re.sub(r'[:,.]', ' ', lowered)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return False
    seen_time_or_date = False
    for token in tokens:
        if token in _TIME_ONLY_CONNECTOR_TOKENS:
            if token in _WEEKDAY_NAMES or token in {'today', 'tomorrow', 'tonight'}:
                seen_time_or_date = True
            continue
        if _TIME_TOKEN_RE.fullmatch(token):
            seen_time_or_date = True
            continue
        return False
    return seen_time_or_date


def _merge_time_only_followup_into_draft(
    draft: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    updated = dict(draft)
    date_label = _extract_latest_weekday_or_relative(text)
    time_label = _extract_time_label(text)
    if date_label:
        updated['date'] = date_label
    if time_label:
        updated['time'] = time_label
    updated['confidence'] = max(float(updated.get('confidence') or 0.0), 0.86)
    updated['source_turns'] = _append_source_turn(updated.get('source_turns'), text)
    updated['expires_at'] = _future_iso(_PENDING_DRAFT_TTL)
    return updated


def _extract_recipient_correction_candidate(
    text: str,
    *,
    current_recipient: str,
) -> str | None:
    lowered = text.casefold()
    if not _CORRECTION_ONLY_PREFIX_RE.search(lowered):
        return None
    if current_recipient and current_recipient.casefold() not in lowered:
        return None
    stripped = _CORRECTION_PREFIX_RE.sub('', text, count=1).strip(' ,.')
    stripped = re.sub(
        rf'^(?:not\s+)?{re.escape(current_recipient)}(?:\s*,\s*|\s+)',
        '',
        stripped,
        count=1,
        flags=re.IGNORECASE,
    ).strip(' ,.')
    candidate = _consume_name_tokens(stripped)
    if not candidate or candidate.casefold() in _STOP_NAME_TOKENS:
        return None
    return candidate


def _resolve_alias_or_keep(
    candidate: str,
    resolve_contact_alias: Callable[[str], dict[str, Any] | None] | None,
) -> str:
    if resolve_contact_alias is None:
        return candidate
    payload = resolve_contact_alias(candidate)
    if not isinstance(payload, dict) or payload.get('ok') is not True:
        return candidate
    if str(payload.get('match') or '').strip().lower() != 'unique':
        return candidate
    alias_used = str(payload.get('alias_used') or '').strip()
    if alias_used:
        return alias_used
    contact = payload.get('contact')
    if isinstance(contact, dict):
        aliases = contact.get('aliases')
        if isinstance(aliases, list) and aliases:
            canonical = str(aliases[0]).strip()
            if canonical:
                return canonical
    return candidate


def _hydrate_result_from_draft(
    result: RecoveryResult,
    *,
    draft: dict[str, Any],
    confidence: float,
) -> None:
    recipient = str(draft.get('recipient') or '').strip()
    draft_kind = str(draft.get('draft_kind') or '').strip()
    prompt = _render_pending_draft_prompt(draft)
    if not prompt:
        return
    result.recovered_text = prompt
    result.resolved_slots['recipient'] = recipient
    result.confidence = confidence
    result.context_updates['last_recipient'] = recipient
    result.context_updates['pending_draft'] = dict(draft)
    if draft.get('contact_id') is not None:
        result.context_updates['last_contact_id'] = draft.get('contact_id')
    channel = str(draft.get('channel') or '').strip()
    if channel:
        result.context_updates['last_channel'] = channel
    if draft_kind == 'reminder_message':
        result.risk_level = 'medium'
        result.resolved_slots['action_kind'] = 'reminder_draft'
        topic = str(draft.get('topic') or '').strip()
        if topic:
            result.resolved_slots['topic'] = topic
            result.context_updates['last_task'] = topic
        if draft.get('date'):
            result.resolved_slots['date_label'] = draft.get('date')
        if draft.get('time'):
            result.resolved_slots['time_label'] = draft.get('time')
    elif draft_kind == 'outbound_message':
        result.risk_level = 'high'
        body = str(draft.get('message_body') or '').strip()
        if body:
            result.resolved_slots['message_body'] = body
            result.context_updates['last_message_body'] = body
            result.context_updates['last_task'] = body


def _merge_state_from_context_and_turns(*, context: dict[str, Any], recent_turns: list[Any]) -> dict[str, Any]:
    now = _utc_now()
    state = _sanitize_recovery_state(dict(context.get('recovery_state') or {}), now=now)
    recent_turn_ids: list[str] = []
    for turn in reversed(recent_turns):
        content = getattr(turn, 'content', '') or ''
        turn_id = str(getattr(turn, 'turn_id', '') or '').strip()
        if turn_id:
            recent_turn_ids.append(turn_id)
        normalized = _normalize_text(content)
        if not normalized:
            continue
        if not state.get('last_recipient'):
            candidate = _extract_explicit_contact_candidate(normalized)
            if candidate and candidate.casefold() not in _STOP_NAME_TOKENS:
                state['last_recipient'] = candidate
                state['last_recipient_source'] = 'recent_turn'
                state['confidence'] = max(float(state.get('confidence') or 0.0), 0.74)
        if not state.get('last_unit'):
            unit = _extract_latest_unit_reference(normalized)
            if unit:
                state['last_unit'] = unit
        if not state.get('last_time_label'):
            time_label = _extract_latest_weekday_or_relative(normalized)
            if time_label:
                state['last_time_label'] = time_label
        if not state.get('last_time'):
            time_value = _extract_time_label(normalized)
            if time_value:
                state['last_time'] = time_value
        if not state.get('last_date'):
            date_value = _extract_latest_weekday_or_relative(normalized)
            if date_value:
                state['last_date'] = date_value
        if not state.get('last_action_kind'):
            action_kind = _infer_action_kind(normalized)
            if action_kind:
                state['last_action_kind'] = action_kind
                state['last_action_kind_source'] = 'recent_turn'
                state['risk_level'] = _classify_risk(normalized)
        if not state.get('last_message_body') and _looks_like_outbound_message(normalized):
            recipient = str(state.get('last_recipient') or '').strip()
            body = _extract_outbound_message_body(normalized, recipient)
            if body:
                state['last_message_body'] = body
                state['last_message_body_source'] = 'recent_turn'
        if not state.get('last_channel'):
            channel = _extract_channel_hint(normalized)
            if channel:
                state['last_channel'] = channel
        if not state.get('last_task'):
            if 'follow up' in normalized or 'remind me' in normalized:
                topic = _extract_reminder_topic(normalized, str(state.get('last_recipient') or '').strip())
                if topic:
                    state['last_task'] = topic
    if recent_turn_ids:
        existing_ids = [
            str(item).strip()
            for item in state.get('source_turn_ids') or []
            if str(item).strip()
        ]
        state['source_turn_ids'] = (existing_ids + list(reversed(recent_turn_ids)))[-6:]
    return state


def _apply_contact_resolution(
    result: RecoveryResult,
    *,
    state: dict[str, Any],
    resolve_contact_alias: Callable[[str], dict[str, Any] | None] | None,
) -> None:
    candidate = _extract_explicit_contact_candidate(result.normalized_text)
    if not candidate or resolve_contact_alias is None:
        return
    payload = resolve_contact_alias(candidate)
    if not isinstance(payload, dict) or payload.get('ok') is not True:
        return

    match_type = str(payload.get('match') or '').strip().lower()
    if match_type == 'unique':
        canonical = str(payload.get('alias_used') or '').strip()
        if not canonical:
            contact = payload.get('contact')
            if isinstance(contact, dict):
                aliases = contact.get('aliases')
                if isinstance(aliases, list) and aliases:
                    canonical = str(aliases[0]).strip()
        if not canonical:
            return
        result.recovered_text = _replace_first_case_insensitive(result.recovered_text, candidate, canonical)
        result.resolved_slots['recipient'] = canonical
        result.confidence = max(result.confidence, 0.87 if result.risk_level == 'high' else 0.8)
        result.context_updates.update({
            'last_recipient': canonical,
            'last_recipient_source': 'explicit',
            'recipient_candidates': [canonical],
        })
        contact = payload.get('contact')
        if isinstance(contact, dict):
            if contact.get('id') is not None:
                result.context_updates['last_contact_id'] = contact.get('id')
        if _looks_like_outbound_message(result.normalized_text):
            body = _extract_outbound_message_body(result.recovered_text, canonical)
            if body:
                result.context_updates['last_message_body'] = body
                result.resolved_slots.setdefault('message_body', body)
    elif match_type == 'ambiguous':
        labels = _serialize_candidate_labels(payload.get('candidates'))
        if not labels:
            return
        result.context_updates['recipient_candidates'] = labels
        result.context_updates['recipient_alias_query'] = candidate
        result.context_updates['last_confirmation_options'] = labels
        if result.risk_level == 'high':
            result.outcome = 'soft_confirm'
            result.confidence = 0.42
            result.missing_slot = 'recipient'
            result.clarification_text = _render_people_clarification(labels)


def _apply_pronoun_resolution(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    lowered = result.normalized_text.casefold()
    pending_draft = _coerce_pending_draft(state.get('pending_draft'))
    current_action_kind = _infer_action_kind(result.normalized_text) or str(result.resolved_slots.get('action_kind') or '').strip()
    last_action_kind = str(state.get('last_action_kind') or '').strip()
    if 'other one' in lowered:
        candidates = [str(item).strip() for item in state.get('recipient_candidates') or [] if str(item).strip()]
        previous = str(
            (pending_draft or {}).get('recipient')
            or state.get('last_recipient')
            or ''
        ).strip()
        if len(candidates) == 2 and previous in candidates:
            resolved = candidates[1] if candidates[0] == previous else candidates[0]
            result.recovered_text = _replace_other_one_reference(result.recovered_text, resolved)
            result.resolved_slots['recipient'] = resolved
            result.corrections_applied.append('recipient')
            result.confidence = max(result.confidence, 0.88)
            result.context_updates['last_recipient'] = resolved
            _record_context_usage(
                result,
                source='pending_draft' if pending_draft else str(state.get('last_recipient_source') or 'global'),
                confidence=max(float((pending_draft or {}).get('confidence') or 0.0), float(state.get('confidence') or 0.0), result.confidence),
            )
            return
        if result.risk_level == 'high':
            result.outcome = 'hard_clarify'
            result.confidence = min(result.confidence, 0.3)
            result.missing_slot = 'recipient'
            if previous:
                result.clarification_text = f'You said not {previous} — who should I send it to?'
            else:
                result.clarification_text = 'Who should I send it to?'
            return

    if not _contains_recipient_pronoun(lowered):
        return
    previous, source, source_confidence = _get_context_recipient_candidate(state)
    if not previous:
        return
    if result.risk_level == 'high' and source != 'pending_draft':
        same_chain = bool(current_action_kind and last_action_kind and current_action_kind == last_action_kind)
        if not same_chain or source_confidence < 0.82:
            result.context_updates['weak_context_blocked'] = True
            return
    result.recovered_text = _replace_pronoun_target(result.recovered_text, previous)
    result.resolved_slots['recipient'] = previous
    if current_action_kind == 'contact_check':
        result.recovered_text = f'check if {previous} replied'
    result.confidence = max(result.confidence, 0.9 if source == 'pending_draft' else 0.78)
    result.context_updates['last_recipient'] = previous
    result.context_updates['last_recipient_source'] = source
    _record_context_usage(
        result,
        source=source,
        confidence=max(source_confidence, result.confidence),
    )


def _apply_correction_memory(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    lowered = result.normalized_text.casefold()
    if not _CORRECTION_ONLY_PREFIX_RE.search(lowered):
        return

    latest_time = _extract_latest_weekday_or_relative(lowered)
    if latest_time and state.get('last_action_kind'):
        result.recovered_text = f'use {latest_time} instead'
        result.resolved_slots['time_label'] = latest_time
        result.corrections_applied.append('time_label')
        result.confidence = max(result.confidence, 0.84)
        result.context_updates['last_time_label'] = latest_time
        _record_context_usage(
            result,
            source='global',
            confidence=max(float(state.get('confidence') or 0.0), result.confidence),
        )
        return

    latest_unit = _extract_latest_unit_reference(lowered)
    if latest_unit and state.get('last_action_kind'):
        result.recovered_text = f'use {latest_unit} instead'
        result.resolved_slots['unit_reference'] = latest_unit
        result.corrections_applied.append('unit_reference')
        result.confidence = max(result.confidence, 0.84)
        result.context_updates['last_unit'] = latest_unit
        _record_context_usage(
            result,
            source='global',
            confidence=max(float(state.get('confidence') or 0.0), result.confidence),
        )


def _apply_inline_time_correction(result: RecoveryResult) -> None:
    matches = list(_INLINE_TIME_CORRECTION_RE.finditer(result.normalized_text))
    if not matches:
        return
    latest = matches[-1].group(1)
    replacement = latest.title() if latest not in {'today', 'tomorrow', 'tonight'} else latest
    result.resolved_slots['time_label'] = replacement
    if 'time_label' not in result.corrections_applied:
        result.corrections_applied.append('time_label')
    result.context_updates['last_time_label'] = replacement
    result.confidence = max(result.confidence, 0.8)


def _apply_generic_object_resolution(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    if result.resolved_slots.get('action_kind') == 'reminder_draft':
        return
    lowered = result.recovered_text.casefold()
    if not _looks_like_outbound_message(lowered):
        return
    pending_draft = _coerce_pending_draft(state.get('pending_draft'))
    recipient = str(
        result.resolved_slots.get('recipient')
        or (pending_draft or {}).get('recipient')
        or state.get('last_recipient')
        or ''
    ).strip()
    message_body = _extract_outbound_message_body(result.recovered_text, recipient)
    if not message_body:
        return
    body_lower = message_body.casefold()
    if body_lower not in _GENERIC_OBJECT_PATTERNS:
        return

    previous_body = str(
        (pending_draft or {}).get('message_body')
        or state.get('last_message_body')
        or ''
    ).strip()
    if previous_body:
        result.recovered_text = result.recovered_text.replace(message_body, previous_body, 1)
        result.resolved_slots.setdefault('message_body', previous_body)
        result.confidence = max(result.confidence, 0.76)
        source = 'pending_draft' if pending_draft and pending_draft.get('message_body') else str(state.get('last_message_body_source') or 'global')
        _record_context_usage(
            result,
            source=source,
            confidence=max(float((pending_draft or {}).get('confidence') or 0.0), float(state.get('confidence') or 0.0), result.confidence),
        )
        return

    if result.risk_level == 'high':
        result.outcome = 'hard_clarify'
        result.confidence = min(result.confidence, 0.38)
        result.missing_slot = 'message_body' if recipient else 'recipient_and_message_body'
        if recipient:
            result.clarification_text = f'You mean send {recipient} what exactly?'
        else:
            result.clarification_text = 'What exactly should I send, and to whom?'


def _capture_turn_state(result: RecoveryResult) -> None:
    action_kind = result.context_updates.get('last_action_kind')
    if action_kind is None:
        action_kind = (
            str(result.resolved_slots.get('action_kind') or '').strip()
            or _infer_action_kind(result.recovered_text)
            or _infer_action_kind(result.normalized_text)
        )
        if action_kind == 'reminder_draft':
            action_kind = 'follow_up'
    if action_kind:
        result.context_updates['last_action_kind'] = action_kind

    if 'last_time_label' not in result.context_updates:
        time_label = _extract_time_label(result.recovered_text) or _extract_latest_weekday_or_relative(result.recovered_text)
        if time_label:
            result.context_updates['last_time_label'] = time_label

    if 'last_unit' not in result.context_updates:
        unit = _extract_latest_unit_reference(result.recovered_text)
        if unit:
            result.context_updates['last_unit'] = unit

    if _looks_like_outbound_message(result.recovered_text) and 'last_message_body' not in result.context_updates:
        recipient = str(result.resolved_slots.get('recipient') or result.context_updates.get('last_recipient') or '').strip()
        body = _extract_outbound_message_body(result.recovered_text, recipient)
        if body:
            result.context_updates['last_message_body'] = body
            result.context_updates['last_task'] = body

    channel = _extract_channel_hint(result.recovered_text) or _extract_channel_hint(result.normalized_text)
    if channel:
        result.context_updates['last_channel'] = channel

    if 'last_date' not in result.context_updates:
        date_label = _extract_latest_weekday_or_relative(result.recovered_text)
        if date_label:
            result.context_updates['last_date'] = date_label
    if 'last_time' not in result.context_updates:
        time_value = _extract_time_label(result.recovered_text)
        if time_value:
            result.context_updates['last_time'] = time_value

    topic = str(result.resolved_slots.get('topic') or '').strip()
    if topic:
        result.context_updates['last_task'] = topic
    elif result.resolved_slots.get('rental_subject'):
        result.context_updates['last_task'] = result.resolved_slots['rental_subject']

    if result.needs_clarification:
        options = [
            str(item).strip()
            for item in result.context_updates.get('recipient_candidates') or []
            if str(item).strip()
        ]
        result.context_updates['last_confirmation_options'] = options
        result.context_updates['last_ambiguity'] = {
            'missing_slot': result.missing_slot or '',
            'question': result.clarification_text or '',
            'options': options,
        }
    else:
        result.context_updates['last_ambiguity'] = None
        result.context_updates['last_confirmation_options'] = None

    result.context_updates['last_action_result'] = result.outcome


def _capture_pending_draft(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    if 'pending_draft' in result.context_updates:
        return
    recipient = str(result.resolved_slots.get('recipient') or '').strip()
    if not recipient:
        return

    lowered = result.recovered_text.casefold()
    action_kind = str(result.context_updates.get('last_action_kind') or state.get('last_action_kind') or '').strip()

    if 'remind me' in lowered or 'follow up' in lowered:
        topic = _extract_reminder_topic(result.recovered_text, recipient)
        draft = {
            'draft_kind': 'reminder_message',
            'action_kind': action_kind or 'follow_up',
            'recipient': recipient,
            'contact_id': result.context_updates.get('last_contact_id') or state.get('last_contact_id'),
            'topic': topic,
            'channel': 'whatsapp' if 'whatsapp' in lowered else 'message',
            'time': _extract_time_label(result.recovered_text),
            'date': _extract_latest_weekday_or_relative(result.recovered_text),
            'corrections': list(result.corrections_applied),
            'source_turns': _append_source_turn(
                (_coerce_pending_draft(state.get('pending_draft')) or {}).get('source_turns'),
                result.raw_text,
            ),
            'source_turn_ids': _merge_source_turn_ids(
                (_coerce_pending_draft(state.get('pending_draft')) or {}).get('source_turn_ids'),
                state.get('source_turn_ids'),
            ),
            'confidence': max(result.confidence, 0.8),
            'risk_level': result.risk_level,
            'expires_at': _future_iso(_PENDING_DRAFT_TTL),
        }
        result.context_updates['pending_draft'] = draft
        return

    if _looks_like_outbound_message(result.recovered_text):
        body = str(result.resolved_slots.get('message_body') or '').strip()
        if not body:
            body = _extract_outbound_message_body(result.recovered_text, recipient) or ''
        if not body:
            return
        draft = {
            'draft_kind': 'outbound_message',
            'action_kind': action_kind or 'outbound_message',
            'recipient': recipient,
            'contact_id': result.context_updates.get('last_contact_id') or state.get('last_contact_id'),
            'message_body': body,
            'channel': 'whatsapp' if 'whatsapp' in lowered else 'message',
            'time': _extract_time_label(result.recovered_text),
            'date': _extract_latest_weekday_or_relative(result.recovered_text),
            'corrections': list(result.corrections_applied),
            'source_turns': _append_source_turn(
                (_coerce_pending_draft(state.get('pending_draft')) or {}).get('source_turns'),
                result.raw_text,
            ),
            'source_turn_ids': _merge_source_turn_ids(
                (_coerce_pending_draft(state.get('pending_draft')) or {}).get('source_turn_ids'),
                state.get('source_turn_ids'),
            ),
            'confidence': max(result.confidence, 0.8),
            'risk_level': result.risk_level,
            'expires_at': _future_iso(_PENDING_DRAFT_TTL),
        }
        result.context_updates['pending_draft'] = draft


def _apply_negated_recipient_block(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    if not _looks_like_outbound_message(result.normalized_text):
        return
    if 'recipient' in result.corrections_applied:
        return
    if _RECIPIENT_NEGATION_RE.search(result.normalized_text) is None:
        return

    pending_draft = _coerce_pending_draft(state.get('pending_draft'))
    rejected = str(
        result.resolved_slots.get('recipient')
        or (pending_draft or {}).get('recipient')
        or result.context_updates.get('last_recipient')
        or state.get('last_recipient')
        or ''
    ).strip()
    body = str(result.resolved_slots.get('message_body') or '').strip()
    if body:
        result.context_updates['pending_message_body'] = body
    if result.resolved_slots.get('time_label'):
        result.context_updates['pending_time_label'] = result.resolved_slots['time_label']

    result.outcome = 'hard_clarify'
    result.confidence = min(result.confidence, 0.2)
    result.missing_slot = 'recipient'
    result.context_updates['recipient_negated_unresolved'] = True
    result.context_updates['negated_recipient_label'] = rejected
    result.context_updates['negated_recipient_reason'] = (
        'other_one' if 'other one' in result.normalized_text.casefold() else 'recipient_negated'
    )
    result.resolved_slots['negated_recipient_label'] = rejected
    result.resolved_slots['negated_recipient_reason'] = result.context_updates['negated_recipient_reason']
    if rejected:
        result.clarification_text = f'You said not {rejected} — who should I send it to?'
    else:
        result.clarification_text = 'Who should I send it to?'


def _canonicalize_outbound_message(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    normalized_lowered = result.normalized_text.casefold()
    if (
        'remind me' in normalized_lowered
        or 'follow up' in normalized_lowered
        or result.resolved_slots.get('action_kind') == 'reminder_draft'
    ):
        return
    if not _looks_like_outbound_message(result.recovered_text):
        return
    pending_draft = _coerce_pending_draft(state.get('pending_draft'))
    recipient = str(
        result.resolved_slots.get('recipient')
        or (pending_draft or {}).get('recipient')
        or state.get('last_recipient')
        or ''
    ).strip()
    if not recipient:
        return
    body = _extract_outbound_message_body(result.recovered_text, recipient)
    if not body:
        return
    body = _normalize_outbound_message_body(
        body,
        preferred_time=str(result.resolved_slots.get('time_label') or '').strip() or None,
    )
    if not body:
        return
    result.recovered_text = f'send message to {recipient}: {body}'
    result.resolved_slots['message_body'] = body
    result.confidence = max(result.confidence, 0.82)


def _finalize_context_updates(result: RecoveryResult, state: dict[str, Any]) -> None:
    merged = {
        key: value
        for key, value in dict(state).items()
        if value not in (None, '', [])
    }
    for key, value in result.context_updates.items():
        if value in (None, '', []):
            merged.pop(key, None)
            continue
        merged[key] = value
    merged['confidence'] = round(max(float(merged.get('confidence') or 0.0), result.confidence), 3)
    merged['risk_level'] = result.risk_level
    merged['expires_at'] = _future_iso(_RECOVERY_STATE_TTL)
    if 'last_time_label' in merged:
        label = str(merged.get('last_time_label') or '').strip()
        if label:
            if _extract_time_label(label):
                merged.setdefault('last_time', label)
            else:
                merged.setdefault('last_date', label)
    if 'last_date' not in merged:
        date_label = _extract_latest_weekday_or_relative(result.recovered_text)
        if date_label:
            merged['last_date'] = date_label
    if 'last_time' not in merged:
        time_value = _extract_time_label(result.recovered_text)
        if time_value:
            merged['last_time'] = time_value
    if 'source_turn_ids' not in merged:
        merged['source_turn_ids'] = [
            str(item).strip()
            for item in state.get('source_turn_ids') or []
            if str(item).strip()
        ][-6:]
    result.context_updates = merged


def _looks_like_outbound_message(text: str) -> bool:
    lowered = text.casefold()
    return any(f'{verb} ' in lowered for verb in _OUTBOUND_VERBS)


def _extract_explicit_contact_candidate(text: str) -> str | None:
    lowered = text.casefold()
    prefixes = (
        'send a message to ',
        'send message to ',
        'message to ',
        'follow up with ',
        'tell ',
        'send ',
        'message ',
        'text ',
        'ask ',
        'check if ',
    )
    for marker in prefixes:
        idx = lowered.find(marker)
        if idx == -1:
            continue
        tail = text[idx + len(marker):].strip()
        tail = re.sub(
            r'^(?:a\s+)?(?:message|reminder)\s+to\s+',
            '',
            tail,
            count=1,
            flags=re.IGNORECASE,
        )
        candidate = _consume_name_tokens(tail)
        if candidate:
            return candidate
    match = re.search(r'\bcheck if\s+([a-z][a-z0-9_-]*)\s+replied\b', lowered, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1)
    return None


def _consume_name_tokens(text: str) -> str | None:
    tokens = [token.strip(".,!?;:'\"") for token in text.split()]
    picked: list[str] = []
    for token in tokens[:3]:
        if not token:
            continue
        if token.casefold() in _STOP_NAME_TOKENS:
            break
        picked.append(token)
    return ' '.join(picked).strip() or None


def _serialize_candidate_labels(candidates: Any) -> list[str]:
    labels: list[str] = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        aliases = item.get('aliases')
        if isinstance(aliases, list) and aliases:
            label = str(aliases[0]).strip()
            if label:
                labels.append(label)
                continue
        phone_last4 = str(item.get('phone_last4') or '').strip()
        if phone_last4:
            labels.append(f'contact ending in {phone_last4}')
    return labels


def _render_people_clarification(labels: list[str]) -> str:
    if len(labels) == 1:
        return f'You mean {labels[0]}, right?'
    if len(labels) == 2:
        return f'You mean {labels[0]} or {labels[1]}?'
    return 'Which person do you mean exactly?'


def _replace_first_case_insensitive(text: str, needle: str, replacement: str) -> str:
    pattern = re.compile(rf'\b{re.escape(needle)}\b', re.IGNORECASE)
    return pattern.sub(replacement, text, count=1)


def _contains_recipient_pronoun(text: str) -> bool:
    return bool(re.search(r'\b(?:her|him|them|he|she)\b', text))


def _replace_pronoun_target(text: str, recipient: str) -> str:
    for pattern in (
        r'\b(send|tell|message|text|ask)\s+(her|him|them|he|she)\b',
        r'\bfollow up with\s+(her|him|them|he|she)\b',
        r'\b(check if)\s+(her|him|them|he|she)\s+replied\b',
        r'\b(did)\s+(her|him|them|he|she)\s+reply\b',
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        matched = match.group(0)
        replacement = matched.replace(match.groups()[-1], recipient)
        return text.replace(matched, replacement, 1)
    return text


def _replace_other_one_reference(text: str, recipient: str) -> str:
    updated = re.sub(r'\bthe other one\b', recipient, text, count=1, flags=re.IGNORECASE)
    updated = re.sub(r'\b(?:not her|not him|not them)\b', '', updated, count=1, flags=re.IGNORECASE)
    updated = _CORRECTION_PREFIX_RE.sub('', updated, count=1).strip(' ,.')
    if not updated:
        return f'use {recipient} instead'
    if updated.casefold() == recipient.casefold():
        return f'use {recipient} instead'
    return updated


def _extract_outbound_message_body(text: str, recipient: str) -> str | None:
    lowered = text.casefold()
    for verb in _OUTBOUND_VERBS:
        marker = f'{verb} '
        idx = lowered.find(marker)
        if idx == -1:
            continue
        tail = text[idx + len(marker):].strip()
        if recipient:
            tail = re.sub(
                rf'^(?:a\s+)?(?:message|reminder)\s+to\s+{re.escape(recipient)}\b',
                '',
                tail,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
        if recipient:
            pattern = re.compile(rf'^{re.escape(recipient)}\b', re.IGNORECASE)
            tail = pattern.sub('', tail, count=1).strip()
        else:
            candidate = _consume_name_tokens(tail)
            if candidate:
                tail = re.sub(rf'^{re.escape(candidate)}\b', '', tail, count=1, flags=re.IGNORECASE).strip()
        if tail:
            return tail
    return None


def _extract_time_label(text: str) -> str | None:
    match = re.search(r'\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', text, flags=re.IGNORECASE)
    if match is None:
        return None
    hour = match.group(1)
    minute = match.group(2)
    meridiem = match.group(3)
    value = hour
    if minute:
        value += f':{minute}'
    if meridiem:
        value += f' {meridiem.upper()}'
    return value


def _extract_channel_hint(text: str) -> str | None:
    lowered = (text or '').casefold()
    if 'whatsapp' in lowered:
        return 'whatsapp'
    if 'sms' in lowered or 'text ' in lowered:
        return 'sms' if 'sms' in lowered else 'message'
    if 'message' in lowered or 'tell ' in lowered or 'send ' in lowered:
        return 'message'
    return None


def _extract_reminder_topic(text: str, recipient: str) -> str:
    match = re.search(r'\babout\s+(.+)$', text, flags=re.IGNORECASE)
    if match is not None:
        topic = match.group(1).strip(" .,!?:;")
        topic = re.sub(r'^(?:the|a|an)\s+', '', topic, count=1, flags=re.IGNORECASE)
        return topic

    body = _extract_outbound_message_body(text, recipient) or ''
    body = re.sub(
        rf'^(?:to\s+)?{re.escape(recipient)}\b',
        '',
        body,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" .,!?:;")
    return body


def _append_source_turn(existing: Any, text: str) -> list[str]:
    turns = [str(item).strip() for item in existing or [] if str(item).strip()]
    if text.strip():
        turns.append(text.strip())
    return turns[-4:]


def _merge_source_turn_ids(existing: Any, current: Any) -> list[str]:
    values = [str(item).strip() for item in existing or [] if str(item).strip()]
    values.extend(str(item).strip() for item in current or [] if str(item).strip())
    deduped: list[str] = []
    for item in values:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[-6:]


def _render_pending_draft_prompt(draft: dict[str, Any]) -> str:
    draft_kind = str(draft.get('draft_kind') or '').strip()
    recipient = str(draft.get('recipient') or '').strip()
    if not draft_kind or not recipient:
        return ''
    date_label = str(draft.get('date') or '').strip()
    time_label = str(draft.get('time') or '').strip()
    if draft_kind == 'reminder_message':
        channel = 'WhatsApp' if str(draft.get('channel') or '').strip().casefold() == 'whatsapp' else 'message'
        topic = str(draft.get('topic') or '').strip()
        time_parts = [part for part in (date_label, f'at {time_label}' if time_label else '') if part]
        timing = ' '.join(time_parts).strip()
        prompt = 'remind me'
        if timing:
            prompt += f' {timing}'
        prompt += f' to send {recipient} a {channel}'
        if topic:
            prompt += f' about {topic}'
        return prompt
    if draft_kind == 'outbound_message':
        body = str(draft.get('message_body') or '').strip()
        if not body:
            return ''
        if date_label and date_label.casefold() not in body.casefold():
            body = f'{body} {date_label}'.strip()
        if time_label and time_label.casefold() not in body.casefold():
            body = f'{body} at {time_label}'.strip()
        return f'send message to {recipient}: {body}'
    return ''


def _normalize_outbound_message_body(body: str, *, preferred_time: str | None) -> str:
    cleaned = ' '.join((body or '').strip().split())
    if not cleaned:
        return ''
    if preferred_time:
        corrected_value = preferred_time.lower() if preferred_time in {'today', 'tomorrow', 'tonight'} else preferred_time
        cleaned = _INLINE_TIME_CORRECTION_RE.sub('', cleaned)
        time_token_re = re.compile(
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight)\b',
            re.IGNORECASE,
        )
        if time_token_re.search(cleaned):
            cleaned = time_token_re.sub(corrected_value, cleaned, count=1)
        elif corrected_value.casefold() not in cleaned.casefold():
            cleaned = f'{cleaned} {corrected_value}'
    cleaned = re.sub(r'\b(?:nah|no|wait no|actually no)\s+not\s+(?:her|him|them|that one)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bthe other one\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(?:nah|no|wait no|actually no)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,.;:!?")
    return cleaned


def _extract_latest_weekday_or_relative(text: str) -> str | None:
    latest_match: tuple[int, str] | None = None
    for token in _WEEKDAY_NAMES:
        for match in re.finditer(rf'\b{re.escape(token)}\b', text, flags=re.IGNORECASE):
            latest_match = (match.start(), token)
    if latest_match is None:
        return None
    value = latest_match[1]
    return value.title() if value not in {'today', 'tomorrow', 'tonight'} else value


def _sanitize_recovery_state(payload: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    timestamp = now or _utc_now()
    state = dict(payload or {})
    if _is_expired(state.get('expires_at'), now=timestamp):
        return {}
    pending = _coerce_pending_draft(state.get('pending_draft'), now=timestamp)
    if pending is None:
        state.pop('pending_draft', None)
    else:
        state['pending_draft'] = pending
    state['source_turn_ids'] = [
        str(item).strip()
        for item in state.get('source_turn_ids') or []
        if str(item).strip()
    ][-6:]
    if not isinstance(state.get('last_ambiguity'), dict):
        state.pop('last_ambiguity', None)
    state['last_confirmation_options'] = [
        str(item).strip()
        for item in state.get('last_confirmation_options') or []
        if str(item).strip()
    ]
    return state


def _infer_action_kind(text: str) -> str:
    lowered = (text or '').casefold().strip()
    if not lowered:
        return ''
    if _recover_rental_status_subject(lowered, state={}) is not None:
        return 'rental_status_check'
    if 'follow up' in lowered or 'remind me' in lowered:
        return 'follow_up'
    if _looks_like_contact_reply_check(lowered):
        return 'contact_check'
    if _looks_like_outbound_message(lowered):
        return 'outbound_message'
    return ''


def _looks_like_contact_reply_check(text: str) -> bool:
    lowered = (text or '').casefold()
    return bool(
        re.search(r'\bcheck if\s+.+\s+replied\b', lowered)
        or re.search(r'\bdid\s+.+\s+reply\b', lowered)
        or re.search(r'\bdid\s+.+\s+replied\b', lowered)
    )


def _is_contact_chain_action(kind: str) -> bool:
    return kind in {'follow_up', 'outbound_message', 'contact_check'}


def _reset_incompatible_context_for_turn(result: RecoveryResult, *, state: dict[str, Any]) -> None:
    current_action_kind = _infer_action_kind(result.normalized_text)
    draft = _coerce_pending_draft(state.get('pending_draft'))
    if not current_action_kind:
        return
    if draft and current_action_kind == str(draft.get('action_kind') or '').strip():
        return
    if _is_time_only_followup(result.normalized_text):
        return
    if _CORRECTION_ONLY_PREFIX_RE.search(result.normalized_text.casefold()):
        return
    if _contains_recipient_pronoun(result.normalized_text.casefold()):
        return
    previous_action_kind = str(state.get('last_action_kind') or '').strip()
    if draft and current_action_kind != str(draft.get('action_kind') or '').strip():
        for key in _CONTACT_CONTEXT_KEYS:
            state.pop(key, None)
    elif previous_action_kind and _is_contact_chain_action(previous_action_kind) and current_action_kind != previous_action_kind:
        for key in _CONTACT_CONTEXT_KEYS:
            state.pop(key, None)


def _get_context_recipient_candidate(state: dict[str, Any]) -> tuple[str, str, float]:
    pending_draft = _coerce_pending_draft(state.get('pending_draft'))
    if pending_draft:
        return (
            str(pending_draft.get('recipient') or '').strip(),
            'pending_draft',
            float(pending_draft.get('confidence') or state.get('confidence') or 0.0),
        )
    recipient = str(state.get('last_recipient') or '').strip()
    if not recipient:
        return '', '', 0.0
    source = str(state.get('last_recipient_source') or 'global').strip() or 'global'
    return recipient, source, float(state.get('confidence') or 0.0)


def _record_context_usage(
    result: RecoveryResult,
    *,
    source: str,
    confidence: float,
) -> None:
    current = str(result.context_updates.get('context_source') or '').strip()
    if current:
        current_priority = _CONTEXT_SOURCE_PRIORITY.get(current, 0)
        incoming_priority = _CONTEXT_SOURCE_PRIORITY.get(source, 0)
        if incoming_priority < current_priority:
            return
    result.context_updates['used_context'] = True
    result.context_updates['context_source'] = source
    result.context_updates['confidence'] = max(float(result.context_updates.get('confidence') or 0.0), confidence)


def _extract_latest_unit_reference(text: str) -> str | None:
    matches = list(re.finditer(
        r'\bunit\s+([a-z0-9][a-z0-9\s-]*?)(?=[\.,;\n]|$)',
        text,
        flags=re.IGNORECASE,
    ))
    if not matches:
        return None
    raw = ' '.join(matches[-1].group(1).strip().split())
    combined_match = re.fullmatch(r'(\d+)\s+and\s+(\d+)', raw, flags=re.IGNORECASE)
    if combined_match is not None:
        left, right = combined_match.groups()
        if left.endswith('0' * len(right)) and len(left) > len(right):
            digits = f'{left[:-len(right)]}{right}'
        else:
            digits = f'{left}{right}'
        return f'Unit {digits}'
    direct_match = re.search(r'(\d{1,6})', raw)
    if direct_match is not None:
        return f'Unit {direct_match.group(1)}'
    return None
