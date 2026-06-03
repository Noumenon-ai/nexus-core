"""V3.5 tool-calling dispatcher.

Entry point that replaces the V2 intent-classifier switchboard. Loop:
  1. Load TELOS for user (if file present)
  2. mem0.search on the user's message
  3. Build system prompt: persona + TELOS + retrieved memories
  4. Call LLM with the union tool catalog
  5. If LLM returns text — that's the final reply
  6. If LLM returns tool_calls:
     - For each call: look up spec by name
       - If spec.requires_approval=True: SAFETY BOUNDARY — do NOT invoke.
         Create Approval row via approval_service.request, return the
         approval prompt as the final reply (with Approve/Cancel buttons).
       - Else: invoke spec.fn(**args, user_id=...) and feed result back
         into the loop as a function-response turn.
  7. Hard cap of 10 iterations — terminates with safe fallback reply.
  8. After final reply: schedule mem0.add(user_msg + assistant_reply) as
     a fire-and-forget background task.

Per V3.5 user directive: this file is the dispatcher; the existing
intent_classifier / assistant_router / handlers stay in place (their
deletion is V3.9 cleanup). Wired into pipeline/unified.py behind a feature
flag.
"""
from __future__ import annotations

import asyncio
import dateparser
from difflib import SequenceMatcher
import inspect
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional, Protocol
from zoneinfo import ZoneInfo

from models import ConversationTurn, User
from pipeline.types import InlineButton, ServiceResponse
from repositories.approvals_repository import ApprovalsRepository
from repositories.conversation_turns_repository import ConversationTurnsRepository
from repositories.proactive_notifications_repository import (
    ProactiveNotificationsRepository,
)
from services.approval_service import ApprovalService
from services.capability_registry import CapabilityRegistry
from services.clarification_state import (
    ActiveClarification,
    build_recovery_clarification,
    build_vague_clarification_state,
    refresh_clarification,
    render_option_labels,
    resolve_clarification_answer,
)
from services.conversation_service import ConversationService
from services.conversational_recovery import (
    ConversationalRecoveryLayer,
    RecoveryResult,
    build_recovery_prompt_block,
)
from services.delivery_truth import (
    DISPATCHED_DELIVERY_STATES,
    RETRY_IN_PROGRESS_STATES,
)
from services.self_correction import detect_self_correction
from services.destructive_intent_classifier import (
    classify as _classify_destructive,
    render_default_preview as _render_destructive_preview,
)
from services.fallback_manager import (
    PROVIDER_UNAVAILABLE_USER_TEXT,
    FallbackContext,
    FallbackManager,
)
from services.human_confirmation_style import HumanConfirmationStyle
from services.reminder_duplicates import (
    ReminderDuplicateCluster,
    cluster_duplicate_reminders,
    render_duplicate_audit_text,
)
from services.runtime_identity import (
    get_runtime_identity,
    render_runtime_status_text,
)
from services.telegram_streaming import StreamingSession
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult, render_approval_preview
from services.vague_clarification import build_vague_clarification
from utils.dates import app_now, format_local_datetime, utc_now
from utils.i18n import Translator

logger = logging.getLogger(__name__)
_HUMAN_CONFIRMATION_STYLE = HumanConfirmationStyle()


_MAX_DEFAULT_ITERATIONS = 10

_MAX_ARCHIVED_CONTEXT_TURNS = 20

_REMINDER_BODY_SIMILARITY_THRESHOLD = 0.88
_POST_APPROVAL_CLEANUP_MATCH_THRESHOLD = 0.88
_POST_APPROVAL_DIGEST_CONTEXT_MAX_AGE = timedelta(hours=12)
_REMINDER_DEDUPE_STOPWORDS = {'a', 'an', 'the'}
_POST_APPROVAL_CLEANUP_STOPWORDS = {
    'a',
    'an',
    'the',
    'my',
    'our',
    'your',
    'task',
    'tasks',
    'reminder',
    'reminders',
    'complete',
    'completed',
    'done',
    'finished',
    'remove',
    'removed',
    'delete',
    'deleted',
    'clear',
    'mark',
}
_TIME_CONTEXT_HINTS = (
    'today',
    'tomorrow',
    'tonight',
    'morning',
    'afternoon',
    'evening',
    'monday',
    'tuesday',
    'wednesday',
    'thursday',
    'friday',
    'saturday',
    'sunday',
)
_ISSUE_CONTEXT_HINTS = (
    'water damage',
    'leak',
    'repair',
    'maintenance',
    'urgent',
    'invoice',
    'late fee',
    'rent',
    'estimate',
    'quote',
)
_GENERIC_TARGET_WORDS = {
    'her',
    'him',
    'them',
    'someone',
    'somebody',
    'anyone',
    'anybody',
    'thing',
    'this',
    'that',
    'it',
}
_CHECK_CASE_CLARIFICATION = (
    'What item or case should I check, and which tenant or unit is it about?'
)
_WEEKDAY_NAME_TO_INDEX = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6,
}
_POST_APPROVAL_CLEANUP_PREFIXES = (
    'i already did ',
    'already did ',
    "i've already done ",
    'i have already done ',
    "i've done ",
    'i have done ',
    'done ',
    'completed ',
    'finished ',
)
_POST_APPROVAL_CLEANUP_TRAILING_PHRASES = (
    ' remove it',
    ' delete it',
    ' clear it',
    ' mark done',
    ' mark complete',
    ' mark completed',
)
_POST_APPROVAL_CLEANUP_GENERIC_TARGETS = {'it', 'this', 'that', 'them'}



_GENERIC_TOOL_STAGE = 'Working on it...'

_APPROVED_CONTACT_REMINDER_PROVIDER_UNAVAILABLE = (
    'Approved action could not be executed because no configured '
    'contact-reminder provider is available.'
)
_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC = 20.0
_GENERIC_PROVIDER_FAILURE_TEXT = PROVIDER_UNAVAILABLE_USER_TEXT
_AUDIT_PROVIDER_FAILURE_GUIDANCE = (
    "I couldn't reach my provider for that. Run one of these local checks:\n"
    "- `python3 -m audit.chaos_audit_runner --quick`\n"
    "- `python3 -m audit.utility_audit_runner --quick`"
)
_GENERAL_PROVIDER_UNAVAILABLE = PROVIDER_UNAVAILABLE_USER_TEXT
_SECOND_CHANCE_VAGUE_CLARIFICATION = build_vague_clarification


def _looks_like_calendar_request(text: str) -> bool:
    lowered = f' {(text or "").casefold()} '
    calendar_terms = (' calendar ', ' event ', ' meeting ', ' appointment ')
    action_terms = (
        ' check ',
        ' what ',
        " what's ",
        ' show ',
        ' list ',
        ' add ',
        ' create ',
        ' schedule ',
        ' update ',
        ' move ',
        ' cancel ',
        ' delete ',
    )
    return any(term in lowered for term in calendar_terms) and any(term in lowered for term in action_terms)


def _looks_like_whatsapp_send_request(text: str) -> bool:
    lowered = f' {(text or "").casefold()} '
    if ' whatsapp ' not in lowered:
        return False
    send_terms = (' send ', ' tell ', ' text ', ' message ', ' ask ')
    return any(term in lowered for term in send_terms)


def _normalize_direct_command_text(text: str) -> str:
    normalized = ' '.join((text or '').strip().casefold().split())
    normalized = normalized.replace("what's", 'whats')
    normalized = re.sub(r'[?!.,<>]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return re.sub(r'^/reminders@[a-z0-9_]+', '/reminders', normalized)


def _classify_direct_reminder_read_command(text: str) -> str | None:
    normalized = _normalize_direct_command_text(text)
    if not normalized:
        return None

    if 'duplicates' in normalized and 'reminder' in normalized:
        if normalized.startswith('/reminders') or normalized.startswith('reminders'):
            return 'duplicates'

    if normalized.startswith('/reminders'):
        return 'list'

    if normalized in {
        'reminders',
        'reminders list',
        'show reminders',
        'show my reminders',
        'list reminders',
        'list my reminders',
        'whats my reminders',
        'what are my reminders',
    }:
        return 'list'

    if re.match(r'^(?:whats|what are)\s+my\s+reminders\b', normalized):
        return 'list'
    if re.match(r'^(?:show|list)\s+(?:my\s+)?reminders\b', normalized):
        return 'list'

    return None

# V3.7 streaming: human-readable stage strings shown to the user via
# Telegram edit-message while the dispatcher works. Coverage of every
# tool registered by `services.dispatcher_registry.build_dispatcher_registry`
# is enforced by `tests/phase_v3_7/test_dispatcher_streaming_invariants.py`
# — a missing entry here fails that test, which is the test working as
# designed (per spec halt condition: source-inspection invariant).
TOOL_STAGE_MESSAGES: dict[str, str] = {
    # V3.2 reads (8 real)
    'get_current_time': 'Checking the time...',
    'list_active_reminders': 'Checking your reminders...',
    'list_pending_tasks': 'Looking up your tasks...',
    'list_completed_tasks': 'Looking up your completed tasks...',
    'list_memories': 'Looking through what I remember...',
    'get_email_summary': 'Reading your emails...',
    'get_telos': 'Reading your priorities...',
    'get_active_approvals': 'Checking your pending approvals...',
    # V3.2 stubs (4 — Google read, deferred until service layer ships)
    'list_calendar_events': 'Checking your calendar...',
    'check_freebusy': 'Checking your free/busy...',
    'list_google_tasks': 'Looking up your Google tasks...',
    'lookup_contact': 'Looking up that contact...',
    # V3.3 writes (5 real)
    'create_reminder': 'Setting up that reminder...',
    'create_task': 'Adding that to your tasks...',
    'mark_task_done': 'Marking that task done...',
    'save_user_memory': 'Saving that to memory...',
    'set_user_preference': 'Updating your preference...',
    # V3.3 stubs (3 — Google write, deferred)
    'create_calendar_event': 'Adding that to your calendar...',
    'update_calendar_event': 'Updating that calendar event...',
    'create_contact': 'Adding that contact...',
    # V3.4 destructive (5 real — approval-gated; stage shows BEFORE
    # the approval prompt is rendered, never as the final user-facing
    # message)
    'delete_reminder': 'Preparing to delete that reminder...',
    'delete_task': 'Preparing to delete that task...',
    'forget_user_memory': 'Preparing to forget that memory...',
    'disconnect_google': 'Preparing to disconnect your Google account...',
    'append_telos_update': 'Preparing to update your priorities...',
    # V3.4 stubs (2 — Google destructive, deferred)
    'delete_calendar_event': 'Preparing to delete that calendar event...',
    'send_telegram_message': 'Preparing to send that message...',
    # V3.8 TELOS onboarding (4 — non-destructive guided-flow tools)
    'start_telos_onboarding': 'Starting your TELOS onboarding...',
    'answer_telos_question': 'Recording that for your TELOS...',
    'view_my_telos': 'Reading your TELOS file...',
    'cancel_telos_onboarding': 'Pausing your TELOS onboarding...',
    # V3.upgrade content tools (3 — keyless public APIs)
    'lookup_wikipedia': 'Looking that up on Wikipedia...',
    'get_weather': 'Checking the weather...',
    'get_news_headlines': 'Pulling the news...',
    # Phase 4 file-system reads (3)
    'read_file': 'Reading that file…',
    'list_directory': 'Listing that directory…',
    'search_files': 'Searching for files…',
    # Phase 4 file-system + terminal (2 — approval-gated destructive)
    'write_file': 'Preparing to write that file...',
    'run_terminal_command': 'Preparing to run that command...',
}


def _approved_contact_provider_failure_text() -> str:
    return _APPROVED_CONTACT_REMINDER_PROVIDER_UNAVAILABLE


def _is_global_provider_failure_text(text: str) -> bool:
    return (text or '').strip() == _GENERIC_PROVIDER_FAILURE_TEXT


def _is_approved_contact_send_payload(payload: dict[str, Any]) -> bool:
    matched_tools = {
        str(name).strip().lower()
        for name in (payload.get('matched_tools') or [])
        if str(name).strip()
    }
    if 'send_message' in matched_tools:
        return True

    original_prompt = str(payload.get('original_prompt') or '').strip().lower()
    if not original_prompt:
        return False

    has_contact_target = any(
        token in original_prompt
        for token in ('phone', 'sms', 'whatsapp', 'text', 'message')
    )
    return 'remind' in original_prompt and has_contact_target


def _looks_like_contact_provider_wiring_error(text: str) -> bool:
    lowered = (text or '').strip().lower()
    if not lowered:
        return False
    if 'provider error' in lowered:
        return True
    if 'not_configured' in lowered or 'no configured' in lowered:
        return True
    if 'legacy_mcp' in lowered:
        return True
    if 'mcp' in lowered and any(
        token in lowered for token in ('error', 'failed', 'not found', 'wiring')
    ):
        return True
    return False


def _looks_like_audit_request(text: str) -> bool:
    lowered = (text or '').lower()
    return any(
        token in lowered for token in (
            'self-audit',
            'self audit',
            'chaos audit',
            'utility audit',
            'run an audit',
            'run a self-audit',
            'debug nexus',
            'audit nexus',
        )
    )


def _looks_like_time_request(text: str) -> bool:
    lowered = (text or '').lower()
    return any(
        token in lowered for token in (
            'what time is it',
            'current time',
            'time is it',
        )
    )


def _normalize_prompt_text(text: str) -> str:
    return ' '.join((text or '').strip().lower().split())


_ROLE_LABEL_PREFIX_RE = re.compile(
    r'^\s*(?:user|assistant|nexus|the ai|system)\s*:\s*(.+?)\s*$',
    re.IGNORECASE,
)
_CONFIRMATION_TOKEN_RE = re.compile(
    r'^(?:yes|no|1|2|3|option\s+[1-3]|first|second|third|the first one|the second one|the third one|that one|yes that)$',
    re.IGNORECASE,
)
_BARE_CONFIRMATION_RE = re.compile(
    r'^(?:yes|ok|okay|do it|approved|approve|go ahead|sure)$',
    re.IGNORECASE,
)
_THREAD_BINDING_TTL = timedelta(hours=2)
_THREAD_CLARIFICATION_MISMATCH_TEXT = (
    "I'm not sure which request you're answering. Say it again with the target."
)
_THREAD_APPROVAL_STALE_TEXT = (
    'That approval is no longer current. I did not run it.'
)
_THREAD_APPROVAL_OLDER_TEXT = (
    'That approval belongs to an older request. Please send the request again.'
)
_THREAD_CONFIRMATION_AMBIGUOUS_TEXT = 'Yes to which request?'


def _sanitize_role_contaminated_text(text: str) -> tuple[str, bool]:
    match = _ROLE_LABEL_PREFIX_RE.match(text or '')
    if match is None:
        return (text or '').strip(), False
    return match.group(1).strip(), True


def _looks_like_confirmation_token(text: str) -> bool:
    return _CONFIRMATION_TOKEN_RE.match(' '.join((text or '').strip().split())) is not None


def _looks_like_bare_confirmation(text: str) -> bool:
    return _BARE_CONFIRMATION_RE.match(' '.join((text or '').strip().split())) is not None


def _parse_thread_revision(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_iso_timestamp_expired(value: Any, *, now: datetime | None = None) -> bool:
    if not value:
        return False
    candidate = str(value).strip()
    if not candidate:
        return False
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now or utc_now()) > parsed.astimezone(timezone.utc)


def _coerce_active_thread(payload: Any, *, now: datetime | None = None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    thread_id = str(payload.get('thread_id') or '').strip()
    if not thread_id:
        return None
    if _is_iso_timestamp_expired(payload.get('expires_at'), now=now):
        return None
    return {
        'thread_id': thread_id,
        'thread_kind': str(payload.get('thread_kind') or 'unknown').strip() or 'unknown',
        'thread_revision': _parse_thread_revision(payload.get('thread_revision') or 0),
        'status': str(payload.get('status') or 'active').strip() or 'active',
        'created_at': str(payload.get('created_at') or '').strip(),
        'updated_at': str(payload.get('updated_at') or '').strip(),
        'expires_at': str(payload.get('expires_at') or '').strip(),
        'source_turn_ids': [
            str(item).strip()
            for item in payload.get('source_turn_ids') or []
            if str(item).strip()
        ][-6:],
        'fingerprint': str(payload.get('fingerprint') or '').strip(),
        'recipient': str(payload.get('recipient') or '').strip(),
        'unit_reference': str(payload.get('unit_reference') or '').strip(),
        'task': str(payload.get('task') or '').strip(),
        'message_body': str(payload.get('message_body') or '').strip(),
        'date_label': str(payload.get('date_label') or '').strip(),
        'time_label': str(payload.get('time_label') or '').strip(),
    }


def _active_thread_from_recovery_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    return _coerce_active_thread(state.get('active_thread'))


def _thread_binding_payload(thread: dict[str, Any] | None) -> dict[str, Any]:
    if not thread:
        return {}
    return {
        'thread_id': str(thread.get('thread_id') or '').strip(),
        'thread_kind': str(thread.get('thread_kind') or 'unknown').strip() or 'unknown',
        'thread_revision': _parse_thread_revision(thread.get('thread_revision') or 0),
    }


def _thread_signature_fields(
    *,
    recovery: RecoveryResult,
    previous_thread: dict[str, Any] | None,
) -> dict[str, str]:
    previous_thread = previous_thread or {}
    used_context = bool(recovery.context_updates.get('used_context'))
    recipient = str(recovery.resolved_slots.get('recipient') or '').strip()
    if not recipient and (recovery.context_updates.get('recipient_explicit') or used_context or recovery.missing_slot == 'recipient'):
        recipient = str(recovery.context_updates.get('last_recipient') or '').strip()
    if not recipient and used_context:
        recipient = str(previous_thread.get('recipient') or '').strip()

    unit_reference = str(
        recovery.resolved_slots.get('unit_reference')
        or recovery.context_updates.get('last_unit')
        or ''
    ).strip()
    if not unit_reference and used_context:
        unit_reference = str(previous_thread.get('unit_reference') or '').strip()

    task = str(
        recovery.resolved_slots.get('topic')
        or recovery.resolved_slots.get('rental_subject')
        or recovery.context_updates.get('last_task')
        or ''
    ).strip()
    if not task and used_context:
        task = str(previous_thread.get('task') or '').strip()

    message_body = str(
        recovery.resolved_slots.get('message_body')
        or recovery.context_updates.get('last_message_body')
        or ''
    ).strip()
    if not message_body and used_context:
        message_body = str(previous_thread.get('message_body') or '').strip()

    date_label = str(
        recovery.resolved_slots.get('date_label')
        or recovery.context_updates.get('last_date')
        or ''
    ).strip()
    if not date_label and used_context:
        date_label = str(previous_thread.get('date_label') or '').strip()

    time_label = str(
        recovery.resolved_slots.get('time_label')
        or recovery.context_updates.get('last_time_label')
        or ''
    ).strip()
    if not time_label and used_context:
        time_label = str(previous_thread.get('time_label') or '').strip()

    return {
        'recipient': recipient,
        'unit_reference': unit_reference,
        'task': task,
        'message_body': message_body,
        'date_label': date_label,
        'time_label': time_label,
    }


def _infer_thread_kind(
    *,
    recovery: RecoveryResult,
    working_text: str,
    social_reply: bool,
    previous_thread: dict[str, Any] | None,
) -> str:
    if social_reply:
        return 'social'
    action_kind = str(
        recovery.resolved_slots.get('action_kind')
        or recovery.context_updates.get('last_action_kind')
        or ''
    ).strip()
    mapping = {
        'rental_status_check': 'rental_status',
        'follow_up': 'reminder',
        'reminder_draft': 'reminder',
        'outbound_message': 'contact_send',
        'contact_check': 'delivery_status',
    }
    if action_kind in mapping:
        return mapping[action_kind]
    lowered = (working_text or recovery.normalized_text or '').casefold()
    if 'birthday' in lowered or 'bday' in lowered:
        return 'social'
    if ' llc' in f' {lowered} ':
        return 'llc_status'
    if any(token in lowered for token in ('remove it', 'delete it', 'mark it done', 'complete it')):
        return 'cleanup'
    if recovery.context_updates.get('used_context') and previous_thread:
        return str(previous_thread.get('thread_kind') or 'unknown').strip() or 'unknown'
    return 'unknown'


def _looks_like_thread_continuation(
    *,
    recovery: RecoveryResult,
    social_reply: bool,
    clarification_metadata: dict[str, Any],
) -> bool:
    if social_reply:
        return True
    if clarification_metadata.get('clarification_answer_resolved'):
        return True
    if recovery.context_updates.get('used_context'):
        return True
    if recovery.corrections_applied:
        return True
    lowered = recovery.normalized_text.casefold()
    if _looks_like_bare_confirmation(lowered):
        return True
    if any(token in lowered for token in ('other one', 'wait no', 'actually no')):
        return True
    return len(lowered.split()) <= 4 and bool(lowered)


def _has_explicit_thread_conflict(
    *,
    previous_thread: dict[str, Any] | None,
    signature: dict[str, str],
    recovery: RecoveryResult,
) -> bool:
    if previous_thread is None:
        return False
    previous_recipient = str(previous_thread.get('recipient') or '').strip()
    current_recipient = str(signature.get('recipient') or '').strip()
    if (
        recovery.context_updates.get('recipient_explicit')
        and previous_recipient
        and current_recipient
        and current_recipient != previous_recipient
    ):
        return True
    previous_unit = str(previous_thread.get('unit_reference') or '').strip()
    current_unit = str(signature.get('unit_reference') or '').strip()
    if previous_unit and current_unit and current_unit != previous_unit and 'unit_reference' in recovery.corrections_applied:
        return True
    return False


def _build_thread_fingerprint(*, thread_kind: str, signature: dict[str, str], missing_slot: str | None) -> str:
    return json.dumps(
        {
            'thread_kind': thread_kind,
            'recipient': signature.get('recipient') or '',
            'unit_reference': signature.get('unit_reference') or '',
            'task': signature.get('task') or '',
            'message_body': signature.get('message_body') or '',
            'date_label': signature.get('date_label') or '',
            'time_label': signature.get('time_label') or '',
            'missing_slot': str(missing_slot or '').strip(),
        },
        separators=(',', ':'),
        sort_keys=True,
    )


def _select_active_thread(
    *,
    previous_thread: dict[str, Any] | None,
    recovery: RecoveryResult,
    working_text: str,
    social_reply: bool,
    clarification_metadata: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    current_kind = _infer_thread_kind(
        recovery=recovery,
        working_text=working_text,
        social_reply=social_reply,
        previous_thread=previous_thread,
    )
    signature = _thread_signature_fields(
        recovery=recovery,
        previous_thread=previous_thread,
    )
    fingerprint = _build_thread_fingerprint(
        thread_kind=current_kind,
        signature=signature,
        missing_slot=recovery.missing_slot,
    )
    continue_thread = False
    if previous_thread is not None and previous_thread.get('thread_kind') == current_kind:
        continue_thread = (
            not _has_explicit_thread_conflict(
                previous_thread=previous_thread,
                signature=signature,
                recovery=recovery,
            )
            and _looks_like_thread_continuation(
                recovery=recovery,
                social_reply=social_reply,
                clarification_metadata=clarification_metadata,
            )
        )

    if continue_thread:
        thread_id = str(previous_thread.get('thread_id') or '').strip()
        created_at = str(previous_thread.get('created_at') or now.isoformat()).strip() or now.isoformat()
        revision = _parse_thread_revision(previous_thread.get('thread_revision') or 1)
        if fingerprint != str(previous_thread.get('fingerprint') or ''):
            revision += 1
    else:
        thread_id = str(uuid.uuid4())
        created_at = now.isoformat()
        revision = 1

    return {
        'thread_id': thread_id,
        'thread_kind': current_kind,
        'thread_revision': revision,
        'status': 'active',
        'created_at': created_at,
        'updated_at': now.isoformat(),
        'expires_at': (now + _THREAD_BINDING_TTL).isoformat(),
        'source_turn_ids': [
            str(item).strip()
            for item in recovery.context_updates.get('source_turn_ids') or []
            if str(item).strip()
        ][-6:],
        'fingerprint': fingerprint,
        **signature,
    }


def _thread_with_status(
    thread: dict[str, Any] | None,
    *,
    status: str,
    now: datetime,
) -> dict[str, Any] | None:
    if not thread:
        return None
    updated = dict(thread)
    updated['status'] = status
    updated['updated_at'] = now.isoformat()
    updated['expires_at'] = (now + _THREAD_BINDING_TTL).isoformat()
    return updated


def _thread_binding_matches(
    *,
    expected_thread_id: str,
    expected_thread_revision: int,
    current_thread: dict[str, Any] | None,
    required_status: str | None,
) -> tuple[bool, str]:
    if not expected_thread_id:
        return True, ''
    if current_thread is None:
        return False, _THREAD_APPROVAL_OLDER_TEXT
    if _is_iso_timestamp_expired(current_thread.get('expires_at')):
        return False, _THREAD_APPROVAL_OLDER_TEXT
    if str(current_thread.get('thread_id') or '').strip() != expected_thread_id:
        return False, _THREAD_APPROVAL_OLDER_TEXT
    if _parse_thread_revision(current_thread.get('thread_revision') or 0) != expected_thread_revision:
        return False, _THREAD_APPROVAL_STALE_TEXT
    if required_status and str(current_thread.get('status') or '').strip() != required_status:
        return False, _THREAD_APPROVAL_STALE_TEXT
    return True, ''


def _is_explicit_memory_request(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    return any(
        phrase in lowered
        for phrase in (
            'remember ',
            'save this',
            'save that',
            'save my ',
            'keep this in mind',
            "don't forget",
            'do not forget',
            'note that',
        )
    )


def _looks_like_birthday_statement(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    return (
        ('birthday' in lowered or 'bday' in lowered)
        and any(token in lowered for token in ('tomorrow', 'today', 'my birthday', 'my bday'))
    )


def _looks_like_non_celebration_birthday_reply(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    has_birthday = 'birthday' in lowered or 'bday' in lowered
    has_joke_or_noncelebration = any(
        token in lowered
        for token in (
            "don't celebrate",
            'dont celebrate',
            'not a celebration',
            'just told you to see how you react',
            'hahaha',
            'haha',
        )
    )
    wants_wish = 'wish' in lowered
    return has_birthday and has_joke_or_noncelebration and wants_wish


def _render_birthday_memory_confirmation(*, app_timezone: str) -> str:
    tomorrow = app_now(app_timezone) + timedelta(days=1)
    date_label = tomorrow.strftime('%B %d').replace(' 0', ' ')
    return _HUMAN_CONFIRMATION_STYLE.render_birthday_memory_confirmation(
        date_label=date_label,
    )


def _memory_confirmation_subject(args: dict[str, Any]) -> str:
    value = str(args.get('value') or '').strip()
    if value:
        return value
    key = str(args.get('key') or 'that').strip().replace('_', ' ')
    return key or 'that'


def _should_block_personal_memory_save(*, prompt_text: str, tool_name: str) -> bool:
    if tool_name not in {'save_user_memory', 'set_user_preference'}:
        return False
    return not _is_explicit_memory_request(prompt_text)


def _render_memory_confirmation_prompt(
    *,
    prompt_text: str,
    args: dict[str, Any],
    app_timezone: str,
) -> str:
    if _looks_like_birthday_statement(prompt_text):
        return _render_birthday_memory_confirmation(app_timezone=app_timezone)
    return _HUMAN_CONFIRMATION_STYLE.render_memory_confirmation(
        subject=_memory_confirmation_subject(args=args),
    )


def _looks_like_claimed_memory_save(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    return any(
        phrase in lowered
        for phrase in (
            'saved ',
            "i'll have it next year",
            'remember that',
            'remembered',
        )
    )


def _render_social_reply(*, text: str, app_timezone: str) -> str | None:
    if _looks_like_non_celebration_birthday_reply(text):
        return _HUMAN_CONFIRMATION_STYLE.render_birthday_wish_reply()
    if _looks_like_birthday_statement(text) and not _is_explicit_memory_request(text):
        return _render_birthday_memory_confirmation(app_timezone=app_timezone)
    return None


def _has_specific_target_context(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    if not lowered:
        return False
    if re.search(r'\bunit\s+[a-z0-9-]+\b', lowered):
        return True
    if any(token in lowered for token in ('tenant', 'vendor', 'hoa', 'landlord')):
        return True
    for match in re.finditer(r'\b(?:if|with|to|tell|ask|message|send)\s+([a-z][a-z0-9_-]*)\b', lowered):
        candidate = match.group(1)
        if candidate not in _GENERIC_TARGET_WORDS:
            return True
    return False


def _prompt_has_rich_reminder_context(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    if not lowered:
        return False

    has_reminder_intent = any(
        phrase in lowered
        for phrase in ('remind me', 'follow up', 'reminder')
    )
    if not has_reminder_intent:
        return False

    has_time_context = bool(re.search(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', lowered)) or any(
        token in lowered for token in _TIME_CONTEXT_HINTS
    )
    if not has_time_context:
        return False

    if not _has_specific_target_context(lowered):
        return False

    return any(token in lowered for token in _ISSUE_CONTEXT_HINTS)


def _has_contact_reminder_intent(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    if not lowered:
        return False
    if not any(token in lowered for token in (' whatsapp', 'sms ', ' on whatsapp', ' via whatsapp', ' on sms', ' via sms')):
        return False
    return lowered.startswith('remind ') or ' reminder ' in lowered


def _should_emit_vague_clarification(*, text: str, clarification: str | None) -> bool:
    if not clarification:
        return False
    if clarification == _CHECK_CASE_CLARIFICATION and _prompt_has_rich_reminder_context(text):
        return False
    return True


def _invoke_local_time_fallback(registry: ToolRegistry, *, user_id: str) -> str | None:
    spec = registry.get('get_current_time')
    if spec is None or spec.requires_approval:
        return None
    try:
        result = spec.fn(user_id=user_id)
    except TypeError:
        try:
            result = spec.fn()
        except Exception:
            return None
    except Exception:
        return None

    if inspect.iscoroutine(result):
        return None

    if isinstance(result, ToolResult):
        if not result.success or not isinstance(result.data, dict):
            return None
        data = result.data
    elif isinstance(result, dict):
        data = result
    else:
        return None

    iso = data.get('iso')
    timezone = data.get('timezone')
    if not isinstance(iso, str) or not isinstance(timezone, str):
        return None
    return f'Current time: {iso} ({timezone})'


def _recovery_metadata_payload(result: RecoveryResult) -> dict[str, Any]:
    payload = dict(result.context_updates)
    payload.update(
        {
            'outcome': result.outcome,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'resolved_slots': dict(result.resolved_slots),
            'corrections_applied': list(result.corrections_applied),
        }
    )
    return payload


def _runtime_reminder_service():
    from scheduler import _RUNTIME  # type: ignore

    runtime = _RUNTIME
    service = getattr(runtime, 'reminder_service', None) if runtime is not None else None
    if service is None:
        raise RuntimeError('ReminderService runtime is not available.')
    return service


def _runtime_session_factory():
    from scheduler import _RUNTIME  # type: ignore

    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError('Runtime session factory is not available.')

    reminder_service = getattr(runtime, 'reminder_service', None)
    reminders_repository = getattr(reminder_service, 'reminders_repository', None)
    session_factory = getattr(reminders_repository, 'session_factory', None)
    if session_factory is not None:
        return session_factory

    users_repository = getattr(runtime, 'users_repository', None)
    session_factory = getattr(users_repository, 'session_factory', None)
    if session_factory is not None:
        return session_factory

    raise RuntimeError('Runtime session factory is not available.')


async def handle_contact_reminder_retry_callback(
    *, reminder_id: str, telegram_id: int,
) -> dict[str, Any]:
    from models import Reminder, User

    session_factory = _runtime_session_factory()
    with session_factory() as session:
        reminder = session.get(Reminder, reminder_id)
        if reminder is None:
            return {'ok': False, 'error': 'not_found'}
        owner = session.get(User, reminder.user_id)
        if owner is None or owner.telegram_id != telegram_id:
            return {'ok': False, 'error': 'not_authorized'}
        if reminder.target_contact_id is None:
            return {'ok': False, 'error': 'not_contact_reminder'}
        if reminder.status == 'cancelled':
            return {'ok': False, 'error': 'cancelled'}
        if reminder.delivery_status in RETRY_IN_PROGRESS_STATES:
            return {'ok': False, 'error': 'already_retrying'}
        if reminder.delivery_status != 'failed':
            return {'ok': False, 'error': 'not_retryable'}
        reminder.delivery_status = 'requested'
        session.commit()

    service = _runtime_reminder_service()
    retry_fn = getattr(service, 'retry_failed_contact_reminder', None)
    if retry_fn is None:
        retry_fn = getattr(service, 'fire_reminder')
    try:
        ok = await retry_fn(reminder_id)
    except Exception:
        with session_factory() as session:
            reminder = session.get(Reminder, reminder_id)
            if (reminder is not None
                    and reminder.status != 'cancelled'
                    and reminder.delivery_status in RETRY_IN_PROGRESS_STATES):
                reminder.delivery_status = 'failed'
                reminder.updated_at = utc_now()
                session.commit()
        raise
    if not ok:
        with session_factory() as session:
            reminder = session.get(Reminder, reminder_id)
            if (reminder is not None
                    and reminder.status != 'cancelled'
                    and reminder.delivery_status in RETRY_IN_PROGRESS_STATES):
                reminder.delivery_status = 'failed'
                reminder.updated_at = utc_now()
                session.commit()
    return {'ok': bool(ok)}


async def handle_contact_reminder_cancel_callback(
    *, reminder_id: str, telegram_id: int,
) -> dict[str, Any]:
    from models import Reminder, User

    session_factory = _runtime_session_factory()
    with session_factory() as session:
        reminder = session.get(Reminder, reminder_id)
        if reminder is None:
            return {'ok': False, 'error': 'not_found'}
        owner = session.get(User, reminder.user_id)
        if owner is None or owner.telegram_id != telegram_id:
            return {'ok': False, 'error': 'not_authorized'}
        if reminder.target_contact_id is None:
            return {'ok': False, 'error': 'not_contact_reminder'}
        if reminder.delivery_status in DISPATCHED_DELIVERY_STATES:
            return {'ok': False, 'error': 'already_sent'}
        if reminder.delivery_status == 'cancelled' and reminder.status == 'cancelled':
            return {'ok': True}
        reminder.delivery_status = 'cancelled'
        reminder.status = 'cancelled'
        reminder.updated_at = utc_now()
        session.commit()
    return {'ok': True}


class LLMClient(Protocol):
    """Interface the dispatcher consumes. The real Gemini integration lives
    on BrainRouter.generate_with_tools; tests inject scripted stubs."""

    async def generate_with_tools(
        self,
        *,
        user_id: str,
        system_prompt: str,
        contents: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


class Mem0Client(Protocol):
    def search(self, query: str, *, user_id, limit: int = 5) -> Any: ...
    def add(self, messages: list[dict[str, Any]], *, user_id) -> Any: ...


@dataclass(slots=True)
class DispatcherInput:
    user: User
    text: str
    translator: Optional[Translator] = None
    # V3.7 streaming. None = no streaming (preserves V3.5/V3.6 behavior
    # for the 18 existing dispatcher test sites that construct
    # DispatcherInput without this field). When present, the dispatcher
    # drives session.update("Thinking...") at entry, session.update(stage)
    # per tool-call, and session.finalize(final_text) before returning.
    streaming_session: Optional[StreamingSession] = None
    # H2-039 FIX 1: when True, skip the destructive-intent classifier gate.
    # Set by the dispatcher itself when re-firing after the user taps
    # Approve on a destructive_message_gate prompt. External callers leave
    # this False; tests can set True to bypass the gate when testing
    # downstream behavior independently.
    bypass_destructive_approval: bool = False
    # H2 MaxFix: lets provider-failure normalization distinguish a normal
    # prompt from a resumed approved workflow.
    post_approval_resume: bool = False


@dataclass(slots=True)
class DispatcherOutput:
    text: str
    iterations: int
    buttons: list[InlineButton] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_memory_id(result: Any) -> str | None:
    """Best-effort extraction of mem0's memory_id from `Memory.add` return.

    mem0's return shape varies across versions. Common shapes:
      - {'results': [{'id': '...', 'memory': '...', 'event': 'ADD'}, ...]}
      - {'id': '...'}
      - bare string id
      - None / missing

    Returns the first id we can find, else None. None is a safe sentinel:
    `mark_mem0_persisted` still records `mem0_persisted_at` so the row
    drops out of the pending index, and absence of `mem0_memory_id` just
    means later dedup against mem0 will need a content-based lookup.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        if isinstance(result.get('id'), str):
            return result['id']
        results_field = result.get('results')
        if isinstance(results_field, list) and results_field:
            first = results_field[0]
            if isinstance(first, dict) and isinstance(first.get('id'), str):
                return first['id']
            if isinstance(first, str):
                return first or None
    return None


def _serialize_memory_chunk(item: Any) -> str:
    """mem0.search results may be {'memory': str} dicts or strings; coerce
    safely to a single-line preview string."""
    if isinstance(item, dict):
        text = item.get('memory') or item.get('text') or json.dumps(item, ensure_ascii=False)
    else:
        text = str(item)
    return text.strip().replace('\n', ' ')


def _conversation_turn_role_for_llm(role: str) -> str:
    return 'model' if role == 'assistant' else 'user'


def _build_llm_contents_from_archive(
    turns: list[ConversationTurn],
    *,
    current_user_text: str,
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for turn in turns:
        text = turn.content.strip()
        if not text:
            continue
        contents.append({
            'role': _conversation_turn_role_for_llm(turn.role),
            'parts': [{'text': text}],
        })

    if not contents:
        return [{'role': 'user', 'parts': [{'text': current_user_text}]}]

    last_part = contents[-1]['parts'][0]
    if contents[-1]['role'] != 'user' or last_part.get('text') != current_user_text:
        contents.append({'role': 'user', 'parts': [{'text': current_user_text}]})

    return contents


def _select_archived_context_turns(
    turns: list[ConversationTurn],
    *,
    current_turn_id: str,
    limit: int,
) -> list[ConversationTurn]:
    history_only = [turn for turn in turns if turn.turn_id != current_turn_id]
    if len(history_only) <= limit:
        return history_only
    return history_only[-limit:]


def _parse_reminder_tool_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith('Z'):
        candidate = candidate[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_reminder_body_for_dedupe(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', value.lower())
    tokens = [token for token in cleaned.split() if token not in _REMINDER_DEDUPE_STOPWORDS]
    return ' '.join(tokens)


def _reminder_tool_args_are_duplicates(left_args: dict[str, Any], right_args: dict[str, Any]) -> bool:
    left_when = _parse_reminder_tool_datetime(left_args.get('next_fire_at'))
    right_when = _parse_reminder_tool_datetime(right_args.get('next_fire_at'))
    if left_when is None or right_when is None:
        return False
    if abs((left_when - right_when).total_seconds()) > 60:
        return False
    if (left_args.get('recurrence') or None) != (right_args.get('recurrence') or None):
        return False
    left_body = _normalize_reminder_body_for_dedupe(left_args.get('body'))
    right_body = _normalize_reminder_body_for_dedupe(right_args.get('body'))
    if not left_body or not right_body:
        return False
    if left_body == right_body:
        return True
    return SequenceMatcher(None, left_body, right_body).ratio() >= _REMINDER_BODY_SIMILARITY_THRESHOLD


def _collapse_duplicate_reminder_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if tool_call.get('name') != 'create_reminder':
            collapsed.append(tool_call)
            continue
        args = dict(tool_call.get('arguments') or {})
        replaced = False
        for index, existing in enumerate(collapsed):
            if existing.get('name') != 'create_reminder':
                continue
            existing_args = dict(existing.get('arguments') or {})
            if _reminder_tool_args_are_duplicates(existing_args, args):
                collapsed[index] = tool_call
                replaced = True
                break
        if not replaced:
            collapsed.append(tool_call)
    return collapsed


def _next_local_weekday_time(
    *,
    now: datetime,
    weekday: int,
    hour: int,
    minute: int,
    app_timezone: str,
) -> datetime:
    local_now = now.astimezone(ZoneInfo(app_timezone))
    days_ahead = (weekday - local_now.weekday()) % 7
    candidate = (local_now + timedelta(days=days_ahead)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if candidate <= local_now:
        candidate += timedelta(days=7)
    return candidate.astimezone(timezone.utc)


def _extract_followup_datetime(
    *,
    text: str,
    now: datetime,
    app_timezone: str,
) -> tuple[datetime, str] | None:
    lowered = _normalize_prompt_text(text)
    schedule_scope = lowered
    schedule_anchor = max(
        lowered.rfind(token)
        for token in ('remind me', 'follow up', 'reminder')
    )
    if schedule_anchor != -1:
        schedule_scope = lowered[schedule_anchor:]
    if 'morning' in schedule_scope:
        latest_weekday_match: tuple[int, str, int] | None = None
        for weekday_name, weekday_index in _WEEKDAY_NAME_TO_INDEX.items():
            for match in re.finditer(rf'\b{weekday_name}\b', schedule_scope):
                latest_weekday_match = (match.start(), weekday_name, weekday_index)
        if latest_weekday_match is not None:
            _, weekday_name, weekday_index = latest_weekday_match
            return (
                _next_local_weekday_time(
                    now=now,
                    weekday=weekday_index,
                    hour=9,
                    minute=0,
                    app_timezone=app_timezone,
                ),
                f'{weekday_name.title()} morning',
            )
        if 'tomorrow' in schedule_scope:
            local_now = now.astimezone(ZoneInfo(app_timezone))
            candidate = (local_now + timedelta(days=1)).replace(
                hour=9,
                minute=0,
                second=0,
                microsecond=0,
            )
            return candidate.astimezone(timezone.utc), 'tomorrow morning'
    return None


def _extract_followup_target_name(text: str) -> str | None:
    for pattern in (
        r'\bcheck if\s+([a-z][a-z0-9_-]*)\s+replied\b',
        r'\bfollow up with\s+([a-z][a-z0-9_-]*)\b',
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).strip().title()
    return None


def _normalize_unit_reference(raw_value: str) -> str | None:
    cleaned = ' '.join((raw_value or '').strip().split())
    if not cleaned:
        return None
    combined_match = re.fullmatch(r'(\d+)\s+and\s+(\d+)', cleaned, flags=re.IGNORECASE)
    if combined_match is not None:
        left, right = combined_match.groups()
        if left.endswith('0' * len(right)) and len(left) > len(right):
            normalized_digits = f'{left[:-len(right)]}{right}'
        else:
            normalized_digits = f'{left}{right}'
        return f'Unit {normalized_digits}'
    direct_match = re.search(r'(\d{1,6})', cleaned)
    if direct_match is not None:
        return f'Unit {direct_match.group(1)}'
    return None


def _extract_latest_unit_reference(text: str) -> str | None:
    matches = list(re.finditer(
        r'\bunit\s+([a-z0-9][a-z0-9\s-]*?)(?=[\.,;\n]|$)',
        text,
        flags=re.IGNORECASE,
    ))
    if not matches:
        return None
    return _normalize_unit_reference(matches[-1].group(1))


def _extract_issue_summary(text: str) -> str | None:
    lowered = _normalize_prompt_text(text)
    if 'water damage' in lowered:
        return 'urgent water damage' if 'urgent' in lowered else 'water damage'
    if 'leak' in lowered:
        return 'urgent leak' if 'urgent' in lowered else 'leak'
    if 'repair' in lowered:
        return 'urgent repair' if 'urgent' in lowered else 'repair'
    if 'maintenance' in lowered:
        return 'urgent maintenance issue' if 'urgent' in lowered else 'maintenance issue'
    return None


def _has_outbound_message_intent(text: str) -> bool:
    lowered = _normalize_prompt_text(text)
    return any(
        token in lowered
        for token in ('tell ', 'send ', 'ask ', 'message ')
    )


def _extract_contact_alias_candidates(raw_alias: str) -> list[str]:
    cleaned = ' '.join((raw_alias or '').strip().split()).strip(" '\".,!?")
    if not cleaned:
        return []
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        normalized = ' '.join(candidate.strip().split()).strip(" '\".,!?")
        key = normalized.casefold()
        if normalized and key not in seen:
            candidates.append(normalized)
            seen.add(key)

    add(cleaned)
    lowered = cleaned.casefold()
    for prefix in ('my ', 'our ', 'the '):
        if lowered.startswith(prefix):
            add(cleaned[len(prefix):])
    for marker in ("'s ", "’s "):
        if marker in cleaned:
            add(cleaned.split(marker, 1)[1])
    return candidates


def _render_contact_reminder_time_label(
    *,
    schedule_text: str,
    fire_at: datetime,
    app_timezone: str,
) -> str:
    cleaned = ' '.join((schedule_text or '').strip().split())
    lowered = cleaned.casefold()
    if lowered.startswith('in '):
        return lowered
    if any(token in lowered for token in ('tomorrow', 'today', 'tonight')):
        return lowered
    return format_local_datetime(fire_at, app_timezone)


def _normalize_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_digest_cleanup_candidates(message: str) -> list[dict[str, Any]]:
    lines = [' '.join(line.strip().split()) for line in (message or '').splitlines()]
    section: str | None = None
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(*, kind: str, label: str, descriptor: str) -> None:
        cleaned = ' '.join((label or '').strip().split()).strip(" \t\r\n.,!?;:'\"")
        if not cleaned:
            return
        key = (kind, cleaned.casefold())
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            'kind': kind,
            'label': cleaned,
            'descriptor': descriptor,
        })

    task_patterns = (
        ('task', 'overdue task', re.compile(r'^(?:Overdue|באיחור):\s+(.+)$', re.IGNORECASE)),
        ('task', 'pending task', re.compile(r'^(?:Pending|בהמתנה):\s+(.+)$', re.IGNORECASE)),
        ('task', 'today task', re.compile(r'^(?:Today|היום):\s+(.+?)\s+\(.+\)$', re.IGNORECASE)),
        ('task', 'upcoming task', re.compile(r'^(?:Upcoming|בהמשך):\s+(.+?)\s+\(.+\)$', re.IGNORECASE)),
        ('reminder', 'active reminder', re.compile(r'^(?:Reminder|תזכורת)\s+.+:\s+(.+)$', re.IGNORECASE)),
    )

    for line in lines:
        if not line:
            continue
        lowered = line.casefold()
        if lowered in {'reminders today:', 'תזכורות להיום:'}:
            section = 'reminders'
            continue
        if lowered in {'top tasks:', 'משימות עיקריות:'}:
            section = 'tasks'
            continue

        if section == 'reminders':
            reminder_match = re.match(r'^-\s+.+:\s+(.+)$', line)
            if reminder_match is not None:
                add_candidate(
                    kind='reminder',
                    label=reminder_match.group(1),
                    descriptor='active reminder',
                )
                continue
            section = None

        if section == 'tasks':
            matched = False
            for kind, descriptor, pattern in task_patterns:
                result = pattern.match(line)
                if result is None:
                    continue
                add_candidate(
                    kind=kind,
                    label=result.group(1),
                    descriptor=descriptor,
                )
                matched = True
                break
            if matched:
                continue
            if lowered in {'you are clear for now.', 'אתה פנוי לעת עתה.'}:
                section = None
                continue
            section = None

    return candidates


def _cleanup_timeout_target_phrase(value: str) -> str | None:
    cleaned = ' '.join((value or '').strip().split()).strip(" \t\r\n.,!?;:'\"-")
    if not cleaned:
        return None
    cleaned = re.sub(r'^(?:the|my|our|your)\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+(?:task|tasks|reminder|reminders)\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\r\n.,!?;:'\"-")
    return cleaned or None


def _extract_post_approval_cleanup_target(text: str) -> str | None:
    cleaned = ' '.join((text or '').strip().split()).strip(" \t\r\n.,!?;:'\"")
    if not cleaned:
        return None

    mark_match = re.match(
        r'^(?:please\s+)?mark\s+(?P<target>.+?)\s+(?:done|complete(?:d)?)$',
        cleaned,
        flags=re.IGNORECASE,
    )
    if mark_match is not None:
        return _cleanup_timeout_target_phrase(mark_match.group('target'))

    lowered = cleaned.casefold()
    for prefix in _POST_APPROVAL_CLEANUP_PREFIXES:
        if not lowered.startswith(prefix):
            continue
        remainder = cleaned[len(prefix):]
        remainder_lower = remainder.casefold()
        for suffix in _POST_APPROVAL_CLEANUP_TRAILING_PHRASES:
            if remainder_lower.endswith(suffix):
                remainder = remainder[:-len(suffix)]
                break
        return _cleanup_timeout_target_phrase(remainder)

    trailing_mark_match = re.match(
        r'^(?P<target>.+?)\s+mark\s+(?:it\s+)?(?:done|complete(?:d)?)$',
        cleaned,
        flags=re.IGNORECASE,
    )
    if trailing_mark_match is not None:
        return _cleanup_timeout_target_phrase(trailing_mark_match.group('target'))
    return None


def _normalize_post_approval_cleanup_text(value: str) -> str:
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', (value or '').casefold())
    tokens = [
        token
        for token in cleaned.split()
        if token not in _POST_APPROVAL_CLEANUP_STOPWORDS
    ]
    return ' '.join(tokens)


def _is_generic_post_approval_cleanup_target(query: str) -> bool:
    normalized = _normalize_post_approval_cleanup_text(query)
    return normalized in _POST_APPROVAL_CLEANUP_GENERIC_TARGETS


def _score_post_approval_cleanup_match(*, query: str, candidate: str) -> float:
    query_norm = _normalize_post_approval_cleanup_text(query)
    candidate_norm = _normalize_post_approval_cleanup_text(candidate)
    if not query_norm or not candidate_norm:
        return 0.0

    query_tokens = query_norm.split()
    candidate_tokens = candidate_norm.split()
    if not query_tokens or not candidate_tokens:
        return 0.0

    if len(query_tokens) == 1:
        return 1.0 if query_norm == candidate_norm else 0.0

    if query_norm == candidate_norm:
        return 1.0
    if query_norm in candidate_norm:
        return 0.97

    shared = set(query_tokens) & set(candidate_tokens)
    if len(shared) != len(set(query_tokens)):
        return 0.0

    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    if ratio < 0.55:
        return 0.0
    return max(0.94, ratio)


def _post_approval_cleanup_candidate_descriptor(
    *,
    kind: str,
    due_at: str | None,
    now: datetime,
) -> str:
    if kind == 'task':
        parsed_due = _parse_reminder_tool_datetime(due_at)
        if parsed_due is not None and parsed_due < now:
            return 'overdue task'
        return 'pending task'
    return 'active reminder'


async def _list_post_approval_cleanup_matches(
    registry: ToolRegistry,
    *,
    user_id: str,
    query: str,
    now: datetime,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    task_spec = registry.get('list_pending_tasks')
    if task_spec is not None and not task_spec.requires_approval:
        payload = await _invoke_tool_payload(task_spec, args={'user_id': user_id})
        data = payload.get('data') if payload else None
        tasks = data.get('tasks') if isinstance(data, dict) else None
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                title = str(task.get('title') or '').strip()
                item_id = str(task.get('id') or '').strip()
                if not title or not item_id:
                    continue
                score = _score_post_approval_cleanup_match(query=query, candidate=title)
                if score < _POST_APPROVAL_CLEANUP_MATCH_THRESHOLD:
                    continue
                matches.append({
                    'kind': 'task',
                    'id': item_id,
                    'label': title,
                    'descriptor': _post_approval_cleanup_candidate_descriptor(
                        kind='task',
                        due_at=task.get('due_at'),
                        now=now,
                    ),
                    'score': score,
                })

    reminder_spec = registry.get('list_active_reminders')
    if reminder_spec is not None and not reminder_spec.requires_approval:
        payload = await _invoke_tool_payload(reminder_spec, args={'user_id': user_id})
        data = payload.get('data') if payload else None
        reminders = data.get('reminders') if isinstance(data, dict) else None
        if isinstance(reminders, list):
            for reminder in reminders:
                if not isinstance(reminder, dict):
                    continue
                body = str(reminder.get('body') or '').strip()
                item_id = str(reminder.get('id') or '').strip()
                if not body or not item_id:
                    continue
                score = _score_post_approval_cleanup_match(query=query, candidate=body)
                if score < _POST_APPROVAL_CLEANUP_MATCH_THRESHOLD:
                    continue
                matches.append({
                    'kind': 'reminder',
                    'id': item_id,
                    'label': body,
                    'descriptor': _post_approval_cleanup_candidate_descriptor(
                        kind='reminder',
                        due_at=reminder.get('next_fire_at'),
                        now=now,
                    ),
                    'score': score,
                })

    kind_priority = {'task': 0, 'reminder': 1}
    return sorted(
        matches,
        key=lambda item: (
            -float(item.get('score') or 0.0),
            kind_priority.get(str(item.get('kind')), 99),
            str(item.get('label') or '').casefold(),
        ),
    )


async def _apply_post_approval_cleanup_match(
    registry: ToolRegistry,
    *,
    user_id: str,
    match: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    if match.get('kind') == 'task':
        spec = registry.get('mark_task_done')
        if spec is None:
            return {
                'kind': 'cleanup_failed',
                'query': query,
                'label': match.get('label'),
            }
        payload = await _invoke_tool_payload(
            spec,
            args={'user_id': user_id, 'query': match['id']},
        )
        data = payload.get('data') if payload else None
        if not payload or payload.get('success') is not True:
            return {
                'kind': 'cleanup_failed',
                'query': query,
                'label': match.get('label'),
            }
        if not isinstance(data, dict) or data.get('matched') is not True:
            return {
                'kind': 'cleanup_failed',
                'query': query,
                'label': match.get('label'),
            }
        return {
            'kind': 'task_cleanup',
            'task_id': match['id'],
            'title': match['label'],
            'query': query,
        }

    spec = registry.get('delete_reminder')
    if spec is None:
        return {
            'kind': 'cleanup_failed',
            'query': query,
            'label': match.get('label'),
        }
    # Approval already happened at the message-gate level, so execute the
    # bound destructive reminder tool directly against the exact matched ID.
    payload = await _invoke_tool_payload(
        spec,
        args={'user_id': user_id, 'query': match['id']},
    )
    data = payload.get('data') if payload else None
    if not payload or payload.get('success') is not True:
        return {
            'kind': 'cleanup_failed',
            'query': query,
            'label': match.get('label'),
        }
    if not isinstance(data, dict) or data.get('cancelled') is not True:
        return {
            'kind': 'cleanup_failed',
            'query': query,
            'label': match.get('label'),
        }
    return {
        'kind': 'reminder_cleanup',
        'reminder_id': match['id'],
        'title': match['label'],
        'query': query,
    }


async def _resolve_post_approval_digest_cleanup_target(
    *,
    proactive_notifications_repository: ProactiveNotificationsRepository | None,
    registry: ToolRegistry,
    user_id: str,
    now: datetime,
) -> dict[str, Any]:
    if proactive_notifications_repository is None:
        return {'kind': 'digest_cleanup_missing'}

    latest = proactive_notifications_repository.latest_for_user(
        user_id=user_id,
        notification_type='morning_briefing',
    )
    sent_at = _normalize_utc_datetime(getattr(latest, 'last_sent_at', None))
    if latest is None or sent_at is None:
        return {'kind': 'digest_cleanup_missing'}
    if now - sent_at > _POST_APPROVAL_DIGEST_CONTEXT_MAX_AGE:
        return {'kind': 'digest_cleanup_missing'}

    digest_candidates = _parse_digest_cleanup_candidates(latest.message)
    if len(digest_candidates) != 1:
        if not digest_candidates:
            return {'kind': 'digest_cleanup_missing'}
        return {
            'kind': 'digest_cleanup_ambiguous',
            'matches': digest_candidates,
        }

    digest_candidate = digest_candidates[0]
    matches = await _list_post_approval_cleanup_matches(
        registry,
        user_id=user_id,
        query=digest_candidate['label'],
        now=now,
    )
    scoped_matches = [
        match for match in matches
        if match.get('kind') == digest_candidate.get('kind')
    ]
    if len(scoped_matches) != 1:
        if not scoped_matches:
            return {'kind': 'digest_cleanup_missing'}
        return {
            'kind': 'digest_cleanup_ambiguous',
            'matches': scoped_matches,
        }
    return await _apply_post_approval_cleanup_match(
        registry,
        user_id=user_id,
        match=scoped_matches[0],
        query=digest_candidate['label'],
    )


async def _execute_post_approval_timeout_cleanup_fallback(
    *,
    proactive_notifications_repository: ProactiveNotificationsRepository | None,
    registry: ToolRegistry,
    user_id: str,
    original_prompt: str,
    now: datetime,
) -> dict[str, Any] | None:
    query = _extract_post_approval_cleanup_target(original_prompt)
    if query is None:
        return None

    if _is_generic_post_approval_cleanup_target(query):
        return await _resolve_post_approval_digest_cleanup_target(
            proactive_notifications_repository=proactive_notifications_repository,
            registry=registry,
            user_id=user_id,
            now=now,
        )

    matches = await _list_post_approval_cleanup_matches(
        registry,
        user_id=user_id,
        query=query,
        now=now,
    )
    if not matches:
        return {
            'kind': 'cleanup_not_found',
            'query': query,
        }
    if len(matches) > 1:
        return {
            'kind': 'cleanup_ambiguous',
            'query': query,
            'matches': matches,
        }
    return await _apply_post_approval_cleanup_match(
        registry,
        user_id=user_id,
        match=matches[0],
        query=query,
    )


def _parse_contact_reminder_timeout_fallback_args(
    *,
    text: str,
    now: datetime,
    app_timezone: str,
) -> dict[str, Any] | None:
    patterns = (
        re.compile(
            r'^\s*remind\s+(?P<alias>.+?)\s+(?:on|via)\s+'
            r'(?P<channel>whatsapp|sms)\s+(?P<schedule>.+?)\s+to\s+'
            r'(?P<body>.+?)\s*$',
            flags=re.IGNORECASE,
        ),
        re.compile(
            r'^\s*(?:send|text|message)\s+(?P<alias>.+?)\s+'
            r'(?:a\s+)?reminder\s+(?:on|via)\s+'
            r'(?P<channel>whatsapp|sms)\s+(?P<schedule>.+?)\s+to\s+'
            r'(?P<body>.+?)\s*$',
            flags=re.IGNORECASE,
        ),
    )
    match = next((pat.match((text or '').strip()) for pat in patterns if pat.match((text or '').strip())), None)
    if match is None:
        return None

    alias_candidates = _extract_contact_alias_candidates(match.group('alias'))
    if not alias_candidates:
        return None

    channel = match.group('channel').strip().lower()
    schedule_text = ' '.join(match.group('schedule').strip().split()).strip(" '\"")
    body = ' '.join(match.group('body').strip().split()).rstrip('.!?')
    if not schedule_text or not body:
        return None

    local_now = now.astimezone(ZoneInfo(app_timezone))
    parsed = dateparser.parse(
        schedule_text,
        settings={
            'TIMEZONE': app_timezone,
            'RETURN_AS_TIMEZONE_AWARE': True,
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': local_now,
        },
    )
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(app_timezone))
    fire_at = parsed.astimezone(timezone.utc)
    return {
        'kind': 'contact_reminder',
        'created': False,
        'alias_candidates': alias_candidates,
        'alias_label': alias_candidates[-1],
        'channel': channel,
        'body': body,
        'fire_at': fire_at.isoformat(),
        'time_label': _render_contact_reminder_time_label(
            schedule_text=schedule_text,
            fire_at=fire_at,
            app_timezone=app_timezone,
        ),
    }


async def _resolve_contact_alias_for_timeout_fallback(
    registry: ToolRegistry,
    *,
    user_id: str,
    alias_candidates: list[str],
) -> dict[str, Any] | None:
    spec = registry.get('resolve_contact_alias')
    if spec is None or spec.requires_approval:
        return None

    ambiguous: dict[str, Any] | None = None
    for candidate in alias_candidates:
        payload = await _invoke_tool_payload(
            spec,
            args={'user_id': user_id, 'query': candidate},
        )
        if not payload or payload.get('success') is not True:
            continue
        data = payload.get('data')
        if not isinstance(data, dict) or data.get('ok') is not True:
            continue
        match = data.get('match')
        if match == 'unique':
            return {
                'match': 'unique',
                'alias_query': candidate,
                'contact': data.get('contact'),
                'alias_used': data.get('alias_used'),
            }
        if match == 'ambiguous' and ambiguous is None:
            ambiguous = {
                'match': 'ambiguous',
                'alias_query': candidate,
                'candidates': data.get('candidates') or [],
            }
    if ambiguous is not None:
        return ambiguous
    return {
        'match': 'none',
        'alias_query': alias_candidates[-1],
    }


def _format_contact_alias_label(contact_data: dict[str, Any] | None) -> str:
    if not isinstance(contact_data, dict):
        return 'that contact'
    aliases = [
        str(alias).strip()
        for alias in (contact_data.get('aliases') or [])
        if str(alias).strip()
    ]
    if not aliases:
        return 'that contact'
    return '/'.join(aliases[:2])


async def _execute_post_approval_timeout_contact_reminder_fallback(
    *,
    registry: ToolRegistry,
    user_id: str,
    original_prompt: str,
    app_timezone: str,
    now: datetime,
) -> dict[str, Any] | None:
    parsed = _parse_contact_reminder_timeout_fallback_args(
        text=original_prompt,
        now=now,
        app_timezone=app_timezone,
    )
    if parsed is None:
        return None

    resolution = await _resolve_contact_alias_for_timeout_fallback(
        registry,
        user_id=user_id,
        alias_candidates=list(parsed['alias_candidates']),
    )
    if resolution is None:
        return None

    if resolution.get('match') == 'none':
        return {
            'kind': 'contact_missing',
            'created': False,
            'channel': parsed['channel'],
            'alias_label': resolution.get('alias_query') or parsed['alias_label'],
            'reminder_id': None,
        }

    if resolution.get('match') == 'ambiguous':
        return {
            'kind': 'contact_ambiguous',
            'created': False,
            'channel': parsed['channel'],
            'alias_label': resolution.get('alias_query') or parsed['alias_label'],
            'reminder_id': None,
        }

    contact_data = resolution.get('contact')
    spec = registry.get('create_contact_reminder')
    if spec is None or spec.requires_approval:
        return None
    payload = await _invoke_tool_payload(
        spec,
        args={
            'user_id': user_id,
            'alias': resolution.get('alias_query') or parsed['alias_label'],
            'body': parsed['body'],
            'fire_at': parsed['fire_at'],
            'channel': parsed['channel'],
        },
    )
    if not payload or payload.get('success') is not True:
        return None
    data = payload.get('data')
    if not isinstance(data, dict) or data.get('ok') is not True:
        return None
    reminder_id = data.get('reminder_id')
    if not reminder_id:
        return None
    preview = data.get('preview') if isinstance(data.get('preview'), dict) else {}
    return {
        'kind': 'contact_reminder',
        'created': True,
        'reminder_id': reminder_id,
        'channel': str(preview.get('channel') or parsed['channel']).strip().lower(),
        'body': parsed['body'],
        'time_label': parsed['time_label'],
        'contact_label': _format_contact_alias_label(contact_data),
        'target_contact_id': (
            contact_data.get('id') if isinstance(contact_data, dict) else None
        ),
    }


def _build_timeout_fallback_reminder_args(
    *,
    text: str,
    now: datetime,
    app_timezone: str,
) -> dict[str, Any] | None:
    if not _prompt_has_rich_reminder_context(text):
        return None
    when = _extract_followup_datetime(text=text, now=now, app_timezone=app_timezone)
    target_name = _extract_followup_target_name(text)
    issue_summary = _extract_issue_summary(text)
    unit_reference = _extract_latest_unit_reference(text)
    if when is None or target_name is None or issue_summary is None or unit_reference is None:
        return None
    when_dt, when_label = when
    body = f'Follow up with {target_name} about {issue_summary} in {unit_reference}.'
    return {
        'body': body,
        'next_fire_at': when_dt.isoformat(),
        'time_label': when_label,
        'target_name': target_name,
        'issue_summary': issue_summary,
        'unit_reference': unit_reference,
    }


async def _invoke_tool_payload(spec, *, args: dict[str, Any]) -> dict[str, Any] | None:
    try:
        result = spec.fn(**args)
    except TypeError:
        args_no_user = {k: v for k, v in args.items() if k != 'user_id'}
        try:
            result = spec.fn(**args_no_user)
        except Exception:
            return None
    except Exception:
        return None

    if inspect.iscoroutine(result):
        try:
            result = await result
        except Exception:
            return None

    if isinstance(result, ToolResult):
        return {
            'success': result.success,
            'data': result.data,
            'announcement': result.announcement,
            'error': result.error,
        }
    return {'success': True, 'data': result}


async def _find_existing_duplicate_reminder(
    registry: ToolRegistry,
    *,
    user_id: str,
    body: str,
    next_fire_at: str,
) -> dict[str, Any] | None:
    spec = registry.get('list_active_reminders')
    if spec is None or spec.requires_approval:
        return None
    payload = await _invoke_tool_payload(spec, args={'user_id': user_id})
    if not payload or payload.get('success') is not True:
        return None
    data = payload.get('data')
    reminders = data.get('reminders') if isinstance(data, dict) else None
    if not isinstance(reminders, list):
        return None
    candidate_args = {'body': body, 'next_fire_at': next_fire_at, 'recurrence': None}
    for reminder in reminders:
        if not isinstance(reminder, dict):
            continue
        existing_args = {
            'body': reminder.get('body'),
            'next_fire_at': reminder.get('next_fire_at'),
            'recurrence': reminder.get('recurrence'),
        }
        if _reminder_tool_args_are_duplicates(existing_args, candidate_args):
            return reminder
    return None


async def _execute_post_approval_timeout_internal_reminder_fallback(
    *,
    registry: ToolRegistry,
    user_id: str,
    original_prompt: str,
    app_timezone: str,
    now: datetime,
) -> dict[str, Any] | None:
    reminder_args = _build_timeout_fallback_reminder_args(
        text=original_prompt,
        now=now,
        app_timezone=app_timezone,
    )
    if reminder_args is None:
        return None

    duplicate = await _find_existing_duplicate_reminder(
        registry,
        user_id=user_id,
        body=reminder_args['body'],
        next_fire_at=reminder_args['next_fire_at'],
    )
    if duplicate is not None:
        return {
            'created': False,
            'reminder_id': duplicate.get('id'),
            'body': reminder_args['body'],
            'time_label': reminder_args['time_label'],
            'target_name': reminder_args['target_name'],
            'issue_summary': reminder_args['issue_summary'],
            'unit_reference': reminder_args['unit_reference'],
        }

    spec = registry.get('create_reminder')
    if spec is None or spec.requires_approval:
        return None
    payload = await _invoke_tool_payload(
        spec,
        args={
            'user_id': user_id,
            'body': reminder_args['body'],
            'next_fire_at': reminder_args['next_fire_at'],
        },
    )
    if not payload or payload.get('success') is not True:
        return None
    data = payload.get('data')
    if not isinstance(data, dict) or data.get('created') is not True:
        return None
    return {
        'created': True,
        'reminder_id': data.get('reminder_id'),
        'body': reminder_args['body'],
        'time_label': reminder_args['time_label'],
        'target_name': reminder_args['target_name'],
        'issue_summary': reminder_args['issue_summary'],
        'unit_reference': reminder_args['unit_reference'],
    }


def _build_system_prompt(
    *,
    persona: str,
    telos: Optional[str],
    memories: list[Any],
    language: str,
    now: datetime,
    app_timezone: str,
    recovery_note: str | None = None,
) -> str:
    """Compose the dispatcher system prompt.

    H2-014 (V3.5.1): always inject ground-truth current time so Gemini
    is not forced to choose between calling get_current_time or falling
    back to its training-data prior. The factual-grounding clause
    explicitly instructs Gemini to use these values (or a fresh
    get_current_time call) verbatim, never invent.
    """
    parts = [persona]
    user_now = now.astimezone(ZoneInfo(app_timezone))
    parts.append(
        '## Current time\n'
        f'UTC: {now.isoformat()}\n'
        f'User timezone ({app_timezone}): {user_now.isoformat()}\n'
        'Use these values when stating the current date or time in any '
        'user-facing reply. Do NOT invent or estimate timestamps from prior '
        'knowledge. If you need a more precise time mid-turn, call the '
        'get_current_time tool and use its returned `iso` field verbatim.'
    )
    # H2-043 FIX A: cross-source capability introspection. Pre-H2-043 the LLM
    # answered "what Gmails do you have access to?" by reading whichever
    # single surface (nexus-email OAuth state OR dashboard quick_links) it
    # happened to recall, and missed the other. The describe_my_access tool
    # in nexus-self merges both surfaces; instruct the LLM to call it first
    # on any capability/access question so the answer is honest.
    parts.append(
        '## Capability questions\n'
        'When the user asks about capabilities, access, or "what can you do '
        'with X" (email, messaging, drive, etc.), call '
        '`mcp__nexus-self__describe_my_access(domain="<domain>")` FIRST and '
        'answer from its return value. Do not guess from a single surface — '
        'the tool merges API-authenticated state (e.g. Gmail OAuth accounts) '
        'with browser-clickable dashboard quick_links so the answer covers '
        'both.'
    )
    # Always echo self-corrections. When the user revises themselves inside a
    # single message ("June 2... no June 4", "5pm wait 6pm", "tell Dan, actually
    # Dana"), silently applying the latest value leaves the user unsure the
    # change registered. State it explicitly. This is unconditional: the model
    # reads the raw user text, so it covers correction forms the deterministic
    # recovery layer does not flag (e.g. numeric calendar dates).
    parts.append(
        '## Self-corrections\n'
        'If the user corrects themselves within a single message (e.g. '
        '"June 2... no June 4", "5pm wait 6pm", "tell Dan, actually Dana"), '
        'state the change explicitly in your confirmation on its own line, '
        'formatted exactly as:\n'
        'Corrected <field>: <old value> -> <new value>\n'
        'Use the superseded value as <old value> and the value you acted on as '
        '<new value> (e.g. "Corrected date: June 2 -> June 4"). Never silently '
        'apply a self-correction. If the message contains no self-correction, '
        'omit this line entirely.'
    )
    if telos:
        parts.append('## User TELOS\n' + telos.strip())
    if memories:
        bullet_lines = [f'- {_serialize_memory_chunk(m)}' for m in memories]
        parts.append('## Retrieved memories\n' + '\n'.join(bullet_lines))
    if recovery_note:
        parts.append(recovery_note)
    parts.append(f'## Language\nReply in: {language}')
    return '\n\n'.join(parts)


def _coerce_tool_catalog(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Render every registered spec into the LLM-facing schema.
    Compact dict form intentional — the LLM tool-call wire format
    method translates this into Gemini-specific functionDeclarations."""
    catalog = []
    for spec in registry.all():
        catalog.append({
            'name': spec.name,
            'description': spec.description or spec.name,
            'parameters': spec.parameters or {'type': 'object', 'properties': {}, 'required': []},
            'requires_approval': spec.requires_approval,
        })
    return catalog


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run a sync callable without relying on asyncio's global executor.

    In this environment, both `asyncio.to_thread()` and short-lived
    ThreadPoolExecutor wrappers can leave pytest subprocesses hanging after the
    awaited work already finished. Use a one-off daemon thread and a simple
    completion event instead.
    """
    result_box: dict[str, object] = {}
    done = threading.Event()

    def worker():
        try:
            result_box['result'] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - re-raised by awaiter
            result_box['exception'] = exc
        finally:
            done.set()

    threading.Thread(
        target=worker,
        daemon=True,
        name='tool-dispatcher-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


@dataclass(slots=True)
class ToolDispatcher:
    llm: LLMClient
    registry: ToolRegistry
    telos_service: TelosService
    mem0: Mem0Client
    approval_service: ApprovalService
    conversation_turns_repository: ConversationTurnsRepository
    conversation_service: ConversationService | None = None
    # H2-039 FIX 1: needed for the gate's approve/cancel callback routing
    # and for the existing Phase 4 button-callback wire that was missing
    # in production. Optional so existing tests that construct ToolDispatcher
    # without it continue to pass; gate degrades to no-op when absent.
    approvals_repository: Optional[ApprovalsRepository] = None
    proactive_notifications_repository: Optional[ProactiveNotificationsRepository] = None
    max_iterations: int = _MAX_DEFAULT_ITERATIONS
    app_timezone: str = 'UTC'
    # H2-046 Part 0: archival now fires-and-forgets via asyncio.create_task so
    # the (much slower) claude-based entity extraction doesn't add latency to
    # every reply. Tests that need to assert the post-archival DB state can
    # await `wait_for_archival_idle()` after their handle() call.
    _inflight_archival_tasks: set[asyncio.Task] = field(default_factory=set,
                                                       repr=False, compare=False)
    _duplicate_reminder_audit_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    recovery_layer: ConversationalRecoveryLayer = field(
        default_factory=ConversationalRecoveryLayer,
        repr=False,
        compare=False,
    )
    capability_registry: CapabilityRegistry = field(
        default_factory=CapabilityRegistry,
        repr=False,
        compare=False,
    )
    fallback_manager: FallbackManager = field(
        default_factory=FallbackManager,
        repr=False,
        compare=False,
    )
    persona: str = (
        "You are Nexus, a personal assistant. Use the user's TELOS and "
        "retrieved memories to ground every reply in their actual situation. "
        "Call tools to answer questions about reminders, tasks, memories, "
        "email, calendar, etc. When asking the user to take a destructive "
        "action (delete, send, disconnect), the dispatcher will surface a "
        "tap-to-approve prompt — do not pretend the action ran before "
        "approval. Never claim to be a large language model."
    )

    def _destructive_approval_enabled(self) -> bool:
        settings = getattr(self.capability_registry, 'settings', None)
        approval = getattr(settings, 'approval', None)
        return bool(getattr(approval, 'destructive_approval_enabled', True))

    def _load_recent_recovery_turns(self, *, user_id: str) -> list[ConversationTurn]:
        try:
            return self.conversation_turns_repository.list_recent_for_user(
                user_id=user_id,
                limit=6,
            )
        except Exception:
            logger.warning(
                'dispatcher_recovery_history_load_failed',
                extra={'user_id': user_id},
            )
            return []

    def _resolve_contact_alias_for_recovery(
        self,
        *,
        user_id: str,
        query: str,
    ) -> dict[str, Any] | None:
        spec = self.registry.get('resolve_contact_alias')
        if spec is None or spec.requires_approval:
            return None
        try:
            result = spec.fn(user_id=user_id, query=query)
        except TypeError:
            return None
        except Exception:
            logger.warning(
                'dispatcher_recovery_alias_resolution_failed',
                extra={'user_id': user_id},
            )
            return None

        if inspect.iscoroutine(result):
            return None
        return result if isinstance(result, dict) else None

    def _load_active_clarification(self, *, user_id: str) -> ActiveClarification | None:
        if self.conversation_service is None:
            return None
        try:
            payload = self.conversation_service.get_active_clarification(user_id)
        except Exception:
            logger.warning(
                'dispatcher_active_clarification_load_failed',
                extra={'user_id': user_id},
            )
            return None
        if not isinstance(payload, dict):
            return None
        clarification = ActiveClarification.from_dict(payload)
        if clarification is not None:
            return clarification
        try:
            self.conversation_service.clear_active_clarification(user_id)
        except Exception:
            logger.warning(
                'dispatcher_active_clarification_clear_failed',
                extra={'user_id': user_id},
            )
        return None

    def _store_active_clarification(
        self,
        *,
        user_id: str,
        clarification: ActiveClarification,
    ) -> None:
        if self.conversation_service is None:
            return
        try:
            self.conversation_service.store_active_clarification(
                user_id,
                clarification=clarification.to_dict(),
                topic=clarification.question,
            )
        except Exception:
            logger.warning(
                'dispatcher_active_clarification_store_failed',
                extra={'user_id': user_id},
            )

    def _clear_active_clarification(self, *, user_id: str) -> None:
        if self.conversation_service is None:
            return
        try:
            self.conversation_service.clear_active_clarification(user_id)
        except Exception:
            logger.warning(
                'dispatcher_active_clarification_clear_failed',
                extra={'user_id': user_id},
            )

    def _archive_short_circuit_reply(
        self,
        *,
        user_id: str,
        user_text: str,
        assistant_text: str,
        created_at: datetime | None = None,
    ) -> str:
        timestamp = created_at or utc_now()
        try:
            conversation_id = self.conversation_turns_repository.resolve_conversation_id(
                user_id=user_id,
                now=timestamp,
            )
            user_turn_id = self.conversation_turns_repository.insert(
                user_id=user_id,
                role='user',
                content=user_text,
                conversation_id=conversation_id,
                created_at=timestamp,
            )
            self.conversation_turns_repository.insert(
                user_id=user_id,
                role='assistant',
                content=assistant_text,
                conversation_id=conversation_id,
                created_at=timestamp,
            )
            return user_turn_id
        except Exception:
            logger.warning(
                'dispatcher_short_circuit_archive_failed',
                extra={'user_id': user_id},
            )
            return ''

    def _persist_recovery_context(
        self,
        *,
        user_id: str,
        result: RecoveryResult,
    ) -> None:
        if self.conversation_service is None:
            return
        recovery_updates = {
            key: value
            for key, value in dict(result.context_updates or {}).items()
            if value not in (None, '', [])
        }
        if not recovery_updates:
            return
        try:
            self.conversation_service.store_recovery_context(
                user_id,
                recovery_updates=recovery_updates,
                topic=result.recovered_text or result.raw_text,
                replace=True,
            )
        except Exception:
            logger.warning(
                'dispatcher_recovery_context_persist_failed',
                extra={'user_id': user_id},
            )

    def _load_recovery_context(self, *, user_id: str) -> dict[str, Any]:
        if self.conversation_service is None:
            return {}
        try:
            return self.conversation_service.get_recovery_context(user_id)
        except Exception:
            logger.warning(
                'dispatcher_recovery_context_load_failed',
                extra={'user_id': user_id},
            )
            return {}

    def _load_current_thread(self, *, user_id: str) -> dict[str, Any] | None:
        return _active_thread_from_recovery_state(
            self._load_recovery_context(user_id=user_id),
        )

    def _store_thread_state(
        self,
        *,
        user_id: str,
        recovery_state: dict[str, Any],
        thread: dict[str, Any] | None,
        topic: str | None = None,
    ) -> None:
        if self.conversation_service is None:
            return
        updated = dict(recovery_state or {})
        if thread is None:
            updated.pop('active_thread', None)
        else:
            updated['active_thread'] = dict(thread)
        try:
            self.conversation_service.store_recovery_context(
                user_id,
                recovery_updates=updated,
                topic=topic,
                replace=True,
            )
        except Exception:
            logger.warning(
                'dispatcher_thread_state_store_failed',
                extra={'user_id': user_id},
            )

    def _bind_clarification_to_thread(
        self,
        *,
        clarification: ActiveClarification,
        thread: dict[str, Any] | None,
        requested_slot: str | None,
        created_at: datetime,
    ) -> ActiveClarification:
        if thread is None:
            return clarification
        clarification.thread_id = str(thread.get('thread_id') or '').strip()
        clarification.thread_kind = str(thread.get('thread_kind') or 'unknown').strip() or 'unknown'
        clarification.thread_revision = _parse_thread_revision(
            thread.get('thread_revision') or 0
        )
        clarification.requested_slot = str(
            requested_slot or clarification.requested_slot or ''
        ).strip()
        clarification.expires_at = (
            created_at + timedelta(minutes=20)
        ).astimezone(timezone.utc).isoformat()
        return clarification

    def _load_pending_approvals(self, *, user_id: str) -> list[Any]:
        if self.approvals_repository is None:
            return []
        try:
            return self.approvals_repository.list_active_pending_for_user(user_id)
        except Exception:
            logger.warning(
                'dispatcher_pending_approvals_load_failed',
                extra={'user_id': user_id},
            )
            return []

    def _parse_approval_payload(self, approval_row: Any) -> dict[str, Any]:
        try:
            return json.loads(approval_row.payload_json or '{}')
        except Exception:
            return {}

    async def _load_active_reminders_payload(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, Any]] | None:
        spec = self.registry.get('list_active_reminders')
        if spec is None or spec.requires_approval:
            return None
        payload = await _invoke_tool_payload(spec, args={'user_id': user_id})
        if not payload or payload.get('success') is not True:
            return None
        data = payload.get('data')
        reminders = data.get('reminders') if isinstance(data, dict) else None
        if not isinstance(reminders, list):
            return []
        return [item for item in reminders if isinstance(item, dict)]

    async def _load_duplicate_reminder_clusters(
        self,
        *,
        user_id: str,
    ) -> list[ReminderDuplicateCluster]:
        reminders = await self._load_active_reminders_payload(user_id=user_id)
        if reminders is None:
            return []
        return cluster_duplicate_reminders(
            reminders,
            app_timezone=self.app_timezone,
        )

    def _store_duplicate_reminder_audit(
        self,
        *,
        user_id: str,
        clusters: list[ReminderDuplicateCluster],
    ) -> None:
        """Persist the duplicate audit to TWO places.

        1. Process-local in-memory cache (fast path, no DB round-trip
           when the same worker handles the follow-up command).
        2. conversation_service recovery_state — cross-process safe.
           The clusters live at the recovery_state top level
           ('duplicate_reminder_audit' key) because
           `_coerce_active_thread` strips fields it doesn't recognize,
           which would drop the clusters if they lived inside the
           active_thread. The active_thread of kind 'duplicate_audit'
           is the binding pointer that 'keep newest' reads to link the
           cleanup approval back to this audit run.

        A later /reminders duplicates supersedes the prior audit by
        bumping its revision and replacing both keys.

        If the new audit produces zero clusters, this is a no-op (nothing
        for 'keep newest' to bind to, and the no-duplicates branch of
        /reminders duplicates should stay stateless).
        """
        if not clusters:
            self._duplicate_reminder_audit_cache.pop(user_id, None)
            return
        now = utc_now()
        cluster_dicts = [cluster.to_dict() for cluster in clusters]
        expires_at = (now + timedelta(minutes=30)).isoformat()
        recovery_state = self._load_recovery_context(user_id=user_id)
        previous = _active_thread_from_recovery_state(recovery_state)
        prior_revision = 0
        if previous is not None and str(previous.get('thread_kind') or '') == 'duplicate_audit':
            prior_revision = _parse_thread_revision(previous.get('thread_revision') or 0)
        audit_thread = {
            'thread_id': str(uuid.uuid4()),
            'thread_kind': 'duplicate_audit',
            'thread_revision': prior_revision + 1,
            'status': 'audited',
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'expires_at': expires_at,
            'source_turn_ids': [],
        }
        in_memory_payload = {
            'clusters': cluster_dicts,
            'expires_at': expires_at,
            'audit_thread_id': audit_thread['thread_id'],
            'audit_thread_revision': audit_thread['thread_revision'],
        }
        self._duplicate_reminder_audit_cache[user_id] = in_memory_payload
        if self.conversation_service is None:
            return
        updated_state = dict(recovery_state or {})
        updated_state['active_thread'] = dict(audit_thread)
        updated_state['duplicate_reminder_audit'] = {
            'clusters': cluster_dicts,
            'expires_at': expires_at,
            'audit_thread_id': audit_thread['thread_id'],
            'audit_thread_revision': audit_thread['thread_revision'],
        }
        try:
            self.conversation_service.store_recovery_context(
                user_id,
                recovery_updates=updated_state,
                topic='duplicate reminder audit',
                replace=True,
            )
        except Exception:
            logger.warning(
                'dispatcher_duplicate_audit_persist_failed',
                extra={'user_id': user_id},
            )

    def _load_duplicate_reminder_audit(self, *, user_id: str) -> dict[str, Any] | None:
        """In-memory cache first; fall back to persisted recovery_state.

        Returning None means "no fresh audit available — ask the user
        to re-run /reminders duplicates". The returned payload always
        carries 'clusters', 'expires_at', 'audit_thread_id', and
        'audit_thread_revision' so 'keep newest' can bind safely.
        """
        payload = self._duplicate_reminder_audit_cache.get(user_id)
        if isinstance(payload, dict):
            if _is_iso_timestamp_expired(payload.get('expires_at')):
                self._duplicate_reminder_audit_cache.pop(user_id, None)
            else:
                return payload
        # Cross-process fallback. Clusters live at the top of
        # recovery_state (not in active_thread) — see _store above.
        recovery_state = self._load_recovery_context(user_id=user_id)
        persisted = (
            recovery_state.get('duplicate_reminder_audit')
            if isinstance(recovery_state, dict)
            else None
        )
        if not isinstance(persisted, dict):
            return None
        if _is_iso_timestamp_expired(persisted.get('expires_at')):
            return None
        clusters = persisted.get('clusters')
        if not isinstance(clusters, list):
            return None
        # If active_thread is still the matching duplicate_audit pointer,
        # prefer the thread id from there (authoritative); otherwise
        # trust the persisted payload's own audit_thread_id.
        thread = _active_thread_from_recovery_state(recovery_state)
        audit_thread_id = str(persisted.get('audit_thread_id') or '').strip()
        audit_thread_revision = _parse_thread_revision(
            persisted.get('audit_thread_revision') or 0
        )
        if (thread is not None
                and str(thread.get('thread_kind') or '') == 'duplicate_audit'
                and str(thread.get('thread_id') or '').strip()):
            audit_thread_id = str(thread.get('thread_id') or '').strip()
            audit_thread_revision = _parse_thread_revision(
                thread.get('thread_revision') or 0
            )
        rehydrated = {
            'clusters': clusters,
            'expires_at': persisted.get('expires_at'),
            'audit_thread_id': audit_thread_id,
            'audit_thread_revision': audit_thread_revision,
        }
        self._duplicate_reminder_audit_cache[user_id] = rehydrated
        return rehydrated

    def _clear_duplicate_reminder_audit(self, *, user_id: str) -> None:
        self._duplicate_reminder_audit_cache.pop(user_id, None)

    def _render_direct_reminder_list_text(
        self,
        *,
        reminders: list[dict[str, Any]],
        translator: Translator,
    ) -> str:
        if not reminders:
            return translator.t('reminder_list_empty')

        lines = ['Upcoming reminders:']
        for reminder in reminders[:20]:
            body = str(reminder.get('body') or '').strip() or 'Reminder'
            when_text = ''
            raw_when = str(reminder.get('next_fire_at') or '').strip()
            if raw_when:
                candidate = raw_when.replace('Z', '+00:00')
                try:
                    when_text = format_local_datetime(
                        datetime.fromisoformat(candidate),
                        self.app_timezone,
                    )
                except ValueError:
                    when_text = raw_when
            if when_text:
                lines.append(f'- {when_text}: {body}')
            else:
                lines.append(f'- {body}')
        return '\n'.join(lines)

    async def _handle_direct_reminder_read_command(
        self,
        *,
        command_kind: str,
        user: User,
        translator: Translator,
    ) -> DispatcherOutput:
        reminders = await self._load_active_reminders_payload(user_id=user.id)
        if reminders is None:
            if command_kind == 'duplicates':
                text = (
                    "I couldn't inspect reminder duplicates because the "
                    'reminder-read path is unavailable right now. '
                    'I did not change anything.'
                )
                metadata = {'duplicate_reminder_audit_unavailable': True}
            else:
                text = (
                    "I couldn't list reminders because the reminder-read path "
                    'is unavailable right now. I did not change anything.'
                )
                metadata = {'reminder_list_unavailable': True}
            return DispatcherOutput(
                text=text,
                iterations=0,
                metadata=metadata,
            )

        if command_kind == 'duplicates':
            clusters = cluster_duplicate_reminders(
                reminders,
                app_timezone=self.app_timezone,
            )
            self._store_duplicate_reminder_audit(user_id=user.id, clusters=clusters)
            return DispatcherOutput(
                text=render_duplicate_audit_text(
                    clusters,
                    app_timezone=self.app_timezone,
                ),
                iterations=0,
                metadata={
                    'duplicate_reminder_audit': True,
                    'duplicate_cluster_count': len(clusters),
                    'reminder_read_only': True,
                },
            )

        return DispatcherOutput(
            text=self._render_direct_reminder_list_text(
                reminders=reminders,
                translator=translator,
            ),
            iterations=0,
            metadata={
                'reminder_list': True,
                'reminder_count': len(reminders),
                'reminder_read_only': True,
            },
        )

    def _render_duplicate_cleanup_preview(self, *, clusters: list[dict[str, Any]]) -> str:
        total_delete = sum(
            max(0, len(list(cluster.get('reminder_ids') or [])) - 1)
            for cluster in clusters
        )
        total_keep = sum(
            1 for cluster in clusters
            if str(cluster.get('newest_reminder_id') or '').strip()
        )
        return (
            f'Keep {total_keep} newest reminder{"s" if total_keep != 1 else ""} '
            f'and cancel {total_delete} duplicate row{"s" if total_delete != 1 else ""}?'
        )

    def _execute_duplicate_reminder_cleanup(
        self,
        *,
        user_id: str,
        payload: dict[str, Any],
    ) -> ServiceResponse:
        clusters = [
            item for item in payload.get('clusters') or []
            if isinstance(item, dict)
        ]
        delete_spec = self.registry.get('delete_reminder')
        if delete_spec is None:
            return ServiceResponse(
                text="I couldn't clean those duplicates because the delete_reminder tool isn't available.",
            )
        removed_bodies: list[str] = []
        for cluster in clusters:
            keep_id = str(cluster.get('newest_reminder_id') or '').strip()
            for reminder_id in cluster.get('reminder_ids') or []:
                reminder_id = str(reminder_id).strip()
                if not reminder_id or reminder_id == keep_id:
                    continue
                try:
                    result = delete_spec.fn(user_id=user_id, query=reminder_id)
                except TypeError:
                    result = delete_spec.fn(query=reminder_id)
                if isinstance(result, ToolResult):
                    data = result.data if isinstance(result.data, dict) else {}
                    if data.get('cancelled'):
                        removed_bodies.append(str(data.get('body') or '').strip())
        self._clear_duplicate_reminder_audit(user_id=user_id)
        if not removed_bodies:
            return ServiceResponse(
                text="I didn't cancel any duplicate reminders because I couldn't find the old rows anymore.",
            )
        removed_count = len(removed_bodies)
        return ServiceResponse(
            text=(
                f'Cleaned up {removed_count} duplicate reminder'
                f'{"s" if removed_count != 1 else ""}. '
                'I kept the newest row in each cluster.'
            ),
        )

    async def handle(self, input_data: DispatcherInput) -> DispatcherOutput:
        user = input_data.user
        raw_text = input_data.text
        text = raw_text
        translator = input_data.translator or Translator(getattr(user, 'language', 'en'))
        streaming = input_data.streaming_session

        normalized_command = ' '.join((text or '').strip().split()).casefold()
        if normalized_command in {'/status runtime', 'status runtime'}:
            return DispatcherOutput(
                text=render_runtime_status_text(get_runtime_identity()),
                iterations=0,
                metadata={'runtime_identity': True},
            )
        if normalized_command in {'/status capabilities', 'status capabilities'}:
            return DispatcherOutput(
                text=self.capability_registry.render_status_text(
                    user=user,
                    registry=self.registry,
                ),
                iterations=0,
                metadata={'capability_status': True},
            )
        reminder_read_command = _classify_direct_reminder_read_command(text)
        if reminder_read_command is not None:
            return await self._handle_direct_reminder_read_command(
                command_kind=reminder_read_command,
                user=user,
                translator=translator,
            )

        # H2-039 FIX 1 — Telegram approval button callback fast path.
        # When the user taps Approve or Cancel on an approval prompt
        # surfaced by either (a) the destructive-message-gate below, or
        # (b) the Phase 4 per-tool `requires_approval` branch later in
        # this handler, the callback round-trip arrives back here as
        # text="approval:approve:<id>" or "approval:cancel:<id>". The
        # UnifiedPipeline normalizes button kind into a text turn. We
        # intercept before any LLM/tool work.
        if text.startswith('approval:approve:') or text.startswith('approval:cancel:'):
            return await self._handle_approval_callback(
                callback_data=text, user=user, translator=translator,
            )
        if text.startswith('contact_reminder_retry:'):
            result = await handle_contact_reminder_retry_callback(
                reminder_id=text.split(':', 1)[1],
                telegram_id=user.telegram_id,
            )
            return DispatcherOutput(
                text=('Retrying that failed contact reminder now.'
                      if result.get('ok')
                      else 'That contact reminder could not be retried.'),
                iterations=0,
            )
        if text.startswith('contact_reminder_cancel:'):
            result = await handle_contact_reminder_cancel_callback(
                reminder_id=text.split(':', 1)[1],
                telegram_id=user.telegram_id,
            )
            return DispatcherOutput(
                text=('Cancelled that failed contact reminder.'
                      if result.get('ok')
                      else 'That contact reminder could not be cancelled.'),
                iterations=0,
            )
        if normalized_command in {'clean duplicates', 'delete duplicates', 'keep newest'}:
            # Load the audit (in-memory cache → recovery_state fallback).
            # `_load_duplicate_reminder_audit` returns the persisted
            # audit_thread_id when it had to read from recovery_state, so
            # we can bind the new cleanup_thread to the same audit run.
            audit = self._load_duplicate_reminder_audit(user_id=user.id)
            clusters = [
                item for item in (audit or {}).get('clusters') or []
                if isinstance(item, dict)
            ]
            if not clusters:
                return DispatcherOutput(
                    text='Run /reminders duplicates first so I can inspect the current reminder rows.',
                    iterations=0,
                    metadata={'duplicate_cleanup_missing_audit': True},
                )
            recovery_state = self._load_recovery_context(user_id=user.id)
            current_thread = _active_thread_from_recovery_state(recovery_state)
            # Identify the source duplicate_audit thread (for traceability
            # in the approval payload + metadata). Prefer the
            # recovery_state thread because it is the source of truth
            # across processes; fall back to the cached audit thread id
            # when the in-memory cache served the clusters.
            audit_source_thread: dict[str, Any] | None = None
            if (current_thread is not None
                    and str(current_thread.get('thread_kind') or '') == 'duplicate_audit'):
                audit_source_thread = current_thread
            audit_thread_id = ''
            audit_thread_revision = 0
            if audit_source_thread is not None:
                audit_thread_id = str(audit_source_thread.get('thread_id') or '').strip()
                audit_thread_revision = _parse_thread_revision(
                    audit_source_thread.get('thread_revision') or 0
                )
            elif isinstance(audit, dict):
                audit_thread_id = str(audit.get('audit_thread_id') or '').strip()
                audit_thread_revision = _parse_thread_revision(
                    audit.get('audit_thread_revision') or 0
                )
            # Build / refresh the cleanup_thread. If the active thread is
            # already a cleanup (user is re-confirming), bump its revision
            # so the approval payload reflects the latest user intent. In
            # every other case (None, duplicate_audit, or some other kind)
            # we create a fresh cleanup_thread linked to the audit.
            now = utc_now()
            if (current_thread is not None
                    and str(current_thread.get('thread_kind') or '') == 'cleanup'):
                cleanup_thread = _thread_with_status(
                    current_thread,
                    status='pending_approval',
                    now=now,
                )
                # Preserve the audit linkage if already present.
                if 'audit_thread_id' not in cleanup_thread and audit_thread_id:
                    cleanup_thread['audit_thread_id'] = audit_thread_id
                    cleanup_thread['audit_thread_revision'] = audit_thread_revision
            else:
                cleanup_thread = {
                    'thread_id': str(uuid.uuid4()),
                    'thread_kind': 'cleanup',
                    'thread_revision': 1,
                    'status': 'pending_approval',
                    'created_at': now.isoformat(),
                    'updated_at': now.isoformat(),
                    'expires_at': (now + _THREAD_BINDING_TTL).isoformat(),
                    'source_turn_ids': [],
                    'audit_thread_id': audit_thread_id,
                    'audit_thread_revision': audit_thread_revision,
                }
            sr = self.approval_service.request(
                user,
                action_type='duplicate_reminder_cleanup',
                preview_text=self._render_duplicate_cleanup_preview(clusters=clusters),
                payload={
                    'clusters': clusters,
                    'audit_thread_id': audit_thread_id,
                    'audit_thread_revision': audit_thread_revision,
                    **_thread_binding_payload(cleanup_thread),
                },
                translator=translator,
            )
            if cleanup_thread is not None:
                self._store_thread_state(
                    user_id=user.id,
                    recovery_state=recovery_state,
                    thread=cleanup_thread,
                    topic='duplicate reminder cleanup',
                )
            return DispatcherOutput(
                text=sr.text,
                iterations=0,
                buttons=list(sr.buttons or []),
                metadata={
                    'destructive_gate': True,
                    'duplicate_cleanup_approval': True,
                    'approval_thread_id': cleanup_thread.get('thread_id') if cleanup_thread else '',
                    'selected_thread_id': cleanup_thread.get('thread_id') if cleanup_thread else '',
                    'selected_thread_kind': cleanup_thread.get('thread_kind') if cleanup_thread else 'cleanup',
                    'thread_revision': cleanup_thread.get('thread_revision') if cleanup_thread else 0,
                    'audit_thread_id': audit_thread_id,
                    'audit_thread_revision': audit_thread_revision,
                    'thread_status_before': (
                        current_thread.get('status') if current_thread is not None else ''
                    ),
                    'thread_status_after': cleanup_thread.get('status') if cleanup_thread else 'pending_approval',
                },
            )

        sanitized_text, role_contaminated = _sanitize_role_contaminated_text(text)
        role_contaminated_confirmation = (
            role_contaminated and _looks_like_confirmation_token(sanitized_text)
        )
        if role_contaminated and sanitized_text and not role_contaminated_confirmation:
            text = sanitized_text

        clarification_metadata: dict[str, Any] = {}
        recovery_context = self._load_recovery_context(user_id=user.id)
        current_thread = _active_thread_from_recovery_state(recovery_context)
        active_clarification = self._load_active_clarification(user_id=user.id)
        pending_approvals = self._load_pending_approvals(user_id=user.id)
        bare_confirmation = _looks_like_bare_confirmation(text)

        if bare_confirmation:
            pending_targets: list[tuple[str, Any]] = []
            if active_clarification is not None:
                pending_targets.append(('clarification', active_clarification))
            pending_targets.extend(('approval', approval) for approval in pending_approvals)
            if len(pending_targets) != 1:
                self._archive_short_circuit_reply(
                    user_id=user.id,
                    user_text=raw_text,
                    assistant_text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                    created_at=utc_now(),
                )
                return DispatcherOutput(
                    text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                    iterations=0,
                    buttons=[],
                    metadata={'thread_confirmation_blocked': True},
                )
            pending_kind, pending_target = pending_targets[0]
            if pending_kind == 'approval':
                approval_payload = self._parse_approval_payload(pending_target)
                matches_thread, _ = _thread_binding_matches(
                    expected_thread_id=str(approval_payload.get('thread_id') or '').strip(),
                    expected_thread_revision=_parse_thread_revision(
                        approval_payload.get('thread_revision') or 0
                    ),
                    current_thread=current_thread,
                    required_status='pending_approval',
                )
                if not matches_thread:
                    self._archive_short_circuit_reply(
                        user_id=user.id,
                        user_text=raw_text,
                        assistant_text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                        created_at=utc_now(),
                    )
                    return DispatcherOutput(
                        text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                        iterations=0,
                        buttons=[],
                        metadata={'thread_confirmation_blocked': True},
                    )
                return await self._handle_approval_callback(
                    callback_data=f'approval:approve:{pending_target.id}',
                    user=user,
                    translator=translator,
                )

        if active_clarification is not None:
            clarification_thread_matches = True
            if active_clarification.thread_id:
                clarification_thread_matches = (
                    current_thread is not None
                    and not _is_iso_timestamp_expired(current_thread.get('expires_at'))
                    and str(current_thread.get('thread_id') or '').strip() == active_clarification.thread_id
                    and _parse_thread_revision(current_thread.get('thread_revision') or 0)
                    == active_clarification.thread_revision
                )
                if (
                    current_thread is None
                    or str(current_thread.get('status') or '').strip()
                    not in {'pending_clarification', 'active'}
                ):
                    clarification_thread_matches = False
            if not clarification_thread_matches:
                likely_thread_answer = _looks_like_confirmation_token(text) or len(text.split()) <= 8
                self._clear_active_clarification(user_id=user.id)
                if likely_thread_answer:
                    self._archive_short_circuit_reply(
                        user_id=user.id,
                        user_text=raw_text,
                        assistant_text=_THREAD_CLARIFICATION_MISMATCH_TEXT,
                        created_at=utc_now(),
                    )
                    return DispatcherOutput(
                        text=_THREAD_CLARIFICATION_MISMATCH_TEXT,
                        iterations=0,
                        buttons=[],
                        metadata={
                            'clarification_thread_mismatch': True,
                            'clarification_thread_id': active_clarification.thread_id,
                        },
                    )
                active_clarification = None
            if active_clarification is None:
                pass
            elif bare_confirmation and pending_approvals:
                self._archive_short_circuit_reply(
                    user_id=user.id,
                    user_text=raw_text,
                    assistant_text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                    created_at=utc_now(),
                )
                return DispatcherOutput(
                    text=_THREAD_CONFIRMATION_AMBIGUOUS_TEXT,
                    iterations=0,
                    buttons=[],
                    metadata={'thread_confirmation_blocked': True},
                )
            elif role_contaminated_confirmation:
                follow_up_text = _HUMAN_CONFIRMATION_STYLE.render_clarification_follow_up(
                    question=active_clarification.question,
                    options=render_option_labels(active_clarification),
                )
                refreshed = refresh_clarification(
                    active_clarification,
                    question=follow_up_text,
                    source_turn_id=self._archive_short_circuit_reply(
                        user_id=user.id,
                        user_text=raw_text,
                        assistant_text=follow_up_text,
                        created_at=utc_now(),
                    ),
                    created_at=utc_now(),
                )
                self._store_active_clarification(
                    user_id=user.id,
                    clarification=refreshed,
                )
                return DispatcherOutput(
                    text=_HUMAN_CONFIRMATION_STYLE.render_role_contamination_guard()
                    + ' '
                    + follow_up_text,
                    iterations=0,
                    buttons=[],
                    metadata={
                        'role_contamination_guard': True,
                        'clarification_follow_up': True,
                        'clarification_id': refreshed.clarification_id,
                        'clarification_thread_id': refreshed.thread_id,
                        'thread_status_before': (
                            str(current_thread.get('status') or '')
                            if current_thread is not None
                            else ''
                        ),
                    },
                )
            elif active_clarification is not None:
                follow_up_text = _HUMAN_CONFIRMATION_STYLE.render_clarification_follow_up(
                    question=active_clarification.question,
                    options=render_option_labels(active_clarification),
                )
                stale_text = _HUMAN_CONFIRMATION_STYLE.render_stale_clarification()
                resolution = resolve_clarification_answer(
                    active_clarification,
                    answer_text=text,
                    now=utc_now(),
                    follow_up_text=follow_up_text,
                    stale_text=stale_text,
                )
                if resolution.action == 'resolved':
                    self._clear_active_clarification(user_id=user.id)
                    text = resolution.resolved_text or text
                    clarification_metadata = {
                        'clarification_answer_resolved': True,
                        'clarification_id': active_clarification.clarification_id,
                        'clarification_option_id': resolution.selected_option_id,
                        'clarification_option_destructive': resolution.destructive,
                        'clarification_thread_id': active_clarification.thread_id,
                        'clarification_thread_revision': active_clarification.thread_revision,
                    }
                elif resolution.action == 'follow_up':
                    follow_up_now = utc_now()
                    source_turn_id = self._archive_short_circuit_reply(
                        user_id=user.id,
                        user_text=raw_text,
                        assistant_text=resolution.follow_up_text or follow_up_text,
                        created_at=follow_up_now,
                    )
                    refreshed = refresh_clarification(
                        active_clarification,
                        question=resolution.follow_up_text or follow_up_text,
                        source_turn_id=source_turn_id,
                        created_at=follow_up_now,
                    )
                    self._store_active_clarification(
                        user_id=user.id,
                        clarification=refreshed,
                    )
                    return DispatcherOutput(
                        text=resolution.follow_up_text or follow_up_text,
                        iterations=0,
                        buttons=[],
                        metadata={
                            'clarification_follow_up': True,
                            'clarification_id': refreshed.clarification_id,
                            'clarification_source': active_clarification.kind,
                            'clarification_thread_id': refreshed.thread_id,
                        },
                    )
                elif resolution.action == 'stale':
                    self._clear_active_clarification(user_id=user.id)
                    stale_reply = resolution.follow_up_text or stale_text
                    self._archive_short_circuit_reply(
                        user_id=user.id,
                        user_text=raw_text,
                        assistant_text=stale_reply,
                        created_at=utc_now(),
                    )
                    return DispatcherOutput(
                        text=stale_reply,
                        iterations=0,
                        buttons=[],
                        metadata={
                            'clarification_stale': True,
                            'clarification_source': active_clarification.kind,
                            'clarification_thread_id': active_clarification.thread_id,
                        },
                    )
                else:
                    self._clear_active_clarification(user_id=user.id)
        elif role_contaminated_confirmation:
            guard_text = _HUMAN_CONFIRMATION_STYLE.render_role_contamination_guard()
            self._archive_short_circuit_reply(
                user_id=user.id,
                user_text=raw_text,
                assistant_text=guard_text,
                created_at=utc_now(),
            )
            return DispatcherOutput(
                text=guard_text,
                iterations=0,
                buttons=[],
                metadata={'role_contamination_guard': True},
            )
        recent_recovery_turns = self._load_recent_recovery_turns(user_id=user.id)
        recovery = self.recovery_layer.recover(
            text=text,
            context={'recovery_state': recovery_context},
            recent_turns=recent_recovery_turns,
            resolve_contact_alias=lambda query: self._resolve_contact_alias_for_recovery(
                user_id=user.id,
                query=query,
            ),
        )
        working_text = recovery.recovered_text or text
        recovery_prompt_block = build_recovery_prompt_block(recovery)

        social_reply = _render_social_reply(
            text=working_text,
            app_timezone=self.app_timezone,
        )
        thread_now = utc_now()
        selected_thread = _select_active_thread(
            previous_thread=current_thread,
            recovery=recovery,
            working_text=working_text,
            social_reply=bool(social_reply),
            clarification_metadata=clarification_metadata,
            now=thread_now,
        )
        recovery.context_updates['active_thread'] = selected_thread
        self._persist_recovery_context(user_id=user.id, result=recovery)
        current_thread = selected_thread
        if social_reply:
            social_reply = _HUMAN_CONFIRMATION_STYLE.compress_reply(text=social_reply)
            self._archive_short_circuit_reply(
                user_id=user.id,
                user_text=raw_text,
                assistant_text=social_reply,
                created_at=utc_now(),
            )
            return DispatcherOutput(
                text=social_reply,
                iterations=0,
                buttons=[],
                metadata={
                    'social_reply': True,
                    'role_contamination_stripped': role_contaminated and not role_contaminated_confirmation,
                    'selected_thread_id': selected_thread.get('thread_id'),
                    'selected_thread_kind': selected_thread.get('thread_kind'),
                    'thread_revision': selected_thread.get('thread_revision'),
                },
            )

        if recovery.needs_clarification and recovery.clarification_text:
            clarification_text = _HUMAN_CONFIRMATION_STYLE.render_specific_clarification(
                recovered_intent=working_text,
                confidence=recovery.confidence,
                risk_level=recovery.risk_level,
                missing_slot=recovery.missing_slot,
                resolved_slots=dict(recovery.resolved_slots),
                selected_clarification_option=recovery.selected_clarification_option,
                existing_text=recovery.clarification_text,
            ) or recovery.clarification_text
            clarification_now = utc_now()
            source_turn_id = self._archive_short_circuit_reply(
                user_id=user.id,
                user_text=raw_text,
                assistant_text=clarification_text,
                created_at=clarification_now,
            )
            clarification_state = build_recovery_clarification(
                result=recovery,
                source_turn_id=source_turn_id,
                created_at=clarification_now,
            )
            if clarification_state is not None:
                clarification_state = self._bind_clarification_to_thread(
                    clarification=clarification_state,
                    thread=selected_thread,
                    requested_slot=recovery.missing_slot,
                    created_at=clarification_now,
                )
                self._store_active_clarification(
                    user_id=user.id,
                    clarification=clarification_state,
                )
            pending_clarification_thread = _thread_with_status(
                selected_thread,
                status='pending_clarification',
                now=clarification_now,
            )
            if pending_clarification_thread is not None:
                self._store_thread_state(
                    user_id=user.id,
                    recovery_state=dict(recovery.context_updates),
                    thread=pending_clarification_thread,
                    topic=clarification_text,
                )
            return DispatcherOutput(
                text=clarification_text,
                iterations=0,
                buttons=[],
                metadata={
                    'recovery_clarification': True,
                    'vague_clarification': True,
                    'recovery_outcome': recovery.outcome,
                    'recovery_confidence': recovery.confidence,
                    'recovery_slots': dict(recovery.resolved_slots),
                    'clarification_id': (
                        clarification_state.clarification_id
                        if clarification_state is not None
                        else ''
                    ),
                    'selected_thread_id': selected_thread.get('thread_id'),
                    'selected_thread_kind': selected_thread.get('thread_kind'),
                    'thread_revision': selected_thread.get('thread_revision'),
                    'clarification_thread_id': (
                        clarification_state.thread_id
                        if clarification_state is not None
                        else selected_thread.get('thread_id')
                    ),
                    'thread_status_before': selected_thread.get('status'),
                    'thread_status_after': (
                        pending_clarification_thread.get('status')
                        if pending_clarification_thread is not None
                        else selected_thread.get('status')
                    ),
                },
            )

        if recovery.resolved_slots.get('action_kind') == 'rental_status_check':
            self._clear_active_clarification(user_id=user.id)
            subject = str(recovery.resolved_slots.get('rental_subject') or 'your rental records').strip()
            rentals_capability = self.capability_registry.get_capability(
                'rentals_read',
                user=user,
                registry=self.registry,
            )
            capability_decision = self.fallback_manager.decide_capability(
                context=FallbackContext(
                    route='rental_status',
                    stage='prelude',
                    provider='local',
                    root_reason=rentals_capability.state,
                    raw_text=raw_text,
                    recovered_text=working_text,
                    recovery_metadata=_recovery_metadata_payload(recovery),
                    capability=rentals_capability,
                    capability_name='rentals_read',
                    details={'subject': subject},
                )
            )
            return DispatcherOutput(
                text=capability_decision.user_text,
                iterations=0,
                buttons=[],
                metadata={
                    'rental_status_auto_resolved': True,
                    'capability_checked': 'rentals_read',
                    'capability_state': rentals_capability.state,
                    'capability': rentals_capability.to_dict(),
                    'recovery_applied': True,
                    'recovery_confidence': recovery.confidence,
                    'recovery_slots': dict(recovery.resolved_slots),
                    **clarification_metadata,
                },
            )

        if _looks_like_calendar_request(working_text):
            capability_name = 'calendar_write' if _classify_destructive(working_text).is_destructive else 'calendar_read'
            calendar_capability = self.capability_registry.get_capability(
                capability_name,
                user=user,
                registry=self.registry,
            )
            if not calendar_capability.safe_to_attempt:
                capability_decision = self.fallback_manager.decide_capability(
                    context=FallbackContext(
                        route=capability_name,
                        stage='prelude',
                        provider='local',
                        root_reason=calendar_capability.state,
                        raw_text=raw_text,
                        recovered_text=working_text,
                        recovery_metadata=_recovery_metadata_payload(recovery),
                        capability=calendar_capability,
                        capability_name=capability_name,
                    )
                )
                reply = capability_decision.user_text
                self._archive_short_circuit_reply(
                    user_id=user.id,
                    user_text=raw_text,
                    assistant_text=reply,
                    created_at=utc_now(),
                )
                return DispatcherOutput(
                    text=reply,
                    iterations=0,
                    buttons=[],
                    metadata={
                        'capability_checked': capability_name,
                        'capability_state': calendar_capability.state,
                        'capability': calendar_capability.to_dict(),
                    },
                )

        if _looks_like_whatsapp_send_request(working_text):
            whatsapp_capability = self.capability_registry.get_capability(
                'whatsapp_send',
                user=user,
                registry=self.registry,
            )
            if (
                not whatsapp_capability.safe_to_attempt
                and whatsapp_capability.state in {'auth_required', 'service_down', 'not_wired'}
            ):
                capability_decision = self.fallback_manager.decide_capability(
                    context=FallbackContext(
                        route='contact_send',
                        stage='prelude',
                        provider='local',
                        root_reason=whatsapp_capability.state,
                        raw_text=raw_text,
                        recovered_text=working_text,
                        recovery_metadata=_recovery_metadata_payload(recovery),
                        capability=whatsapp_capability,
                        capability_name='whatsapp_send',
                    )
                )
                reply = capability_decision.user_text
                self._archive_short_circuit_reply(
                    user_id=user.id,
                    user_text=raw_text,
                    assistant_text=reply,
                    created_at=utc_now(),
                )
                return DispatcherOutput(
                    text=reply,
                    iterations=0,
                    buttons=[],
                    metadata={
                        'capability_checked': 'whatsapp_send',
                        'capability_state': whatsapp_capability.state,
                        'capability': whatsapp_capability.to_dict(),
                    },
                )

        vague_clarification = None
        if not recovery.suppress_vague_clarification:
            vague_clarification = build_vague_clarification(working_text)
        if _should_emit_vague_clarification(
            text=working_text,
            clarification=vague_clarification,
        ):
            logger.info(
                'dispatcher_vague_clarification_emitted',
                extra={'user_id': user.id},
            )
            clarification_now = utc_now()
            source_turn_id = self._archive_short_circuit_reply(
                user_id=user.id,
                user_text=raw_text,
                assistant_text=vague_clarification,
                created_at=clarification_now,
            )
            clarification_state = build_vague_clarification_state(
                original_text=working_text,
                clarification_text=vague_clarification,
                source_turn_id=source_turn_id,
                created_at=clarification_now,
            )
            clarification_state = self._bind_clarification_to_thread(
                clarification=clarification_state,
                thread=selected_thread,
                requested_slot='',
                created_at=clarification_now,
            )
            self._store_active_clarification(
                user_id=user.id,
                clarification=clarification_state,
            )
            pending_clarification_thread = _thread_with_status(
                selected_thread,
                status='pending_clarification',
                now=clarification_now,
            )
            if pending_clarification_thread is not None:
                self._store_thread_state(
                    user_id=user.id,
                    recovery_state=dict(recovery.context_updates),
                    thread=pending_clarification_thread,
                    topic=vague_clarification,
                )
            return DispatcherOutput(
                text=vague_clarification,
                iterations=0,
                buttons=[],
                metadata={
                    'vague_clarification': True,
                    'recovery_applied': working_text != raw_text,
                    'recovery_confidence': recovery.confidence,
                    'clarification_id': clarification_state.clarification_id,
                    'selected_thread_id': selected_thread.get('thread_id'),
                    'selected_thread_kind': selected_thread.get('thread_kind'),
                    'thread_revision': selected_thread.get('thread_revision'),
                    'clarification_thread_id': clarification_state.thread_id,
                    'thread_status_before': selected_thread.get('status'),
                    'thread_status_after': (
                        pending_clarification_thread.get('status')
                        if pending_clarification_thread is not None
                        else selected_thread.get('status')
                    ),
                },
            )

        # H2-039 FIX 1 — Destructive intent gate (Pattern B).
        # The MCP path (brain_router → claude -p) executes its entire
        # tool loop inside a subprocess; the bot never sees individual
        # tool calls. Per-tool approval is unreachable there. As a
        # mitigation, we classify the USER's prompt before invoking the
        # LLM. If the prompt looks destructive (delete / send / move /
        # write / etc., or matches a registered destructive tool name)
        # AND the caller didn't bypass, we surface an approval prompt
        # via ApprovalService.request and short-circuit. On approve, the
        # callback path above re-fires this dispatcher with bypass=True.
        if (self._destructive_approval_enabled()
                and not input_data.bypass_destructive_approval
                and self.approvals_repository is not None):
            intent = _classify_destructive(working_text)
            if intent.is_destructive:
                # H2-047 Fix 2: advisory intents (multi-intent numbered prompts
                # today) skip approval_service entirely. The user sees the
                # template text with no Approve/Cancel buttons — neither
                # acting on the multi-action prompt nor pretending it could
                # be approved as one bundle.
                if intent.is_advisory_only:
                    logger.info(
                        'dispatcher_destructive_advisory_emitted',
                        extra={
                            'user_id': user.id,
                            'matched_verbs': list(intent.matched_verbs),
                            'confidence': intent.confidence,
                        },
                    )
                    return DispatcherOutput(
                        text=intent.suggested_approval_template,
                        iterations=0,
                        buttons=[],
                        metadata={'destructive_advisory': True,
                                  'matched_verbs': list(intent.matched_verbs),
                                  'recovery_applied': working_text != raw_text,
                                  'recovery_confidence': recovery.confidence,
                                  **clarification_metadata},
                    )
                preview = _render_destructive_preview(working_text, intent)
                approval_thread = _thread_with_status(
                    selected_thread,
                    status='pending_approval',
                    now=utc_now(),
                )
                sr = self.approval_service.request(
                    user,
                    action_type='destructive_message_gate',
                    preview_text=preview,
                    payload={
                        'original_prompt': working_text,
                        'original_prompt_raw': raw_text,
                        'recovery': {
                            'outcome': recovery.outcome,
                            'confidence': recovery.confidence,
                            'resolved_slots': dict(recovery.resolved_slots),
                            'corrections_applied': list(recovery.corrections_applied),
                            'recipient_negated_unresolved': bool(
                                recovery.context_updates.get('recipient_negated_unresolved')
                            ),
                            'negated_recipient_label': recovery.context_updates.get('negated_recipient_label'),
                            'negated_recipient_reason': recovery.context_updates.get('negated_recipient_reason'),
                        },
                        'matched_tools': list(intent.matched_tools),
                        'matched_verbs': list(intent.matched_verbs),
                        'user_id': user.id,
                        **_thread_binding_payload(approval_thread),
                    },
                    translator=translator,
                )
                if approval_thread is not None:
                    self._store_thread_state(
                        user_id=user.id,
                        recovery_state=dict(recovery.context_updates),
                        thread=approval_thread,
                        topic=working_text,
                    )
                logger.info(
                    'dispatcher_destructive_gate_armed',
                    extra={
                        'user_id': user.id,
                        'matched_tools': list(intent.matched_tools),
                        'matched_verbs': list(intent.matched_verbs),
                        'confidence': intent.confidence,
                    },
                )
                return DispatcherOutput(
                    text=sr.text, iterations=0,
                    buttons=list(sr.buttons or []),
                    metadata={'destructive_gate': True,
                              'matched_tools': list(intent.matched_tools),
                              'recovery_applied': working_text != raw_text,
                              'recovery_confidence': recovery.confidence,
                              'selected_thread_id': selected_thread.get('thread_id'),
                              'selected_thread_kind': selected_thread.get('thread_kind'),
                              'thread_revision': selected_thread.get('thread_revision'),
                              'approval_thread_id': (
                                  approval_thread.get('thread_id')
                                  if approval_thread is not None
                                  else selected_thread.get('thread_id')
                              ),
                              'thread_status_before': selected_thread.get('status'),
                              'thread_status_after': (
                                  approval_thread.get('status')
                                  if approval_thread is not None
                                  else selected_thread.get('status')
                              ),
                              **clarification_metadata},
                )

        # V3.7 streaming entry beat — fires before TELOS / mem0 / LLM
        # so the user sees a placeholder appear within ~50ms instead of
        # staring at "typing..." for the whole turn. Throttle inside
        # StreamingSession caps at 2 events / sec.
        if streaming is not None:
            await streaming.update('Thinking...')

        # V3.6 forward-only conversation archive: write the inbound user
        # turn before any LLM/tool work. The assistant turn is written
        # after the LLM loop composes final_text. Both turn_ids are then
        # passed to the mem0 persistence path so the archive row gets
        # marked persisted on success.
        turn_now = utc_now()
        conversation_id = self.conversation_turns_repository.resolve_conversation_id(
            user_id=user.id, now=turn_now,
        )
        user_turn_id = self.conversation_turns_repository.insert(
            user_id=user.id,
            role='user',
            content=text,
            conversation_id=conversation_id,
            created_at=turn_now,
        )

        try:
            archived_turns = self.conversation_turns_repository.list_recent_for_user(
                user_id=user.id,
                limit=_MAX_ARCHIVED_CONTEXT_TURNS + 1,
                conversation_id=conversation_id,
            )
            archived_turns = _select_archived_context_turns(
                archived_turns,
                current_turn_id=user_turn_id,
                limit=_MAX_ARCHIVED_CONTEXT_TURNS,
            )
        except Exception:
            logger.warning('dispatcher_conversation_history_load_failed', extra={'user_id': user.id, 'conversation_id': conversation_id})
            archived_turns = []

        telos = self.telos_service.load(user.id)
        try:
            memories = self.mem0.search(working_text, user_id=user.id, limit=5) or []
        except Exception:
            logger.warning('dispatcher_mem0_search_failed', extra={'user_id': user.id})
            memories = []

        system_prompt = _build_system_prompt(
            persona=self.persona,
            telos=telos,
            memories=list(memories),
            language=translator.language,
            now=utc_now(),
            app_timezone=self.app_timezone,
            recovery_note=recovery_prompt_block,
        )
        tool_catalog = _coerce_tool_catalog(self.registry)
        contents = _build_llm_contents_from_archive(archived_turns, current_user_text=working_text)

        final_text = ''
        buttons: list[InlineButton] = []
        reply_metadata: dict[str, Any] = {
            'recovery_applied': working_text != raw_text,
            'recovery_confidence': recovery.confidence,
            **clarification_metadata,
        }
        if recovery.resolved_slots:
            reply_metadata['recovery_slots'] = dict(recovery.resolved_slots)
        if recovery.corrections_applied:
            reply_metadata['recovery_corrections'] = list(recovery.corrections_applied)
        iterations = 0
        # H2-046: track recent iteration tool-call shapes so we can
        # (a) early-break when Claude is stuck retrying unknown tools and
        # (b) emit a useful diagnostic log if we do hit the iteration cap.
        # Each entry is the list of tool names attempted in that iteration.
        iteration_tool_history: list[list[str]] = []
        created_reminder_calls: list[dict[str, Any]] = []
        consecutive_all_unknown_iterations = 0
        early_break_reason: str | None = None
        memory_confirmation_prompt: str | None = None
        forced_terminal_reply: str | None = None

        for _ in range(self.max_iterations):
            iterations += 1
            response = await self.llm.generate_with_tools(
                user_id=user.id,
                system_prompt=system_prompt,
                contents=contents,
                tool_catalog=tool_catalog,
            )
            tool_calls = _collapse_duplicate_reminder_tool_calls(
                response.get('tool_calls') or []
            )
            if not tool_calls:
                final_text = response.get('text', '') or ''
                is_provider_failure_text = _is_global_provider_failure_text(final_text)
                normalized_text, normalized_mode, structured_failure = self.fallback_manager.normalize_provider_failure(
                    context=FallbackContext(
                        route='',
                        stage='post_approval' if input_data.post_approval_resume else 'tool_loop',
                        provider='brain_router',
                        root_reason='provider_unavailable',
                        raw_text=raw_text,
                        recovered_text=working_text,
                        recovery_metadata=_recovery_metadata_payload(recovery),
                        details={
                            'has_contact_reminder_intent': _has_contact_reminder_intent(working_text),
                            'has_outbound_message_intent': _has_outbound_message_intent(working_text),
                            'has_rich_reminder_context': _prompt_has_rich_reminder_context(working_text),
                            'is_time_request': _looks_like_time_request(working_text),
                            'is_audit_request': _looks_like_audit_request(working_text),
                        },
                    ),
                    provider_text=final_text,
                    is_provider_failure_text=is_provider_failure_text,
                    local_time_text=(
                        _invoke_local_time_fallback(self.registry, user_id=user.id)
                        if is_provider_failure_text and _looks_like_time_request(working_text)
                        else None
                    ),
                    audit_guidance=_AUDIT_PROVIDER_FAILURE_GUIDANCE,
                    vague_clarification=_SECOND_CHANCE_VAGUE_CLARIFICATION(working_text),
                    post_approval_resume=input_data.post_approval_resume,
                )
                if normalized_text is not None:
                    final_text = normalized_text
                    reply_metadata['provider_failure_normalized'] = normalized_mode
                    if structured_failure is not None:
                        reply_metadata['structured_failure'] = structured_failure.to_metadata()
                        logger.warning(
                            'dispatcher_provider_failure_normalized '
                            'raw_reason=%s route=%s stage=%s provider=%s '
                            'fallback=%s root_reason=%s',
                            structured_failure.technical_reason,
                            structured_failure.route,
                            structured_failure.stage,
                            structured_failure.provider,
                            structured_failure.fallback,
                            structured_failure.root_reason,
                        )
                break

            # H2-046: per-iteration log so journalctl shows the tool sequence
            # without needing to add ad-hoc prints during incident response.
            iteration_tool_names = [
                (tc.get('name') or '') for tc in tool_calls
            ]
            iteration_tool_history.append(iteration_tool_names)
            logger.info(
                'dispatcher_tool_invocation_round',
                extra={
                    'iteration': iterations,
                    'tools': iteration_tool_names,
                    'user_id': user.id,
                },
            )

            # H2-046: early-break when the model is stuck calling tools that
            # don't exist in the registry. Three consecutive iterations where
            # EVERY tool_call hits the unknown_tool branch (line below) means
            # the model has no reachable way to fulfill the request — better
            # UX to surface that immediately than wait through max_iterations.
            all_unknown_this_iter = all(
                self.registry.get(name) is None for name in iteration_tool_names
            ) if iteration_tool_names else False
            if all_unknown_this_iter:
                consecutive_all_unknown_iterations += 1
            else:
                consecutive_all_unknown_iterations = 0
            if consecutive_all_unknown_iterations >= 3:
                early_break_reason = 'consecutive_unknown_tools'
                logger.warning(
                    'dispatcher_early_break_unknown_tools',
                    extra={
                        'user_id': user.id,
                        'iterations': iterations,
                        'recent_tool_names': iteration_tool_history[-3:],
                    },
                )
                final_text = (
                    "I'm having trouble figuring out which tool to use for "
                    "that. Could you rephrase what you want?"
                )
                break

            # Append the model turn that emitted the tool calls so the
            # next call sees the full conversation history.
            contents.append({
                'role': 'model',
                'parts': [{'functionCall': {'name': tc['name'], 'args': tc.get('arguments') or {}}} for tc in tool_calls],
            })

            approval_emitted = False
            tool_response_parts = []

            for tool_call in tool_calls:
                name = tool_call.get('name', '')
                args = dict(tool_call.get('arguments') or {})
                spec = self.registry.get(name)

                # V3.7 streaming: per-tool-call human-readable stage.
                # Unknown tool names fall back to the generic stage so
                # the user still sees motion. Stage fires whether or
                # not the tool is approval-gated — for destructive
                # tools, the "Preparing to..." stage shows BEFORE the
                # approval prompt is rendered.
                if streaming is not None:
                    await streaming.update(TOOL_STAGE_MESSAGES.get(name, _GENERIC_TOOL_STAGE))

                if spec is None:
                    tool_response_parts.append({
                        'functionResponse': {
                            'name': name,
                            'response': {'error': 'unknown_tool', 'name': name},
                        }
                    })
                    continue

                if spec.requires_approval and self._destructive_approval_enabled():
                    # SAFETY BOUNDARY — do NOT invoke spec.fn.
                    preview = render_approval_preview(spec, args)
                    approval_thread = _thread_with_status(
                        selected_thread,
                        status='pending_approval',
                        now=utc_now(),
                    )
                    payload = {
                        'tool_name': spec.name,
                        'arguments': args,
                        **_thread_binding_payload(approval_thread),
                    }
                    sr = self.approval_service.request(
                        user,
                        action_type=spec.name,
                        preview_text=preview,
                        payload=payload,
                        translator=translator,
                    )
                    if approval_thread is not None:
                        self._store_thread_state(
                            user_id=user.id,
                            recovery_state=dict(recovery.context_updates),
                            thread=approval_thread,
                            topic=working_text,
                        )
                    final_text = sr.text
                    buttons = list(sr.buttons or [])
                    reply_metadata.update({
                        'approval_thread_id': (
                            approval_thread.get('thread_id')
                            if approval_thread is not None
                            else selected_thread.get('thread_id')
                        ),
                        'selected_thread_id': selected_thread.get('thread_id'),
                        'selected_thread_kind': selected_thread.get('thread_kind'),
                        'thread_revision': selected_thread.get('thread_revision'),
                        'thread_status_before': selected_thread.get('status'),
                        'thread_status_after': (
                            approval_thread.get('status')
                            if approval_thread is not None
                            else selected_thread.get('status')
                        ),
                    })
                    approval_emitted = True
                    break

                if _should_block_personal_memory_save(
                    prompt_text=working_text,
                    tool_name=name,
                ):
                    memory_confirmation_prompt = _render_memory_confirmation_prompt(
                        prompt_text=working_text,
                        args=args,
                        app_timezone=self.app_timezone,
                    )
                    reply_metadata['memory_confirmation_required'] = True
                    tool_response_parts.append({
                        'functionResponse': {
                            'name': name,
                            'response': {
                                'success': True,
                                'data': {
                                    'saved': False,
                                    'needs_confirmation': True,
                                },
                                'announcement': memory_confirmation_prompt,
                                'error': None,
                            },
                        }
                    })
                    if len(tool_calls) == 1:
                        forced_terminal_reply = memory_confirmation_prompt
                        approval_emitted = True
                        break
                    continue

                if name == 'create_reminder':
                    duplicate = next(
                        (
                            existing for existing in created_reminder_calls
                            if _reminder_tool_args_are_duplicates(existing, args)
                        ),
                        None,
                    )
                    if duplicate is not None:
                        tool_response_parts.append({
                            'functionResponse': {
                                'name': name,
                                'response': {
                                    'success': True,
                                    'data': {
                                        'created': False,
                                        'deduplicated': True,
                                        'reminder_id': duplicate.get('reminder_id'),
                                        'next_fire_at': duplicate.get('next_fire_at'),
                                    },
                                    'announcement': 'Duplicate reminder suppressed; using the latest reminder from this request.',
                                    'error': None,
                                },
                            }
                        })
                        continue

                # Non-destructive tool — invoke and capture result. The
                # dispatcher supports BOTH sync and async tool functions
                # (H2-011 extension lesson — V3.2.5 Google wrappers will
                # be async). Coroutine returns are awaited before the
                # ToolResult check so we never feed a raw coroutine to
                # the next LLM turn.
                args_with_user = dict(args)
                args_with_user.setdefault('user_id', user.id)
                try:
                    result = spec.fn(**args_with_user)
                except TypeError:
                    # Tool may not accept user_id (e.g. get_current_time).
                    args_no_user = {k: v for k, v in args_with_user.items() if k != 'user_id'}
                    try:
                        result = spec.fn(**args_no_user)
                    except Exception as exc:
                        logger.warning('dispatcher_tool_invocation_failed', extra={'tool': name, 'error_class': type(exc).__name__})
                        result = ToolResult.fail(f'tool_invocation_failed: {type(exc).__name__}')
                except Exception as exc:
                    logger.warning('dispatcher_tool_invocation_failed', extra={'tool': name, 'error_class': type(exc).__name__})
                    result = ToolResult.fail(f'tool_invocation_failed: {type(exc).__name__}')

                if inspect.iscoroutine(result):
                    try:
                        result = await result
                    except Exception as exc:
                        logger.warning('dispatcher_async_tool_invocation_failed', extra={'tool': name, 'error_class': type(exc).__name__})
                        result = ToolResult.fail(f'tool_invocation_failed: {type(exc).__name__}')

                if isinstance(result, ToolResult):
                    payload = {
                        'success': result.success,
                        'data': result.data,
                        'announcement': result.announcement,
                        'error': result.error,
                    }
                else:
                    payload = {'success': True, 'data': result}

                if (
                    name == 'create_reminder'
                    and payload.get('success') is True
                    and isinstance(payload.get('data'), dict)
                    and payload['data'].get('created') is True
                ):
                    created_reminder_calls.append({
                        'body': args.get('body'),
                        'next_fire_at': args.get('next_fire_at'),
                        'recurrence': args.get('recurrence'),
                        'reminder_id': payload['data'].get('reminder_id'),
                    })

                tool_response_parts.append({
                    'functionResponse': {'name': name, 'response': payload},
                })

            if approval_emitted:
                if forced_terminal_reply is not None:
                    final_text = forced_terminal_reply
                break

            # Feed all tool responses back to the model in one user turn.
            contents.append({'role': 'user', 'parts': tool_response_parts})

        else:
            # Loop exhausted without break — hit the iteration cap.
            # H2-046: log rich diagnostic context so the next occurrence in
            # production can be traced without having to add ad-hoc prints.
            logger.warning(
                'dispatcher_iteration_cap_hit',
                extra={
                    'user_id': user.id,
                    'iterations': iterations,
                    'last_iterations_tools': iteration_tool_history[-3:],
                    'prompt_chars': len(text or ''),
                    'tool_catalog_size': len(tool_catalog) if tool_catalog else 0,
                },
            )
            reminder_read_command = _classify_direct_reminder_read_command(raw_text)
            if reminder_read_command == 'duplicates':
                final_text = (
                    "I couldn't inspect reminder duplicates because the "
                    'reminder-read path looped. I did not change anything.'
                )
            elif reminder_read_command == 'list':
                final_text = (
                    "I couldn't list reminders because the reminder-read path "
                    'looped. I did not change anything.'
                )
            else:
                final_text = "I hit my iteration limit working on that and couldn't finish it. Please try rephrasing your request."

        if memory_confirmation_prompt and _looks_like_claimed_memory_save(final_text):
            final_text = memory_confirmation_prompt
            reply_metadata['memory_confirmation_required'] = True
            reply_metadata['memory_save_blocked'] = True

        final_text = _HUMAN_CONFIRMATION_STYLE.compress_reply(text=final_text)
        if not final_text.strip():
            fallback_text = _render_social_reply(
                text=working_text,
                app_timezone=self.app_timezone,
            ) or _HUMAN_CONFIRMATION_STYLE.render_no_silent_reply(
                user_text=working_text,
            )
            final_text = _HUMAN_CONFIRMATION_STYLE.compress_reply(text=fallback_text)
            reply_metadata['silent_reply_fallback'] = True

        # Deterministic self-correction backstop. The system prompt already
        # asks the model to echo an in-message revision ("June 2 no June 4"),
        # but we guarantee it on reminder confirmations regardless of how the
        # model phrased its reply. Scoped to turns that actually created/updated
        # a reminder so general chat never gets a spurious "Corrected" line.
        if created_reminder_calls and 'corrected' not in final_text.casefold():
            correction = detect_self_correction(raw_text)
            if correction is not None:
                field, old_value, new_value = correction
                final_text = (
                    f'{final_text.rstrip()}\n'
                    f'Corrected {field}: {old_value} -> {new_value}'
                )
                reply_metadata['self_correction_echoed'] = {
                    'field': field,
                    'old': old_value,
                    'new': new_value,
                }

        if role_contaminated and not role_contaminated_confirmation:
            reply_metadata['role_contamination_stripped'] = True

        # V3.6 assistant-turn archive: persist the assistant reply BEFORE
        # the mem0.add call so that even if mem0 fails, the conversation
        # archive contains both halves. The `mem0_persisted_at` partial
        # index marks this row as recoverable for future retry.
        assistant_turn_id = self.conversation_turns_repository.insert(
            user_id=user.id,
            role='assistant',
            content=final_text,
            conversation_id=conversation_id,
            created_at=utc_now(),
        )

        # H2-046 Part 0: archive in the background so the (~3s) claude-based
        # entity extraction we're about to swap in doesn't block the reply.
        # The Task is parked on _inflight_archival_tasks so tests can
        # deterministically await it via wait_for_archival_idle(). In
        # production nothing awaits this — the reply has already been
        # delivered and the archive can fail or finish on its own time.
        archive_task = asyncio.create_task(self._archive_memory_async(
            user_id=user.id,
            user_text=text,
            assistant_text=final_text,
            user_turn_id=user_turn_id,
            assistant_turn_id=assistant_turn_id,
        ))
        self._inflight_archival_tasks.add(archive_task)
        archive_task.add_done_callback(self._inflight_archival_tasks.discard)

        # V3.7 streaming: deliver the final reply via finalize() so the
        # placeholder gets edited to the real text (or split across
        # messages for >4000 chars). The DispatcherOutput still carries
        # `final_text` per the existing contract — the caller layer
        # (UnifiedPipeline / TelegramBot) reads `streaming.final_sent`
        # to decide whether to skip its own text-send. That guard
        # prevents the duplicate-send race where streaming has already
        # delivered the text and `_send_output` would deliver it again.
        metadata: dict[str, Any] = dict(reply_metadata)
        if streaming is not None:
            await streaming.finalize(final_text)
            metadata['streamed'] = True

        return DispatcherOutput(
            text=final_text, iterations=iterations, buttons=buttons, metadata=metadata,
        )

    async def _handle_approval_callback(
        self, *, callback_data: str, user: User, translator: Translator,
    ) -> DispatcherOutput:
        """Route an approval:{approve,cancel}:<id> button-tap.

        Two action_types matter here:
          - 'destructive_message_gate' (NEW, H2-039 FIX 1): on approve, look
            up the original prompt from the approval payload and re-fire
            this dispatcher with bypass_destructive_approval=True. On cancel,
            tell the user nothing was changed.
          - everything else (Phase 4 per-tool approvals via the dispatcher's
            requires_approval branch): on approve, route through
            approval_service.execute with make_post_approval_executor.
            On cancel, route through approval_service.cancel.

        Either way the original user text was a button callback string,
        NOT a prompt to the LLM — so we never invoke the LLM here directly.
        """
        if self.approvals_repository is None:
            return DispatcherOutput(
                text=translator.t('approval_cancelled'), iterations=0,
            )

        parts = callback_data.split(':', 2)
        if len(parts) != 3:
            return DispatcherOutput(
                text=translator.t('approval_cancelled'), iterations=0,
            )
        verb, approval_id = parts[1], parts[2]

        # Cancel branch.
        if verb == 'cancel':
            sr = self.approval_service.cancel(approval_id, user.id, translator)
            return DispatcherOutput(text=sr.text, iterations=0,
                                    buttons=list(sr.buttons or []))

        # Approve branch — verb == 'approve'. Two action_types diverge.
        row = self.approvals_repository.get(approval_id)
        if row is None or row.user_id != user.id:
            return DispatcherOutput(
                text=translator.t('approval_cancelled'), iterations=0,
            )
        action_type = row.action_type
        payload = self._parse_approval_payload(row)
        current_thread = self._load_current_thread(user_id=user.id)
        if self.conversation_service is None:
            matches_thread, mismatch_text = True, ''
        else:
            matches_thread, mismatch_text = _thread_binding_matches(
                expected_thread_id=str(payload.get('thread_id') or '').strip(),
                expected_thread_revision=_parse_thread_revision(
                    payload.get('thread_revision') or 0
                ),
                current_thread=current_thread,
                required_status='pending_approval',
            )
        if not matches_thread:
            try:
                self.approvals_repository.update_status(
                    approval_id,
                    status='cancelled',
                    cancelled_at=utc_now(),
                )
            except Exception:
                logger.warning(
                    'dispatcher_stale_approval_cancel_failed',
                    extra={'user_id': user.id, 'approval_id': approval_id},
                )
            return DispatcherOutput(
                text=mismatch_text,
                iterations=0,
                buttons=[],
                metadata={
                    'approval_thread_mismatch': True,
                    'approval_thread_id': str(payload.get('thread_id') or '').strip(),
                },
            )

        if action_type != 'destructive_message_gate':
            if action_type == 'duplicate_reminder_cleanup':
                sr = self.approval_service.execute(
                    approval_id,
                    user.id,
                    lambda _action_type, callback_payload: self._execute_duplicate_reminder_cleanup(
                        user_id=user.id,
                        payload=callback_payload,
                    ),
                    translator,
                )
                if current_thread is not None:
                    refreshed_state = self._load_recovery_context(user_id=user.id)
                    self._store_thread_state(
                        user_id=user.id,
                        recovery_state=refreshed_state,
                        thread=_thread_with_status(
                            current_thread,
                            status='active',
                            now=utc_now(),
                        ),
                        topic='duplicate reminder cleanup',
                    )
                return DispatcherOutput(text=sr.text, iterations=0,
                                        buttons=list(sr.buttons or []))
            # Phase 4 per-tool approval — sync executor path.
            executor = make_post_approval_executor(self.registry)
            sr = self.approval_service.execute(approval_id, user.id,
                                               executor, translator)
            if current_thread is not None:
                refreshed_state = self._load_recovery_context(user_id=user.id)
                self._store_thread_state(
                    user_id=user.id,
                    recovery_state=refreshed_state,
                    thread=_thread_with_status(
                        current_thread,
                        status='active',
                        now=utc_now(),
                    ),
                    topic=str(payload.get('tool_name') or action_type),
                )
            return DispatcherOutput(text=sr.text, iterations=0,
                                    buttons=list(sr.buttons or []))

        # Destructive-message-gate approval — atomically claim and re-fire
        # the dispatcher with the original prompt and bypass=True. We
        # bypass approval_service.execute here because re-firing requires
        # awaiting the async pipeline, which the sync executor contract
        # doesn't support.
        claimed = self.approvals_repository.claim_for_execution(approval_id)
        if claimed is None:
            return DispatcherOutput(
                text=translator.t('approval_already_expired'), iterations=0,
            )
        if claimed.status != 'approved':
            return DispatcherOutput(
                text=translator.t('approval_status', status=claimed.status),
                iterations=0,
            )
        recovery_payload = payload.get('recovery') if isinstance(payload.get('recovery'), dict) else {}
        original_prompt = (payload.get('original_prompt') or '').strip()
        if not original_prompt:
            return DispatcherOutput(
                text=translator.t('approval_cancelled'), iterations=0,
            )

        # Mark executed BEFORE re-fire so a downstream failure doesn't leave
        # the approval row in 'approved' indefinitely (next approval-sweep
        # would then over-expire it). If the re-fire raises, we still log
        # but the approval is closed.
        try:
            self.approvals_repository.mark_executed(approval_id,
                                                    executed_at=utc_now())
        except ValueError:
            pass

        logger.info(
            'dispatcher_destructive_gate_approved',
            extra={'user_id': user.id, 'approval_id': approval_id},
        )

        # Re-fire the original prompt through the dispatcher with the
        # bypass flag set so the gate doesn't re-trigger.
        try:
            post_out = await asyncio.wait_for(
                self.handle(DispatcherInput(
                    user=user, text=original_prompt, translator=translator,
                    streaming_session=None,
                    bypass_destructive_approval=True,
                    post_approval_resume=True,
                )),
                timeout=_POST_APPROVAL_CONTINUATION_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            fallback_decision = await self.fallback_manager.decide_post_approval_timeout(
                context=FallbackContext(
                    route='',
                    stage='post_approval',
                    provider='brain_router',
                    root_reason='timeout',
                    raw_text=original_prompt,
                    recovered_text=original_prompt,
                    recovery_metadata=dict(recovery_payload or {}),
                    details={
                        'has_contact_reminder_intent': _has_contact_reminder_intent(original_prompt),
                        'has_outbound_message_intent': _has_outbound_message_intent(original_prompt),
                        'has_rich_reminder_context': _prompt_has_rich_reminder_context(original_prompt),
                    },
                ),
                contact_reminder_fallback=lambda: _execute_post_approval_timeout_contact_reminder_fallback(
                    registry=self.registry,
                    user_id=user.id,
                    original_prompt=original_prompt,
                    app_timezone=self.app_timezone,
                    now=utc_now(),
                ),
                cleanup_fallback=lambda: _execute_post_approval_timeout_cleanup_fallback(
                    proactive_notifications_repository=self.proactive_notifications_repository,
                    registry=self.registry,
                    user_id=user.id,
                    original_prompt=original_prompt,
                    now=utc_now(),
                ),
                internal_reminder_fallback=lambda: _execute_post_approval_timeout_internal_reminder_fallback(
                    registry=self.registry,
                    user_id=user.id,
                    original_prompt=original_prompt,
                    app_timezone=self.app_timezone,
                    now=utc_now(),
                ),
            )
            structured_failure = fallback_decision.structured_failure
            assert structured_failure is not None
            logger.warning(
                'dispatcher_post_approval_continuation_timed_out '
                'raw_reason=%s route=%s stage=%s provider=%s fallback=%s '
                'root_reason=%s safe_action_taken=%s unsafe_action_blocked=%s',
                structured_failure.technical_reason,
                fallback_decision.log_fields.get('route', structured_failure.route),
                fallback_decision.log_fields.get('stage', structured_failure.stage),
                fallback_decision.log_fields.get('provider', structured_failure.provider),
                fallback_decision.log_fields.get('fallback_type', structured_failure.fallback),
                fallback_decision.log_fields.get('root_reason', structured_failure.root_reason),
                fallback_decision.log_fields.get('safe_action_taken', structured_failure.safe_action_taken),
                fallback_decision.log_fields.get('unsafe_action_blocked', structured_failure.unsafe_action_blocked),
                extra={
                    'user_id': user.id,
                    'approval_id': approval_id,
                    'prompt_chars': len(original_prompt),
                    **fallback_decision.log_fields,
                },
            )
            if fallback_decision.payload is not None:
                logger.info(
                    'dispatcher_post_approval_timeout_local_fallback_succeeded',
                    extra={
                        'user_id': user.id,
                        'approval_id': approval_id,
                        **fallback_decision.log_fields,
                        'fallback_kind': fallback_decision.payload.get('kind'),
                        'fallback_query': fallback_decision.payload.get('query'),
                        'task_id': fallback_decision.payload.get('task_id'),
                        'reminder_id': fallback_decision.payload.get('reminder_id'),
                        'fallback_title': fallback_decision.payload.get('title'),
                        'fallback_reminder_created': fallback_decision.payload.get('created'),
                    },
                )
                return DispatcherOutput(
                    text=fallback_decision.user_text,
                    iterations=0,
                    buttons=[],
                    metadata={
                        'post_approval_timeout': True,
                        'post_approval_local_fallback': True,
                        'outbound_send_blocked': _has_outbound_message_intent(original_prompt),
                        'fallback_reminder_created': bool(fallback_decision.payload.get('created')),
                        'structured_failure': structured_failure.to_metadata(),
                    },
                )
            return DispatcherOutput(
                text=fallback_decision.user_text,
                iterations=0,
                buttons=[],
                metadata={
                    'post_approval_timeout': True,
                    'structured_failure': structured_failure.to_metadata(),
                },
            )
        if (_is_approved_contact_send_payload(payload)
                and _looks_like_contact_provider_wiring_error(post_out.text)):
            metadata = dict(post_out.metadata or {})
            metadata['approved_contact_provider_failure'] = True
            return DispatcherOutput(
                text=_approved_contact_provider_failure_text(),
                iterations=post_out.iterations,
                buttons=[],
                metadata=metadata,
            )
        return post_out

    async def _archive_memory_async(
        self,
        *,
        user_id: str,
        user_text: str,
        assistant_text: str,
        user_turn_id: str,
        assistant_turn_id: str,
    ) -> None:
        """H2-046 Part 0: background memory archival. Both `mem0.add` (legacy)
        and the upcoming `LocalMemoryService.add` are synchronous and can take
        seconds; this method wraps them in asyncio.to_thread so they never
        block the event loop. Failure is logged, never raised — the reply
        has already been delivered."""
        messages = [
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': assistant_text},
        ]
        try:
            result = await _run_blocking_without_default_executor(
                self.mem0.add,
                messages,
                user_id=user_id,
            )
        except Exception:
            logger.warning('dispatcher_memory_add_failed', extra={'user_id': user_id})
            return

        memory_id = _extract_memory_id(result)
        try:
            await _run_blocking_without_default_executor(
                self.conversation_turns_repository.mark_mem0_persisted,
                turn_ids=[user_turn_id, assistant_turn_id],
                memory_id=memory_id,
            )
        except Exception:
            logger.warning(
                'dispatcher_archive_mark_persisted_failed',
                extra={'user_id': user_id, 'turn_ids': [user_turn_id, assistant_turn_id]},
            )

    async def wait_for_archival_idle(self) -> None:
        """Block until every in-flight archival task started by this dispatcher
        instance has finished (or raised). Used by tests that assert the
        post-archival DB state; production code never calls this."""
        if not self._inflight_archival_tasks:
            return
        await asyncio.gather(*self._inflight_archival_tasks, return_exceptions=True)


def make_post_approval_executor(
    registry: ToolRegistry,
    *,
    user_id_resolver: Callable[[str], str] = lambda uid: uid,
) -> Callable[[str, dict], ServiceResponse]:
    """Build the executor passed to approval_service.execute() when the
    user taps Approve. The executor receives (action_type, payload) where
    action_type is the tool name and payload contains 'tool_name' +
    'arguments' as packed by the dispatcher. It looks up the spec, calls
    spec.fn with user_id injected, and returns a ServiceResponse derived
    from the tool's announcement.
    """

    def executor(action_type: str, payload: dict) -> ServiceResponse:
        spec = registry.get(action_type) or registry.get(payload.get('tool_name', ''))
        if spec is None:
            return ServiceResponse(text='Tool not found.')
        args = dict(payload.get('arguments') or {})
        args.setdefault('user_id', payload.get('user_id') or args.get('user_id'))
        try:
            result = spec.fn(**args)
        except TypeError:
            args.pop('user_id', None)
            try:
                result = spec.fn(**args)
            except Exception as exc:
                return ServiceResponse(text=f'Tool failed: {type(exc).__name__}')
        except Exception as exc:
            return ServiceResponse(text=f'Tool failed: {type(exc).__name__}')
        if isinstance(result, ToolResult):
            return ServiceResponse(text=result.announcement or result.error or 'Done.')
        return ServiceResponse(text='Done.')

    return executor
