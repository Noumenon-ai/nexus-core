from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from services.conversational_recovery import RecoveryResult


_MAX_CLARIFICATION_AGE = timedelta(minutes=20)
_ANSWER_TOKEN_STOPWORDS = {
    'a',
    'an',
    'the',
    'to',
    'about',
    'my',
    'your',
    'their',
    'it',
    'them',
    'that',
    'this',
    'one',
    'option',
    'please',
}
_ORDINAL_TO_INDEX = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
}


@dataclass(slots=True)
class ClarificationOption:
    option_id: str
    label: str
    resolution_text: str
    destructive: bool = False
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClarificationOption:
        return cls(
            option_id=str(payload.get('option_id') or '').strip(),
            label=str(payload.get('label') or '').strip(),
            resolution_text=str(payload.get('resolution_text') or '').strip(),
            destructive=bool(payload.get('destructive')),
            aliases=[
                str(alias).strip()
                for alias in (payload.get('aliases') or [])
                if str(alias).strip()
            ],
        )


@dataclass(slots=True)
class ActiveClarification:
    clarification_id: str
    question: str
    options: list[ClarificationOption]
    created_at: str
    expires_at: str
    source_turn_id: str
    risk_level: str
    primary_option_id: str | None = None
    kind: str = 'generic'
    original_text: str = ''
    requested_slot: str = ''
    thread_id: str = ''
    thread_kind: str = 'unknown'
    thread_revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'clarification_id': self.clarification_id,
            'question': self.question,
            'options': [option.to_dict() for option in self.options],
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'source_turn_id': self.source_turn_id,
            'risk_level': self.risk_level,
            'primary_option_id': self.primary_option_id,
            'kind': self.kind,
            'original_text': self.original_text,
            'requested_slot': self.requested_slot,
            'thread_id': self.thread_id,
            'thread_kind': self.thread_kind,
            'thread_revision': self.thread_revision,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActiveClarification | None:
        clarification_id = str(payload.get('clarification_id') or '').strip()
        question = str(payload.get('question') or '').strip()
        created_at = str(payload.get('created_at') or '').strip()
        expires_at = str(payload.get('expires_at') or '').strip()
        source_turn_id = str(payload.get('source_turn_id') or '').strip()
        risk_level = str(payload.get('risk_level') or '').strip()
        if not clarification_id or not question or not created_at or not risk_level:
            return None
        options = [
            ClarificationOption.from_dict(item)
            for item in (payload.get('options') or [])
            if isinstance(item, dict)
        ]
        try:
            thread_revision = int(payload.get('thread_revision') or 0)
        except (TypeError, ValueError):
            thread_revision = 0
        return cls(
            clarification_id=clarification_id,
            question=question,
            options=[item for item in options if item.option_id and item.label and item.resolution_text],
            created_at=created_at,
            expires_at=expires_at,
            source_turn_id=source_turn_id,
            risk_level=risk_level,
            primary_option_id=str(payload.get('primary_option_id') or '').strip() or None,
            kind=str(payload.get('kind') or 'generic').strip() or 'generic',
            original_text=str(payload.get('original_text') or '').strip(),
            requested_slot=str(payload.get('requested_slot') or '').strip(),
            thread_id=str(payload.get('thread_id') or '').strip(),
            thread_kind=str(payload.get('thread_kind') or 'unknown').strip() or 'unknown',
            thread_revision=thread_revision,
            metadata=dict(payload.get('metadata') or {}),
        )


@dataclass(slots=True)
class ClarificationResolution:
    action: str
    resolved_text: str | None = None
    selected_option_id: str | None = None
    destructive: bool = False
    follow_up_text: str | None = None
    clarification: ActiveClarification | None = None


def build_rental_status_clarification(
    *,
    subject: str,
    original_text: str,
    source_turn_id: str,
    created_at: datetime,
) -> ActiveClarification:
    option_one_resolution = _render_rental_status_resolution_text(subject)
    option_one = ClarificationOption(
        option_id='check_status',
        label=f'check whether {subject} were updated',
        resolution_text=option_one_resolution,
        aliases=[
            'check whether they were updated',
            'were they updated',
            'check whether updated',
            'check status',
        ],
    )
    option_two = ClarificationOption(
        option_id='update_records',
        label='update the rental records now',
        resolution_text='update my rental records now',
        destructive=True,
        aliases=[
            'update records',
            'update records now',
            'update them now',
            'update rentals now',
        ],
    )
    option_three = ClarificationOption(
        option_id='send_updates',
        label='send updates about the rentals',
        resolution_text='send updates about my rentals',
        destructive=True,
        aliases=[
            'send updates',
            'send rental updates',
            'send updates about rentals',
        ],
    )
    return ActiveClarification(
        clarification_id=str(uuid.uuid4()),
        question=(
            "Do you mean:\n"
            f"1. {option_one.label},\n"
            f"2. {option_two.label},\n"
            f"3. {option_three.label},\n"
            "or something else?"
        ),
        options=[option_one, option_two, option_three],
        created_at=created_at.astimezone(timezone.utc).isoformat(),
        expires_at=(created_at + _MAX_CLARIFICATION_AGE).astimezone(timezone.utc).isoformat(),
        source_turn_id=source_turn_id,
        risk_level='medium',
        primary_option_id=option_one.option_id,
        kind='rental_status_menu',
        original_text=original_text,
        requested_slot='rental_status_intent',
        metadata={'rental_subject': subject},
    )


def build_recovery_clarification(
    *,
    result: RecoveryResult,
    source_turn_id: str,
    created_at: datetime,
) -> ActiveClarification | None:
    question = str(result.clarification_text or '').strip()
    if not question:
        return None

    recipient_candidates = [
        str(item).strip()
        for item in result.context_updates.get('recipient_candidates') or []
        if str(item).strip()
    ]
    alias_query = str(result.context_updates.get('recipient_alias_query') or '').strip()
    original_text = result.recovered_text or result.raw_text
    if recipient_candidates and alias_query:
        options = [
            ClarificationOption(
                option_id=f'recipient_{index}',
                label=label,
                resolution_text=_replace_first_case_insensitive(
                    original_text,
                    alias_query,
                    label,
                ),
                destructive=result.risk_level == 'high',
                aliases=[label],
            )
            for index, label in enumerate(recipient_candidates, start=1)
        ]
        return ActiveClarification(
            clarification_id=str(uuid.uuid4()),
            question=question,
            options=options,
            created_at=created_at.astimezone(timezone.utc).isoformat(),
            expires_at=(created_at + _MAX_CLARIFICATION_AGE).astimezone(timezone.utc).isoformat(),
            source_turn_id=source_turn_id,
            risk_level=result.risk_level,
            primary_option_id=options[0].option_id if options else None,
            kind='recipient_candidates',
            original_text=original_text,
            requested_slot=result.missing_slot or 'recipient',
            metadata={'recipient_alias_query': alias_query},
        )

    return ActiveClarification(
        clarification_id=str(uuid.uuid4()),
        question=question,
        options=[],
        created_at=created_at.astimezone(timezone.utc).isoformat(),
        expires_at=(created_at + _MAX_CLARIFICATION_AGE).astimezone(timezone.utc).isoformat(),
        source_turn_id=source_turn_id,
        risk_level=result.risk_level,
        kind='generic',
        original_text=original_text,
        requested_slot=result.missing_slot or '',
        metadata={'missing_slot': result.missing_slot or ''},
    )


def build_generic_clarification(
    *,
    question: str,
    original_text: str,
    source_turn_id: str,
    created_at: datetime,
    risk_level: str,
    kind: str = 'generic',
    options: list[ClarificationOption] | None = None,
    primary_option_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ActiveClarification:
    return ActiveClarification(
        clarification_id=str(uuid.uuid4()),
        question=question,
        options=list(options or []),
        created_at=created_at.astimezone(timezone.utc).isoformat(),
        expires_at=(created_at + _MAX_CLARIFICATION_AGE).astimezone(timezone.utc).isoformat(),
        source_turn_id=source_turn_id,
        risk_level=risk_level,
        primary_option_id=primary_option_id,
        kind=kind,
        original_text=original_text,
        requested_slot='',
        metadata=dict(metadata or {}),
    )


def build_vague_clarification_state(
    *,
    original_text: str,
    clarification_text: str,
    source_turn_id: str,
    created_at: datetime,
) -> ActiveClarification:
    subject = _recover_rental_subject(original_text)
    if clarification_text.startswith('Do you mean:\n') and subject:
        return build_rental_status_clarification(
            subject=subject,
            original_text=original_text,
            source_turn_id=source_turn_id,
            created_at=created_at,
        )
    return build_generic_clarification(
        question=clarification_text,
        original_text=original_text,
        source_turn_id=source_turn_id,
        created_at=created_at,
        risk_level='medium',
        kind='generic',
    )


def refresh_clarification(
    clarification: ActiveClarification,
    *,
    question: str,
    source_turn_id: str,
    created_at: datetime,
) -> ActiveClarification:
    return ActiveClarification(
        clarification_id=str(uuid.uuid4()),
        question=question,
        options=list(clarification.options),
        created_at=created_at.astimezone(timezone.utc).isoformat(),
        expires_at=(created_at + _MAX_CLARIFICATION_AGE).astimezone(timezone.utc).isoformat(),
        source_turn_id=source_turn_id,
        risk_level=clarification.risk_level,
        primary_option_id=clarification.primary_option_id,
        kind=clarification.kind,
        original_text=clarification.original_text,
        requested_slot=clarification.requested_slot,
        thread_id=clarification.thread_id,
        thread_kind=clarification.thread_kind,
        thread_revision=clarification.thread_revision,
        metadata=dict(clarification.metadata),
    )


def resolve_clarification_answer(
    clarification: ActiveClarification,
    *,
    answer_text: str,
    now: datetime,
    follow_up_text: str,
    stale_text: str,
) -> ClarificationResolution:
    if _is_stale(clarification, now=now):
        if _looks_like_answer(answer_text):
            return ClarificationResolution(
                action='stale',
                follow_up_text=stale_text,
            )
        return ClarificationResolution(action='ignore')

    option = _match_option(clarification, answer_text)
    if option is not None:
        return ClarificationResolution(
            action='resolved',
            resolved_text=option.resolution_text,
            selected_option_id=option.option_id,
            destructive=option.destructive,
        )

    if not clarification.options:
        return ClarificationResolution(action='ignore')

    if _looks_like_answer(answer_text):
        return ClarificationResolution(
            action='follow_up',
            follow_up_text=follow_up_text,
        )
    return ClarificationResolution(action='ignore')


def render_option_labels(clarification: ActiveClarification) -> list[str]:
    return [option.label for option in clarification.options]


def _is_stale(clarification: ActiveClarification, *, now: datetime) -> bool:
    try:
        created_at = datetime.fromisoformat(clarification.created_at)
    except ValueError:
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc) - created_at.astimezone(timezone.utc) > _MAX_CLARIFICATION_AGE


def _match_option(
    clarification: ActiveClarification,
    answer_text: str,
) -> ClarificationOption | None:
    normalized = _normalize(answer_text)
    if not normalized:
        return None

    numeric_match = re.fullmatch(r'(?:option\s+)?([1-9]\d*)', normalized)
    if numeric_match:
        index = int(numeric_match.group(1)) - 1
        if 0 <= index < len(clarification.options):
            return clarification.options[index]

    for ordinal, index in _ORDINAL_TO_INDEX.items():
        if re.search(rf'\b{ordinal}\b', normalized):
            option_index = index - 1
            if 0 <= option_index < len(clarification.options):
                return clarification.options[option_index]

    if ('other one' in normalized or normalized == 'other') and len(clarification.options) == 2:
        primary = clarification.primary_option_id or clarification.options[0].option_id
        if clarification.options[0].option_id == primary:
            return clarification.options[1]
        return clarification.options[0]

    if normalized in {'yes that', 'that one', 'yes', 'yep', 'yeah'}:
        primary = clarification.primary_option_id
        if primary:
            for option in clarification.options:
                if option.option_id == primary:
                    return option

    best_option: ClarificationOption | None = None
    best_score = 0.0
    second_best = 0.0
    for option in clarification.options:
        score = _score_option_match(normalized, option)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_option = option
        elif score > second_best:
            second_best = score

    if best_option is None:
        return None
    if best_score < 0.74:
        return None
    if best_score - second_best < 0.08:
        return None
    return best_option


def _score_option_match(normalized_answer: str, option: ClarificationOption) -> float:
    candidates = [option.label, option.resolution_text, *option.aliases]
    best = 0.0
    answer_tokens = _content_tokens(normalized_answer)
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        if not normalized_candidate:
            continue
        if normalized_answer == normalized_candidate:
            return 1.0
        candidate_tokens = _content_tokens(normalized_candidate)
        if answer_tokens and candidate_tokens:
            overlap = len(answer_tokens & candidate_tokens) / max(1, len(answer_tokens))
            if answer_tokens <= candidate_tokens and len(answer_tokens) >= 2:
                overlap = max(overlap, 0.9)
            best = max(best, overlap)
        best = max(best, SequenceMatcher(None, normalized_answer, normalized_candidate).ratio())
    return best


def _looks_like_answer(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if re.fullmatch(r'(?:option\s+)?[1-9]\d*', normalized):
        return True
    if any(re.search(rf'\b{ordinal}\b', normalized) for ordinal in _ORDINAL_TO_INDEX):
        return True
    if any(
        phrase in normalized
        for phrase in (
            'yes that',
            'that one',
            'other one',
            'option',
        )
    ):
        return True
    return len(normalized.split()) <= 8


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r'[a-z0-9]+', text.casefold())
        if token not in _ANSWER_TOKEN_STOPWORDS
    }


def _normalize(text: str) -> str:
    cleaned = ' '.join((text or '').strip().lower().split())
    cleaned = cleaned.rstrip('.!?')
    return cleaned


def _replace_first_case_insensitive(text: str, needle: str, replacement: str) -> str:
    pattern = re.compile(rf'\b{re.escape(needle)}\b', re.IGNORECASE)
    return pattern.sub(replacement, text, count=1)


def _recover_rental_subject(text: str) -> str | None:
    lowered = _normalize(text)
    patterns = (
        r'^(?:did you update|did you do|update)\s+(?:my|the)?\s*(?:(\d+)\s+)?rentals?\b',
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


def _render_rental_status_resolution_text(subject: str) -> str:
    match = re.match(r'^your\s+(\d+)\s+rental records$', subject)
    if match is not None:
        return f'check whether my {match.group(1)} rentals were updated'
    return 'check whether my rentals were updated'
