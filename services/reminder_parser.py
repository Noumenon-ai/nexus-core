from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import dateparser
import json as _json
import logging
from dateutil.rrule import rrulestr

from services.brain_router import BrainRouter
from utils.dates import app_now, to_utc
from utils.prompt_templates import reminder_parser_prompt


logger = logging.getLogger(__name__)

try:
    from recurrent.event_parser import RecurringEvent
except Exception:
    RecurringEvent = None


RECURRENCE_HINTS = {'every', 'daily', 'weekly', 'weekdays', 'weekends', 'each'}
DAY_MAP = {
    'monday': 'MO', 'mondays': 'MO',
    'tuesday': 'TU', 'tuesdays': 'TU',
    'wednesday': 'WE', 'wednesdays': 'WE',
    'thursday': 'TH', 'thursdays': 'TH',
    'friday': 'FR', 'fridays': 'FR',
    'saturday': 'SA', 'saturdays': 'SA',
    'sunday': 'SU', 'sundays': 'SU',
}
TIME_PATTERN = re.compile(r'\b(?:\d{1,2})(?::\d{2})?\s*(?:am|pm)?\b|\bnoon\b|\bmidnight\b', re.IGNORECASE)


@dataclass(slots=True)
class ReminderParseResult:
    type: str
    body: str
    confidence: float
    datetime_utc: datetime | None = None
    rrule: str | None = None
    clarifying_question: str | None = None
    source_layer: str | None = None
    defaulted_time: bool = False
    natural_text: str | None = None


class ReminderParser:
    def __init__(self, brain_router: BrainRouter, app_timezone: str) -> None:
        # H2-047 Wave 2: LLM-layer reminder parsing now goes through
        # brain_router (Claude CLI) instead of ai_service (Gemini). The
        # dateparser-first path is unchanged — only the fallback LLM call
        # site was swapped.
        self.brain_router = brain_router
        self.app_timezone = app_timezone

    async def parse(self, user_id: str, text: str, *, context_subject: str | None = None) -> ReminderParseResult:
        try:
            schedule_text, body = self._split_schedule_and_body(text, context_subject=context_subject)
        except ValueError:
            return await self._parse_with_llm(user_id, text.strip(), '')
        recurring = self._parse_recurring(schedule_text, body)
        if recurring is not None:
            return recurring
        one_shot = self._parse_one_shot(schedule_text, body)
        if one_shot is not None:
            return one_shot
        return await self._parse_with_llm(user_id, schedule_text, body)

    def compute_next_slot(self, rrule_text: str, *, after: datetime | None = None) -> datetime:
        after_time = after or app_now(self.app_timezone)
        rule = rrulestr(rrule_text, dtstart=after_time)
        next_slot = rule.after(after_time, inc=False)
        if next_slot is None:
            raise ValueError('RRULE did not produce a future slot')
        return to_utc(next_slot)

    def _split_schedule_and_body(self, text: str, *, context_subject: str | None = None) -> tuple[str, str]:
        lowered = text.strip()
        triggers = r'(?:remind me|remind|reminder|set\s+a\s+reminder|set\s+reminder|add\s+a\s+reminder|add\s+reminder|create\s+a\s+reminder|create\s+reminder|schedule\s+a\s+reminder|schedule\s+reminder)'
        schedule_starts = r'(?:in\s|at\s|on\s|by\s|tomorrow\b|today\b|tonight\b|next\s|every\s|each\s|after\s|this\s)'
        match = re.match(rf'^{triggers}\s+(.+?)\s+to\s+(.+)$', lowered, re.IGNORECASE)
        if match and re.match(rf'^{schedule_starts}', match.group(1).strip(), re.IGNORECASE):
            return match.group(1).strip(), match.group(2).strip()
        match = re.match(rf'^{triggers}\s+to\s+(.+?)\s+({schedule_starts}.+)$', lowered, re.IGNORECASE)
        if match:
            return match.group(2).strip(), match.group(1).strip()
        match = re.match(rf'^{triggers}\s+(.+?)\s+to\s+(.+)$', lowered, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        if 'about that' in lowered.lower() and context_subject:
            return lowered, context_subject
        raise ValueError('Reminder text must include a schedule and body')

    def _parse_recurring(self, schedule_text: str, body: str) -> ReminderParseResult | None:
        normalized = schedule_text.lower()
        if not any(hint in normalized for hint in RECURRENCE_HINTS) and not any(day in normalized for day in DAY_MAP):
            return None
        time_dt, defaulted_time = self._parse_time_component(schedule_text)
        byhour = time_dt.hour
        byminute = time_dt.minute
        byday = None
        freq = 'WEEKLY'
        natural_text = schedule_text
        if 'weekday' in normalized:
            byday = 'MO,TU,WE,TH,FR'
        elif 'weekend' in normalized:
            byday = 'SA,SU'
        elif 'daily' in normalized or 'every day' in normalized:
            freq = 'DAILY'
        else:
            for word, code in DAY_MAP.items():
                if word in normalized:
                    byday = code
                    break
        if byday is None and freq == 'WEEKLY':
            if RecurringEvent is not None:
                try:
                    parser = RecurringEvent(now_date=app_now(self.app_timezone).date())
                    parser.parse(schedule_text)
                except Exception:
                    pass
            return None
        if freq == 'DAILY':
            rrule = f'FREQ=DAILY;BYHOUR={byhour};BYMINUTE={byminute}'
        else:
            rrule = f'FREQ={freq};BYDAY={byday};BYHOUR={byhour};BYMINUTE={byminute}'
        next_fire_at = self.compute_next_slot(rrule, after=app_now(self.app_timezone))
        return ReminderParseResult(
            type='recurring',
            body=body,
            confidence=0.95,
            datetime_utc=next_fire_at,
            rrule=rrule,
            source_layer='recurrent',
            defaulted_time=defaulted_time,
            natural_text=natural_text,
        )

    def _parse_one_shot(self, schedule_text: str, body: str) -> ReminderParseResult | None:
        from utils.language import dateparser_languages, detect_language
        defaulted_time = not bool(TIME_PATTERN.search(schedule_text))
        settings = {
            'TIMEZONE': self.app_timezone,
            'RETURN_AS_TIMEZONE_AWARE': True,
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': app_now(self.app_timezone),
        }
        parsed = dateparser.parse(schedule_text, settings=settings, languages=dateparser_languages(detect_language(schedule_text)))
        if parsed is None:
            return None
        if defaulted_time:
            parsed = parsed.replace(hour=9, minute=0, second=0, microsecond=0)
        return ReminderParseResult(
            type='one_shot',
            body=body,
            confidence=0.9,
            datetime_utc=to_utc(parsed),
            source_layer='dateparser',
            defaulted_time=defaulted_time,
            natural_text=schedule_text,
        )

    async def _parse_with_llm(self, user_id: str, schedule_text: str, body: str) -> ReminderParseResult:
        # H2-047 Wave 2: call brain_router with a structured-output prompt;
        # parse JSON from the response. If the response isn't parseable
        # JSON the dateparser-only paths upstream have already failed, so
        # we treat the whole turn as ambiguous and surface the clarifying
        # question — same fallback shape generate_json used to give us.
        fallback = {'type': 'ambiguous', 'body': body, 'confidence': 0.0}
        prompt = (
            f"{reminder_parser_prompt()}\n\n"
            f"Schedule text:\n{schedule_text}\n\n"
            f"Reminder body:\n{body}\n\n"
            "Respond with a single JSON object only. No prose around it."
        )
        try:
            result = await self.brain_router.generate(prompt, user_id=user_id)
            raw = (result.text or '').strip()
            payload = self._extract_json(raw) or fallback
        except Exception:
            logger.warning('reminder_parser_llm_failed', extra={'user_id': user_id})
            payload = fallback
        parse_type = payload.get('type', 'ambiguous')
        raw_confidence = payload.get('confidence', 0.0)
        try:
            confidence = float(raw_confidence) if not isinstance(raw_confidence, str) else float(raw_confidence.strip())
        except (TypeError, ValueError):
            confidence = 0.0
        if parse_type == 'ambiguous' or confidence < 0.7:
            return ReminderParseResult(
                type='ambiguous',
                body=body,
                confidence=confidence,
                clarifying_question='What time should I use for that reminder?',
                source_layer='llm',
            )
        if parse_type == 'recurring':
            rrule = payload.get('rrule')
            if not rrule:
                return ReminderParseResult(type='ambiguous', body=body, confidence=0.0, clarifying_question='What schedule should I use?', source_layer='llm')
            next_fire_at = self.compute_next_slot(rrule, after=app_now(self.app_timezone))
            return ReminderParseResult(type='recurring', body=payload.get('body', body), confidence=confidence, datetime_utc=next_fire_at, rrule=rrule, source_layer='llm', natural_text=schedule_text)
        datetime_iso = payload.get('datetime_iso')
        if not datetime_iso:
            return ReminderParseResult(type='ambiguous', body=body, confidence=0.0, clarifying_question='What time should I use?', source_layer='llm')
        try:
            parsed = datetime.fromisoformat(datetime_iso)
        except (TypeError, ValueError):
            return ReminderParseResult(type='ambiguous', body=body, confidence=0.0, clarifying_question='What time should I use?', source_layer='llm')
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(self.app_timezone))
        return ReminderParseResult(type='one_shot', body=payload.get('body', body), confidence=confidence, datetime_utc=to_utc(parsed), source_layer='llm', natural_text=schedule_text)

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Pull a JSON object out of Claude's response. Tries the whole
        string first, then a fenced ```json block, then any {...} match.
        Returns None on failure so the caller can fall back to ambiguous."""
        if not raw:
            return None
        candidates: list[str] = [raw.strip()]
        fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        brace = re.search(r'\{.*\}', raw, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
        for candidate in candidates:
            try:
                parsed = _json.loads(candidate)
            except (_json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _parse_time_component(self, schedule_text: str):
        explicit = TIME_PATTERN.search(schedule_text)
        if not explicit:
            return app_now(self.app_timezone).replace(hour=9, minute=0, second=0, microsecond=0), True
        parsed = dateparser.parse(explicit.group(0), settings={'TIMEZONE': self.app_timezone, 'RETURN_AS_TIMEZONE_AWARE': True})
        if parsed is None:
            return app_now(self.app_timezone).replace(hour=9, minute=0, second=0, microsecond=0), True
        return parsed, False
