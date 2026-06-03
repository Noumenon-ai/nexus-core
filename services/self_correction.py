"""Deterministic self-correction detection.

Detects in-message self-corrections such as "June 2 no June 4", "5pm change it
to 6pm", or "monday scratch that move it to the 4th", and reports them as a
normalized (field, old, new) triple. The dispatcher uses this to append a
"Corrected <field>: <old> -> <new>" line to reminder confirmations so a
revision is never silently applied.

Precision model: the two values are paired only when the text strictly between
them consists ENTIRELY of correction connectors (e.g. "no", "wait", "scratch
that", "move it to"). Any real content word in the gap blocks the pair, so a
range like "from 5pm to 6pm" or a list like "monday and tuesday" never fires.
Date representations (calendar date, weekday, relative day, bare ordinal) pair
with each other; clock times pair only with clock times.

Direction matters. Forward connectors ("A no B", "A change it to B") keep the
second value: old=A, new=B. Backward connectors ("A not B", "A instead of B",
"A rather than B") keep the FIRST value: old=B, new=A. Either way the echo
reads "Corrected <field>: <rejected> -> <kept>".
"""
from __future__ import annotations

import re

_MONTHS = (
    r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|'
    r'dec(?:ember)?)'
)
_MONTHDAY = rf'{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?'
_WEEKDAY = r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
_RELATIVE = r'(?:today|tomorrow|tonight)'
_ORDINAL = r'(?:the\s+)?\d{1,2}(?:st|nd|rd|th)'
_TIME = r'\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}'

# (kind, family, value-pattern). family is the user-facing field.
_KINDS: tuple[tuple[str, str, str], ...] = (
    ('monthday', 'date', _MONTHDAY),
    ('weekday', 'date', _WEEKDAY),
    ('relative', 'date', _RELATIVE),
    ('ordinal', 'date', _ORDINAL),
    ('time', 'time', _TIME),
)

# Connector PHRASES. Multi-word triggers ("move it to") are whole phrases, not
# loose words, so common words they contain ("to", "it") cannot match alone.
# FORWARD: "A <conn> B" keeps B (old=A, new=B).
_FORWARD_PHRASE = (
    r'(?:'
    r'wait|actually|no|nope|nah|sorry|rather|instead|correction|'
    r'hold\s+on|on\s+second\s+thought|scratch\s+that|my\s+bad|'
    r'i\s+mean|i\s+meant|meant|'
    r"let'?s\s+(?:do|say|make\s+it)|"
    r'make\s+(?:it|that)|'
    r'change\s+(?:it\s+|that\s+)?to|'
    r'move\s+(?:it\s+|that\s+)?to'
    r')'
)
# BACKWARD: "A <conn> B" keeps A (old=B, new=A). "rather than"/"instead of" are
# checked here; bare "rather"/"instead" stay forward above.
_BACKWARD_PHRASE = r'(?:not|and\s+not|instead\s+of|rather\s+than)'

# A gap qualifies only if it is one-or-more connector phrases separated by
# whitespace/commas (and nothing else).
_FORWARD_RE = re.compile(rf'[\s,]*(?:{_FORWARD_PHRASE}[\s,]*)+', re.IGNORECASE)
_BACKWARD_RE = re.compile(rf'[\s,]*(?:{_BACKWARD_PHRASE}[\s,]*)+', re.IGNORECASE)


def _normalize(kind: str, value: str) -> str:
    cleaned = ' '.join(value.split())
    if kind == 'time':
        return re.sub(
            r'\s*(am|pm)$',
            lambda m: f' {m.group(1).upper()}',
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
    if kind == 'ordinal':
        # Keep the natural "the 4th" / "4th" lower-cased; digits untouched.
        return cleaned.lower()
    # monthday / weekday / relative: title-case words, digits/ordinals intact.
    return ' '.join(part.capitalize() for part in cleaned.split())


def _date_family_tokens(text: str) -> list[tuple[int, int, str]]:
    """All date-representation value spans, de-overlapped (longest wins)."""
    raw: list[tuple[int, int, str]] = []
    for kind, family, pattern in _KINDS:
        if family != 'date':
            continue
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw.append((m.start(), m.end(), kind))
    raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, str]] = []
    for start, end, kind in raw:
        if any(start < k_end and end > k_start for k_start, k_end, _ in kept):
            continue  # overlaps an already-kept (longer/earlier) span
        kept.append((start, end, kind))
    kept.sort(key=lambda t: t[0])
    return kept


def _time_tokens(text: str) -> list[tuple[int, int, str]]:
    return [
        (m.start(), m.end(), 'time')
        for m in re.finditer(_TIME, text, re.IGNORECASE)
    ]


def detect_self_correction(text: str) -> tuple[str, str, str] | None:
    """Return (field, old, new) for the latest in-message self-correction.

    field is 'date' or 'time'. Returns None when no confident correction is
    present, or when old and new normalize to the same value.
    """
    if not text:
        return None

    best: tuple[int, str, str, str] | None = None  # (end_pos, field, old, new)
    for family, tokens in (
        ('date', _date_family_tokens(text)),
        ('time', _time_tokens(text)),
    ):
        for prev, curr in zip(tokens, tokens[1:]):
            gap = text[prev[1]:curr[0]]
            if _FORWARD_RE.fullmatch(gap):
                old_tok, new_tok = prev, curr  # keep the second value
            elif _BACKWARD_RE.fullmatch(gap):
                old_tok, new_tok = curr, prev  # keep the first value
            else:
                continue
            old = _normalize(old_tok[2], text[old_tok[0]:old_tok[1]])
            new = _normalize(new_tok[2], text[new_tok[0]:new_tok[1]])
            if old.casefold() == new.casefold():
                continue
            if best is None or curr[1] > best[0]:
                best = (curr[1], family, old, new)

    if best is None:
        return None
    _end, field, old, new = best
    return field, old, new
