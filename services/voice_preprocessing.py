from __future__ import annotations

import re
from datetime import date


_EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001F5FF'
    '\U0001F600-\U0001F64F'
    '\U0001F680-\U0001F6FF'
    '\U0001F700-\U0001F77F'
    '\U0001F780-\U0001F7FF'
    '\U0001F800-\U0001F8FF'
    '\U0001F900-\U0001F9FF'
    '\U0001FA00-\U0001FAFF'
    '\u2600-\u26FF'
    '\u2700-\u27BF'
    ']+'
)
_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})\b')
_EN_TIME_RE = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b', re.IGNORECASE)
_DATE_RE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
_GMT_RE = re.compile(r'\bGMT\s*([+-])\s*(\d{1,2})\b', re.IGNORECASE)
_CURRENCY_RE = re.compile(r'([$₪])(\d+)\b')
_HEADER_RE = re.compile(r'^\s{0,3}#{1,6}\s*', re.MULTILINE)
_LIST_ITEM_RE = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+', re.MULTILINE)
_CODE_FENCE_RE = re.compile(r'```.*?```', re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`([^`]+)`')
_MARKDOWN_DECORATION_RE = re.compile(r'(\*\*|__|\*|_|~~)')
_MULTISPACE_RE = re.compile(r'\s+')
_HYPHENATED_PATH_RE = re.compile(r'(?:(?:[A-Za-z]:\\)|(?:\.{1,2}/)|/)[^\s]{2,}')
_TABLE_RE = re.compile(r'^\s*\|.+\|\s*$', re.MULTILINE)
_STACK_TRACE_RE = re.compile(r'(^Traceback \(most recent call last\)|File ".+", line \d+)', re.MULTILINE)
_API_KEY_RE = re.compile(r'sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{20,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}', re.IGNORECASE)
_TOKEN_RE = re.compile(r'Bearer\s+\S+|access_token=\S+|refresh_token=\S+|id_token=\S+|oauth', re.IGNORECASE)
_NIQQUD_RE = re.compile(r'[\u0591-\u05C7]')

_EN_MONTHS = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
}
_HE_MONTHS = {
    1: 'ינואר',
    2: 'פברואר',
    3: 'מרץ',
    4: 'אפריל',
    5: 'מאי',
    6: 'יוני',
    7: 'יולי',
    8: 'אוגוסט',
    9: 'ספטמבר',
    10: 'אוקטובר',
    11: 'נובמבר',
    12: 'דצמבר',
}
_EN_NUMBER_WORDS = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eighteen',
    19: 'nineteen',
    20: 'twenty',
    30: 'thirty',
    40: 'forty',
    50: 'fifty',
    60: 'sixty',
    70: 'seventy',
    80: 'eighty',
    90: 'ninety',
}
_EN_ORDINAL_DAYS = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth',
    6: 'sixth',
    7: 'seventh',
    8: 'eighth',
    9: 'ninth',
    10: 'tenth',
    11: 'eleventh',
    12: 'twelfth',
    13: 'thirteenth',
    14: 'fourteenth',
    15: 'fifteenth',
    16: 'sixteenth',
    17: 'seventeenth',
    18: 'eighteenth',
    19: 'nineteenth',
    20: 'twentieth',
    21: 'twenty-first',
    22: 'twenty-second',
    23: 'twenty-third',
    24: 'twenty-fourth',
    25: 'twenty-fifth',
    26: 'twenty-sixth',
    27: 'twenty-seventh',
    28: 'twenty-eighth',
    29: 'twenty-ninth',
    30: 'thirtieth',
    31: 'thirty-first',
}
_HE_SMALL_MASC = {1: 'אחד', 2: 'שניים', 3: 'שלושה', 4: 'ארבעה', 5: 'חמישה', 6: 'שישה', 7: 'שבעה', 8: 'שמונה', 9: 'תשעה'}
_HE_SMALL_FEM = {1: 'אחת', 2: 'שתיים', 3: 'שלוש', 4: 'ארבע', 5: 'חמש', 6: 'שש', 7: 'שבע', 8: 'שמונה', 9: 'תשע'}

_ABBREVIATIONS = {
    r'\bMon\b': 'Monday',
    r'\bTue\b': 'Tuesday',
    r'\bWed\b': 'Wednesday',
    r'\bThu\b': 'Thursday',
    r'\bFri\b': 'Friday',
    r'\bSat\b': 'Saturday',
    r'\bSun\b': 'Sunday',
    r'\bDr\.': 'Doctor',
    r'\bSt\.': 'Street',
}


def preprocess_english(text: str) -> str:
    spoken = _normalize_common(text)
    spoken = _EMAIL_RE.sub(lambda match: f"{match.group(1)} at {match.group(2)} dot {match.group(3)}", spoken)
    spoken = _URL_RE.sub('the link I sent', spoken)
    spoken = _GMT_RE.sub(lambda match: f"G-M-T {'plus' if match.group(1) == '+' else 'minus'} {_number_to_words_en(int(match.group(2)))}", spoken)
    spoken = _CURRENCY_RE.sub(_replace_currency_en, spoken)
    spoken = _DATE_RE.sub(_replace_date_en, spoken)
    spoken = _EN_TIME_RE.sub(_replace_time_en, spoken)
    for pattern, replacement in _ABBREVIATIONS.items():
        spoken = re.sub(pattern, replacement, spoken)
    spoken = re.sub(r'\b([0-9])\b', lambda match: _number_to_words_en(int(match.group(1))), spoken)
    return _finalize(spoken)


def preprocess_hebrew(text: str) -> str:
    spoken = _normalize_common(text)
    spoken = spoken.replace('׳', "'").replace('״', '"')
    spoken = _URL_RE.sub('הקישור ששלחתי', spoken)
    spoken = _DATE_RE.sub(_replace_date_he, spoken)
    spoken = _EN_TIME_RE.sub(_replace_time_he, spoken)
    for pattern, replacement in _ABBREVIATIONS.items():
        spoken = re.sub(pattern, replacement, spoken)
    # KNOWN LIMITATION: Hebrew noun gender is not inferred during voice preprocessing.
    spoken = _NIQQUD_RE.sub('', spoken)
    return _finalize(spoken)


def preprocess_for_language(text: str, language: str) -> str:
    if language == 'he':
        return preprocess_hebrew(text)
    return preprocess_english(text)


def chunk_synthesis_text(text: str, *, max_chars: int = 500) -> list[str]:
    cleaned = _finalize(text)
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    chunks: list[str] = []
    current = ''
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f'{current} {sentence}'
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ''
        if len(sentence) <= max_chars:
            current = sentence
            continue
        words = sentence.split()
        running = ''
        for word in words:
            candidate = word if not running else f'{running} {word}'
            if len(candidate) <= max_chars:
                running = candidate
            else:
                if running:
                    chunks.append(running)
                running = word[:max_chars]
                remainder = word[max_chars:]
                while remainder:
                    chunks.append(remainder[:max_chars])
                    remainder = remainder[max_chars:]
                    running = ''
        if running:
            current = running
    if current:
        chunks.append(current)
    return chunks


def should_skip_voice(text: str) -> bool:
    list_items = len(_LIST_ITEM_RE.findall(text))
    if list_items >= 5:
        return True
    if _URL_RE.search(text):
        return True
    if _HYPHENATED_PATH_RE.search(text):
        return True
    if _API_KEY_RE.search(text) or _TOKEN_RE.search(text):
        return True
    if _TABLE_RE.search(text) or _looks_structured_table(text):
        return True
    if '```' in text:
        return True
    if ('[telegram_inline_keyboard]' in text and re.search(r'approval|approve|cancel|אישור|אשר|בטל', text, re.IGNORECASE)) or 'Approval needed:' in text or 'נדרש אישור:' in text:
        return True
    if _STACK_TRACE_RE.search(text):
        return True
    return False


def should_rewrite_for_voice(text: str) -> bool:
    if len(text.strip()) > 80:
        return True
    if _LIST_ITEM_RE.search(text):
        return True
    if _HEADER_RE.search(text):
        return True
    if '```' in text or _INLINE_CODE_RE.search(text):
        return True
    if _URL_RE.search(text):
        return True
    return False


def is_confirmation_text(text: str) -> bool:
    normalized = _finalize(text).lower()
    confirmations = {
        'done',
        'got it',
        'sure',
        'okay',
        'ok',
        'all set',
        'task added',
        'marked done',
        'voice replies are on',
        'voice replies are off',
        'בוצע',
        'בטח',
        'אוקיי',
        'סגור',
    }
    return normalized in confirmations


def is_parsed_event_preview(text: str) -> bool:
    normalized = _finalize(text)
    return bool(
        re.search(r"I'll remind you .*Correct\?", normalized, re.IGNORECASE)
        or re.search(r'אזכיר לך .*נכון\?', normalized)
    )


def _normalize_common(text: str) -> str:
    spoken = _EMOJI_RE.sub(' ', text)
    spoken = _LINK_RE.sub(lambda match: match.group(1), spoken)
    spoken = _CODE_FENCE_RE.sub(lambda match: match.group(0).replace('```', ' '), spoken)
    spoken = _INLINE_CODE_RE.sub(lambda match: match.group(1), spoken)
    spoken = _HEADER_RE.sub('', spoken)
    spoken = re.sub(r'^>\s*', '', spoken, flags=re.MULTILINE)
    spoken = _MARKDOWN_DECORATION_RE.sub('', spoken)
    return spoken


def _finalize(text: str) -> str:
    return _MULTISPACE_RE.sub(' ', text).strip()


def _replace_currency_en(match: re.Match[str]) -> str:
    symbol = match.group(1)
    amount = int(match.group(2))
    unit = 'dollars' if symbol == '$' else 'shekels'
    return f'{_number_to_words_en(amount)} {unit}'


def _replace_date_en(match: re.Match[str]) -> str:
    year, month, day = (int(part) for part in match.groups())
    date(year, month, day)
    return f"{_EN_MONTHS[month]} {_EN_ORDINAL_DAYS[day]}"


def _replace_date_he(match: re.Match[str]) -> str:
    year, month, day = (int(part) for part in match.groups())
    date(year, month, day)
    return f'{day} ב{_HE_MONTHS[month]} {year}'


def _replace_time_en(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2) or '0')
    meridiem = match.group(3).upper()
    parts = [_number_to_words_en(hour)]
    if minute:
        parts.append(_number_to_words_en(minute))
    parts.append(meridiem)
    return ' '.join(parts)


def _replace_time_he(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2) or '0')
    meridiem = match.group(3).upper()
    parts = [_small_hebrew_number(hour, feminine=True)]
    if minute:
        parts.append(_number_to_words_en(minute))
    parts.append(meridiem)
    return ' '.join(parts)


def _small_hebrew_number(value: int, *, feminine: bool) -> str:
    table = _HE_SMALL_FEM if feminine else _HE_SMALL_MASC
    return table.get(value, str(value))


def _number_to_words_en(value: int) -> str:
    if value in _EN_NUMBER_WORDS:
        return _EN_NUMBER_WORDS[value]
    if value < 100:
        tens = (value // 10) * 10
        units = value % 10
        return f'{_EN_NUMBER_WORDS[tens]}-{_EN_NUMBER_WORDS[units]}'
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f'{_EN_NUMBER_WORDS[hundreds]} hundred'
        return f'{_EN_NUMBER_WORDS[hundreds]} hundred {_number_to_words_en(remainder)}'
    return str(value)


def _looks_structured_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    multi_column = 0
    for line in lines:
        if line.count('|') >= 2 or len(re.split(r'\s{2,}|\t', line.strip())) >= 3:
            multi_column += 1
    return multi_column >= 3
