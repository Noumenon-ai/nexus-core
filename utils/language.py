from __future__ import annotations

import re

_HEBREW_REGEX = re.compile(r'[֐-׿]')


def detect_language(text: str) -> str:
    if not text:
        return 'en'
    if _HEBREW_REGEX.search(text):
        return 'he'
    return 'en'


def dateparser_languages(detected: str) -> list[str]:
    if detected == 'he':
        return ['he', 'en']
    return ['en']
