from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urljoin, urlparse


_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f]+')
_WHITESPACE = re.compile(r'\s+')


def sanitize_plaintext(value: str | None, *, max_length: int) -> str:
    if not value:
        return ''
    cleaned = _CONTROL_CHARS.sub(' ', value)
    cleaned = _WHITESPACE.sub(' ', cleaned).strip()
    if len(cleaned) <= max_length:
        return cleaned
    if max_length <= 3:
        return cleaned[:max_length]
    return cleaned[: max_length - 3].rstrip() + '...'


def sanitize_url(value: str | None, *, max_length: int = 2048) -> str:
    candidate = sanitize_plaintext(value, max_length=max_length)
    if not candidate:
        return ''
    if candidate.startswith('/'):
        candidate = urljoin('https://duckduckgo.com', candidate)
    if candidate.startswith('//'):
        candidate = f'https:{candidate}'
    parsed = urlparse(candidate)
    if parsed.netloc.endswith('duckduckgo.com'):
        redirected = parse_qs(parsed.query).get('uddg', [None])[0]
        if redirected:
            candidate = unquote(redirected)
            parsed = urlparse(candidate)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return ''
    return candidate[:max_length]
