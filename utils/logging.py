from __future__ import annotations

import logging
import re
from typing import Iterable


class RedactionFilter(logging.Filter):
    def __init__(self, patterns: Iterable[str]) -> None:
        super().__init__()
        escaped = [re.escape(pattern) for pattern in patterns if pattern]
        joined = '|'.join(escaped)
        self._pair_regex = re.compile(
            rf'(?P<prefix>(?:["\']?(?:{joined})["\']?)\s*[:=]\s*)(?P<quote>["\']?)(?P<value>[^"\'\s,;]+)(?P=quote)',
            re.IGNORECASE,
        ) if escaped else None
        self._bearer_regex = re.compile(r'\bBearer\b\s+[A-Za-z0-9._-]+', re.IGNORECASE)
        self._url_secret_regex = re.compile(
            r'([?&](?:key|apikey|api_key|access_token|token|auth|secret)=)[^\s&"\'<>]+',
            re.IGNORECASE,
        )
        self._telegram_token_regex = re.compile(r'/bot\d{8,12}:[A-Za-z0-9_-]{20,}/', re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self._redact(arg))
                else:
                    s = str(arg)
                    r = self._redact(s)
                    new_args.append(r if r != s else arg)
            record.args = tuple(new_args)
        return True

    def _redact(self, value: str) -> str:
        redacted = value
        if self._pair_regex is not None:
            redacted = self._pair_regex.sub(r'\g<prefix>\g<quote>[REDACTED]\g<quote>', redacted)
        redacted = self._url_secret_regex.sub(r'\1[REDACTED]', redacted)
        redacted = self._telegram_token_regex.sub('/bot[REDACTED]/', redacted)
        return self._bearer_regex.sub('Bearer [REDACTED]', redacted)


def configure_logging(level: str, redact_patterns: Iterable[str]) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
    redaction = RedactionFilter(redact_patterns)
    for handler in root.handlers:
        handler.addFilter(redaction)
