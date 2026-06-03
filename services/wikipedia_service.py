"""V3.upgrade — Wikipedia summary lookup.

Uses Wikipedia's REST API (no key required). Returns short summary plus
the article URL so the user can click through. Cache-friendly; rate-limit
respected via per-call timeout.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


_USER_AGENT = 'NexusBot/1.0 (personal-use; httpx)'
_BASE = 'https://en.wikipedia.org/api/rest_v1/page/summary/'


@dataclass(slots=True)
class WikipediaSummary:
    title: str
    extract: str
    url: str

    def to_dict(self) -> dict:
        return {'title': self.title, 'extract': self.extract, 'url': self.url}


class WikipediaService:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None, timeout: float = 10.0) -> None:
        self._http_client = http_client
        self._timeout = timeout

    async def lookup(self, topic: str) -> WikipediaSummary | None:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError('topic must be non-empty')
        slug = topic.strip().replace(' ', '_')
        url = f'{_BASE}{slug}'
        if self._http_client is not None:
            response = await self._http_client.get(url, headers={'User-Agent': _USER_AGENT})
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers={'User-Agent': _USER_AGENT})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return WikipediaSummary(
            title=str(data.get('title') or topic),
            extract=str(data.get('extract') or ''),
            url=str((data.get('content_urls') or {}).get('desktop', {}).get('page') or ''),
        )
