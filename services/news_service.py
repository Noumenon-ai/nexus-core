"""V3.upgrade — News headlines via Google News RSS (no API key).

Parses the RSS feed using stdlib xml.etree.ElementTree (no feedparser
dependency). Returns top N headlines with source + URL.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx


_USER_AGENT = 'NexusBot/1.0 (personal-use; httpx)'
_RSS_BASE = 'https://news.google.com/rss'


@dataclass(slots=True)
class NewsHeadline:
    title: str
    source: str
    url: str
    published: str

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'source': self.source,
            'url': self.url,
            'published': self.published,
        }


class NewsService:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None, timeout: float = 10.0) -> None:
        self._http_client = http_client
        self._timeout = timeout

    async def top_headlines(self, *, max_results: int = 5, locale: str = 'en-US:en') -> list[NewsHeadline]:
        hl, ceid = self._split_locale(locale)
        params = {'hl': hl, 'gl': hl.split('-')[1] if '-' in hl else 'US', 'ceid': ceid}
        url = _RSS_BASE
        client = self._http_client
        if client is None:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                resp = await c.get(url, params=params, headers={'User-Agent': _USER_AGENT})
        else:
            resp = await client.get(url, params=params, headers={'User-Agent': _USER_AGENT})
        resp.raise_for_status()
        return self._parse_rss(resp.text, max_results=max_results)

    @staticmethod
    def _split_locale(locale: str) -> tuple[str, str]:
        if ':' in locale:
            hl, lang = locale.split(':', 1)
            return hl, f'{hl.split("-")[1] if "-" in hl else "US"}:{lang}'
        return 'en-US', 'US:en'

    @staticmethod
    def _parse_rss(text: str, *, max_results: int) -> list[NewsHeadline]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []
        channel = root.find('channel')
        if channel is None:
            return []
        out: list[NewsHeadline] = []
        for item in channel.findall('item')[:max_results]:
            title = (item.findtext('title') or '').strip()
            link = (item.findtext('link') or '').strip()
            pub = (item.findtext('pubDate') or '').strip()
            source_elem = item.find('source')
            source = source_elem.text.strip() if source_elem is not None and source_elem.text else ''
            if title:
                out.append(NewsHeadline(title=title, source=source, url=link, published=pub))
        return out
