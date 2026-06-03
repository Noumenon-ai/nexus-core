from __future__ import annotations

from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from utils.text_safety import sanitize_plaintext, sanitize_url


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchClient:
    def __init__(self, *, backend: str, serpapi_key: str = '') -> None:
        self.backend = backend
        self.serpapi_key = serpapi_key

    async def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if self.backend == 'serpapi':
            return await self._search_serpapi(query, limit=limit)
        return await self._search_ddg(query, limit=limit)

    async def _search_ddg(self, query: str, *, limit: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=10.0, headers={'User-Agent': 'NexusV2/1.0'}) as client:
            response = await client.post('https://html.duckduckgo.com/html/', data={'q': query})
            response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results: list[SearchResult] = []
        for item in soup.select('.result'):
            link = item.select_one('.result__a')
            snippet = item.select_one('.result__snippet')
            if link is None:
                continue
            href = self._normalize_url(link.get('href') or '')
            title = sanitize_plaintext(link.get_text(' ', strip=True), max_length=160)
            snippet_text = sanitize_plaintext(snippet.get_text(' ', strip=True) if snippet is not None else '', max_length=320)
            if title and href:
                results.append(SearchResult(title=title, url=href, snippet=snippet_text))
            if len(results) >= limit:
                break
        return results

    async def _search_serpapi(self, query: str, *, limit: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                'https://serpapi.com/search.json',
                params={'q': query, 'api_key': self.serpapi_key, 'num': limit},
            )
            response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get('organic_results', [])[:limit]:
            title = sanitize_plaintext(item.get('title', ''), max_length=160)
            url = sanitize_url(item.get('link', ''))
            snippet = sanitize_plaintext(item.get('snippet', ''), max_length=320)
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results

    def _normalize_url(self, raw_url: str) -> str:
        return sanitize_url(raw_url)
