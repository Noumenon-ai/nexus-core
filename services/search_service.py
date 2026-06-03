from __future__ import annotations

from models import User
from pipeline.types import ServiceResponse
from utils.i18n import Translator
from utils.web_search import WebSearchClient


class SearchService:
    def __init__(self, web_search_client: WebSearchClient, *, enabled: bool = True, max_query_chars: int = 256) -> None:
        self.web_search_client = web_search_client
        self.enabled = enabled
        self.max_query_chars = max_query_chars

    async def handle(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        if not self.enabled:
            return ServiceResponse(text=translator.t('search_unavailable'))
        query = self._extract_query(text)
        if not query:
            return ServiceResponse(text=translator.t('search_need_query'))
        query = query[:self.max_query_chars].strip()
        try:
            results = await self.web_search_client.search(query, limit=5)
        except Exception:
            return ServiceResponse(text=translator.t('search_unavailable'))
        if not results:
            return ServiceResponse(text=translator.t('search_unavailable'))
        lines = [f"- {item.title}: {item.snippet} ({item.url})" for item in results[:3]]
        summary = translator.t('search_results_intro', query=query) + '\n' + '\n'.join(lines)
        return ServiceResponse(text=summary, voice_appropriate=False, metadata={'results': results})

    def _extract_query(self, text: str) -> str:
        normalized = text.strip()
        lowered = normalized.lower()
        if lowered in {'search', 'search for', 'look up'}:
            return ''
        for prefix in ('search for ', 'search ', 'look up '):
            if lowered.startswith(prefix):
                return normalized[len(prefix):].strip()
        return normalized
