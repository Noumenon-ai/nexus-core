from __future__ import annotations

import hashlib
import json
import re

try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def token_sort_ratio(left: str, right: str) -> float:
            left_sorted = ' '.join(sorted(left.lower().split()))
            right_sorted = ' '.join(sorted(right.lower().split()))
            return SequenceMatcher(None, left_sorted, right_sorted).ratio() * 100

    fuzz = _FallbackFuzz()

from config import Settings
from models import User
from pipeline.types import ServiceResponse
from repositories.memories_repository import MemoriesRepository
from repositories.opportunity_signals_repository import OpportunitySignalsRepository
from services.search_service import SearchService
from utils.i18n import Translator
from utils.web_search import SearchResult


class OpportunityService:
    def __init__(self, settings: Settings, memories_repository: MemoriesRepository, signals_repository: OpportunitySignalsRepository, search_service: SearchService) -> None:
        self.settings = settings
        self.memories_repository = memories_repository
        self.signals_repository = signals_repository
        self.search_service = search_service

    async def handle(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        normalized = text.strip().lower()
        if normalized.startswith('stop watching '):
            topic = self._suffix_after_prefix(text, 'stop watching')
            if not topic:
                return ServiceResponse(text=translator.t('watch_need_topic'))
            self.memories_repository.delete(user_id=user.id, key=self._interest_key(topic))
            return ServiceResponse(text=translator.t('watch_removed'))
        topic = self._suffix_after_prefix(text, 'watch for')
        if not topic:
            return ServiceResponse(text=translator.t('watch_need_topic'))
        topic = topic[:self.settings.search.web_search_max_query_chars].strip()
        category = self._category_for_topic(topic)
        key = self._interest_key(topic)
        self.memories_repository.upsert(user_id=user.id, memory_type='opportunity_interest', key=key, value=json.dumps({'topic': topic, 'category': category}, separators=(',', ':')), confidence=1.0, source='explicit')
        return ServiceResponse(text=translator.t('watch_saved'))

    async def scan_user(self, user: User, translator: Translator) -> list[str]:
        interests = self.memories_repository.list_by_user(user.id, memory_type='opportunity_interest')
        recent_titles = self.signals_repository.list_recent_titles(user_id=user.id, days=14)
        outputs: list[str] = []
        for interest in interests:
            payload = json.loads(interest.value)
            category = payload['category']
            if self.signals_repository.count_recent_by_category(user_id=user.id, category=category, days=7) >= self.settings.search.radar_max_signals_per_category_per_week:
                continue
            response = await self.search_service.handle(user, payload['topic'], translator)
            for result in response.metadata.get('results', [])[:5]:
                confidence = self._score_signal(payload['topic'], result)
                if confidence < self.settings.search.radar_min_confidence:
                    continue
                summary = self._suggestion_text(result.snippet, translator)
                if self.is_duplicate(result.title, recent_titles):
                    self.signals_repository.create(user_id=user.id, category=category, title=result.title, summary=summary, source=result.url, confidence=confidence, dedupe_hash=self.dedupe_key(result.title), status='suppressed')
                    continue
                suggestion = translator.t('opportunity_suggestion', title=result.title, summary=summary, url=result.url)
                self.signals_repository.create(user_id=user.id, category=category, title=result.title, summary=suggestion, source=result.url, confidence=confidence, dedupe_hash=self.dedupe_key(result.title), status='delivered')
                outputs.append(suggestion)
                recent_titles.append(result.title)
                break
        return outputs

    def _interest_key(self, topic: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', topic.lower()).strip('_')
        return f'opportunity_{slug[:64]}'

    def _category_for_topic(self, topic: str) -> str:
        lowered = topic.lower()
        if 'stock' in lowered:
            return 'stock'
        if 'business' in lowered:
            return 'business'
        if 'real estate' in lowered or 'property' in lowered:
            return 'real_estate'
        if 'finance' in lowered:
            return 'finance'
        if 'news' in lowered:
            return 'news'
        return 'general'

    def _score_signal(self, topic: str, result: SearchResult) -> float:
        combined = f"{result.title} {result.snippet}".lower()
        tokens = [token for token in re.split(r'\W+', topic.lower()) if token]
        if not tokens:
            return 0.0
        matches = sum(1 for token in tokens if token in combined)
        return min(1.0, 0.4 + (matches / len(tokens)) * 0.6)

    def _suggestion_text(self, snippet: str, translator: Translator) -> str:
        text = snippet.replace('you should', 'may be worth reviewing').replace('buy', 'review').replace('sell', 'review')
        return text or translator.t('opportunity_snippet_fallback')

    def _suffix_after_prefix(self, text: str, prefix: str) -> str:
        lowered = text.lower()
        index = lowered.find(prefix)
        if index == -1:
            return text.strip()
        return text[index + len(prefix):].strip()

    @staticmethod
    def dedupe_key(title: str) -> str:
        cleaned = re.sub(r'[^\w\s]', '', title.lower())
        tokens = sorted(cleaned.split())
        return hashlib.sha256(' '.join(tokens).encode()).hexdigest()

    @classmethod
    def is_duplicate(cls, new_title: str, recent_titles: list[str]) -> bool:
        if cls.dedupe_key(new_title) in {cls.dedupe_key(title) for title in recent_titles}:
            return True
        for title in recent_titles:
            if fuzz.token_sort_ratio(new_title, title) >= 85:
                return True
        return False
