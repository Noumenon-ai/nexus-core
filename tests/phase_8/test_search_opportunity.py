from __future__ import annotations

import pytest

from scheduler import NexusScheduler
from utils.i18n import Translator
from utils.web_search import SearchResult


@pytest.mark.asyncio
async def test_search_service_summarizes_results(container, monkeypatch):
    async def fake_search(query: str, limit: int = 5):
        return [SearchResult(title='OpenAI ships update', url='https://example.com/a', snippet='A useful update landed today.')]
    monkeypatch.setattr(container.search_service.web_search_client, 'search', fake_search)
    response = await container.search_service.handle(container.users_repository.get_or_create(111), 'search for OpenAI update', Translator())
    assert 'strongest matches' in response.text.lower()
    assert 'https://example.com/a' in response.text


@pytest.mark.asyncio
async def test_search_service_requires_query(container):
    response = await container.search_service.handle(container.users_repository.get_or_create(111), 'search ', Translator())
    assert 'what should i search for' in response.text.lower()


@pytest.mark.asyncio
async def test_watch_interest_is_saved_and_scan_returns_suggestion(container, monkeypatch):
    user = container.users_repository.get_or_create(111)
    await container.opportunity_service.handle(user, 'watch for AI stock opportunities', Translator())
    async def fake_handle(user, text, translator):
        from pipeline.types import ServiceResponse
        return ServiceResponse(text='ok', metadata={'results': [SearchResult(title='AI stocks surge', url='https://example.com/surge', snippet='A new filing may be worth reviewing.')]})
    monkeypatch.setattr(container.opportunity_service.search_service, 'handle', fake_handle)
    results = await container.opportunity_service.scan_user(user, Translator())
    assert results
    assert 'suggestion' in results[0].lower()


def test_opportunity_dedupe_handles_similar_titles(container):
    recent = ['AI stocks surge']
    assert container.opportunity_service.is_duplicate('AI stock surge', recent) is True


def test_opportunity_scheduler_uses_configured_interval(container, monkeypatch):
    scheduler = NexusScheduler(container.settings.core.database_url, container.settings.core.app_timezone)
    scheduled: list[dict] = []

    def fake_add_job(func, trigger, **kwargs):
        scheduled.append({'func': func, 'trigger': trigger, **kwargs})

    monkeypatch.setattr(scheduler.scheduler, 'add_job', fake_add_job)
    scheduler.schedule_opportunity_scans([container.users_repository.get_or_create(111)], interval_hours=container.settings.search.radar_run_interval_hours)
    assert scheduled
    assert scheduled[0]['trigger'] == 'interval'
    assert scheduled[0]['hours'] == container.settings.search.radar_run_interval_hours
