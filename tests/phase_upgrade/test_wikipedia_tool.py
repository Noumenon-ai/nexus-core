"""V3.upgrade — Wikipedia lookup tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.read_tools import make_lookup_wikipedia
from services.wikipedia_service import WikipediaService, WikipediaSummary


@pytest.mark.asyncio
async def test_lookup_wikipedia_returns_summary():
    fake = MagicMock(spec=WikipediaService)
    fake.lookup = AsyncMock(return_value=WikipediaSummary(
        title='Riemann hypothesis',
        extract='In mathematics, the Riemann hypothesis...',
        url='https://en.wikipedia.org/wiki/Riemann_hypothesis',
    ))
    tool = make_lookup_wikipedia(fake)
    result = await tool(topic='Riemann hypothesis')
    assert result.success is True
    assert result.data['found'] is True
    assert result.data['title'] == 'Riemann hypothesis'
    assert 'Riemann' in result.data['extract']
    fake.lookup.assert_awaited_once_with('Riemann hypothesis')


@pytest.mark.asyncio
async def test_lookup_wikipedia_handles_not_found():
    fake = MagicMock(spec=WikipediaService)
    fake.lookup = AsyncMock(return_value=None)
    tool = make_lookup_wikipedia(fake)
    result = await tool(topic='asdfqwerty12345')
    assert result.success is True
    assert result.data['found'] is False


@pytest.mark.asyncio
async def test_lookup_wikipedia_rejects_empty_topic():
    fake = MagicMock(spec=WikipediaService)
    fake.lookup = AsyncMock()
    tool = make_lookup_wikipedia(fake)
    result = await tool(topic='   ')
    assert result.success is False
    fake.lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_wikipedia_handles_http_error():
    fake = MagicMock(spec=WikipediaService)
    fake.lookup = AsyncMock(side_effect=Exception('boom'))
    tool = make_lookup_wikipedia(fake)
    result = await tool(topic='X')
    assert result.success is False
    assert 'failed' in result.error.lower()
