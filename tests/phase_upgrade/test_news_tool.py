"""V3.upgrade — News headlines tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.read_tools import make_get_news_headlines
from services.news_service import NewsHeadline, NewsService


@pytest.mark.asyncio
async def test_get_news_headlines_returns_list():
    fake = MagicMock(spec=NewsService)
    fake.top_headlines = AsyncMock(return_value=[
        NewsHeadline(title='Story A', source='Reuters', url='https://r/a', published='Wed'),
        NewsHeadline(title='Story B', source='AP', url='https://ap/b', published='Wed'),
    ])
    tool = make_get_news_headlines(fake)
    result = await tool()
    assert result.success is True
    assert result.data['count'] == 2
    assert result.data['headlines'][0]['title'] == 'Story A'
    fake.top_headlines.assert_awaited_once_with(max_results=5)


@pytest.mark.asyncio
async def test_get_news_headlines_clamps_max_results():
    fake = MagicMock(spec=NewsService)
    fake.top_headlines = AsyncMock(return_value=[])
    tool = make_get_news_headlines(fake)
    await tool(max_results=99)
    fake.top_headlines.assert_awaited_once_with(max_results=20)


@pytest.mark.asyncio
async def test_get_news_headlines_falls_back_on_invalid_max():
    fake = MagicMock(spec=NewsService)
    fake.top_headlines = AsyncMock(return_value=[])
    tool = make_get_news_headlines(fake)
    await tool(max_results=0)
    fake.top_headlines.assert_awaited_once_with(max_results=5)


@pytest.mark.asyncio
async def test_get_news_headlines_handles_error():
    fake = MagicMock(spec=NewsService)
    fake.top_headlines = AsyncMock(side_effect=Exception('boom'))
    tool = make_get_news_headlines(fake)
    result = await tool()
    assert result.success is False
