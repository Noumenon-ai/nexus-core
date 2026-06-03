"""V3.upgrade — Weather lookup tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.read_tools import make_get_weather
from services.weather_service import CurrentWeather, WeatherService


@pytest.mark.asyncio
async def test_get_weather_returns_current():
    fake = MagicMock(spec=WeatherService)
    fake.current = AsyncMock(return_value=CurrentWeather(
        location='Tel Aviv, Israel',
        temperature_c=22.5,
        conditions='partly cloudy',
        wind_kph=12.0,
    ))
    tool = make_get_weather(fake)
    result = await tool(location='Tel Aviv')
    assert result.success is True
    assert result.data['found'] is True
    assert result.data['temperature_c'] == 22.5
    fake.current.assert_awaited_once_with('Tel Aviv')


@pytest.mark.asyncio
async def test_get_weather_unknown_location():
    fake = MagicMock(spec=WeatherService)
    fake.current = AsyncMock(return_value=None)
    tool = make_get_weather(fake)
    result = await tool(location='Atlantis')
    assert result.success is True
    assert result.data['found'] is False


@pytest.mark.asyncio
async def test_get_weather_rejects_empty_location():
    fake = MagicMock(spec=WeatherService)
    fake.current = AsyncMock()
    tool = make_get_weather(fake)
    result = await tool(location='   ')
    assert result.success is False
    fake.current.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_weather_handles_error():
    fake = MagicMock(spec=WeatherService)
    fake.current = AsyncMock(side_effect=Exception('boom'))
    tool = make_get_weather(fake)
    result = await tool(location='X')
    assert result.success is False
