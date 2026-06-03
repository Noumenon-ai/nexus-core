"""V3.upgrade — Weather lookup via Open-Meteo (no API key).

Two-step flow:
  1. Geocode location string to lat/lon (Open-Meteo geocoding API)
  2. Fetch current + 24h forecast (Open-Meteo forecast API)
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


_GEOCODE = 'https://geocoding-api.open-meteo.com/v1/search'
_FORECAST = 'https://api.open-meteo.com/v1/forecast'

# WMO weather code → human-readable label (subset)
_WMO_CODES = {
    0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
    45: 'fog', 48: 'depositing rime fog',
    51: 'light drizzle', 53: 'moderate drizzle', 55: 'dense drizzle',
    61: 'slight rain', 63: 'moderate rain', 65: 'heavy rain',
    71: 'slight snow', 73: 'moderate snow', 75: 'heavy snow',
    77: 'snow grains',
    80: 'rain showers', 81: 'moderate rain showers', 82: 'violent rain showers',
    85: 'slight snow showers', 86: 'heavy snow showers',
    95: 'thunderstorm', 96: 'thunderstorm w/ slight hail', 99: 'thunderstorm w/ heavy hail',
}


@dataclass(slots=True)
class CurrentWeather:
    location: str
    temperature_c: float
    conditions: str
    wind_kph: float

    def to_dict(self) -> dict:
        return {
            'location': self.location,
            'temperature_c': self.temperature_c,
            'conditions': self.conditions,
            'wind_kph': self.wind_kph,
        }


class WeatherService:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None, timeout: float = 10.0) -> None:
        self._http_client = http_client
        self._timeout = timeout

    async def current(self, location: str) -> CurrentWeather | None:
        if not isinstance(location, str) or not location.strip():
            raise ValueError('location must be non-empty')
        place = await self._geocode(location.strip())
        if place is None:
            return None
        lat, lon, label = place
        client = self._http_client
        if client is None:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                resp = await c.get(_FORECAST, params={'latitude': lat, 'longitude': lon, 'current_weather': 'true'})
        else:
            resp = await client.get(_FORECAST, params={'latitude': lat, 'longitude': lon, 'current_weather': 'true'})
        resp.raise_for_status()
        data = resp.json()
        cw = data.get('current_weather') or {}
        return CurrentWeather(
            location=label,
            temperature_c=float(cw.get('temperature') or 0.0),
            conditions=_WMO_CODES.get(int(cw.get('weathercode') or -1), 'unknown'),
            wind_kph=float(cw.get('windspeed') or 0.0),
        )

    async def _geocode(self, location: str) -> tuple[float, float, str] | None:
        client = self._http_client
        if client is None:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                resp = await c.get(_GEOCODE, params={'name': location, 'count': 1})
        else:
            resp = await client.get(_GEOCODE, params={'name': location, 'count': 1})
        resp.raise_for_status()
        results = (resp.json() or {}).get('results') or []
        if not results:
            return None
        first = results[0]
        label_parts = [first.get('name') or location]
        if first.get('admin1'):
            label_parts.append(str(first.get('admin1')))
        if first.get('country'):
            label_parts.append(str(first.get('country')))
        label = ', '.join(label_parts)
        return float(first['latitude']), float(first['longitude']), label
