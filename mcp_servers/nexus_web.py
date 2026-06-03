"""nexus-web MCP server — Phase 5 / spec Server 8.

Wraps services.browser_agent.BrowserAgent.  Stdio transport.  Spawns a Playwright
Firefox session bound to the calling user's profile for each tool invocation.

Configuration (env vars):
  NEXUS_MCP_DEFAULT_USER_ID    user UUID for tools called without explicit user_id
  NEXUS_USER_1_UUID            user_1 UUID (defaults to the current owner UUID)
  NEXUS_USER_2_UUID            user_2 UUID
  NEXUS_USER_1_FIREFOX_PROFILE absolute path
  NEXUS_USER_2_FIREFOX_PROFILE absolute path under NEXUS_USER_2_PROFILE_ROOT

Tool semantics match NEXUS_MCP_FULL_SPEC.md Server 8.  Approval-gated tools
(book_appointment, fill_web_form) still execute on call — Claude CLI is expected
to gate them at the user-permission layer.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import hashlib
import os
import re
from typing import Any

from fli.models import (
    Airline,
    Airport,
    DateSearchFilters,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
    TripType,
)
from fli.search import DatePrice, SearchDates, SearchFlights
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("nexus-web")
_AIRLINE_LABEL_PATTERN = re.compile(r"[^a-z0-9]+")
MAX_FASTEST_PROBE_DAYS = 3


def _resolve_user_id(user_id: str | None) -> str | None:
    if user_id and user_id.strip():
        return user_id.strip()
    env_uid = os.environ.get("NEXUS_MCP_DEFAULT_USER_ID")
    return env_uid.strip() if env_uid else None


async def _with_agent(user_id: str | None, coro_fn):
    """Boot a BrowserAgent, run coro_fn(agent), return the dict result."""
    uid = _resolve_user_id(user_id)
    if not uid:
        return {"ok": False, "reason": "no_user_id",
                "detail": "pass user_id or set NEXUS_MCP_DEFAULT_USER_ID"}
    from services.browser_agent import BrowserAgent, BrowserProfileError
    try:
        async with BrowserAgent(uid) as agent:
            result = await coro_fn(agent)
    except BrowserProfileError as exc:
        return {"ok": False, "reason": "profile_error", "detail": str(exc)}
    except Exception as exc:
        return {"ok": False, "reason": "agent_failed", "detail": str(exc)[:300]}
    if hasattr(result, "ok"):
        # BrowserActionResult
        out = {"ok": bool(result.ok)}
        if result.reason:
            out["reason"] = result.reason
        if result.detail:
            out["detail"] = result.detail
        if result.data:
            out.update(result.data)
        return out
    return result


def _normalize_airline_label(value: str) -> str:
    return _AIRLINE_LABEL_PATTERN.sub(" ", value.casefold()).strip()


def _format_duration(duration_minutes: int) -> str:
    hours, minutes = divmod(duration_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _resolve_airport_code(code: str, *, field_name: str) -> Airport:
    normalized = (code or "").strip().upper()
    if not normalized:
        raise ValueError(f"{field_name} is required and must be a 3-letter IATA code.")
    try:
        return getattr(Airport, normalized)
    except AttributeError as exc:
        raise ValueError(f"Unknown {field_name} IATA code '{normalized}'.") from exc


def _match_airline(value: str) -> Airline | None:
    candidate = value.strip()
    if not candidate:
        return None
    normalized_candidate = _normalize_airline_label(candidate)
    if normalized_candidate in {
        "us",
        "usa",
        "u s",
        "us airlines",
        "u s airlines",
        "american",
        "american airlines",
    }:
        return None
    enum_key = candidate.upper()
    if enum_key[0].isdigit():
        enum_key = f"_{enum_key}"
    airline = getattr(Airline, enum_key, None)
    if airline is not None:
        return airline
    for item in Airline:
        if _normalize_airline_label(item.value) == normalized_candidate:
            return item
    return None


def _resolve_airlines(airlines: str | None) -> list[Airline] | None:
    if airlines is None or not airlines.strip():
        return None

    us_group_tokens = {
        "us",
        "usa",
        "u s",
        "us airlines",
        "u s airlines",
        "american",
        "american airlines",
    }
    us_carrier_codes = ("AA", "DL", "UA", "B6", "AS", "WN", "NK", "F9", "HA")
    airline_by_code = {item.name.lstrip("_"): item for item in Airline}
    resolved: list[Airline] = []
    raw_chunks = [chunk.strip() for chunk in airlines.split(",") if chunk.strip()]
    chunks = raw_chunks or [airlines.strip()]

    for chunk in chunks:
        normalized_chunk = _normalize_airline_label(chunk)
        if normalized_chunk in us_group_tokens:
            resolved.extend(
                airline_by_code[code]
                for code in us_carrier_codes
                if code in airline_by_code
            )
            continue

        airline = _match_airline(chunk)
        if airline is not None:
            resolved.append(airline)
            continue

        tokens = [token.strip() for token in chunk.split() if token.strip()]
        if len(tokens) > 1 and all(len(token) <= 3 for token in tokens):
            token_matches = [_match_airline(token) for token in tokens]
            if all(match is not None for match in token_matches):
                resolved.extend(match for match in token_matches if match is not None)
                continue

        raise ValueError(
            f"Unknown airline filter '{chunk}'. Use airline names like 'El Al', "
            "airline groups like 'US', or IATA codes like 'LY'."
        )

    unique: list[Airline] = []
    seen: set[str] = set()
    for airline in resolved:
        if airline.name in seen:
            continue
        seen.add(airline.name)
        unique.append(airline)
    return unique


def _build_segment(origin: Airport, destination: Airport, travel_date: str) -> FlightSegment:
    return FlightSegment(
        departure_airport=[[origin, 0]],
        arrival_airport=[[destination, 0]],
        travel_date=travel_date,
    )


def _serialize_cheapest_day(result: DatePrice) -> dict[str, Any]:
    departure_date = result.date[0].strftime("%Y-%m-%d")
    return {
        "kind": "cheapest_day",
        "date": departure_date,
        "price": result.price,
        "currency": result.currency,
    }


def _serialize_fastest_option(travel_date: str, flight: Any) -> dict[str, Any]:
    airlines = []
    for leg in flight.legs:
        airline_code = leg.airline.name.lstrip("_")
        if airline_code not in airlines:
            airlines.append(airline_code)
    return {
        "kind": "fastest_option",
        "date": travel_date,
        "price": flight.price,
        "currency": flight.currency,
        "duration_minutes": flight.duration,
        "duration_text": _format_duration(flight.duration),
        "stops": flight.stops,
        "airlines": airlines,
        "segments": [
            {
                "airline": leg.airline.value,
                "airline_code": leg.airline.name.lstrip("_"),
                "flight_number": leg.flight_number,
                "departure_airport": leg.departure_airport.name,
                "arrival_airport": leg.arrival_airport.name,
                "departure_time": leg.departure_datetime.isoformat(),
                "arrival_time": leg.arrival_datetime.isoformat(),
                "duration_minutes": leg.duration,
            }
            for leg in flight.legs
        ],
    }


def _search_flights_sync(
    *,
    origin: str,
    destination: str,
    date_from: str,
    date_to: str,
    passengers: int,
    airlines: str | None,
) -> dict[str, Any]:
    """Search date prices once, then probe at most MAX_FASTEST_PROBE_DAYS cheapest dates for fastest itineraries."""
    if passengers < 1:
        raise ValueError("passengers must be at least 1.")

    start_date = datetime.strptime(date_from, "%Y-%m-%d").date()
    end_date = datetime.strptime(date_to, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError("date_from must be on or before date_to.")

    origin_airport = _resolve_airport_code(origin, field_name="origin")
    destination_airport = _resolve_airport_code(destination, field_name="destination")
    if origin_airport == destination_airport:
        raise ValueError("origin and destination must be different IATA codes.")

    airline_filters = _resolve_airlines(airlines)
    passenger_info = PassengerInfo(adults=passengers)
    search_dates = SearchDates()
    search_flights = SearchFlights()

    date_filters = DateSearchFilters(
        trip_type=TripType.ONE_WAY,
        passenger_info=passenger_info,
        flight_segments=[_build_segment(origin_airport, destination_airport, date_from)],
        stops=MaxStops.ANY,
        seat_type=SeatType.ECONOMY,
        airlines=airline_filters,
        from_date=date_from,
        to_date=date_to,
    )
    date_results = search_dates.search(date_filters) or []
    if not date_results:
        return {
            "ok": True,
            "results": [],
            "fastest_probed_days": 0,
            "detail": (
                f"No flights found from {origin_airport.name} to {destination_airport.name} "
                f"between {date_from} and {date_to}."
            ),
        }

    cheapest_day = min(
        date_results,
        key=lambda item: (item.price, item.date[0]),
    )

    fastest_payload: dict[str, Any] | None = None
    fastest_key: tuple[int, float, str] | None = None
    fastest_probe_results = sorted(
        date_results,
        key=lambda item: (item.price, item.date[0]),
    )[:MAX_FASTEST_PROBE_DAYS]
    for date_result in fastest_probe_results:
        travel_date = date_result.date[0].strftime("%Y-%m-%d")
        flight_filters = FlightSearchFilters(
            trip_type=TripType.ONE_WAY,
            passenger_info=passenger_info,
            flight_segments=[_build_segment(origin_airport, destination_airport, travel_date)],
            stops=MaxStops.ANY,
            seat_type=SeatType.ECONOMY,
            airlines=airline_filters,
            sort_by=SortBy.DURATION,
        )
        flight_results = search_flights.search(flight_filters, top_n=5) or []
        for flight in flight_results:
            if isinstance(flight, tuple):
                continue
            candidate_key = (flight.duration, flight.price, travel_date)
            if fastest_key is None or candidate_key < fastest_key:
                fastest_key = candidate_key
                fastest_payload = _serialize_fastest_option(travel_date, flight)

    results = [_serialize_cheapest_day(cheapest_day)]
    if fastest_payload is not None:
        results.append(fastest_payload)

    response: dict[str, Any] = {
        "ok": True,
        "results": results,
        "cheapest_day": _serialize_cheapest_day(cheapest_day),
        "fastest_probed_days": len(fastest_probe_results),
    }
    if fastest_payload is not None:
        response["fastest_option"] = fastest_payload
    else:
        response["detail"] = "Found date prices, but no flight itineraries were returned."
    return response


# ---------- read tools ----------

@mcp.tool()
async def fetch_url(url: str, wait_for_network_idle: bool = True, timeout_sec: int = 30,
              user_id: str | None = None) -> dict:
    """Fetch the page at URL. Returns title, visible text, html length + 5KB preview.
    Async — runs inside FastMCP's event loop directly (no nested asyncio.run)."""
    async def _go(agent):
        return await agent.fetch_url(url, wait_for_network_idle=wait_for_network_idle,
                                     timeout_sec=timeout_sec)
    return await _with_agent(user_id, _go)


@mcp.tool()
async def search_web(query: str, num_results: int = 10, user_id: str | None = None) -> dict:
    """Web search via DuckDuckGo HTML endpoint. Returns title + url + snippet per hit."""
    async def _go(agent):
        return await agent.search_web(query=query, num_results=num_results)
    return await _with_agent(user_id, _go)


@mcp.tool()
async def check_website_status(url: str, timeout_sec: int = 30, user_id: str | None = None) -> dict:
    """HEAD-equivalent: returns HTTP status_code + ok bool. Uses full browser (cookies, JS)."""
    async def _go(agent):
        return await agent.check_status(url=url, timeout_sec=timeout_sec)
    return await _with_agent(user_id, _go)


@mcp.tool()
async def screenshot(url: str, output_path: str, full_page: bool = True,
              timeout_sec: int = 30, user_id: str | None = None) -> dict:
    """Navigate to URL and write a PNG screenshot to output_path (must be inside ALLOWED_ROOTS)."""
    async def _go(agent):
        return await agent.screenshot(url=url, output_path=output_path,
                                      full_page=full_page, timeout_sec=timeout_sec)
    return await _with_agent(user_id, _go)


@mcp.tool()
async def scrape_structured_data(url: str, schema: dict, timeout_sec: int = 30,
                          user_id: str | None = None) -> dict:
    """Extract fields per a CSS-selector schema.
    schema example:
      {"title": "h1", "links": {"selector": "a", "attr": "href", "all": true}}
    """
    async def _go(agent):
        return await agent.scrape(url=url, schema=schema, timeout_sec=timeout_sec)
    return await _with_agent(user_id, _go)


@mcp.tool()
async def monitor_url_for_changes(url: str, frequency: str | None = None,
                            user_id: str | None = None) -> dict:
    """Take a fingerprint of the URL's current visible-text + return it.

    True recurring monitoring needs an out-of-band scheduler; this tool gives
    the caller a SHA-256 of the page's visible text at this moment so a future
    call can compare. `frequency` is accepted for API compatibility but not
    acted on here.
    """
    async def _go(agent):
        return await agent.fetch_url(url, wait_for_network_idle=True, timeout_sec=30)
    raw = await _with_agent(user_id, _go)
    if not raw.get("ok"):
        return raw
    text = raw.get("text") or ""
    return {
        "ok": True,
        "url": raw.get("url"), "final_url": raw.get("final_url"),
        "fingerprint_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fingerprint_chars": len(text),
        "title": raw.get("title"),
        "frequency_hint": frequency,
        "note": "Persist this fingerprint and re-call to detect drift.",
    }


@mcp.tool()
async def search_flights(
    origin: str,
    destination: str,
    date_from: str,
    date_to: str,
    passengers: int = 1,
    airlines: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Search Google Flights across a date range and return cheapest and fastest options."""
    _resolve_user_id(user_id)
    try:
        return await asyncio.to_thread(
            _search_flights_sync,
            origin=origin,
            destination=destination,
            date_from=date_from,
            date_to=date_to,
            passengers=passengers,
            airlines=airlines,
        )
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_input", "detail": str(exc)}
    except Exception as exc:
        return {"ok": False, "reason": "search_failed", "detail": str(exc)[:300]}


# ---------- approval-gated tools ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Fill out and submit a web form?"})
async def fill_web_form(url: str, field_values: dict, submit_selector: str | None = None,
                 timeout_sec: int = 30, user_id: str | None = None) -> dict:
    """APPROVAL-GATED: navigate to URL, fill named fields with human-like typing.
    Submits ONLY when submit_selector is provided — leave it None to fill-and-pause."""
    async def _go(agent):
        return await agent.fill_form(url=url, field_values=field_values,
                                     submit_selector=submit_selector,
                                     timeout_sec=timeout_sec)
    return await _with_agent(user_id, _go)


@mcp.tool(meta={"approval_required": True, "approval_template": "Book an appointment?"})
async def book_appointment(service_url: str, preferences: dict,
                    user_id: str | None = None) -> dict:
    """APPROVAL-GATED: Calendly-style booking. Site-specific adapters not yet wired.

    Returns `not_configured` for now — actual booking needs per-site selectors that
    the user provides per service. The recommended path is to use fill_web_form
    with explicit selectors for the chosen service.
    """
    return {
        "ok": False,
        "reason": "not_configured",
        "detail": "book_appointment requires per-site adapters. Use fill_web_form with explicit selectors.",
        "echo": {"service_url": service_url, "preferences": preferences},
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
