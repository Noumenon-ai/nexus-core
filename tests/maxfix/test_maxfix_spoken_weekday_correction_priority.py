from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.tool_dispatcher import _build_timeout_fallback_reminder_args


VOICE_PROMPT = (
    "Tell Sarah I will send someone tomorrow. "
    "Actually no, ask if Thursday works. "
    "Wait no. First check if Mike replied. "
    "Remind me to follow up Friday morning. "
    "No, not Sarah, unit 200 and 4. "
    "Actually this is urgent because of water damage."
)


def test_followup_datetime_prefers_latest_spoken_weekday_correction():
    fixed_now = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)

    reminder_args = _build_timeout_fallback_reminder_args(
        text=VOICE_PROMPT,
        now=fixed_now,
        app_timezone="America/New_York",
    )

    assert reminder_args is not None
    assert reminder_args["time_label"] == "Friday morning"
    assert reminder_args["next_fire_at"] == "2026-05-22T13:00:00+00:00"
    assert "Mike" in reminder_args["body"]
    assert "Unit 204" in reminder_args["body"]
