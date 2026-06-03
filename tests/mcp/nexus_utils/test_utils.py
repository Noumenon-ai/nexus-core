"""Adversarial tests for nexus-utils MCP server (Phase 2.5a)."""
from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_servers.nexus_utils import (
    base64_decode,
    base64_encode,
    convert_units,
    currency_convert,
    current_time,
    do_math,
    flip_coin,
    generate_password,
    hash_text,
    hebcal,
    omer_count,
    qr_code,
    random_pick,
    regex_test,
    roll_dice,
    shabbos_times,
    translate,
    url_decode,
    url_encode,
    uuid,
    weather,
)


# ---------- current_time ----------

def test_current_time_default_utc_returns_iso():
    r = current_time()
    assert r["ok"] is True
    assert r["timezone"] == "UTC"
    assert "T" in r["iso"]


def test_current_time_unknown_tz_rejected():
    r = current_time(timezone="Not/A/Real/Zone")
    assert r["ok"] is False
    assert r["reason"] == "unknown_timezone"


def test_current_time_known_iana_tz_works():
    r = current_time(timezone="America/New_York")
    assert r["ok"] is True
    assert r["timezone"] == "America/New_York"


# ---------- convert_units ----------

def test_convert_units_distance_km_to_mi():
    r = convert_units(value=1.0, from_unit="km", to_unit="mi")
    assert r["ok"] is True
    assert abs(r["value"] - 0.621371) < 1e-4


def test_convert_units_temperature_f_to_c():
    r = convert_units(value=32, from_unit="f", to_unit="c")
    assert r["ok"] is True
    assert abs(r["value"] - 0.0) < 1e-6


def test_convert_units_unknown_unit_rejected():
    r = convert_units(value=1.0, from_unit="parsec", to_unit="km")
    assert r["ok"] is False
    assert r["reason"] == "unknown_from_unit"


def test_convert_units_dimension_mismatch_rejected():
    r = convert_units(value=1.0, from_unit="kg", to_unit="km")
    assert r["ok"] is False
    assert r["reason"] == "dimension_mismatch"


def test_convert_units_non_numeric_value_rejected():
    r = convert_units(value="abc", from_unit="km", to_unit="mi")
    assert r["ok"] is False
    assert r["reason"] == "non_numeric_value"


# ---------- currency_convert ----------

def test_currency_invalid_code_rejected_pre_network():
    r = currency_convert(amount=100, from_currency="DOLLAR", to_currency="EUR")
    assert r["ok"] is False
    assert r["reason"] == "invalid_currency_code"


def test_currency_non_numeric_amount_rejected():
    r = currency_convert(amount="lots", from_currency="USD", to_currency="EUR")
    assert r["ok"] is False
    assert r["reason"] == "non_numeric_amount"


def test_currency_unknown_target_rejected_via_mock():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"rates": {"EUR": 0.9}, "time_last_update_utc": "x"}
    mock_resp.raise_for_status = lambda: None
    with patch("mcp_servers.nexus_utils.httpx.get", return_value=mock_resp):
        r = currency_convert(amount=10, from_currency="USD", to_currency="XYZ")
    assert r["ok"] is False
    assert r["reason"] == "unknown_target_currency"


def test_currency_happy_path_via_mock():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"rates": {"EUR": 0.9}, "time_last_update_utc": "x"}
    mock_resp.raise_for_status = lambda: None
    with patch("mcp_servers.nexus_utils.httpx.get", return_value=mock_resp):
        r = currency_convert(amount=100, from_currency="USD", to_currency="EUR")
    assert r["ok"] is True
    assert r["converted"] == 90.0


# ---------- do_math: SAFE EVAL BOUNDARY ----------

def test_do_math_basic_arithmetic():
    r = do_math("2 + 3 * 4")
    assert r["ok"] is True
    assert r["result"] == 14.0


def test_do_math_parentheses_and_power():
    r = do_math("(2 + 3) ** 2")
    assert r["ok"] is True
    assert r["result"] == 25.0


def test_do_math_division_by_zero():
    r = do_math("1 / 0")
    assert r["ok"] is False
    assert r["reason"] == "division_by_zero"


@pytest.mark.parametrize("attack", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
    "exec('import os')",
    "getattr(1, 'real')",
    "x + 1",            # bare name reference
    "abs(-1)",          # function call
    "lambda: 1",        # lambda
    "[1,2,3]",          # list literal
    "{1:2}",            # dict literal
    "'string'",         # string constant
    "True and False",
])
def test_do_math_rejects_unsafe_input(attack):
    r = do_math(attack)
    assert r["ok"] is False, f"unsafe input was accepted: {attack!r}"


def test_do_math_empty_rejected():
    assert do_math("")["ok"] is False
    assert do_math("   ")["ok"] is False


def test_do_math_too_long_rejected():
    r = do_math("1 + " * 200 + "1")
    assert r["ok"] is False
    assert r["reason"] == "expression_too_long"


def test_do_math_garbage_syntax_rejected():
    r = do_math("2 ++++ ++")
    assert r["ok"] is False
    assert r["reason"] in {"parse_error", "eval_error"}


# ---------- weather ----------

def test_weather_no_location_rejected():
    r = weather()
    assert r["ok"] is False
    assert r["reason"] == "no_location"


def test_weather_unknown_location_via_mock():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = lambda: None
    with patch("mcp_servers.nexus_utils.httpx.get", return_value=mock_resp):
        r = weather(location="Atlantis")
    assert r["ok"] is False
    assert r["reason"] == "location_not_found"


# ---------- translate (stub) ----------

def test_translate_returns_not_configured():
    r = translate(text="hi", target_language="he")
    assert r["ok"] is False
    assert r["reason"] == "not_configured"


# ---------- generate_password ----------

def test_generate_password_default_strong():
    r = generate_password()
    assert r["ok"] is True
    assert len(r["passwords"]) == 1
    assert len(r["passwords"][0]) == 20


def test_generate_password_length_too_short_rejected():
    r = generate_password(length=2)
    assert r["ok"] is False
    assert r["reason"] == "length_out_of_range"


def test_generate_password_length_too_long_rejected():
    r = generate_password(length=10_000)
    assert r["ok"] is False
    assert r["reason"] == "length_out_of_range"


def test_generate_password_count_out_of_range_rejected():
    r = generate_password(count=0)
    assert r["ok"] is False
    r2 = generate_password(count=999)
    assert r2["ok"] is False


def test_generate_password_alphanum_complexity_no_symbols():
    r = generate_password(length=50, complexity="alphanum", count=1)
    assert r["ok"] is True
    pw = r["passwords"][0]
    assert all(c.isalnum() for c in pw)


def test_generate_password_passwords_are_distinct():
    r = generate_password(length=24, count=5)
    assert r["ok"] is True
    assert len(set(r["passwords"])) == 5


# ---------- random_pick / dice / coin ----------

def test_random_pick_empty_options_rejected():
    r = random_pick([])
    assert r["ok"] is False
    assert r["reason"] == "empty_or_invalid_options"


def test_random_pick_not_a_list_rejected():
    r = random_pick("abc")
    assert r["ok"] is False


def test_random_pick_picks_from_list():
    r = random_pick(options=["a", "b", "c"])
    assert r["ok"] is True
    assert r["pick"] in {"a", "b", "c"}


def test_roll_dice_default_d6():
    r = roll_dice()
    assert r["ok"] is True
    assert 1 <= r["rolls"][0] <= 6


def test_roll_dice_sides_too_few_rejected():
    r = roll_dice(sides=1)
    assert r["ok"] is False


def test_roll_dice_count_too_many_rejected():
    r = roll_dice(count=10_000)
    assert r["ok"] is False


def test_flip_coin_count_out_of_range_rejected():
    assert flip_coin(count=0)["ok"] is False
    assert flip_coin(count=10_000)["ok"] is False


def test_flip_coin_returns_balanced_set():
    r = flip_coin(count=10)
    assert r["ok"] is True
    assert r["heads"] + r["tails"] == 10


# ---------- uuid / hash / encode ----------

def test_uuid_returns_valid_uuid_string():
    import re as _re
    r = uuid()
    assert r["ok"] is True
    assert _re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", r["uuid"])


def test_hash_text_sha256_known_vector():
    r = hash_text("abc", algorithm="sha256")
    assert r["ok"] is True
    assert r["hex"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hash_text_unknown_algo_rejected():
    r = hash_text("abc", algorithm="sha9999")
    assert r["ok"] is False
    assert r["reason"] == "unknown_algorithm"


def test_hash_text_non_string_rejected():
    r = hash_text(text=12345)
    assert r["ok"] is False


def test_base64_roundtrip():
    enc = base64_encode("hello world")
    assert enc["ok"] is True
    dec = base64_decode(enc["base64"])
    assert dec["ok"] is True
    assert dec["text"] == "hello world"


def test_base64_decode_invalid_rejected():
    r = base64_decode("not!!!base64===")
    assert r["ok"] is False
    assert r["reason"] == "invalid_base64"


def test_base64_decode_not_utf8_rejected():
    enc = base64_encode("x")
    # Inject bytes that aren't valid UTF-8 when decoded back
    import base64 as b
    raw_b64 = b.b64encode(b"\xff\xfe\xfd").decode("ascii")
    r = base64_decode(raw_b64)
    assert r["ok"] is False
    assert r["reason"] == "not_utf8"


def test_url_encode_decode_roundtrip():
    enc = url_encode("hello world/?&=")
    assert enc["ok"] is True
    assert "%20" in enc["encoded"]
    dec = url_decode(enc["encoded"])
    assert dec["ok"] is True
    assert dec["decoded"] == "hello world/?&="


# ---------- regex_test ----------

def test_regex_test_match_found():
    r = regex_test(pattern=r"foo(\d+)", text="foo123 bar")
    assert r["ok"] is True
    assert r["match"] is True
    assert r["groups"] == ["123"]


def test_regex_test_no_match():
    r = regex_test(pattern=r"^xyz$", text="abc")
    assert r["ok"] is True
    assert r["match"] is False


def test_regex_test_invalid_pattern_rejected():
    r = regex_test(pattern="(", text="abc")
    assert r["ok"] is False
    assert r["reason"] == "invalid_regex"


def test_regex_test_pattern_too_long_rejected():
    r = regex_test(pattern="a" * 5000, text="abc")
    assert r["ok"] is False
    assert r["reason"] == "pattern_too_long"


# ---------- qr_code: path safety ----------

def test_qr_code_outside_allowed_root_rejected(tmp_path):
    r = qr_code(content="hi", output_path="/etc/qr.png")
    assert r["ok"] is False
    assert r["reason"] == "path_denied"


def test_qr_code_relative_path_rejected():
    r = qr_code(content="hi", output_path="relative.png")
    assert r["ok"] is False
    assert r["reason"] == "path_denied"


def _allow_tmp_workspace(monkeypatch, tmp_path):
    workspace = tmp_path.resolve()
    monkeypatch.setattr("services.file_system_tools.ALLOWED_ROOTS", (workspace,))
    return workspace


def test_qr_code_empty_content_rejected(tmp_path):
    r = qr_code(content="", output_path=str(tmp_path / "qr.png"))
    assert r["ok"] is False
    assert r["reason"] == "empty_content"


def test_qr_code_writes_png_inside_allowed_root(tmp_path, monkeypatch):
    target_dir = _allow_tmp_workspace(monkeypatch, tmp_path)
    out = target_dir / "_nexus_utils_test_qr.png"
    try:
        r = qr_code(content="hello world", output_path=str(out))
        assert r["ok"] is True, r
        assert r["size_bytes"] > 0
        # PNG magic header
        with open(out, "rb") as f:
            assert f.read(4) == b"\x89PNG"
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass


def test_qr_code_parent_missing_rejected():
    r = qr_code(content="hi", output_path=str(Path.home() / "no_such_dir_xyz_phase25a" / "q.png"))
    assert r["ok"] is False
    assert r["reason"] == "parent_does_not_exist"


# ---------- hebcal / shabbos / omer (network-mocked smoke) ----------

def test_hebcal_invalid_date_rejected():
    r = hebcal(date="not-a-date")
    assert r["ok"] is False
    assert r["reason"] == "invalid_date"


def test_hebcal_happy_via_mock():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"hy": 5786, "hm": "Iyyar", "hd": 24}
    mock_resp.raise_for_status = lambda: None
    with patch("mcp_servers.nexus_utils.httpx.get", return_value=mock_resp):
        r = hebcal(date="2026-05-12")
    assert r["ok"] is True
    assert r["date"] == "2026-05-12"
    assert r["hebrew"]["hy"] == 5786


def test_shabbos_times_no_location_rejected():
    r = shabbos_times()
    assert r["ok"] is False
    assert r["reason"] == "no_location"


def test_omer_count_invalid_date_rejected():
    r = omer_count(date="not-a-date")
    assert r["ok"] is False


def test_omer_count_via_mock_with_omer_entry():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {"date": "2026-05-12", "category": "omer", "title": "32nd day of the Omer", "memo": "Hod sheb'Hod"},
        ]
    }
    mock_resp.raise_for_status = lambda: None
    with patch("mcp_servers.nexus_utils.httpx.get", return_value=mock_resp):
        r = omer_count(date="2026-05-12")
    assert r["ok"] is True
    assert r["in_omer"] is True
    assert "Omer" in r["title"]
