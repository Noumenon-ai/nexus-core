from __future__ import annotations

from pathlib import Path

import pytest

from services.telos_service import TelosService, load_telos_template


@pytest.fixture
def telos(tmp_path: Path) -> TelosService:
    return TelosService(telos_dir=tmp_path / "telos")


def test_init_creates_directory_with_700_mode(tmp_path: Path):
    target = tmp_path / "telos"
    assert not target.exists()
    TelosService(telos_dir=target)
    assert target.exists()
    assert target.is_dir()
    mode = target.stat().st_mode & 0o777
    assert mode == 0o700, f"expected 0o700, got {oct(mode)}"


def test_load_returns_none_when_file_missing(telos: TelosService):
    assert telos.load(user_id=42) is None


def test_load_returns_contents_when_file_exists(telos: TelosService):
    p = telos.path_for(user_id=42)
    p.write_text("# TELOS — Owner\n## Identity\nBuilder.\n", encoding="utf-8")
    assert telos.load(user_id=42) == "# TELOS — Owner\n## Identity\nBuilder.\n"


def test_load_reads_fresh_each_call_no_cache(telos: TelosService):
    """Spec: 'TELOS file is loaded fresh each turn, NOT cached.
    User edits take effect immediately on next message.'"""
    p = telos.path_for(user_id=7)
    p.write_text("v1", encoding="utf-8")
    assert telos.load(7) == "v1"
    p.write_text("v2", encoding="utf-8")
    assert telos.load(7) == "v2"


def test_per_user_isolation(telos: TelosService):
    """Spec: 'Per-user isolation: User A cannot read User B TELOS.'"""
    telos.path_for(user_id="alice").write_text("alice secrets", encoding="utf-8")
    telos.path_for(user_id="bob").write_text("bob secrets", encoding="utf-8")
    assert telos.load("alice") == "alice secrets"
    assert telos.load("bob") == "bob secrets"


def test_user_id_path_traversal_rejected(telos: TelosService):
    for bad in ("../etc/passwd", "../../foo", "alice/bob", "", "   ", "a\x00b"):
        with pytest.raises(ValueError):
            telos.path_for(user_id=bad)


def test_has_telos_reports_presence(telos: TelosService):
    assert telos.has_telos(99) is False
    telos.path_for(99).write_text("x", encoding="utf-8")
    assert telos.has_telos(99) is True


def test_template_has_required_sections():
    content = load_telos_template()
    assert content.startswith("# TELOS"), "template must start with '# TELOS' heading"
    required_sections = [
        "## Identity",
        "## What I'm Building",
        "## What I Care About",
        "## How I Communicate",
        "## My Day",
        "## My People",
        "## Health & Energy",
        "## Money",
        "## Constraints",
        "## Where I'm Going",
        "## Things I Tend to Forget",
        "## Updates",
    ]
    for section in required_sections:
        assert section in content, f"template missing section: {section}"


def test_template_path_exists():
    """Template file must ship with the repo at resources/telos_template.md."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    template = repo_root / "resources" / "telos_template.md"
    assert template.exists(), f"missing: {template}"


def test_user_id_accepts_int_and_str(telos: TelosService):
    """Telegram user IDs are integers, but filenames may also come from other sources."""
    p_int = telos.path_for(user_id=12345)
    p_str = telos.path_for(user_id="12345")
    assert p_int == p_str
