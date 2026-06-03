"""Phase 8 — handler wiring + image-claude-call adversarial tests.

Real Telegram Update objects are heavyweight to construct; these tests cover
the parts that don't need a live Update: import safety, registration order,
and the static _claude_with_image helper that drives subprocess.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_telegram_bot_imports_with_new_handlers():
    """If filters.Document.MimeType / filters.PHOTO are missing from a stub
    telegram, the import still works thanks to the existing fallback shim."""
    import telegram_bot  # noqa: F401 — pure import smoke test


def test_telegram_bot_handler_registration_order():
    """Document + photo handlers must register BEFORE the catch-all TEXT
    handler, otherwise messages with text captions get caught by the wrong
    handler.  The registration code is the canonical source — read the AST.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent.parent / "telegram_bot.py"
    text = src.read_text()
    # The catch-all TEXT handler must come AFTER the document + photo handlers
    doc_idx = text.find("_handle_document")
    photo_idx = text.find("_handle_photo")
    text_handler_reg_idx = text.find("filters.TEXT & ~filters.COMMAND, self._handle_text")
    assert doc_idx != -1, "document handler missing"
    assert photo_idx != -1, "photo handler missing"
    assert text_handler_reg_idx > doc_idx, "TEXT handler must register after document handler"
    assert text_handler_reg_idx > photo_idx, "TEXT handler must register after photo handler"


@pytest.mark.asyncio
async def test_claude_with_image_invokes_subprocess_with_read_tool():
    """The image handler invokes claude -p with --allowedTools 'Read' and embeds
    the image path in the prompt so Claude reads it via its built-in tool.
    """
    from telegram_bot import TelegramBot
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = b"A photo of a coffee mug on a desk."
    mock_proc.stderr = b""
    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        result = await TelegramBot._claude_with_image(
            image_path=Path("/home/user/nexus_workspace/uploads/foo.jpg"),
            caption="what is this",
        )
    assert "coffee mug" in result
    cmd = mock_run.call_args.args[0]
    assert "claude" in cmd
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert "Read" in cmd[idx + 1]
    prompt = cmd[-1]
    assert "/home/user/nexus_workspace/uploads/foo.jpg" in prompt
    assert "what is this" in prompt


@pytest.mark.asyncio
async def test_claude_with_image_propagates_failure():
    from telegram_bot import TelegramBot
    mock_proc = MagicMock()
    mock_proc.returncode = 2
    mock_proc.stdout = b""
    mock_proc.stderr = b"auth required"
    with patch("subprocess.run", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="exit 2"):
            await TelegramBot._claude_with_image(
                image_path=Path("/tmp/x.jpg"),
                caption="describe",
            )


@pytest.mark.asyncio
async def test_claude_with_image_returns_placeholder_for_empty_output():
    from telegram_bot import TelegramBot
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = b"  \n  "
    mock_proc.stderr = b""
    with patch("subprocess.run", return_value=mock_proc):
        r = await TelegramBot._claude_with_image(
            image_path=Path("/tmp/x.jpg"), caption="describe")
    assert "empty response" in r.lower()


@pytest.mark.asyncio
async def test_claude_with_image_uses_stdin_devnull():
    """The subprocess must inherit stdin=DEVNULL so the Telegram handler
    doesn't accidentally tie up the bot process if claude waits for input.
    """
    from telegram_bot import TelegramBot
    mock_proc = MagicMock(returncode=0, stdout=b"x", stderr=b"")
    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        await TelegramBot._claude_with_image(
            image_path=Path("/tmp/x.jpg"), caption="describe")
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("stdin") == subprocess.DEVNULL
    assert kwargs.get("timeout") == 180


# ---------- PDF extraction smoke (uses real pdfplumber) ----------

def test_pdfplumber_extracts_text_from_generated_pdf(tmp_path):
    """A locally-generated PDF (reportlab) must produce extractable text via
    pdfplumber — the PDF handler relies on this round-trip.
    """
    from reportlab.pdfgen import canvas
    target = tmp_path / "smoke.pdf"
    c = canvas.Canvas(str(target))
    c.drawString(72, 720, "PHASE_8_PDF_INTAKE_OK")
    c.showPage()
    c.save()
    import pdfplumber
    with pdfplumber.open(target) as pdf:
        text = pdf.pages[0].extract_text() or ""
    assert "PHASE_8_PDF_INTAKE_OK" in text
