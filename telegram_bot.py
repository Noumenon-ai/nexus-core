from __future__ import annotations

import asyncio
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
except ImportError:
    class NetworkError(Exception):
        """Fallback telegram network error."""

    class RetryAfter(Exception):
        """Fallback telegram retry error."""

    class TimedOut(Exception):
        """Fallback telegram timeout error."""

    class BadRequest(Exception):
        """Fallback telegram bad request error."""

    class InlineKeyboardButton:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class InlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    class Update:
        pass

    class _UnavailableBuilder:
        def token(self, *_args, **_kwargs):
            raise RuntimeError('python-telegram-bot is not installed')

    class Application:
        @classmethod
        def builder(cls):
            return _UnavailableBuilder()

    class CallbackQueryHandler:
        def __init__(self, *args, **kwargs):
            pass

    class CommandHandler:
        def __init__(self, *args, **kwargs):
            pass

    class MessageHandler:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyContextTypes:
        DEFAULT_TYPE = object

    ContextTypes = _DummyContextTypes()

    class _DummyFilterExpr:
        def __and__(self, other):
            return self

        def __invert__(self):
            return self

    class _DummyFilters:
        VOICE = _DummyFilterExpr()
        TEXT = _DummyFilterExpr()
        COMMAND = _DummyFilterExpr()

    filters = _DummyFilters()

from pipeline.types import PipelineInput, PipelineOutput
from utils.i18n import Translator


logger = logging.getLogger(__name__)


async def _sleep_for_retry(exc: Exception) -> None:
    retry_after = getattr(exc, 'retry_after', 0)
    delay = retry_after.total_seconds() if hasattr(retry_after, 'total_seconds') else float(retry_after or 0)
    if delay > 0:
        await asyncio.sleep(delay)


async def _run_blocking_without_default_executor(fn, /, *args, **kwargs):
    """Run a blocking call without touching asyncio's default executor.

    In this environment, `asyncio.to_thread()` can leave a worker alive long
    enough to make short-lived pytest subprocesses hang on exit even after the
    awaited Claude CLI call has completed. Use a one-off daemon thread and wait
    on a simple event instead.
    """
    result_box: dict[str, object] = {}
    done = threading.Event()

    def worker():
        try:
            result_box['result'] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - re-raised by awaiter
            result_box['exception'] = exc
        finally:
            done.set()

    threading.Thread(
        target=worker,
        daemon=True,
        name='telegram-bot-blocking',
    ).start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    if 'exception' in result_box:
        raise result_box['exception']  # type: ignore[misc]
    return result_box.get('result')


@dataclass(slots=True)
class TelegramMessenger:
    application: Application
    users_repository: any

    async def send_output(self, telegram_id: int, output: PipelineOutput) -> None:
        markup = _build_reply_markup(output)
        await self._send_message_with_retry(chat_id=telegram_id, text=output.text, reply_markup=markup)
        if output.voice_audio:
            await self._send_voice_with_retry(chat_id=telegram_id, voice_audio=output.voice_audio)

    async def _send_message_with_retry(self, *, chat_id: int, text: str, reply_markup=None) -> None:
        attempts = 0
        while True:
            try:
                await self.application.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
                return
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)

    async def _send_voice_with_retry(self, *, chat_id: int, voice_audio: bytes) -> None:
        attempts = 0
        while True:
            try:
                await self.application.bot.send_voice(chat_id=chat_id, voice=voice_audio)
                return
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)


class TelegramBot:
    def __init__(self, *, token: str, pipeline, voice_in_dir: Path, max_voice_file_bytes: int) -> None:
        self.pipeline = pipeline
        self.voice_in_dir = voice_in_dir
        self.max_voice_file_bytes = max_voice_file_bytes
        self.application = Application.builder().token(token).build()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler('start', self._handle_text))
        self.application.add_handler(CommandHandler('status', self._handle_text))
        self.application.add_handler(CommandHandler('disable_proactive', self._handle_text))
        self.application.add_handler(CommandHandler('enable_proactive', self._handle_text))
        self.application.add_handler(CommandHandler('voice_on', self._handle_text))
        self.application.add_handler(CommandHandler('voice_off', self._handle_text))
        # /reminders, /reminders duplicates — routed to the dispatcher's
        # _classify_direct_reminder_read_command. Without this, slash-command
        # /reminders messages are filtered out by the `~filters.COMMAND`
        # below and the user sees no reply.
        self.application.add_handler(CommandHandler('reminders', self._handle_text))
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
        self.application.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        # Phase 8: PDF + image handlers BEFORE the catch-all text handler.
        self.application.add_handler(MessageHandler(filters.Document.MimeType("application/pdf"), self._handle_document))
        self.application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        self.application.add_error_handler(self._handle_error)

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        if isinstance(err, (NetworkError, TimedOut, RetryAfter)):
            logger.warning('telegram_network_transient: %s: %s', type(err).__name__, err)
            return
        if isinstance(err, BadRequest):
            logger.warning('telegram_bad_request: %s', err)
            return
        logger.error('telegram_unhandled_error: %s: %s', type(err).__name__, err, exc_info=err)

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or update.effective_message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        streaming_session = self._maybe_build_streaming_session(chat_id)
        output = await self.pipeline.handle(PipelineInput(
            kind='text', telegram_id=update.effective_user.id,
            text=update.effective_message.text or '',
            username=update.effective_user.username,
            full_name=update.effective_user.full_name,
            streaming_session=streaming_session,
        ))
        await self._send_output(chat_id, output)

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.callback_query is None or update.effective_user is None or update.effective_chat is None:
            return
        await update.callback_query.answer()
        output = await self.pipeline.handle(PipelineInput(kind='button', telegram_id=update.effective_user.id, callback_data=update.callback_query.data or '', username=update.effective_user.username, full_name=update.effective_user.full_name))
        await self._send_output(update.effective_chat.id, output)

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 8: PDF intake.

        Downloads the PDF, extracts text via pdfplumber (best-effort), folds the
        text + the user's caption into a text prompt, and runs through the
        normal pipeline so MCP tools / brain_router behave as if the user typed
        a long descriptive message.
        """
        if (update.message is None or update.effective_user is None
                or update.effective_chat is None or update.message.document is None):
            return
        if update.effective_user.id not in getattr(self.pipeline, 'allowed_telegram_ids', ()):
            return
        doc = update.message.document
        # Size guard — Telegram caps documents at 20 MB anyway; reject larger.
        if doc.file_size and doc.file_size > 20 * 1024 * 1024:
            await self._send_output(update.effective_chat.id, PipelineOutput(
                text="That PDF is over 20MB — too large for this handler."))
            return

        uploads_dir = Path("/home/user/nexus_workspace/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target = uploads_dir / f"{uuid.uuid4()}.pdf"
        try:
            tg_file = await context.bot.get_file(doc.file_id)
            await tg_file.download_to_drive(str(target))
        except Exception as exc:
            logger.warning("pdf_download_failed: %s", exc)
            await self._send_output(update.effective_chat.id, PipelineOutput(
                text=f"Couldn't download the PDF: {exc}"))
            return

        # Extract text (best effort)
        text_excerpt = ""
        page_count = 0
        try:
            import pdfplumber
            with pdfplumber.open(target) as pdf:
                page_count = len(pdf.pages)
                pages: list[str] = []
                for p in pdf.pages[:50]:  # cap at 50 pages for prompt budget
                    pages.append(p.extract_text() or "")
                text_excerpt = "\n\n".join(pages)
        except Exception as exc:
            logger.warning("pdf_extract_failed: %s", exc)
            text_excerpt = ""

        if len(text_excerpt) > 30_000:
            text_excerpt = text_excerpt[:30_000] + "\n\n[... truncated ...]"

        caption = (update.message.caption or "").strip()
        prompt_parts = [
            f"The user just uploaded a PDF named {doc.file_name or 'document.pdf'!r} ({page_count} pages, {doc.file_size} bytes, saved at {target}).",
        ]
        if caption:
            prompt_parts.append(f"User's caption: {caption}")
        else:
            prompt_parts.append("User did not include a caption — give a concise summary of the document.")
        if text_excerpt:
            prompt_parts.append(f"--- BEGIN PDF TEXT ---\n{text_excerpt}\n--- END PDF TEXT ---")
        else:
            prompt_parts.append("(pdfplumber returned no extractable text — the PDF may be image-only/scanned.)")
        composed_prompt = "\n\n".join(prompt_parts)

        chat_id = update.effective_chat.id
        streaming_session = self._maybe_build_streaming_session(chat_id)
        output = await self.pipeline.handle(PipelineInput(
            kind='text', telegram_id=update.effective_user.id,
            text=composed_prompt, username=update.effective_user.username,
            full_name=update.effective_user.full_name,
            streaming_session=streaming_session,
        ))
        await self._send_output(chat_id, output)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 8: image intake.

        Downloads the largest photo, then invokes Claude CLI directly (NOT via
        brain_router's MCP-only path) so the model's built-in Read tool can
        process the image multimodally.  Returns the model's description.
        """
        if (update.message is None or update.effective_user is None
                or update.effective_chat is None or not update.message.photo):
            return
        if update.effective_user.id not in getattr(self.pipeline, 'allowed_telegram_ids', ()):
            return
        photo = update.message.photo[-1]  # largest size
        if photo.file_size and photo.file_size > 10 * 1024 * 1024:
            await self._send_output(update.effective_chat.id, PipelineOutput(
                text="That image is over 10MB — too large for this handler."))
            return
        uploads_dir = Path("/home/user/nexus_workspace/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target = uploads_dir / f"{uuid.uuid4()}.jpg"
        try:
            tg_file = await context.bot.get_file(photo.file_id)
            await tg_file.download_to_drive(str(target))
        except Exception as exc:
            logger.warning("photo_download_failed: %s", exc)
            await self._send_output(update.effective_chat.id, PipelineOutput(
                text=f"Couldn't download the photo: {exc}"))
            return

        caption = (update.message.caption or "describe what's in this image").strip()
        chat_id = update.effective_chat.id
        try:
            text = await self._claude_with_image(image_path=target, caption=caption)
        except Exception as exc:
            logger.warning("photo_claude_failed: %s", exc)
            text = f"I downloaded the image but Claude CLI failed to process it: {exc}"
        await self._send_output(chat_id, PipelineOutput(text=text))

    @staticmethod
    async def _claude_with_image(image_path: Path, caption: str) -> str:
        """Spawn `claude -p` with Read tool allowed + an instruction that points
        at the image path.  Returns stdout text.
        """
        import subprocess
        prompt = (
            f"Use the Read tool to view the image at {image_path}. "
            f"Then respond to the user with: {caption}"
        )
        cmd = ['claude', '--allowedTools', 'Read', '-p', prompt]
        proc = await _run_blocking_without_default_executor(
            subprocess.run, cmd, capture_output=True, timeout=180, stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr.decode(errors='replace')[:300]}")
        return proc.stdout.decode(errors='replace').strip() or "(empty response)"

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_user is None or update.effective_chat is None or update.message.voice is None:
            return
        if update.effective_user.id not in getattr(self.pipeline, 'allowed_telegram_ids', ()): 
            await self._send_output(update.effective_chat.id, PipelineOutput(text=Translator('en').t('security_unauthorized')))
            return
        language_code = getattr(update.effective_user, 'language_code', '') or ''
        translator = Translator('he' if language_code.lower().startswith('he') else 'en')
        file_size = getattr(update.message.voice, 'file_size', None)
        if isinstance(file_size, int) and file_size > self.max_voice_file_bytes:
            max_mb = math.ceil(self.max_voice_file_bytes / (1024 * 1024))
            await self._send_output(update.effective_chat.id, PipelineOutput(text=translator.t('audio_too_large', max_mb=max_mb)))
            return
        file = await context.bot.get_file(update.message.voice.file_id)
        target = self.voice_in_dir / f'{uuid.uuid4()}.ogg'
        await file.download_to_drive(str(target))
        try:
            output = await self.pipeline.handle(PipelineInput(kind='voice', telegram_id=update.effective_user.id, voice_file_path=str(target), username=update.effective_user.username, full_name=update.effective_user.full_name))
        finally:
            target.unlink(missing_ok=True)
        await self._send_output(update.effective_chat.id, output)

    def _maybe_build_streaming_session(self, chat_id: int):
        """V3.7: instantiate a StreamingSession only when the dispatcher
        path is active. The classic intent_classifier path returns one
        complete reply per turn — no progress beats to stream — so
        wiring streaming there would just send empty placeholders.

        Returns `None` when the dispatcher is disabled OR not wired.
        Imports are local to avoid pulling streaming/telegram deps into
        non-streaming code paths.
        """
        dispatcher_enabled = bool(getattr(self.pipeline, 'dispatcher_enabled', False))
        tool_dispatcher = getattr(self.pipeline, 'tool_dispatcher', None)
        if not dispatcher_enabled or tool_dispatcher is None:
            return None
        from services.telegram_streaming import StreamingSession
        return StreamingSession(chat_id=chat_id, telegram_bot=self)

    async def _send_output(self, chat_id: int, output: PipelineOutput) -> None:
        markup = _build_reply_markup(output)
        # V3.7: when the dispatcher already streamed the final text via
        # StreamingSession.finalize, skip the duplicate text send — but
        # still deliver buttons (attached to a fresh message) and voice
        # since those don't ride through streaming.
        streamed = bool(output.metadata.get('streamed'))
        if not streamed:
            await self._send_message_with_retry(chat_id=chat_id, text=output.text, reply_markup=markup)
        elif markup is not None:
            # Buttons can't attach to a finalized streaming message in
            # V3.7 narrow scope — send them as a follow-up message. The
            # text on this follow-up is intentionally minimal so the
            # primary content remains the streamed reply above it.
            await self._send_message_with_retry(chat_id=chat_id, text=' ', reply_markup=markup)
        if output.voice_audio:
            await self._send_voice_with_retry(chat_id=chat_id, voice_audio=output.voice_audio)

    async def _send_message_with_retry(self, *, chat_id: int, text: str, reply_markup=None) -> None:
        attempts = 0
        while True:
            try:
                await self.application.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
                return
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)

    async def _send_voice_with_retry(self, *, chat_id: int, voice_audio: bytes) -> None:
        attempts = 0
        while True:
            try:
                await self.application.bot.send_voice(chat_id=chat_id, voice=voice_audio)
                return
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)

    async def send_text(self, *, chat_id: int, text: str) -> int:
        """V3.7 streaming: send a plain-text message and return its
        message_id so callers can edit it later. Transient telegram
        errors get one retry; persistent errors propagate so the
        caller can fall back."""
        attempts = 0
        while True:
            try:
                msg = await self.application.bot.send_message(chat_id=chat_id, text=text)
                return msg.message_id
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)

    async def edit_text(self, *, chat_id: int, message_id: int, text: str) -> None:
        """V3.7 streaming: edit an existing message in place. Transient
        errors get one retry; BadRequest (message-not-found,
        not-modified, etc.) propagates immediately so the streaming
        session can fall back to sending a new message."""
        attempts = 0
        while True:
            try:
                await self.application.bot.edit_message_text(
                    chat_id=chat_id, message_id=message_id, text=text,
                )
                return
            except (NetworkError, TimedOut, RetryAfter) as exc:
                attempts += 1
                if attempts >= 2:
                    raise
                await _sleep_for_retry(exc)

    def run(self) -> None:
        self.application.run_polling(close_loop=False)


def _build_reply_markup(output: PipelineOutput) -> InlineKeyboardMarkup | None:
    if not output.buttons:
        return None
    return InlineKeyboardMarkup([[InlineKeyboardButton(button.text, callback_data=button.callback_data) for button in output.buttons]])
