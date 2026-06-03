"""V3.9 unified pipeline — single dispatcher path.

Pre-V3.9 classic-path code was deleted. The
NEXUS_DISPATCHER_ENABLED env var stays as a no-op for one release
cycle so deploy environments don't break on missing-var lookups;
removal of the env var itself is a future cleanup.

Tool dispatcher must be wired (non-None) for any text path to work.
If main.py fails to construct the dispatcher (LLM disabled, mem0
unavailable, Google not connected), text messages return a benign
clarify-generic sentinel rather than silently routing to a path
that no longer exists.

See HARDENING_PASS_V2.md H2-022 for the V3.9 spec audit and H2-013
for the resolved bare-token classification bug class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.types import IntentResult, PipelineInput, PipelineOutput
from services.conversation_service import ConversationService
from services.voice_input_service import VoiceInputService
from services.voice_output_service import VoiceOutputService
from utils.i18n import Translator
from utils.language import detect_language


@dataclass(slots=True)
class UnifiedPipeline:
    users_repository: Any
    conversation_service: ConversationService
    voice_input_service: VoiceInputService | None
    voice_output_service: VoiceOutputService | None
    allowed_telegram_ids: tuple[int, ...]
    tool_dispatcher: Any = None

    async def handle(self, input_data: PipelineInput) -> PipelineOutput:
        if input_data.telegram_id not in self.allowed_telegram_ids:
            return PipelineOutput(text=Translator('en').t('security_unauthorized'))
        user = self.users_repository.get_or_create(
            input_data.telegram_id,
            username=input_data.username,
            full_name=input_data.full_name,
        )
        translator = Translator(user.language)
        normalized_text = await self._normalize(input_data)
        if normalized_text is None:
            return PipelineOutput(text=translator.t('audio_unclear'))
        translator = Translator(detect_language(normalized_text))

        if self.tool_dispatcher is None:
            # Production must always wire the dispatcher; this branch
            # only fires when boot-time construction failed (e.g. LLM
            # creds missing). Returning the unauthorized sentinel
            # would leak a wrong reason — return the generic clarify
            # message instead.
            return PipelineOutput(text=translator.t('clarify_generic'))

        from pipeline.tool_dispatcher import DispatcherInput
        dispatch_out = await self.tool_dispatcher.handle(DispatcherInput(
            user=user, text=normalized_text, translator=translator,
            streaming_session=input_data.streaming_session,
        ))
        synthetic_intent = IntentResult(intent_type='general', confidence=1.0)
        self.conversation_service.save_context(
            user.id, intent=synthetic_intent,
            response_text=dispatch_out.text,
            context_updates={'topic': normalized_text},
        )
        return PipelineOutput(
            text=dispatch_out.text,
            buttons=list(dispatch_out.buttons),
            metadata=dict(dispatch_out.metadata),
        )

    async def _normalize(self, input_data: PipelineInput) -> str | None:
        if input_data.kind == 'voice':
            if self.voice_input_service is None:
                return None
            try:
                result = await self.voice_input_service.transcribe(input_data.voice_file_path)
            except Exception:
                return None
            if result.confidence < 0.3 or not result.text.strip():
                return None
            return result.text.strip()
        if input_data.kind == 'button':
            return input_data.callback_data.strip()
        if input_data.kind == 'scheduled':
            payload = input_data.scheduled_payload or {}
            return payload.get('text', '').strip()
        return input_data.text.strip()
