from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


IntentType = Literal['general', 'reminder', 'memory', 'task', 'email', 'search', 'opportunity', 'action', 'system', 'ambiguous']
InputKind = Literal['text', 'voice', 'button', 'scheduled']


@dataclass(slots=True)
class PipelineInput:
    kind: InputKind
    telegram_id: int
    text: str = ''
    callback_data: str = ''
    voice_file_path: str = ''
    scheduled_payload: dict[str, Any] | None = None
    username: str | None = None
    full_name: str | None = None
    # V3.7 streaming session — Optional[StreamingSession] in practice, typed
    # as Any here to keep `pipeline/types.py` free of telegram/streaming
    # imports. UnifiedPipeline forwards this to DispatcherInput when the
    # dispatcher path is active. None for non-dispatcher routes.
    streaming_session: Any = None


@dataclass(slots=True)
class InlineButton:
    text: str
    callback_data: str


@dataclass(slots=True)
class PipelineOutput:
    text: str
    voice_audio: bytes | None = None
    buttons: list[InlineButton] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineContext:
    user_id: str
    last_topic: str | None
    last_entity_type: str | None
    last_entity_value: str | None
    last_intent: str | None
    context: dict[str, Any]
    expires_at: datetime | None


@dataclass(slots=True)
class IntentResult:
    intent_type: IntentType
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    needs_tool: bool = False
    needs_clarification: bool = False
    clarifying_question: str | None = None
    voice_appropriate: bool = True


@dataclass(slots=True)
class ServiceResponse:
    text: str
    voice_appropriate: bool = True
    buttons: list[InlineButton] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
