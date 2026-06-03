from __future__ import annotations

from pipeline.types import ServiceResponse
from services.memory_service import MemoryService
from services.task_service import TaskService
from utils.i18n import Translator


class ActionService:
    def __init__(self, memory_service: MemoryService, task_service: TaskService, *, messenger=None) -> None:
        self.memory_service = memory_service
        self.task_service = task_service
        self.messenger = messenger

    def build_memory_delete_payload(self, user_id: str, key: str, translator: Translator) -> tuple[str, dict]:
        return translator.t('action_preview_delete_memory', key=key), {'user_id': user_id, 'key': key}

    def build_outbound_message_payload(self, user_id: str, message_text: str, translator: Translator) -> tuple[str, dict]:
        return translator.t('action_preview_outbound_message', text=message_text[:80]), {'user_id': user_id, 'text': message_text}

    def execute(self, action_type: str, payload: dict, translator: Translator) -> ServiceResponse:
        if action_type == 'delete_memory':
            return self.memory_service.delete_by_key(payload['user_id'], payload['key'], translator)
        if action_type == 'outbound_message':
            if self.messenger is None:
                return ServiceResponse(text=translator.t('approval_cancelled'))
            self.messenger(payload['user_id'], payload['text'])
            return ServiceResponse(text=translator.t('approval_executed'))
        return ServiceResponse(text=translator.t('approval_cancelled'))
