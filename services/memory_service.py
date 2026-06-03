from __future__ import annotations

import re

from models import User
from pipeline.types import ServiceResponse
from repositories.memories_repository import MemoriesRepository
from utils.i18n import Translator
from utils.safety_rules import contains_secret


class MemoryService:
    def __init__(self, memories_repository: MemoriesRepository) -> None:
        self.memories_repository = memories_repository

    def remember(self, user: User, text: str, translator: Translator) -> ServiceResponse:
        statement = self._suffix_after_prefix(text, 'remember')
        if not statement:
            return ServiceResponse(text=translator.t('memory_need_content'))
        if contains_secret(statement):
            return ServiceResponse(text=translator.t('memory_forbidden_secret'))
        memory_type, key, value = self._normalize(statement)
        self.memories_repository.upsert(
            user_id=user.id,
            memory_type=memory_type,
            key=key,
            value=value,
            confidence=1.0,
            source='explicit',
        )
        return ServiceResponse(text=translator.t('memory_saved'))

    def list_memories(self, user: User, translator: Translator) -> ServiceResponse:
        memories = self.memories_repository.list_by_user(user.id, source='explicit')
        if not memories:
            return ServiceResponse(text=translator.t('memory_list_empty'))
        lines = [f"- {memory.key}: {memory.value}" for memory in memories]
        return ServiceResponse(text='\n'.join(lines), voice_appropriate=False)

    def delete_memory(self, user: User, key: str, translator: Translator) -> ServiceResponse:
        return self.delete_by_key(user.id, key, translator)

    def delete_by_key(self, user_id: str, key: str, translator: Translator) -> ServiceResponse:
        deleted = self.memories_repository.delete(user_id=user_id, key=key)
        if deleted is None:
            return ServiceResponse(text=translator.t('memory_not_found'))
        return ServiceResponse(text=translator.t('memory_deleted'))

    def forget_key_from_text(self, text: str) -> str | None:
        statement = self._suffix_after_prefix(text, ('forget', 'delete memory', 'remove memory'))
        if not statement:
            return None
        _memory_type, key, _value = self._normalize(statement)
        return key

    def _normalize(self, statement: str) -> tuple[str, str, str]:
        lowered = statement.lower()
        match = re.search(r'prefer\s+(morning|afternoon|evening)\s+reminders', lowered)
        if match:
            return 'preference', 'reminder_time_preference', match.group(1)
        slug = re.sub(r'[^a-z0-9]+', '_', lowered).strip('_')[:64] or 'memory'
        return 'fact', slug, statement

    def _suffix_after_prefix(self, text: str, prefix: str | tuple[str, ...]) -> str:
        prefixes = (prefix,) if isinstance(prefix, str) else prefix
        lowered = text.lower()
        for item in prefixes:
            index = lowered.find(item)
            if index != -1:
                return text[index + len(item):].strip()
        return text.strip()
