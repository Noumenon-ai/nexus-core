from __future__ import annotations

import json
from datetime import datetime

from repositories.memories_repository import MemoriesRepository
from utils.i18n import Translator


class HabitService:
    def __init__(self, memories_repository: MemoriesRepository) -> None:
        self.memories_repository = memories_repository

    def record_reminder_creation(self, user_id: str, created_at: datetime) -> None:
        self._update_histogram(user_id, 'reminder_creation_hours', created_at.hour)

    def record_task_completion(self, user_id: str, completed_at: datetime) -> None:
        self._update_histogram(user_id, 'task_completion_hours', completed_at.hour)

    def suggestion_for_user(self, user_id: str, translator: Translator | None = None) -> str | None:
        memory = self.memories_repository.get(user_id=user_id, memory_type='habit', key='reminder_creation_hours')
        if memory is None:
            return None
        payload = json.loads(memory.value)
        if not payload:
            return None
        dominant_hour = max(payload.items(), key=lambda item: item[1])[0]
        translator = translator or Translator('en')
        return translator.t('habit_reminder_hour', hour=f'{int(dominant_hour):02d}:00')

    def detect_deviation(self, user_id: str, current_hour: int, translator: Translator | None = None) -> str | None:
        memory = self.memories_repository.get(user_id=user_id, memory_type='habit', key='task_completion_hours')
        if memory is None:
            return None
        payload = json.loads(memory.value)
        if not payload:
            return None
        dominant_hour = int(max(payload.items(), key=lambda item: item[1])[0])
        if current_hour >= dominant_hour + 1:
            translator = translator or Translator('en')
            return translator.t('habit_deviation')
        return None

    def _update_histogram(self, user_id: str, key: str, hour: int) -> None:
        memory = self.memories_repository.get(user_id=user_id, memory_type='habit', key=key)
        if memory is None:
            payload = {}
            confidence = 0.2
        else:
            payload = json.loads(memory.value)
            confidence = min(memory.confidence + 0.1, 0.9)
        payload[str(hour)] = payload.get(str(hour), 0) + 1
        self.memories_repository.upsert(
            user_id=user_id,
            memory_type='habit',
            key=key,
            value=json.dumps(payload, separators=(',', ':'), sort_keys=True),
            confidence=confidence,
            source='observed',
        )
