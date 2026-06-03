from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pipeline.types import IntentResult, PipelineContext
from repositories.conversation_context_repository import ConversationContextRepository
from utils.dates import utc_now


class ConversationService:
    def __init__(self, context_repository: ConversationContextRepository) -> None:
        self.context_repository = context_repository

    def load_context(self, user_id: str) -> PipelineContext:
        record = self.context_repository.get_active(user_id)
        if record is None:
            return PipelineContext(
                user_id=user_id,
                last_topic=None,
                last_entity_type=None,
                last_entity_value=None,
                last_intent=None,
                context={},
                expires_at=None,
            )
        payload = {}
        if record.context_json:
            try:
                payload = json.loads(record.context_json)
            except json.JSONDecodeError:
                payload = {}
        return PipelineContext(
            user_id=user_id,
            last_topic=record.last_topic,
            last_entity_type=record.last_entity_type,
            last_entity_value=record.last_entity_value,
            last_intent=record.last_intent,
            context=payload,
            expires_at=record.expires_at,
        )

    def save_context(self, user_id: str, *, intent: IntentResult, response_text: str, context_updates: dict[str, Any] | None = None) -> PipelineContext:
        existing = self.load_context(user_id)
        payload = dict(existing.context)
        if context_updates:
            payload.update(context_updates)
        record = self.context_repository.upsert(
            user_id=user_id,
            last_topic=payload.get('topic') or response_text[:120],
            last_entity_type=next(iter(intent.entities.keys()), None) if intent.entities else None,
            last_entity_value=next(iter(intent.entities.values()), None) if intent.entities else None,
            last_intent=intent.intent_type,
            context_json=json.dumps(payload, separators=(',', ':'), sort_keys=True),
        )
        return PipelineContext(
            user_id=user_id,
            last_topic=record.last_topic,
            last_entity_type=record.last_entity_type,
            last_entity_value=record.last_entity_value,
            last_intent=record.last_intent,
            context=payload,
            expires_at=record.expires_at,
        )

    def merge_context(
        self,
        user_id: str,
        *,
        context_updates: dict[str, Any],
        ttl_minutes: int = 30,
        last_topic: str | None = None,
        last_entity_type: str | None = None,
        last_entity_value: str | None = None,
        last_intent: str | None = None,
    ) -> PipelineContext:
        existing = self.load_context(user_id)
        payload = dict(existing.context)
        payload.update(context_updates or {})
        record = self.context_repository.upsert(
            user_id=user_id,
            last_topic=last_topic or existing.last_topic,
            last_entity_type=last_entity_type or existing.last_entity_type,
            last_entity_value=last_entity_value or existing.last_entity_value,
            last_intent=last_intent or existing.last_intent,
            context_json=json.dumps(payload, separators=(',', ':'), sort_keys=True),
            ttl_minutes=ttl_minutes,
        )
        return PipelineContext(
            user_id=user_id,
            last_topic=record.last_topic,
            last_entity_type=record.last_entity_type,
            last_entity_value=record.last_entity_value,
            last_intent=record.last_intent,
            context=payload,
            expires_at=record.expires_at,
        )

    def get_recovery_context(self, user_id: str) -> dict[str, Any]:
        context = self.load_context(user_id)
        payload = context.context.get('recovery_state')
        if not isinstance(payload, dict):
            return {}
        sanitized = self._sanitize_recovery_state(dict(payload))
        if sanitized != payload:
            self.store_recovery_context(
                user_id,
                recovery_updates=sanitized,
                topic=context.last_topic,
                replace=True,
            )
        return sanitized

    def get_active_clarification(self, user_id: str) -> dict[str, Any] | None:
        context = self.load_context(user_id)
        payload = context.context.get('active_clarification')
        if not isinstance(payload, dict):
            return None
        clarification = dict(payload)
        if self._is_expired(clarification.get('expires_at')):
            self.clear_active_clarification(user_id)
            return None
        return clarification

    def store_recovery_context(
        self,
        user_id: str,
        *,
        recovery_updates: dict[str, Any],
        topic: str | None = None,
        ttl_minutes: int = 120,
        replace: bool = False,
    ) -> PipelineContext:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        if replace:
            recovery_state = dict(recovery_updates or {})
        else:
            recovery_state = dict(context.get('recovery_state') or {})
            recovery_state.update(recovery_updates or {})
        context['recovery_state'] = recovery_state
        return self.merge_context(
            user_id,
            context_updates=context,
            ttl_minutes=ttl_minutes,
            last_topic=topic or existing.last_topic,
        )

    def _sanitize_recovery_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = dict(payload or {})
        if self._is_expired(state.get('expires_at')):
            return {}

        pending = state.get('pending_draft')
        if isinstance(pending, dict) and self._is_expired(pending.get('expires_at')):
            state.pop('pending_draft', None)
        elif not isinstance(pending, dict):
            state.pop('pending_draft', None)

        active_thread = state.get('active_thread')
        if isinstance(active_thread, dict):
            thread = dict(active_thread)
            if self._is_expired(thread.get('expires_at')):
                state.pop('active_thread', None)
            else:
                thread['source_turn_ids'] = [
                    str(item).strip()
                    for item in thread.get('source_turn_ids') or []
                    if str(item).strip()
                ][-6:]
                state['active_thread'] = thread
        else:
            state.pop('active_thread', None)

        if 'source_turn_ids' in state and not isinstance(state.get('source_turn_ids'), list):
            state.pop('source_turn_ids', None)
        if 'last_confirmation_options' in state and not isinstance(state.get('last_confirmation_options'), list):
            state.pop('last_confirmation_options', None)
        if 'last_ambiguity' in state and not isinstance(state.get('last_ambiguity'), dict):
            state.pop('last_ambiguity', None)
        return state

    def _is_expired(self, value: Any) -> bool:
        if not value:
            return False
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return utc_now() > parsed.astimezone(timezone.utc)

    def store_active_clarification(
        self,
        user_id: str,
        *,
        clarification: dict[str, Any],
        topic: str | None = None,
        ttl_minutes: int = 20,
    ) -> PipelineContext:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context['active_clarification'] = dict(clarification)
        return self.merge_context(
            user_id,
            context_updates=context,
            ttl_minutes=ttl_minutes,
            last_topic=topic or existing.last_topic,
        )

    def clear_active_clarification(self, user_id: str) -> None:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context.pop('active_clarification', None)
        self.context_repository.upsert(
            user_id=user_id,
            last_topic=existing.last_topic,
            last_entity_type=existing.last_entity_type,
            last_entity_value=existing.last_entity_value,
            last_intent=existing.last_intent,
            context_json=json.dumps(context, separators=(',', ':'), sort_keys=True),
        )

    def store_pending_reminder(self, user_id: str, payload: dict[str, Any]) -> None:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context['pending_reminder'] = payload
        self.context_repository.upsert(
            user_id=user_id,
            last_topic=existing.last_topic,
            last_entity_type='reminder',
            last_entity_value=payload.get('body'),
            last_intent='reminder',
            context_json=json.dumps(context, separators=(',', ':'), sort_keys=True),
            ttl_minutes=30,
        )

    def get_pending_reminder(self, user_id: str) -> dict[str, Any] | None:
        context = self.load_context(user_id)
        pending = context.context.get('pending_reminder')
        if not pending:
            return None
        expires_at = pending.get('expires_at')
        if expires_at:
            try:
                expires_at_value = datetime.fromisoformat(expires_at)
            except ValueError:
                self.clear_pending_reminder(user_id)
                return None
            if expires_at_value.tzinfo is None:
                expires_at_value = expires_at_value.replace(tzinfo=timezone.utc)
            if utc_now() > expires_at_value.astimezone(timezone.utc):
                self.clear_pending_reminder(user_id)
                return None
        return pending

    def clear_pending_reminder(self, user_id: str) -> None:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context.pop('pending_reminder', None)
        self.context_repository.upsert(
            user_id=user_id,
            last_topic=existing.last_topic,
            last_entity_type=existing.last_entity_type,
            last_entity_value=existing.last_entity_value,
            last_intent=existing.last_intent,
            context_json=json.dumps(context, separators=(',', ':'), sort_keys=True),
        )

    def get_duplicate_reminder_audit(self, user_id: str) -> dict[str, Any] | None:
        context = self.load_context(user_id)
        payload = context.context.get('duplicate_reminder_audit')
        if not isinstance(payload, dict):
            return None
        if self._is_expired(payload.get('expires_at')):
            self.clear_duplicate_reminder_audit(user_id)
            return None
        return dict(payload)

    def store_duplicate_reminder_audit(
        self,
        user_id: str,
        *,
        payload: dict[str, Any],
        topic: str | None = None,
        ttl_minutes: int = 30,
    ) -> PipelineContext:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context['duplicate_reminder_audit'] = dict(payload)
        return self.merge_context(
            user_id,
            context_updates=context,
            ttl_minutes=ttl_minutes,
            last_topic=topic or existing.last_topic,
        )

    def clear_duplicate_reminder_audit(self, user_id: str) -> None:
        existing = self.load_context(user_id)
        context = dict(existing.context)
        context.pop('duplicate_reminder_audit', None)
        self.context_repository.upsert(
            user_id=user_id,
            last_topic=existing.last_topic,
            last_entity_type=existing.last_entity_type,
            last_entity_value=existing.last_entity_value,
            last_intent=existing.last_intent,
            context_json=json.dumps(context, separators=(',', ':'), sort_keys=True),
        )
