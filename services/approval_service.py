from __future__ import annotations

import json
from datetime import timedelta
from typing import Callable

from models import User, now_utc
from pipeline.types import InlineButton, ServiceResponse
from repositories.approvals_repository import ApprovalsRepository
from utils.i18n import Translator


class ApprovalService:
    def __init__(self, approvals_repository: ApprovalsRepository, *, message_ttl_minutes: int, system_ttl_minutes: int) -> None:
        self.approvals_repository = approvals_repository
        self.message_ttl_minutes = message_ttl_minutes
        self.system_ttl_minutes = system_ttl_minutes

    def request(self, user: User, *, action_type: str, preview_text: str, payload: dict, translator: Translator) -> ServiceResponse:
        ttl_minutes = self.message_ttl_minutes if action_type == 'outbound_message' else self.system_ttl_minutes
        approval = self.approvals_repository.create(user_id=user.id, action_type=action_type, preview_text=preview_text, payload_json=json.dumps(payload, separators=(',', ':'), sort_keys=True), expires_at=now_utc() + timedelta(minutes=ttl_minutes))
        return ServiceResponse(text=translator.t('approval_prompt', preview=preview_text), buttons=[InlineButton(text=translator.t('approval_button_approve'), callback_data=f'approval:approve:{approval.id}'), InlineButton(text=translator.t('approval_button_cancel'), callback_data=f'approval:cancel:{approval.id}')])

    def cancel(self, approval_id: str, acting_user_id: str, translator: Translator) -> ServiceResponse:
        approval = self.approvals_repository.get(approval_id)
        if approval is None:
            return ServiceResponse(text=translator.t('approval_cancelled'))
        if approval.user_id != acting_user_id:
            return ServiceResponse(text=translator.t('security_unauthorized'))
        if approval.status != 'pending':
            return ServiceResponse(text=translator.t('approval_status', status=approval.status))
        if approval.expires_at <= now_utc():
            self.approvals_repository.update_status(approval_id, status='expired')
            return ServiceResponse(text=translator.t('approval_already_expired'))
        self.approvals_repository.update_status(approval_id, status='cancelled', cancelled_at=now_utc())
        return ServiceResponse(text=translator.t('approval_cancelled'))

    def execute(self, approval_id: str, acting_user_id: str, executor: Callable[[str, dict], ServiceResponse], translator: Translator) -> ServiceResponse:
        current = now_utc()
        approval = self.approvals_repository.claim_for_execution(approval_id, claimed_at=current)
        if approval is None:
            return ServiceResponse(text=translator.t('approval_already_expired'))
        if approval.user_id != acting_user_id:
            if approval.status == 'approved':
                self.approvals_repository.update_status(approval_id, status='pending', approved_at=None)
            return ServiceResponse(text=translator.t('security_unauthorized'))
        if approval.status == 'pending' and approval.expires_at <= current:
            self.approvals_repository.update_status(approval_id, status='expired')
            return ServiceResponse(text=translator.t('approval_already_expired'))
        if approval.status == 'expired':
            return ServiceResponse(text=translator.t('approval_already_expired'))
        if approval.status != 'approved':
            return ServiceResponse(text=translator.t('approval_status', status=approval.status))
        payload = json.loads(approval.payload_json)
        response = executor(approval.action_type, payload)
        self.approvals_repository.mark_executed(approval_id, executed_at=now_utc())
        return response

    def sweep_expired(self, notify: Callable[[str, str], None], translator: Translator) -> int:
        expired = self.approvals_repository.list_pending_expired()
        for approval in expired:
            self.approvals_repository.update_status(approval.id, status='expired')
            notify(approval.user_id, translator.t('approval_expired', action_type=approval.action_type))
        return len(expired)
