from __future__ import annotations

from dataclasses import dataclass

from pipeline.types import IntentResult
from utils.safety_rules import is_sensitive_action


@dataclass(frozen=True, slots=True)
class SafetyAssessment:
    requires_approval: bool
    action_type: str | None = None


class SafetyGate:
    def assess(self, intent: IntentResult, text: str) -> SafetyAssessment:
        normalized = text.strip().lower()
        if intent.intent_type == 'action':
            if normalized.startswith('send ') or normalized.startswith('broadcast '):
                return SafetyAssessment(requires_approval=True, action_type='outbound_message')
            if normalized.startswith('delete '):
                return SafetyAssessment(requires_approval=True, action_type='external_state_change')
        if intent.intent_type == 'memory' and normalized.startswith('forget '):
            return SafetyAssessment(requires_approval=True, action_type='delete_memory')
        return SafetyAssessment(requires_approval=False, action_type=None)
