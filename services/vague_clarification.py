from __future__ import annotations

import re

from services.human_confirmation_style import HumanConfirmationStyle


_HUMAN_CONFIRMATION_STYLE = HumanConfirmationStyle()


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _rental_status_subject(lowered: str) -> str:
    match = re.search(r"\b(?:my|the)\s+(\d+)\s+rentals?\b", lowered)
    if match:
        return f"your {match.group(1)} rental records"
    return "your rental records"


def _looks_like_ambiguous_rental_update(lowered: str) -> bool:
    if "rental" not in lowered:
        return False

    if lowered.startswith("did you update"):
        return True
    if lowered.startswith("were ") and "updated" in lowered:
        return True
    if lowered.startswith("check if") and "updated" in lowered:
        return True
    if lowered.startswith("check whether") and "updated" in lowered:
        return True

    if lowered.startswith("update my rental"):
        return not any(token in lowered for token in (" now", " to ", " as ", " with "))

    return False


def _build_rental_status_clarification(lowered: str) -> str:
    subject = _rental_status_subject(lowered)
    return (
        "Do you mean:\n"
        f"1. check whether {subject} were updated,\n"
        "2. update the rental records now,\n"
        "3. send updates about the rentals,\n"
        "or something else?"
    )


def build_vague_clarification(text: str) -> str | None:
    """Return a targeted clarification for vague prompts.

    The goal is to ask for the missing slot directly instead of falling back
    to a generic "what do you mean?" prompt. Returning None means the caller
    should continue with its normal path.
    """
    lowered = _normalize(text)
    if not lowered:
        return None

    if _looks_like_ambiguous_rental_update(lowered):
        return _build_rental_status_clarification(lowered)

    if "follow up" in lowered and any(token in lowered for token in (" her", " him", " them")):
        return _HUMAN_CONFIRMATION_STYLE.render_specific_clarification(
            recovered_intent=text,
            confidence=0.0,
            risk_level='medium',
            missing_slot='follow_up_target_and_topic',
        )

    if "vendor" in lowered and ("send" in lowered or "message" in lowered):
        if "leak" in lowered:
            return "Which vendor should I contact about the leak, and which tenant or unit is this for?"
        return "Which vendor should I contact, and what exactly should I send them?"

    if any(
        phrase in lowered
        for phrase in (
            "tell him",
            "tell her",
            "message him",
            "message her",
            "send him",
            "send her",
            "send them",
        )
    ):
        return "Who should I message, and what should I say you will check or update?"

    if "check" in lowered and any(
        phrase in lowered
        for phrase in (" handled this", " this", "that thing", "this thing")
    ):
        return "What item or case should I check, and which tenant or unit is it about?"

    if any(phrase in lowered for phrase in ("thing from yesterday", "that thing", "this thing")):
        return "Which person or item do you mean, and which tenant or unit is it about?"

    if "deal with" in lowered and "tenant issue" in lowered:
        return "Which tenant issue do you mean, and what action do you want me to take?"

    if "call someone" in lowered or ("someone" in lowered and "repair" in lowered):
        return "Who should I call, and what repair or issue is this about?"

    return None
