from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StructuredFailure:
    route: str
    stage: str
    provider: str
    fallback: str
    root_reason: str
    safe_action_taken: str
    unsafe_action_blocked: str
    technical_reason: str | None = None

    def to_metadata(self) -> dict[str, str]:
        data = asdict(self)
        return {key: str(value) for key, value in data.items() if value not in (None, '')}


def render_user_facing_failure(
    failure: StructuredFailure,
    *,
    default_text: str | None = None,
) -> str:
    if default_text:
        return default_text

    if failure.route == 'contact_send':
        if failure.root_reason == 'unresolved_recipient':
            return (
                "I couldn't finish the send path because the recipient was still "
                "unclear. I did not send anything."
            )
        if failure.root_reason == 'provider_unavailable':
            return (
                "I couldn't finish the send path because I couldn't reach the "
                'provider. I did not send anything.'
            )
        return (
            "I couldn't finish the send path before the provider timed out. "
            'I did not send anything.'
        )

    if failure.route == 'contact_reminder':
        if failure.root_reason == 'provider_unavailable':
            return (
                "I couldn't finish the reminder path because I couldn't reach "
                "the provider. I didn't schedule or send anything."
            )
        return (
            "I couldn't finish the reminder path before the provider timed out. "
            "I didn't schedule or send anything."
        )

    if failure.route == 'cleanup':
        if failure.root_reason == 'provider_unavailable':
            return (
                "I couldn't finish the cleanup path because I couldn't reach "
                "the provider. I didn't change anything."
            )
        return (
            "I couldn't finish the cleanup path before the provider timed out. "
            "I didn't change anything."
        )

    if failure.route == 'follow_up':
        if failure.root_reason == 'provider_unavailable':
            return (
                "I couldn't finish the follow-up path because I couldn't reach "
                'the provider. I did not send anything.'
            )
        return (
            "I couldn't finish the follow-up path before the provider timed out. "
            'I did not send anything.'
        )

    if failure.root_reason == 'provider_unavailable':
        return (
            "I couldn't finish that approved action because I couldn't reach "
            "the provider. I didn't change anything."
        )
    return (
        "I couldn't finish that approved action before the provider timed out. "
        "I didn't change anything."
    )
