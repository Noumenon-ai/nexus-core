from __future__ import annotations

from services.human_confirmation_style import HumanConfirmationStyle
from services.vague_clarification import build_vague_clarification


def test_renders_rental_status_confirmation():
    style = HumanConfirmationStyle()

    text = style.render_natural_confirmation(
        recovered_intent='check rental update status for your 3 rental records',
        confidence=0.84,
        risk_level='low',
        resolved_slots={
            'action_kind': 'rental_status_check',
            'rental_subject': 'your 3 rental records',
        },
    )

    assert text == (
        "You mean checking whether your 3 rental records were updated. "
        "I'll check what I can see."
    )


def test_renders_negated_recipient_clarification():
    style = HumanConfirmationStyle()

    text = style.render_specific_clarification(
        recovered_intent='send message to Acme Corp: i will do it Friday',
        confidence=0.2,
        risk_level='high',
        missing_slot='recipient',
        resolved_slots={
            'negated_recipient_label': 'Acme Corp',
            'negated_recipient_reason': 'other_one',
        },
    )

    assert text == 'You said not Acme Corp — who should I send it to?'


def test_followup_vague_prompt_uses_human_confirmation_text():
    assert (
        build_vague_clarification('follow up with her tomorrow')
        == 'Who should I follow up with, and what is it about?'
    )


def test_renders_partial_success_safe_reminder_text():
    style = HumanConfirmationStyle()

    text = style.render_partial_success(
        fallback_result={
            'created': True,
            'time_label': 'Friday',
            'target_name': 'Mike',
            'issue_summary': 'water damage',
            'unit_reference': 'Unit 204',
        },
    )

    assert text == (
        "I couldn't finish the send path, but I created the safe reminder part: "
        'Friday follow-up for Mike about water damage in Unit 204. '
        'No message was sent.'
    )
