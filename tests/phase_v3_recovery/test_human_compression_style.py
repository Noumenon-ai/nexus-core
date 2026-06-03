from __future__ import annotations

from services.human_confirmation_style import HumanConfirmationStyle


def _sentence_count(text: str) -> int:
    return sum(text.count(mark) for mark in '.!?')


def test_birthday_confirmation_is_short_and_human():
    style = HumanConfirmationStyle()

    text = style.render_birthday_memory_confirmation(date_label='May 27')

    assert text == (
        "Got it — tomorrow's your birthday, May 27. "
        "Want me to remember that for next year? "
        "I can wish you tomorrow without saving it permanently."
    )
    assert _sentence_count(text) <= 3


def test_birthday_noncelebration_reply_is_short_and_specific():
    style = HumanConfirmationStyle()

    text = style.render_birthday_wish_reply()

    assert text == (
        "Haha got it — I won't treat it like a celebration. "
        "I'll still wish you happy birthday tomorrow."
    )
    assert _sentence_count(text) <= 3


def test_contact_reply_is_compressed_and_has_no_fake_affection():
    style = HumanConfirmationStyle()

    text = style.compress_reply(
        text='your wife is sam. anything you want me to do for her? 💕'
    )

    assert text == 'Got it — Sam. What should I tell her?'
    assert '💕' not in text
    assert _sentence_count(text) <= 3


def test_role_prefix_is_removed_from_reply_text():
    style = HumanConfirmationStyle()

    assert style.compress_reply(text='User: yes') == 'yes'
