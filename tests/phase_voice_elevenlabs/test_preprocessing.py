from __future__ import annotations

from services.voice_preprocessing import preprocess_english, preprocess_hebrew


def test_english_preprocessing_strips_markdown():
    assert preprocess_english('## **Hello** `world` [link](https://example.com)') == 'Hello world link'


def test_english_preprocessing_converts_times():
    assert preprocess_english('Call me at 3:00 PM and again at 9 AM') == 'Call me at three PM and again at nine AM'


def test_english_preprocessing_converts_currency():
    assert preprocess_english('It costs $5 or ₪50') == 'It costs five dollars or fifty shekels'


def test_hebrew_preprocessing_keeps_loanwords_latin():
    processed = preprocess_hebrew('בדוק את Google Calendar ואת WhatsApp מחר')
    assert 'Google Calendar' in processed
    assert 'WhatsApp' in processed


def test_hebrew_preprocessing_strips_geresh():
    assert preprocess_hebrew('ג׳רוויס אמר "שלום"') == 'ג\'רוויס אמר "שלום"'


def test_hebrew_preprocessing_no_niqqud_added():
    processed = preprocess_hebrew('שלום עולם')
    assert processed == 'שלום עולם'


def test_emoji_stripped_both_languages():
    assert preprocess_english('Hello 😊 world') == 'Hello world'
    assert preprocess_hebrew('שלום 😊 עולם') == 'שלום עולם'


def test_preprocess_date_conversion():
    en = preprocess_english('Meeting on 2026-04-30')
    assert 'April thirtieth' in en
    assert '2026-04-30' not in en

    he_iso = preprocess_hebrew('פגישה ב 2026-04-30 בבוקר')
    assert '30 באפריל 2026' in he_iso
    assert '2026-04-30' not in he_iso

    he_native = preprocess_hebrew('פגישה 30 באפריל 2026 בבוקר')
    assert '30 באפריל 2026' in he_native
