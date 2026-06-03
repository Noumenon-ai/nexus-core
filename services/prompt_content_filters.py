"""Shared pre-classification filters for user prompts (H2-046 production fix).

Both ``intent_classifier`` and ``destructive_intent_classifier`` historically
matched their keyword sets against the *entire* user prompt. Telegram's PDF
handler folds the full extracted PDF body (up to 30 KB) into the prompt
between explicit delimiters; the lease PDFs that triggered the production
bugs contained "personal", "financial", "signature", "contract" — sensitive-
route + add_signature_to_pdf gate-armed false positives.

The single fix is to strip embedded document blocks BEFORE the classifiers
match. This module owns the regex + delimiter list so the two callers
behave identically; adding a new delimiter (audio transcript, image
caption, etc.) is one change here, not two scattered changes downstream.
"""
from __future__ import annotations

import re


# Each entry: (begin_marker, end_marker). Matched case-insensitively; the
# entire span (markers included) is replaced with a short placeholder so
# the surrounding instruction text stays intact.
_EMBEDDED_BLOCK_PATTERNS: tuple[tuple[str, str], ...] = (
    ('--- BEGIN PDF TEXT ---', '--- END PDF TEXT ---'),
    ('--- BEGIN AUDIO TRANSCRIPT ---', '--- END AUDIO TRANSCRIPT ---'),
    ('--- BEGIN ATTACHMENT ---', '--- END ATTACHMENT ---'),
)

_PLACEHOLDER = '[embedded content removed for classification]'


def strip_embedded_content(prompt: str) -> str:
    """Return ``prompt`` with every recognized embedded-content block
    replaced by a short placeholder.

    Why a placeholder rather than full removal: keeps the surrounding
    sentence structure intact (e.g. the trailing "User did not include a
    caption — give a concise summary of the document." caption from the PDF
    handler still parses naturally) so the classifier can still see the
    user's *instruction* without being polluted by document body text.

    Non-string / empty inputs pass through unchanged so existing call sites
    that hand off whatever the LLM produced don't need extra guards.
    """
    if not prompt or not isinstance(prompt, str):
        return prompt
    out = prompt
    for begin, end in _EMBEDDED_BLOCK_PATTERNS:
        pattern = re.compile(
            re.escape(begin) + r'.*?' + re.escape(end),
            flags=re.DOTALL | re.IGNORECASE,
        )
        out = pattern.sub(_PLACEHOLDER, out)
    return out
