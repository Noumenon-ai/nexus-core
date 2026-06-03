"""V3.8 TELOS onboarding tools — four @tool functions that drive the
guided conversation flow for users without a TELOS file.

The four tools:
    start_telos_onboarding   — begin or resume the flow
    answer_telos_question    — record one answer, advance section
    view_my_telos            — read the user's current TELOS file
    cancel_telos_onboarding  — pause without losing state

State lives in `telos_onboarding_state` (one row per user, V3.8
schema). Question content lives in
`resources/telos_onboarding_questions.json`. TELOS file writes go
through `TelosService.append`, which already chmods `0o600` on every
append — V3.8 halt condition #1 (mode 600 preservation) is intrinsic
to that path, not a per-tool concern. The `test_telos_file_mode_600_
preserved_after_append` test verifies this end-to-end.

Hebrew users with a missing `he` section in the JSON get an English
fallback PLUS the `HEBREW_FALLBACK_NOTE` prepended to their reply
(V3.8 halt condition #6). See `services/telos_onboarding_content.py`
for the fallback logic.
"""
from __future__ import annotations

from typing import Any, Callable

from repositories.telos_onboarding_state_repository import (
    TelosOnboardingStateRepository,
)
from repositories.users_repository import UsersRepository
from services.telos_onboarding_content import (
    HEBREW_FALLBACK_NOTE,
    find_next_unanswered_question,
    get_section,
    is_section_complete,
    next_section_name,
    section_order,
)
from services.telos_service import TelosService
from services.tool_registry import ToolRegistry, ToolResult, ToolSpec


_PARAMS_USER_SCOPED: dict[str, Any] = {
    'type': 'object',
    'properties': {},
    'required': [],
}

_PARAMS_ANSWER: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'answer': {
            'type': 'string',
            'description': "User's answer to the current TELOS onboarding question.",
        },
    },
    'required': ['answer'],
}


def _user_language(users_repository: UsersRepository, user_id: str) -> str:
    user = users_repository.get_by_id(user_id)
    if user is None:
        return 'en'
    return user.language or 'en'


def _maybe_prepend_fallback_note(text: str, used_fallback: bool) -> str:
    if used_fallback:
        return f'{HEBREW_FALLBACK_NOTE}\n\n{text}'
    return text


def _format_question_prompt(
    section_data: dict[str, Any], question: dict[str, Any], used_fallback: bool,
) -> str:
    intro = section_data.get('intro', '').strip()
    prompt = question.get('prompt', '').strip()
    example = question.get('example', '').strip()
    parts = []
    if intro:
        parts.append(intro)
    parts.append(prompt)
    if example:
        parts.append(f'(Example: {example})')
    parts.append("Reply with your answer, or say 'stop telos' to pause.")
    body = '\n\n'.join(parts)
    return _maybe_prepend_fallback_note(body, used_fallback)


def _render_section_for_telos_file(
    section_name: str, section_data: dict[str, Any], answers: dict[str, str],
) -> str:
    """Build the markdown block that gets appended to the user's TELOS
    file when a section completes. Section header + each question's
    prompt as a sub-header + the user's answer underneath. The user's
    answers stay in their own words and language — we never translate
    or normalize them, which is correct: TELOS is the user's voice."""
    intro = section_data.get('intro', '').strip()
    questions = section_data.get('questions') or []
    lines: list[str] = []
    # Friendly title-case for the section name. e.g. 'decision_rules' → 'Decision Rules'.
    title = section_name.replace('_', ' ').title()
    lines.append(f'\n## {title}')
    if intro:
        lines.append(intro)
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get('id')
        prompt = q.get('prompt', '').strip()
        if not qid or qid not in answers:
            continue
        lines.append(f'\n### {prompt}')
        lines.append(answers[qid])
    lines.append('')  # trailing newline
    return '\n'.join(lines)


def make_telos_onboarding_tools(
    *,
    telos_service: TelosService,
    onboarding_repository: TelosOnboardingStateRepository,
    users_repository: UsersRepository,
) -> list[tuple[Callable[..., ToolResult], dict[str, Any]]]:
    """Build the four onboarding-tool closures bound to the given deps.

    Tools take `user_id: str` (matching every other V3.x tool —
    dispatcher injects this kwarg). User language is resolved per-call
    from `users_repository`.
    """

    def start_telos_onboarding(*, user_id: str) -> ToolResult:
        language = _user_language(users_repository, user_id)
        state = onboarding_repository.get_or_create(user_id)
        if state.completed_at is not None:
            return ToolResult.ok(
                announcement=(
                    "Your TELOS is already complete. To update it, edit the file directly "
                    "or use 'append to telos' to add an Updates entry."
                ),
                data={'status': 'already_complete'},
            )
        # Resuming a paused or fresh flow. Mark started_at if NULL,
        # clear cancelled_at if set.
        onboarding_repository.mark_started(user_id)
        if state.cancelled_at is not None:
            onboarding_repository.clear_cancelled(user_id)
        # Load the current section's content (with Hebrew fallback).
        section_data, used_fallback = get_section(language, state.current_section)
        import json
        try:
            answers = json.loads(state.answers_so_far) if state.answers_so_far else {}
        except json.JSONDecodeError:
            answers = {}
        next_q = find_next_unanswered_question(section_data, answers)
        if next_q is None:
            # Edge: section in state was already fully answered (shouldn't
            # happen if save_answer + advance_section ran correctly, but
            # be robust). Advance to next section here.
            advance_to = next_section_name(state.current_section)
            if advance_to is None:
                onboarding_repository.mark_completed(user_id)
                return ToolResult.ok(
                    announcement=_maybe_prepend_fallback_note(
                        'TELOS onboarding complete. I will use what you wrote to shape my replies.',
                        used_fallback,
                    ),
                    data={'status': 'completed'},
                )
            onboarding_repository.advance_section(user_id, next_section=advance_to)
            section_data, used_fallback = get_section(language, advance_to)
            next_q = find_next_unanswered_question(section_data, answers)
            if next_q is None:
                # Section has zero questions — content bug. Treat as complete.
                onboarding_repository.mark_completed(user_id)
                return ToolResult.ok(
                    announcement='TELOS onboarding complete.',
                    data={'status': 'completed'},
                )
        return ToolResult.ok(
            announcement=_format_question_prompt(section_data, next_q, used_fallback),
            data={
                'status': 'awaiting_answer',
                'section': onboarding_repository.get(user_id).current_section,
                'awaiting_answer_to': next_q['id'],
            },
        )

    def answer_telos_question(answer: str, *, user_id: str) -> ToolResult:
        if not isinstance(answer, str) or not answer.strip():
            return ToolResult.ok(
                announcement='Please send your answer as a non-empty message.',
                data={'status': 'empty_answer'},
            )
        language = _user_language(users_repository, user_id)
        state = onboarding_repository.get(user_id)
        if state is None or state.started_at is None or state.completed_at is not None:
            return ToolResult.ok(
                announcement="No active onboarding. Say 'start telos' to begin.",
                data={'status': 'no_active_flow'},
            )
        # Compute which question is next-unanswered in the current section.
        # The schema doesn't carry an `awaiting_question_id` column (V3.8
        # audit decision: compute on read, simpler than schema bookkeeping).
        section_data, used_fallback = get_section(language, state.current_section)
        import json
        try:
            answers_before = json.loads(state.answers_so_far) if state.answers_so_far else {}
        except json.JSONDecodeError:
            answers_before = {}
        target_question = find_next_unanswered_question(section_data, answers_before)
        if target_question is None:
            # Should not happen — start_telos_onboarding advances past
            # fully-answered sections — but be defensive.
            return ToolResult.ok(
                announcement="Section already complete. Send 'start telos' to continue.",
                data={'status': 'section_already_complete'},
            )
        # Record the answer.
        answers_after = onboarding_repository.record_answer(
            user_id, question_id=target_question['id'], answer=answer.strip(),
        )
        # Section complete? Append to TELOS file and advance.
        if is_section_complete(section_data, answers_after):
            section_md = _render_section_for_telos_file(
                state.current_section, section_data, answers_after,
            )
            telos_service.append(user_id, section_md)
            advance_to = next_section_name(state.current_section)
            if advance_to is None:
                onboarding_repository.mark_completed(user_id)
                return ToolResult.ok(
                    announcement=_maybe_prepend_fallback_note(
                        (
                            'TELOS onboarding complete. I will use what you wrote to '
                            'shape my replies. You can update it anytime by editing '
                            f'the file at .data/telos/{user_id}.md or saying '
                            "'append to telos'."
                        ),
                        used_fallback,
                    ),
                    data={'status': 'completed'},
                )
            onboarding_repository.advance_section(user_id, next_section=advance_to)
            # Show first question of next section.
            next_section_data, next_fallback = get_section(language, advance_to)
            next_q = find_next_unanswered_question(next_section_data, answers_after)
            if next_q is None:
                # Empty section — content bug. Skip onward.
                onboarding_repository.mark_completed(user_id)
                return ToolResult.ok(
                    announcement='TELOS onboarding complete.',
                    data={'status': 'completed'},
                )
            return ToolResult.ok(
                announcement=_format_question_prompt(next_section_data, next_q, next_fallback),
                data={
                    'status': 'awaiting_answer',
                    'section': advance_to,
                    'awaiting_answer_to': next_q['id'],
                },
            )
        # Section not complete — show next question in the same section.
        next_q = find_next_unanswered_question(section_data, answers_after)
        if next_q is None:
            # Defensive: shouldn't reach here because is_section_complete
            # would have been True. Treat as section-complete edge.
            return ToolResult.ok(
                announcement='Section complete; continue with next.',
                data={'status': 'section_advanced'},
            )
        return ToolResult.ok(
            announcement=_format_question_prompt(section_data, next_q, used_fallback),
            data={
                'status': 'awaiting_answer',
                'section': state.current_section,
                'awaiting_answer_to': next_q['id'],
            },
        )

    def view_my_telos(*, user_id: str) -> ToolResult:
        content = telos_service.load(user_id)
        if not content:
            return ToolResult.ok(
                announcement="No TELOS file yet. Say 'start telos' to build one.",
                data={'present': False},
            )
        return ToolResult.ok(
            announcement=content,
            data={'present': True, 'content': content},
        )

    def cancel_telos_onboarding(*, user_id: str) -> ToolResult:
        state = onboarding_repository.get(user_id)
        if state is None or state.started_at is None:
            return ToolResult.ok(
                announcement="No active onboarding to pause.",
                data={'status': 'no_active_flow'},
            )
        if state.completed_at is not None:
            return ToolResult.ok(
                announcement='Onboarding is already complete; nothing to pause.',
                data={'status': 'already_complete'},
            )
        onboarding_repository.mark_cancelled(user_id)
        return ToolResult.ok(
            announcement=(
                "Paused. Your progress is saved. Say 'start telos' anytime "
                "to pick up where you left off."
            ),
            data={'status': 'cancelled', 'section': state.current_section},
        )

    return [
        (
            start_telos_onboarding,
            {
                'name': 'start_telos_onboarding',
                'description': 'Begin or resume the guided TELOS onboarding flow for the user.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            answer_telos_question,
            {
                'name': 'answer_telos_question',
                'description': "Record the user's answer to the current TELOS onboarding question and advance the flow.",
                'parameters': _PARAMS_ANSWER,
            },
        ),
        (
            view_my_telos,
            {
                'name': 'view_my_telos',
                'description': "Read the user's current TELOS file content (or report none yet).",
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
        (
            cancel_telos_onboarding,
            {
                'name': 'cancel_telos_onboarding',
                'description': 'Pause TELOS onboarding without losing answers; resume preserves the current section.',
                'parameters': _PARAMS_USER_SCOPED,
            },
        ),
    ]


def register_telos_onboarding_tools(
    registry: ToolRegistry,
    *,
    telos_service: TelosService,
    onboarding_repository: TelosOnboardingStateRepository,
    users_repository: UsersRepository,
) -> list[ToolSpec]:
    """Register all four V3.8 TELOS onboarding tools into the given registry."""
    specs: list[ToolSpec] = []
    for fn, meta in make_telos_onboarding_tools(
        telos_service=telos_service,
        onboarding_repository=onboarding_repository,
        users_repository=users_repository,
    ):
        specs.append(registry.register(fn, **meta))
    return specs
