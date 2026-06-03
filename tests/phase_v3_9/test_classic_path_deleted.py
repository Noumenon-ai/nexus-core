"""V3.9 bookend tests — assert classic-path code is gone.

Five negative / source-inspection tests:

1. test_intent_classifier_module_does_not_exist — `from pipeline import
   intent_classifier` must raise ImportError. Catches accidental
   restoration via `git checkout` or new module of the same name.
2. test_assistant_router_module_does_not_exist — same shape for
   pipeline.assistant_router.
3. test_dispatcher_is_only_message_handler — UnifiedPipeline.handle
   has no classic-path branch; behaviorally, no intent_classifier or
   assistant_router are reached when the dispatcher is wired. We
   assert the absence of a `dispatcher_enabled` gate by introspecting
   the dataclass fields and the source.
4. test_unified_pipeline_no_intent_classifier_import — source-inspect
   `pipeline/unified.py` text; assert no `import intent_classifier`
   line of any shape.
5. test_unified_pipeline_no_assistant_router_import — same for
   assistant_router.

These tests are bookends: they MUST currently FAIL (modules and
imports still exist pre-deletion). They MUST PASS after V3.9
deletion + refactor lands. The FAIL → PASS transition IS the
deletion proof — the test file change tracks exactly which commit
made classic-path code structurally gone.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UNIFIED_PATH = _PROJECT_ROOT / 'pipeline' / 'unified.py'


def test_intent_classifier_module_does_not_exist():
    """Negative: importing pipeline.intent_classifier must raise.

    Pre-deletion (today, this turn): FAILS — module exists, import
    succeeds, no exception is raised.
    Post-deletion: PASSES — ImportError raised because the file is
    gone.
    """
    with pytest.raises(ImportError):
        from pipeline import intent_classifier  # noqa: F401


def test_assistant_router_module_does_not_exist():
    """Negative: importing pipeline.assistant_router must raise.

    Same FAIL → PASS bookend semantics as
    test_intent_classifier_module_does_not_exist.
    """
    with pytest.raises(ImportError):
        from pipeline import assistant_router  # noqa: F401


def test_dispatcher_is_only_message_handler():
    """UnifiedPipeline must route every text message through the
    V3 dispatcher unconditionally. No dispatcher_enabled gate, no
    classic-path branch, no intent_classifier / assistant_router
    fields on the dataclass.

    Pre-deletion: FAILS — dataclass has `intent_classifier`,
    `assistant_router`, `dispatcher_enabled` fields.
    Post-V3.9: PASSES — dataclass exposes only dispatcher-side
    dependencies (tool_dispatcher, conversation_service, voice
    services, allowed_telegram_ids).
    """
    from pipeline.unified import UnifiedPipeline

    field_names = {f.name for f in dataclasses.fields(UnifiedPipeline)}

    # Forbidden fields — these are the classic-path wiring slots.
    forbidden = {'intent_classifier', 'assistant_router', 'dispatcher_enabled'}
    leaked = forbidden & field_names
    assert leaked == set(), (
        f'Classic-path UnifiedPipeline fields still present: {sorted(leaked)}. '
        f'V3.9 must remove all three.'
    )


def test_unified_pipeline_no_intent_classifier_import():
    """Source-inspect pipeline/unified.py — no `intent_classifier`
    import line of any shape. Catches the case where someone restores
    classic wiring via a deferred import inside `handle()`.

    Pre-deletion: FAILS — `pipeline/unified.py:38` has
    `from pipeline.tool_dispatcher import DispatcherInput` (fine)
    AND the dataclass at line 18 declares `intent_classifier: Any`
    (fails this test's check on the source text).
    Post-V3.9: PASSES.
    """
    source = _UNIFIED_PATH.read_text(encoding='utf-8')
    assert 'intent_classifier' not in source, (
        'pipeline/unified.py still references intent_classifier. '
        'V3.9 must remove all classic-path wiring.'
    )


def test_unified_pipeline_no_assistant_router_import():
    """Source-inspect pipeline/unified.py — no `assistant_router`
    reference. Same shape as
    test_unified_pipeline_no_intent_classifier_import.

    Pre-deletion: FAILS.
    Post-V3.9: PASSES.
    """
    source = _UNIFIED_PATH.read_text(encoding='utf-8')
    assert 'assistant_router' not in source, (
        'pipeline/unified.py still references assistant_router. '
        'V3.9 must remove all classic-path wiring.'
    )
