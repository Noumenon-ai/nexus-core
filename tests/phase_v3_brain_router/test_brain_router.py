"""Tests for services.brain_router — Phase 2."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path as FilePath
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.fallback_manager import PROVIDER_UNAVAILABLE_USER_TEXT
from services.brain_router import BrainResult, BrainRouter
from services.intent_classifier import Path, Provider


def _make_completed(stdout: str, returncode: int = 0, stderr: str = '') -> MagicMock:
    """Mock for subprocess.Popen. H2-040: migrated from subprocess.run mock to
    Popen mock so shutdown drain can track + terminate in-flight subprocesses."""
    m = MagicMock()
    m.returncode = returncode
    m.communicate.return_value = (stdout.encode(), stderr.encode())
    return m


def _make_timeout_popen(cmd: str = 'claude', timeout: int = 60) -> MagicMock:
    """Mock Popen whose first communicate() raises TimeoutExpired and second
    communicate() (called after .kill()) returns empty output. Mirrors the
    real subprocess.Popen contract used by BrainRouter._run_subprocess_sync."""
    m = MagicMock()
    m.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=cmd, timeout=timeout),
        (b'', b''),
    ]
    m.returncode = -9
    return m


@pytest.mark.asyncio
async def test_conversational_routes_to_claude_returns_text():
    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed('hello back')):
        result = await router.generate('hello')
    assert result.text == 'hello back'
    assert result.tool_calls is None
    assert result.provider_used == 'claude'
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_tool_likely_parses_json_tool_calls():
    router = BrainRouter()
    json_output = '{"tool_calls": [{"name": "create_reminder", "arguments": {"body": "call bank", "next_fire_at": "2026-05-11T12:00:00Z"}}]}'
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(json_output)):
        result = await router.generate(
            'remind me at noon to call the bank',
            tool_catalog=[{'name': 'create_reminder', 'description': 'create reminder', 'parameters': {}}],
        )
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]['name'] == 'create_reminder'
    assert result.tool_calls[0]['arguments']['body'] == 'call bank'


@pytest.mark.asyncio
async def test_tool_likely_falls_back_to_text_when_no_json():
    router = BrainRouter()
    text_output = "I'll set that reminder for you right now."
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(text_output)):
        result = await router.generate(
            'remind me at noon to call the bank',
            tool_catalog=[{'name': 'create_reminder', 'description': 'create reminder', 'parameters': {}}],
        )
    assert result.text == text_output
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_tool_call_parser_handles_fenced_json():
    router = BrainRouter()
    fenced = '```json\n{"tool_calls": [{"name": "list_calendar_events", "arguments": {}}]}\n```'
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(fenced)):
        result = await router.generate(
            'what is on my calendar tomorrow',
            tool_catalog=[{'name': 'list_calendar_events', 'description': 'list events', 'parameters': {}}],
        )
    assert result.tool_calls is not None
    assert result.tool_calls[0]['name'] == 'list_calendar_events'


@pytest.mark.asyncio
async def test_code_query_routes_to_codex_strips_banner():
    router = BrainRouter()
    codex_output = "codex 0.120.0\n-----\n\nUse a list comprehension here.\n\ntokens used: 42"
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(codex_output)):
        result = await router.generate('refactor this function with a list comprehension')
    assert result.provider_used == 'codex'
    assert 'list comprehension' in (result.text or '')
    assert 'tokens used' not in (result.text or '')


@pytest.mark.asyncio
async def test_subprocess_uses_devnull_stdin():
    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed('ok')) as mock_run:
        await router.generate('hello')
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs.get('stdin') == subprocess.DEVNULL


@pytest.mark.asyncio
async def test_claude_failure_falls_back_to_codex_without_gemini():
    """Claude failure should use Codex as the hosted second-chance path.
    Gemini remains retired and must not be invoked."""
    fallback = MagicMock()
    fallback.generate_text = MagicMock()  # MUST NOT be called

    router = BrainRouter(gemini_fallback=fallback)  # param accepted but ignored
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(return_value='codex reply')):
        result = await router.generate('hello')
    assert result.fallback_used is True
    assert result.provider_used == 'codex'
    assert result.text == 'codex reply'
    assert not fallback.generate_text.called, (
        'gemini_fallback.generate_text was invoked — H2-047 Wave 2 retired it'
    )


@pytest.mark.asyncio
async def test_claude_failure_retries_once_before_falling_back_to_codex():
    router = BrainRouter()
    with patch.object(
        router,
        '_invoke_claude',
        AsyncMock(
            side_effect=[
                subprocess.TimeoutExpired(cmd='claude', timeout=60),
                subprocess.TimeoutExpired(cmd='claude', timeout=60),
            ]
        ),
    ) as mock_claude, patch.object(
        router,
        '_sleep_before_claude_retry',
        AsyncMock(return_value=None),
    ) as mock_sleep, patch.object(
        router,
        '_invoke_codex',
        AsyncMock(return_value='codex reply'),
    ) as mock_codex:
        result = await router.generate('hello')
    assert result.fallback_used is True
    assert result.provider_used == 'codex'
    assert result.text == 'codex reply'
    assert mock_claude.await_count == 2
    mock_sleep.assert_awaited_once()
    mock_codex.assert_awaited_once()


@pytest.mark.asyncio
async def test_subprocess_failure_no_fallback_returns_clean_user_text():
    """If both hosted providers fail, the user still gets the clean canned
    message — never the raw subprocess exception."""
    router = BrainRouter()
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='codex', timeout=120))):
        result = await router.generate('hello')
    assert result.fallback_used is True
    assert result.provider_used == 'none'
    text = result.text or ''
    assert text == PROVIDER_UNAVAILABLE_USER_TEXT
    assert 'brain_router error' not in text, (
        f'raw exception template leaked to user: {text!r}'
    )


@pytest.mark.asyncio
async def test_breaker_opens_after_three_consecutive_failures():
    router = BrainRouter(breaker_failure_threshold=3, breaker_cooldown_sec=300)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=RuntimeError('claude unavailable'))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=RuntimeError('codex unavailable'))):
        for _ in range(3):
            await router.generate('hello')
    state = router._breakers.get('claude')
    assert state is not None
    assert state.consecutive_failures >= 3
    assert state.open_until_ts > 0


@pytest.mark.asyncio
async def test_breaker_resets_on_success():
    router = BrainRouter()
    router._breakers['claude'] = router._breakers.get('claude', None) or __import__(
        'services.brain_router', fromlist=['_BreakerState']
    )._BreakerState(consecutive_failures=2, open_until_ts=0.0)
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed('ok')):
        await router.generate('hello')
    state = router._breakers['claude']
    assert state.consecutive_failures == 0


@pytest.mark.asyncio
async def test_nonzero_returncode_raises_then_falls_back():
    """H2-046: nonzero subprocess exit code propagates through to _fallback
    which emits the clean canned message. Raw stderr / exit-code details
    go to journalctl only."""
    router = BrainRouter()
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=RuntimeError('subprocess claude returned 1: auth error'))), \
         patch.object(router, '_invoke_codex', AsyncMock(side_effect=RuntimeError('subprocess codex returned 1: auth error'))):
        result = await router.generate('hello')
    assert result.fallback_used is True
    text = result.text or ''
    assert text == PROVIDER_UNAVAILABLE_USER_TEXT
    assert 'returned 1' not in text  # raw subprocess detail must NOT leak
    assert 'auth error' not in text   # stderr must NOT leak


@pytest.mark.asyncio
async def test_sensitive_routes_to_ollama():
    router = BrainRouter(ollama_url='http://localhost:11434')
    fake_response = json.dumps({'response': 'ollama reply'}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda *a: None
    mock_resp.read = lambda: fake_response

    with patch('urllib.request.urlopen', return_value=mock_resp):
        result = await router.generate('this is my private medical journal')
    assert result.provider_used == 'ollama'
    assert result.text == 'ollama reply'


@pytest.mark.asyncio
async def test_tool_call_parser_rejects_malformed_json():
    router = BrainRouter()
    bad = '{"tool_calls": "not_a_list"}'
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(bad)):
        result = await router.generate(
            'remind me at noon',
            tool_catalog=[{'name': 'create_reminder', 'description': 'x', 'parameters': {}}],
        )
    assert result.tool_calls is None
    assert result.text == bad



@pytest.mark.asyncio
async def test_generate_with_tools_returns_dispatcher_dict_shape():
    """LLMProtocol-compatible entrypoint returns {text} or {tool_calls}."""
    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed('plain text reply')):
        result = await router.generate_with_tools(
            user_id='u1',
            system_prompt='you are nexus',
            contents=[{'role': 'user', 'parts': [{'text': 'hello'}]}],
            tool_catalog=[],
        )
    assert isinstance(result, dict)
    assert result.get('text') == 'plain text reply'
    assert 'tool_calls' not in result


@pytest.mark.asyncio
async def test_generate_with_tools_returns_tool_calls_for_structured_output():
    router = BrainRouter()
    json_out = '{"tool_calls": [{"name": "create_reminder", "arguments": {"body": "x", "next_fire_at": "2026-05-11T12:00:00Z"}}]}'
    with patch('services.brain_router.subprocess.Popen', return_value=_make_completed(json_out)):
        result = await router.generate_with_tools(
            user_id='u1',
            system_prompt='you are nexus',
            contents=[{'role': 'user', 'parts': [{'text': 'remind me at noon'}]}],
            tool_catalog=[{'name': 'create_reminder', 'description': 'r', 'parameters': {}}],
        )
    assert 'tool_calls' in result
    assert result['tool_calls'][0]['name'] == 'create_reminder'


@pytest.mark.asyncio
async def test_generate_with_tools_passes_full_history_to_claude():
    """H2-042 fix (replaces an earlier assertion that locked in the bug). The
    Phase 2.5a build added a `_extract_last_user_text` collapse that dropped
    every prior turn before invoking Claude. The pre-H2-042 test asserted
    that behavior — making it a regression-locking test, not a contract
    test. The fixed contract: every prior turn that arrived in `contents`
    must reach Claude's prompt, so pronoun resolution works."""
    router = BrainRouter()
    captured = {}
    def _capture(cmd, **kw):
        captured['cmd'] = cmd
        return _make_completed('ok')
    with patch('services.brain_router.subprocess.Popen', side_effect=_capture):
        await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=[
                {'role': 'user', 'parts': [{'text': 'first message'}]},
                {'role': 'model', 'parts': [{'text': 'reply'}]},
                {'role': 'user', 'parts': [{'text': 'latest message'}]},
            ],
            tool_catalog=[],
        )
    final_arg = captured['cmd'][-1]  # the `-p` payload
    assert 'first message' in final_arg, (
        f'turn 1 missing from Claude prompt — H2-042 regression. Got: {final_arg!r}'
    )
    assert 'reply' in final_arg, (
        f'turn 1 assistant reply missing from Claude prompt. Got: {final_arg!r}'
    )
    assert 'latest message' in final_arg


@pytest.mark.asyncio
async def test_fallback_never_calls_gemini_after_wave_2():
    """H2-047 Wave 2: brain_router._fallback no longer reaches out to any
    Gemini surface. Even when a fallback object is passed (back-compat
    parameter), generate_text MUST NOT be invoked."""
    fallback = MagicMock()
    fallback.generate_text = MagicMock()

    router = BrainRouter(gemini_fallback=fallback)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(return_value='codex reply')):
        result = await router.generate('hello')
    assert not fallback.generate_text.called
    assert result.fallback_used is True
    assert result.provider_used == 'codex'



@pytest.mark.asyncio
async def test_tool_likely_routes_to_claude_mcp_when_config_present(tmp_path, monkeypatch):
    """Phase 2.5a: with NEXUS_MCP_CONFIG pointing at a real file, tool_likely
    turns spawn `claude -p --mcp-config ...` and return the subprocess output as text."""
    fake_config = tmp_path / "mcp.json"
    fake_config.write_text("{}")
    monkeypatch.setenv("NEXUS_MCP_CONFIG", str(fake_config))

    fallback = MagicMock()
    fallback.generate_with_tools = MagicMock()  # should NOT be called
    router = BrainRouter(gemini_fallback=fallback)

    with patch(
        "services.brain_router.subprocess.Popen",
        return_value=_make_completed("Listed 3 files: a.txt, b.txt, c.txt"),
    ) as mock_run:
        result = await router.generate_with_tools(
            user_id="u1",
            system_prompt="sys",
            contents=[{"role": "user", "parts": [{"text": "list the files in my Documents folder"}]}],
            tool_catalog=[{"name": "list_directory", "description": "l", "parameters": {}}],
        )

    assert "text" in result
    assert "Listed 3 files" in result["text"]
    assert mock_run.called, "claude -p subprocess must be invoked when MCP config is present"
    # Verify the command includes --mcp-config and the right config path
    cmd_args = mock_run.call_args.args[0]
    assert "--mcp-config" in cmd_args
    assert str(fake_config) in cmd_args
    assert "--strict-mcp-config" in cmd_args
    appended_prompt = cmd_args[cmd_args.index("--append-system-prompt") + 1]
    assert 'user_id="u1"' in appended_prompt
    # Gemini fallback must NOT have been used
    assert not fallback.generate_with_tools.called


def test_mcp_config_path_defaults_to_current_repo(monkeypatch):
    monkeypatch.delenv("NEXUS_MCP_CONFIG", raising=False)
    expected = FilePath(__file__).resolve().parents[2] / ".mcp.dev.json"
    assert BrainRouter._mcp_config_path() == str(expected)


@pytest.mark.asyncio
async def test_tool_likely_mcp_failure_falls_back_to_codex_without_gemini(tmp_path, monkeypatch, caplog):
    """When Claude MCP fails, the router should retry through Codex with the
    rendered tool catalog. Gemini remains retired."""
    fake_config = tmp_path / "mcp.json"
    fake_config.write_text("{}")
    monkeypatch.setenv("NEXUS_MCP_CONFIG", str(fake_config))

    fallback = MagicMock()
    fallback.generate_with_tools = MagicMock()  # must NOT be called
    fallback.generate_text = MagicMock()        # must NOT be called either
    router = BrainRouter(gemini_fallback=fallback)

    with caplog.at_level('WARNING', logger='services.brain_router'):
        with patch.object(router, '_invoke_claude_with_mcp', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=180))), \
             patch.object(router, '_invoke_codex', AsyncMock(return_value='{"tool_calls": [{"name": "list_directory", "arguments": {"path": "/home"}}]}')):
            result = await router.generate_with_tools(
                user_id="u1",
                system_prompt="sys",
                contents=[{"role": "user", "parts": [{"text": "list my files"}]}],
                tool_catalog=[{"name": "list_directory", "description": "l", "parameters": {}}],
            )
    assert 'tool_calls' in result
    assert result['tool_calls'][0]['name'] == 'list_directory'
    assert not fallback.generate_with_tools.called
    assert not fallback.generate_text.called
    codex_fallback_logs = [r for r in caplog.records
                           if r.message == 'brain_router_claude_mcp_failed']
    assert len(codex_fallback_logs) == 1, (
        f'expected one brain_router_claude_mcp_failed log, got {len(codex_fallback_logs)}'
    )


@pytest.mark.asyncio
async def test_tool_likely_mcp_failure_retries_once_before_codex_fallback(tmp_path, monkeypatch):
    fake_config = tmp_path / "mcp.json"
    fake_config.write_text("{}")
    monkeypatch.setenv("NEXUS_MCP_CONFIG", str(fake_config))

    router = BrainRouter()
    with patch.object(
        router,
        '_invoke_claude_with_mcp',
        AsyncMock(
            side_effect=[
                subprocess.TimeoutExpired(cmd='claude', timeout=180),
                subprocess.TimeoutExpired(cmd='claude', timeout=180),
            ]
        ),
    ) as mock_claude_mcp, patch.object(
        router,
        '_sleep_before_claude_retry',
        AsyncMock(return_value=None),
    ) as mock_sleep, patch.object(
        router,
        '_invoke_codex',
        AsyncMock(
            return_value='{"tool_calls": [{"name": "list_directory", "arguments": {"path": "/home"}}]}'
        ),
    ) as mock_codex:
        result = await router.generate_with_tools(
            user_id="u1",
            system_prompt="sys",
            contents=[{"role": "user", "parts": [{"text": "list my files"}]}],
            tool_catalog=[{"name": "list_directory", "description": "l", "parameters": {}}],
        )
    assert result["tool_calls"][0]["name"] == "list_directory"
    assert mock_claude_mcp.await_count == 2
    mock_sleep.assert_awaited_once()
    mock_codex.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_likely_without_mcp_falls_through_to_claude_cli(monkeypatch):
    """H2-040: tool_likely + MCP disabled falls through to self.generate (Claude
    CLI direct), NOT to Gemini. Previous behavior routed to gemini_fallback;
    that path was retired."""
    monkeypatch.setenv("NEXUS_MCP_CONFIG", "disabled")

    fallback = MagicMock()
    fallback.generate_with_tools = MagicMock()  # must NOT be called
    fallback.generate_text = MagicMock()        # must NOT be called either
    router = BrainRouter(gemini_fallback=fallback)

    json_out = '{"tool_calls": [{"name": "list_directory", "arguments": {"path": "/home"}}]}'
    with patch('services.brain_router.subprocess.Popen',
               return_value=_make_completed(json_out)) as mock_run:
        result = await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=[{'role': 'user', 'parts': [{'text': 'list my home directory'}]}],
            tool_catalog=[{'name': 'list_directory', 'description': 'list dir', 'parameters': {}}],
        )

    assert mock_run.called, 'Claude CLI subprocess should be invoked when MCP disabled'
    assert 'tool_calls' in result
    assert result['tool_calls'][0]['name'] == 'list_directory'
    assert not fallback.generate_with_tools.called
    assert not fallback.generate_text.called


@pytest.mark.asyncio
async def test_user_id_no_longer_threaded_to_fallback():
    """No Gemini surface remains even when Claude fails and Codex takes over."""
    fallback = MagicMock()
    fallback.generate_text = MagicMock()

    router = BrainRouter(gemini_fallback=fallback)
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(return_value='codex reply')):
        result = await router.generate_with_tools(
            user_id='user-42',
            system_prompt='sys',
            contents=[{'role': 'user', 'parts': [{'text': 'hi'}]}],
            tool_catalog=[],
        )
    assert not fallback.generate_text.called
    assert result.get('text') == 'codex reply'


# ---------------------------------------------------------------------------
# H2-040: shutdown drain tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_active_subprocesses():
    """Ensure _active_subprocesses starts empty for every drain-related test
    (class-level state would otherwise leak between tests)."""
    BrainRouter._active_subprocesses.clear()
    yield
    BrainRouter._active_subprocesses.clear()


@pytest.mark.asyncio
async def test_drain_returns_true_when_no_subprocesses_active():
    """Drain with nothing in-flight returns True immediately."""
    assert BrainRouter._active_subprocesses == set()
    result = await BrainRouter.drain(timeout_seconds=1.0)
    assert result is True


@pytest.mark.asyncio
async def test_drain_waits_then_terminates_inflight_subprocess():
    """When a subprocess is still in the active set after the timeout, drain
    calls .terminate() on it and returns False. Models the systemd
    KillMode=mixed scenario where children survive SIGTERM long enough for
    the Python drain to observe them."""
    fake_proc = MagicMock()
    BrainRouter._active_subprocesses.add(fake_proc)

    start = time.monotonic()
    result = await BrainRouter.drain(timeout_seconds=0.3)
    elapsed = time.monotonic() - start

    assert result is False
    assert 0.25 <= elapsed < 1.0, f'drain should wait ~0.3s, waited {elapsed:.2f}s'
    fake_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_drain_returns_true_when_subprocess_completes_before_timeout():
    """Drain polls until the active set empties; if that happens before the
    timeout fires, drain returns True without calling .terminate()."""
    fake_proc = MagicMock()
    BrainRouter._active_subprocesses.add(fake_proc)

    async def _complete_after_delay():
        await asyncio.sleep(0.15)
        BrainRouter._active_subprocesses.discard(fake_proc)

    asyncio.create_task(_complete_after_delay())

    result = await BrainRouter.drain(timeout_seconds=5.0)
    assert result is True
    fake_proc.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_run_subprocess_sync_registers_and_unregisters_proc():
    """Every CLI invocation should add the Popen to _active_subprocesses for
    its lifetime and remove it on completion. Asserts the registration is
    actually wired up — without this, drain would never observe in-flight
    processes."""
    seen_during_call: list[int] = []
    real_communicate = MagicMock(return_value=(b'ok', b''))

    def _capture_popen(*args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.communicate = real_communicate
        # snapshot the set size BEFORE communicate runs and removes us
        seen_during_call.append(len(BrainRouter._active_subprocesses) + 1)
        return m

    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen', side_effect=_capture_popen):
        await router.generate('hello')

    assert seen_during_call == [1], 'subprocess should be in active set during run'
    assert BrainRouter._active_subprocesses == set(), 'subprocess must be removed after run'


@pytest.mark.asyncio
async def test_run_subprocess_sync_unregisters_on_failure():
    """_active_subprocesses must be cleaned up even when the subprocess raises
    (nonzero returncode, timeout, etc.) — otherwise drain would block on
    zombies forever."""
    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen',
               return_value=_make_completed('', returncode=1, stderr='boom')):
        try:
            router._run_subprocess_sync(['claude', '-p', 'hello'], 5)
        except RuntimeError:
            pass
    assert BrainRouter._active_subprocesses == set()


# ---------------------------------------------------------------------------
# H2-042: multi-turn conversation history serialization
# ---------------------------------------------------------------------------


def _extract_user_prompt_from_argv(argv: list[str]) -> str:
    """Pull the value of the `-p` flag from a captured Claude CLI argv."""
    for i, arg in enumerate(argv):
        if arg == '-p' and i + 1 < len(argv):
            return argv[i + 1]
    raise AssertionError(f'no -p flag found in argv: {argv!r}')


def test_serialize_contents_for_claude_empty_returns_empty_string():
    """Empty contents — no crash, no exception, empty result. Caller falls
    back to latest_user_text in production code."""
    assert BrainRouter._serialize_contents_for_claude([]) == ''


def test_serialize_contents_for_claude_single_user_turn():
    """One turn = one line, 'User: <text>'."""
    out = BrainRouter._serialize_contents_for_claude([
        {'role': 'user', 'parts': [{'text': 'hello world'}]},
    ])
    assert out == 'User: hello world'


def test_serialize_contents_for_claude_multi_turn_alternates_roles():
    """Two turns — User + Assistant labels alternate, separated by blank line."""
    out = BrainRouter._serialize_contents_for_claude([
        {'role': 'user',  'parts': [{'text': 'turn 1 user'}]},
        {'role': 'model', 'parts': [{'text': 'turn 1 assistant'}]},
        {'role': 'user',  'parts': [{'text': 'turn 2 user'}]},
    ])
    expected = 'User: turn 1 user\n\nAssistant: turn 1 assistant\n\nUser: turn 2 user'
    assert out == expected


def test_serialize_contents_for_claude_skips_unknown_roles_and_non_text_parts():
    """Function-call / tool-response parts from the legacy Gemini in-process
    tool loop must be skipped so the transcript stays text-only."""
    out = BrainRouter._serialize_contents_for_claude([
        {'role': 'user',  'parts': [{'text': 'real user text'}]},
        {'role': 'model', 'parts': [{'functionCall': {'name': 'x', 'args': {}}}]},
        {'role': 'user',  'parts': [{'functionResponse': {'name': 'x', 'response': {}}}]},
        {'role': 'system', 'parts': [{'text': 'system'}]},  # unknown role → skip
        {'role': 'user',  'parts': [{'text': 'second user text'}]},
    ])
    assert out == 'User: real user text\n\nUser: second user text'


@pytest.mark.asyncio
async def test_mcp_path_serializes_two_turn_pronoun_resolution(tmp_path, monkeypatch):
    """The reported regression: turn 1 names entities ('Anna and Bob'), turn 2
    uses a pronoun ('Add them'). The MCP path must send Claude CLI BOTH
    turns so the pronoun resolves."""
    fake_config = tmp_path / 'mcp.json'
    fake_config.write_text('{}')
    monkeypatch.setenv('NEXUS_MCP_CONFIG', str(fake_config))

    captured: dict = {}

    def _capture_popen(cmd, **kw):
        captured['cmd'] = cmd
        return _make_completed('(claude reply)')

    router = BrainRouter()
    contents = [
        {'role': 'user',  'parts': [{'text': 'Tell me about Anna and Bob.'}]},
        {'role': 'model', 'parts': [{'text': 'Anna and Bob are tenants in unit 3.'}]},
        {'role': 'user',  'parts': [{'text': 'Add them to the registry.'}]},
    ]
    with patch('services.brain_router.subprocess.Popen', side_effect=_capture_popen):
        await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=contents,
            tool_catalog=[{'name': 'add_unit', 'description': 'add', 'parameters': {}}],
        )

    user_prompt = _extract_user_prompt_from_argv(captured['cmd'])
    assert 'Anna and Bob' in user_prompt, (
        f'turn 1 entity names missing from -p argv. Got:\n{user_prompt}'
    )
    assert 'Add them to the registry' in user_prompt, (
        f'current user turn missing from -p argv. Got:\n{user_prompt}'
    )
    assert 'Assistant:' in user_prompt, (
        f'turn 1 assistant reply missing the role label. Got:\n{user_prompt}'
    )


@pytest.mark.asyncio
async def test_mcp_path_serializes_five_turn_history(tmp_path, monkeypatch):
    """All 5 historical turns must appear in the Claude CLI prompt — not just
    the last one and not a window of the last 2."""
    fake_config = tmp_path / 'mcp.json'
    fake_config.write_text('{}')
    monkeypatch.setenv('NEXUS_MCP_CONFIG', str(fake_config))

    captured: dict = {}

    def _capture_popen(cmd, **kw):
        captured['cmd'] = cmd
        return _make_completed('ok')

    router = BrainRouter()
    contents = [
        {'role': 'user',  'parts': [{'text': 'turn-alpha-user'}]},
        {'role': 'model', 'parts': [{'text': 'turn-alpha-model'}]},
        {'role': 'user',  'parts': [{'text': 'turn-bravo-user'}]},
        {'role': 'model', 'parts': [{'text': 'turn-bravo-model'}]},
        {'role': 'user',  'parts': [{'text': 'turn-current-user — list directory contents'}]},
    ]
    with patch('services.brain_router.subprocess.Popen', side_effect=_capture_popen):
        await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=contents,
            tool_catalog=[{'name': 'list_directory', 'description': 'l', 'parameters': {}}],
        )

    user_prompt = _extract_user_prompt_from_argv(captured['cmd'])
    for needle in ('turn-alpha-user', 'turn-alpha-model', 'turn-bravo-user',
                   'turn-bravo-model', 'turn-current-user'):
        assert needle in user_prompt, f'turn marker {needle!r} missing from -p argv'


@pytest.mark.asyncio
async def test_mcp_path_empty_contents_does_not_crash(tmp_path, monkeypatch):
    """Defensive — empty contents falls back to empty latest text, must not
    raise. Real bots never produce this (dispatcher always appends current
    turn), but the helper should tolerate the edge."""
    fake_config = tmp_path / 'mcp.json'
    fake_config.write_text('{}')
    monkeypatch.setenv('NEXUS_MCP_CONFIG', str(fake_config))

    router = BrainRouter()
    with patch('services.brain_router.subprocess.Popen',
               return_value=_make_completed('ok')):
        result = await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=[],
            tool_catalog=[],
        )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_conversational_path_carries_multi_turn_history(monkeypatch):
    """When MCP is disabled and the classifier picks the conversational path
    (not tool_likely), Claude CLI still receives the full transcript."""
    monkeypatch.setenv('NEXUS_MCP_CONFIG', 'disabled')

    captured: dict = {}

    def _capture_popen(cmd, **kw):
        captured['cmd'] = cmd
        return _make_completed('hello back')

    router = BrainRouter()
    contents = [
        {'role': 'user',  'parts': [{'text': 'My name is Alex.'}]},
        {'role': 'model', 'parts': [{'text': 'Nice to meet you, Alex.'}]},
        {'role': 'user',  'parts': [{'text': 'What is my name?'}]},
    ]
    with patch('services.brain_router.subprocess.Popen', side_effect=_capture_popen):
        await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=contents,
            tool_catalog=[],
        )

    # Conversational path goes through self.generate, which uses _build_prompt
    # to assemble the final argv. The serialized history should appear inside
    # the '## User' block.
    final_arg = captured['cmd'][-1]
    assert 'My name is Alex' in final_arg
    assert 'Nice to meet you, Alex' in final_arg
    assert 'What is my name' in final_arg


@pytest.mark.asyncio
async def test_fallback_path_no_longer_calls_gemini_after_wave_2(monkeypatch):
    """MCP-disabled conversational turns should retry through Codex before the
    final canned failure path. Gemini remains retired."""
    monkeypatch.setenv('NEXUS_MCP_CONFIG', 'disabled')

    fallback = MagicMock()
    fallback.generate_text = MagicMock()
    router = BrainRouter(gemini_fallback=fallback)

    contents = [
        {'role': 'user',  'parts': [{'text': 'I drive a Tesla Model 3.'}]},
        {'role': 'model', 'parts': [{'text': 'Got it.'}]},
        {'role': 'user',  'parts': [{'text': 'What car do I drive?'}]},
    ]
    with patch.object(router, '_invoke_claude', AsyncMock(side_effect=subprocess.TimeoutExpired(cmd='claude', timeout=60))), \
         patch.object(router, '_invoke_codex', AsyncMock(return_value='You drive a Tesla Model 3.')):
        result = await router.generate_with_tools(
            user_id='u1',
            system_prompt='sys',
            contents=contents,
            tool_catalog=[],
        )

    assert not fallback.generate_text.called
    assert result.get('text') == 'You drive a Tesla Model 3.'
