"""Phase 2: Brain router — Claude CLI / Codex CLI / Ollama.

Phase 2.5a (2026-05-12): tool_likely turns route to `claude -p` with the
Phase 2.5a MCP servers loaded (--mcp-config). Claude CLI handles the entire
tool-calling loop internally via native function calling.

H2-040 (2026-05-12): Gemini tool-call fallback retired. Free-tier daily quota
(20/day, shared with mem0) made Gemini unreliable as a tool-call fallback —
users got cryptic "Tool-call request to Gemini failed." messages instead of a
clear retry hint. Claude and Codex are the live hosted providers now; when
Claude fails we retry through Codex before surfacing a canned unavailable
message.

One-shot subprocess pattern. No daemon. Same as trading bots.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import logging
import os
from pathlib import Path as FilePath
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from services.fallback_manager import PROVIDER_UNAVAILABLE_USER_TEXT
from services.intent_classifier import IntentDecision, Path, Provider, classify


logger = logging.getLogger(__name__)
_CLAUDE_RETRY_DELAY_SEC = 3.0


_TOOL_OUTPUT_INSTRUCTIONS = """
When you need to call a tool to fulfill the user's request, output ONLY a JSON object on a single line with this exact shape and nothing else:
{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}

If multiple tool calls are needed, include them all in the tool_calls array.

If no tool is needed and you can answer directly, output your answer as plain text (no JSON wrapper).

Do not mix JSON tool calls with prose. Either output a JSON tool_calls object, or output plain text. Never both.
"""


@dataclass
class BrainResult:
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    provider_used: str = ''
    fallback_used: bool = False


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    open_until_ts: float = 0.0


class BrainRouter:
    """Routes AI calls to Claude CLI / Codex CLI / Ollama / Gemini fallback."""

    # H2-040: every CLI subprocess started by this router registers itself here
    # for the duration of its run. The shutdown drain (see classmethod `drain`)
    # polls this set to know when it can let the process exit cleanly. Class-
    # level so a single drain call covers every router instance in the
    # process (in practice there's only one, but the contract is "all
    # in-flight brain_router subprocesses").
    _active_subprocesses: set[subprocess.Popen[bytes]] = set()

    def __init__(
        self,
        *,
        gemini_fallback: Any = None,  # H2-047 Wave 2: parameter kept for
        # back-compat with existing tests that pass it; the value is
        # ignored. _fallback now emits canned text directly without
        # calling Gemini. Safe to drop this parameter entirely once
        # the rest of the test surface is migrated.
        ollama_url: str | None = None,
        claude_binary: str = 'claude',
        codex_binary: str = 'codex',
        claude_timeout_sec: int = 60,
        codex_timeout_sec: int = 120,
        breaker_failure_threshold: int = 3,
        breaker_cooldown_sec: int = 300,
    ):
        del gemini_fallback  # ignored; H2-047 Wave 2 removed Gemini calls
        self.ollama_url = ollama_url
        self.claude_binary = claude_binary
        self.codex_binary = codex_binary
        self.claude_timeout_sec = claude_timeout_sec
        self.codex_timeout_sec = codex_timeout_sec
        self.breaker_failure_threshold = breaker_failure_threshold
        self.breaker_cooldown_sec = breaker_cooldown_sec
        self._breakers: dict[str, _BreakerState] = {}

    async def generate(
        self,
        prompt: str,
        *,
        tool_catalog: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        user_id: str | None = None,
    ) -> BrainResult:
        decision = classify(prompt)
        logger.info(
            'brain_router_decision',
            extra={
                'provider': decision.provider.value,
                'path': decision.path.value,
                'matched_keywords': list(decision.matched_keywords),
            },
        )

        # H2-046: when classifier routes to OLLAMA (sensitive-keyword path)
        # but ollama isn't configured in the deploy, transparently re-route
        # to Claude CLI rather than raising "ollama_url not configured" up
        # through _fallback. Claude CLI is a local subprocess — preserves
        # the privacy intent of the sensitive route at least as well as
        # Ollama would have. Log the demotion so we don't lose the signal.
        if decision.provider is Provider.OLLAMA and not self.ollama_url:
            logger.info(
                'brain_router_ollama_unavailable_routing_to_claude',
                extra={'matched_keywords': list(decision.matched_keywords)},
            )
            decision = IntentDecision(
                provider=Provider.CLAUDE,
                path=decision.path,
                matched_keywords=decision.matched_keywords,
            )
        return await self._generate_from_decision(
            prompt=prompt,
            tool_catalog=tool_catalog,
            system_prompt=system_prompt,
            decision=decision,
            user_id=user_id,
        )

    async def _generate_from_decision(
        self,
        *,
        prompt: str,
        tool_catalog: list[dict[str, Any]] | None,
        system_prompt: str | None,
        decision: IntentDecision,
        user_id: str | None,
        fallback_used: bool = False,
    ) -> BrainResult:
        full_prompt = self._build_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
            decision=decision,
        )
        providers_to_try = self._provider_chain(decision.provider)
        last_exc: Exception | None = None

        for index, provider in enumerate(providers_to_try):
            try:
                raw = await self._invoke_provider_with_retry(provider, full_prompt)
                provider_used = provider.value
                used_fallback = fallback_used or index > 0
                self._record_success(provider_used)
                logger.info(
                    'brain_router_invoked',
                    extra={
                        'provider': provider_used,
                        'path': decision.path.value,
                        'output_chars': len(raw) if raw else 0,
                        'tool_catalog_size': len(tool_catalog) if tool_catalog else 0,
                        'fallback_used': used_fallback,
                    },
                )

                if decision.path is Path.TOOL_LIKELY and tool_catalog:
                    parsed = self._try_parse_tool_calls(raw)
                    if parsed is not None:
                        return BrainResult(
                            tool_calls=parsed,
                            provider_used=provider_used,
                            fallback_used=used_fallback,
                        )

                return BrainResult(
                    text=raw,
                    provider_used=provider_used,
                    fallback_used=used_fallback,
                )

            except Exception as exc:
                last_exc = exc
                next_provider = None
                if index + 1 < len(providers_to_try):
                    next_provider = providers_to_try[index + 1].value
                logger.warning(
                    'brain_router_provider_failed',
                    extra={
                        'provider': provider.value,
                        'error': str(exc),
                        'next_provider': next_provider,
                    },
                )
                self._record_failure(provider.value)

        if last_exc is None:
            last_exc = RuntimeError('brain_router provider chain exhausted without attempt')
        return await self._fallback(prompt, tool_catalog, system_prompt, last_exc, user_id=user_id)

    def _build_prompt(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        tool_catalog: list[dict[str, Any]] | None,
        decision: IntentDecision,
    ) -> str:
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if decision.path is Path.TOOL_LIKELY and tool_catalog:
            parts.append('## Available tools')
            for tool in tool_catalog:
                name = tool.get('name', '')
                desc = tool.get('description', '')
                params = tool.get('parameters', {})
                parts.append(f'- {name}: {desc}')
                if params:
                    parts.append(f'  parameters: {json.dumps(params)}')
            parts.append(_TOOL_OUTPUT_INSTRUCTIONS)
        parts.append('## User')
        parts.append(prompt)
        return '\n\n'.join(parts)

    async def _run_blocking(self, fn, /, *args):
        """Run a blocking call without touching asyncio's global default executor.

        In this environment, `asyncio.to_thread()` can leave the interpreter
        hung after the awaited coroutine already completed. Use a short-lived
        dedicated executor per call so the worker thread is explicitly shut
        down before the event loop exits.
        """
        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='brain-router') as pool:
            return await loop.run_in_executor(pool, call)

    async def _invoke_claude(self, prompt: str) -> str:
        self._check_breaker('claude')
        return await self._run_blocking(self._run_subprocess_sync,
            [self.claude_binary, '-p', prompt],
            self.claude_timeout_sec,
        )

    async def _invoke_codex(self, prompt: str) -> str:
        self._check_breaker('codex')
        raw = await self._run_blocking(self._run_subprocess_sync,
            [self.codex_binary, 'exec', prompt],
            self.codex_timeout_sec,
        )
        return self._strip_codex_banner(raw)

    async def _invoke_ollama(self, prompt: str) -> str:
        if not self.ollama_url:
            raise RuntimeError('ollama_url not configured')
        import urllib.request
        body = json.dumps({'model': 'llama3.2', 'prompt': prompt, 'stream': False}).encode()
        req = urllib.request.Request(
            f'{self.ollama_url}/api/generate',
            data=body,
            headers={'Content-Type': 'application/json'},
        )
        def _do_call() -> str:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
                return payload.get('response', '')
        return await self._run_blocking(_do_call)

    @staticmethod
    def _provider_chain(primary_provider: Provider) -> tuple[Provider, ...]:
        if primary_provider is Provider.CLAUDE:
            return (Provider.CLAUDE, Provider.CODEX)
        if primary_provider is Provider.CODEX:
            return (Provider.CODEX, Provider.CLAUDE)
        return (Provider.OLLAMA,)

    @staticmethod
    def _should_retry_claude_failure(exc: Exception) -> bool:
        return 'circuit breaker open' not in str(exc).lower()

    async def _sleep_before_claude_retry(self) -> None:
        await asyncio.sleep(_CLAUDE_RETRY_DELAY_SEC)

    async def _invoke_claude_with_retry(self, prompt: str) -> str:
        for attempt in range(2):
            try:
                return await self._invoke_claude(prompt)
            except Exception as exc:
                if attempt == 0 and self._should_retry_claude_failure(exc):
                    logger.warning(
                        'brain_router_claude_retry_scheduled',
                        extra={
                            'error': str(exc)[:200],
                            'delay_sec': _CLAUDE_RETRY_DELAY_SEC,
                        },
                    )
                    await self._sleep_before_claude_retry()
                    continue
                raise
        raise RuntimeError('claude retry loop exhausted')

    async def _invoke_claude_with_mcp_retry(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        mcp_config_path: str,
        user_id: str | None = None,
        timeout_sec: int | None = None,
    ) -> str:
        for attempt in range(2):
            try:
                return await self._invoke_claude_with_mcp(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    mcp_config_path=mcp_config_path,
                    user_id=user_id,
                    timeout_sec=timeout_sec,
                )
            except Exception as exc:
                if attempt == 0 and self._should_retry_claude_failure(exc):
                    logger.warning(
                        'brain_router_claude_mcp_retry_scheduled',
                        extra={
                            'error': str(exc)[:200],
                            'delay_sec': _CLAUDE_RETRY_DELAY_SEC,
                        },
                    )
                    await self._sleep_before_claude_retry()
                    continue
                raise
        raise RuntimeError('claude mcp retry loop exhausted')

    async def _invoke_provider_with_retry(self, provider: Provider, prompt: str) -> str:
        if provider is Provider.CLAUDE:
            return await self._invoke_claude_with_retry(prompt)
        return await self._invoke_provider(provider, prompt)

    async def _invoke_provider(self, provider: Provider, prompt: str) -> str:
        if provider is Provider.CLAUDE:
            return await self._invoke_claude(prompt)
        if provider is Provider.CODEX:
            return await self._invoke_codex(prompt)
        if provider is Provider.OLLAMA:
            return await self._invoke_ollama(prompt)
        raise RuntimeError(f'unknown provider: {provider}')

    def _run_subprocess_sync(self, cmd: list[str], timeout_sec: int) -> str:
        # H2-040: Popen + tracked-set pattern so the shutdown drain can wait
        # for / terminate this process. Previously used subprocess.run which
        # gives no handle on the Popen, so the drain had no way to observe
        # in-flight CLI calls.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        BrainRouter._active_subprocesses.add(proc)
        stdout_b = b''
        stderr_b = b''
        try:
            try:
                stdout_b, stderr_b = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
        finally:
            BrainRouter._active_subprocesses.discard(proc)
        if proc.returncode != 0:
            raise RuntimeError(
                f'subprocess {cmd[0]} returned {proc.returncode}: {stderr_b.decode(errors="replace")[:500]}'
            )
        return stdout_b.decode(errors='replace').strip()

    @classmethod
    async def drain(cls, timeout_seconds: float = 10.0) -> bool:
        """Wait for in-flight CLI subprocesses to finish, terminate on timeout.

        Returns True when every tracked subprocess exited on its own within
        ``timeout_seconds``; False when at least one had to be terminated.

        H2-040: invoked from main.py's post_stop hook so SIGTERM-triggered
        shutdowns don't leave a `claude -p` mid-tool-loop. Effective only when
        the systemd unit uses ``KillMode=mixed``; the default
        ``KillMode=control-group`` would deliver SIGTERM to every PID in the
        cgroup, killing the children before this drain can observe them.
        """
        deadline = time.monotonic() + timeout_seconds
        while cls._active_subprocesses and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if cls._active_subprocesses:
            for proc in list(cls._active_subprocesses):
                try:
                    proc.terminate()
                except Exception:
                    pass
            return False
        return True

    @staticmethod
    def _strip_codex_banner(raw: str) -> str:
        lines = raw.splitlines()
        body_lines = []
        skip_header = True
        for line in lines:
            if skip_header:
                if line.strip().startswith('codex') or 'tokens used' in line.lower() or line.startswith('-'):
                    continue
                if not line.strip():
                    continue
                skip_header = False
            if 'tokens used' in line.lower():
                break
            body_lines.append(line)
        return '\n'.join(body_lines).strip() or raw

    @staticmethod
    def _try_parse_tool_calls(raw: str) -> list[dict[str, Any]] | None:
        if not raw:
            return None
        candidates = []
        candidates.append(raw.strip())
        match = re.search(r'\{[^{}]*"tool_calls"[^{}]*\[.*?\][^{}]*\}', raw, re.DOTALL)
        if match:
            candidates.append(match.group(0))
        fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'tool_calls' in parsed:
                calls = parsed['tool_calls']
                if isinstance(calls, list) and all(
                    isinstance(c, dict) and 'name' in c for c in calls
                ):
                    return calls
        return None

    def _check_breaker(self, provider: str) -> None:
        state = self._breakers.get(provider)
        if state is None:
            return
        if state.open_until_ts > asyncio.get_event_loop().time():
            raise RuntimeError(f'{provider} circuit breaker open')

    def _record_failure(self, provider: str) -> None:
        state = self._breakers.setdefault(provider, _BreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.breaker_failure_threshold:
            state.open_until_ts = asyncio.get_event_loop().time() + self.breaker_cooldown_sec
            logger.warning(
                'brain_router_breaker_opened',
                extra={'provider': provider, 'cooldown_sec': self.breaker_cooldown_sec},
            )

    def _record_success(self, provider: str) -> None:
        state = self._breakers.get(provider)
        if state is not None:
            state.consecutive_failures = 0
            state.open_until_ts = 0.0

    async def _fallback(
        self,
        prompt: str,
        tool_catalog: list[dict[str, Any]] | None,
        system_prompt: str | None,
        original_exc: Exception,
        *,
        user_id: str | None = None,
    ) -> BrainResult:
        # H2-047 Wave 2: Gemini fallback retired entirely. Every brain_router
        # provider failure now emits a single clean canned message; the
        # original exception goes to journalctl. The retired surface
        # (gemini_fallback.generate_text) was the last live Gemini call
        # site in the routing layer.
        del prompt, tool_catalog, system_prompt, user_id  # unused after Wave 2
        logger.warning(
            'brain_router_fallback_engaged',
            extra={'original_error': str(original_exc)[:200]},
        )
        return BrainResult(
            text=PROVIDER_UNAVAILABLE_USER_TEXT,
            provider_used='none',
            fallback_used=True,
        )
    async def generate_with_tools(
        self,
        *,
        user_id: str,
        system_prompt: str,
        contents: list[dict[str, Any]],
        tool_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """LLMProtocol-compatible entrypoint.

        Routing:
          1. TOOL_LIKELY + MCP enabled → `claude -p --mcp-config ...`
                Claude CLI executes the entire tool-calling loop via MCP.
                On failure: retry through Codex with the rendered tool catalog.
          2. TOOL_LIKELY + MCP disabled → conversational provider path via
                self.generate; tool catalog rendered into the prompt.
          3. otherwise → conversational path via self.generate.
        Returns {'text': str} on the MCP path (Claude handled tools internally).
        """
        # H2-042: classify on the latest user turn only — routing intent must
        # come from the current message, not from stale keywords in earlier
        # turns. The serialized transcript carries the full multi-turn history
        # forward to whichever path (MCP / Claude CLI / Gemini fallback)
        # eventually invokes the LLM.
        latest_user_text = self._extract_last_user_text(contents)
        serialized_history = self._serialize_contents_for_claude(contents)
        # Fall back to the latest user text alone if serialization produced
        # nothing (defensive — should only happen with empty/malformed
        # contents). Prevents passing '' as the LLM prompt.
        prompt_for_llm = serialized_history or latest_user_text
        decision = classify(latest_user_text)

        if decision.path is Path.TOOL_LIKELY:
            mcp_config_path = self._mcp_config_path()
            if mcp_config_path:
                logger.info(
                    'brain_router_tool_dispatch_to_claude_mcp',
                    extra={'mcp_config': mcp_config_path, 'tool_catalog_size': len(tool_catalog)},
                )
                try:
                    text = await self._invoke_claude_with_mcp_retry(
                        user_prompt=prompt_for_llm,
                        system_prompt=system_prompt,
                        mcp_config_path=mcp_config_path,
                        user_id=user_id,
                    )
                    self._record_success('claude_mcp')
                    return {'text': text}
                except Exception as exc:
                    logger.warning(
                        'brain_router_claude_mcp_failed',
                        extra={'error': str(exc)[:200]},
                    )
                    self._record_failure('claude_mcp')
                    logger.info(
                        'brain_router_claude_mcp_falling_back_to_codex',
                        extra={'original_error': str(exc)[:200]},
                    )
                    fallback_decision = IntentDecision(
                        provider=Provider.CODEX,
                        path=Path.TOOL_LIKELY,
                        matched_keywords=decision.matched_keywords,
                    )
                    result = await self._generate_from_decision(
                        prompt=prompt_for_llm,
                        tool_catalog=tool_catalog,
                        system_prompt=system_prompt,
                        decision=fallback_decision,
                        user_id=user_id,
                        fallback_used=True,
                    )
                    if result.tool_calls is not None:
                        return {'tool_calls': result.tool_calls}
                    return {'text': result.text or ''}

        result = await self.generate(
            prompt_for_llm,
            tool_catalog=tool_catalog,
            system_prompt=system_prompt,
            user_id=user_id,
        )
        if result.tool_calls is not None:
            return {'tool_calls': result.tool_calls}
        return {'text': result.text or ''}

    @staticmethod
    def _mcp_config_path() -> str | None:
        """Resolve the MCP config file path. Returns None when MCP routing is disabled.

        Resolution order:
          1. NEXUS_MCP_CONFIG=disabled  → None (Gemini-only routing)
          2. NEXUS_MCP_CONFIG=<path>    → use that path if file exists
          3. (default)                    → current repo .mcp.dev.json, then .mcp.json
        """
        env_val = os.environ.get('NEXUS_MCP_CONFIG')
        if env_val == 'disabled':
            return None
        if env_val:
            return env_val if os.path.exists(env_val) else None
        project_root = FilePath(__file__).resolve().parent.parent
        for candidate in (project_root / '.mcp.dev.json', project_root / '.mcp.json'):
            if candidate.exists():
                return str(candidate)
        return None

    async def _invoke_claude_with_mcp(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        mcp_config_path: str,
        user_id: str | None = None,
        timeout_sec: int | None = None,
    ) -> str:
        """Invoke `claude -p` with all Phase 2.5a MCP servers + an extended timeout for tool turns."""
        self._check_breaker('claude_mcp')
        timeout = timeout_sec if timeout_sec is not None else max(self.claude_timeout_sec, 180)
        # --permission-mode bypassPermissions: Pattern B gate (H2-039) handles
        # destructive-intent approval at the bot/dispatcher level BEFORE
        # invoking Claude CLI. Without bypassPermissions, Claude in -p mode
        # may decline to call MCP tools (especially when the catalog includes
        # meta={"approval_required": True} on neighboring tools), responding
        # in natural language asking the user to approve. The bot's gate is
        # the actual safety boundary; Claude can execute freely once the
        # subprocess is invoked. --allowedTools still restricts the surface
        # to mcp__nexus-* (no Bash/Edit/built-ins).
        effective_system_prompt = system_prompt
        if user_id:
            effective_system_prompt = (
                f"{system_prompt}\n\n"
                "## Tool call user identity\n"
                "When calling any tool that accepts a `user_id` argument, "
                f"pass `user_id=\"{user_id}\"` explicitly. Do not rely on "
                "server defaults or omit the field."
            )
        cmd = [
            self.claude_binary,
            '--strict-mcp-config',
            '--mcp-config', mcp_config_path,
            '--permission-mode', 'bypassPermissions',
            '--allowedTools',
            'mcp__nexus-utils mcp__nexus-filesystem mcp__nexus-knowledge mcp__nexus-pdf-docs mcp__nexus-reminders-tasks mcp__nexus-calendar mcp__nexus-email mcp__nexus-contacts mcp__nexus-web mcp__nexus-messaging mcp__nexus-rentals mcp__nexus-business mcp__nexus-self',
            '--append-system-prompt', effective_system_prompt,
            '-p', user_prompt,
        ]
        return await self._run_blocking(self._run_subprocess_sync, cmd, timeout)

    @staticmethod
    def _extract_last_user_text(contents: list[dict[str, Any]]) -> str:
        """Extract the latest user message text from a Gemini-shaped contents
        array. Used for *classification only* (intent routing must read intent
        from the current turn, not from stale keywords in earlier turns) —
        H2-042. For the LLM prompt itself, use _serialize_contents_for_claude."""
        for entry in reversed(contents):
            if entry.get('role') == 'user':
                parts = entry.get('parts', [])
                texts = []
                for p in parts:
                    if isinstance(p, dict) and 'text' in p:
                        texts.append(p['text'])
                    elif isinstance(p, str):
                        texts.append(p)
                if texts:
                    return '\n'.join(texts)
        return ''

    @staticmethod
    def _serialize_contents_for_claude(contents: list[dict[str, Any]]) -> str:
        """Render a Gemini-shaped multi-turn contents array as a plain-text
        transcript that `claude -p` consumes.

        Output format (H2-042 fix for the Phase 2.5a context-blindness bug):

            User: <turn 1 user text>

            Assistant: <turn 1 model text>

            User: <current user text>

        Why this exists: Claude CLI in `-p` mode takes a single string and has
        no native multi-turn input. Without a serialized transcript, every
        tool_likely turn lost its conversation history at the brain_router
        boundary — the dispatcher built a multi-turn `contents` array, this
        class collapsed it to the latest user text, Claude saw a context-
        free prompt, and pronoun resolution broke. ("Add them" with no clue
        what "them" referred to.)

        Empty contents → empty string. Function-call / tool-response parts
        (which only show up in the legacy Gemini in-process tool loop, never
        in the MCP path) are skipped so the transcript stays text-only.
        """
        lines: list[str] = []
        for entry in contents:
            role = entry.get('role')
            if role == 'user':
                label = 'User'
            elif role == 'model':
                label = 'Assistant'
            else:
                continue
            parts = entry.get('parts', [])
            texts: list[str] = []
            for p in parts:
                if isinstance(p, dict) and 'text' in p:
                    text = p.get('text') or ''
                    if text:
                        texts.append(text)
                elif isinstance(p, str) and p:
                    texts.append(p)
            if not texts:
                continue
            lines.append(f'{label}: {" ".join(texts)}')
        return '\n\n'.join(lines)
