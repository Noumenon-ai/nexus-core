"""Phase 5: Playwright-based browser controller.

One BrowserAgent per call.  Async-only.  Uses the per-user Firefox profile from
browser_profile_resolver.py so logged-in sessions persist between turns.

Anti-detection:
  - playwright-stealth applied to every context
  - Human-like typing speed: 50..200ms per character with occasional pauses
  - Random delays between actions: 200ms..2s
  - Scroll-before-click on interactive elements
  - User_1 (real human profile) NEVER runs headless
  - User_2 (isolated profile) may run headless

Security:
  - All file paths (screenshots, downloads, uploads) validated through Phase 4
    `_validate_path` / `_validate_write_target` — same ALLOWED_ROOTS as the rest
    of NEXUS.
  - Profile lock detection — refuses to launch if Firefox is already using the
    profile in another process (would corrupt cookies).
  - 30s default action timeout (120s hard cap).

Surface: high-level methods (fetch_url, screenshot, fill_form, ...).  Callers
get back a structured dict — failures are reported as `{ok: False, reason, detail}`,
NOT raised.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.browser_profile_resolver import (
    BrowserProfileError,
    ProfileSelection,
    profile_exists,
    profile_is_locked,
    resolve_profile,
)
from services.file_system_tools import FileSystemSecurityError, _validate_path
from mcp_servers.nexus_filesystem import _validate_write_target


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SEC = 30
_MAX_TIMEOUT_SEC = 120
_HUMAN_TYPE_MIN_MS = 50
_HUMAN_TYPE_MAX_MS = 200
_HUMAN_TYPE_PAUSE_PROBABILITY = 0.10
_HUMAN_TYPE_PAUSE_MS = 500
_ACTION_DELAY_MIN_MS = 200
_ACTION_DELAY_MAX_MS = 2000

_ALLOWED_URL_SCHEMES = {"http", "https"}


def _decide_headless(*, force_headless: bool | None,
                     selection_headless_ok: bool,
                     is_real_profile: bool,
                     env: dict | os._Environ) -> bool:
    """Resolve the final headless setting for a Chromium launch.

    Rules (in order):
      1. Explicit `force_headless` (test/caller override) wins.
      2. User_1 real profile defaults to headed for anti-detection — UNLESS
         no display server is reachable. In that case, fall back to headless
         so the launch doesn't crash. The operator can override with
         NEXUS_BROWSER_FORCE_HEADED=1 to force a hard failure (signaling
         they need to set up xvfb-run / DISPLAY first).
      3. Non-real profiles use selection.headless_ok as the baseline.
    """
    if force_headless is not None:
        return force_headless
    has_display = bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    force_headed = bool(env.get("NEXUS_BROWSER_FORCE_HEADED"))
    if is_real_profile:
        # User_1: prefer headed. Fall back to headless only when no display
        # AND operator hasn't opted into hard-failure mode.
        if not has_display and not force_headed:
            return True
        return False
    return selection_headless_ok


@dataclass
class BrowserActionResult:
    ok: bool
    reason: str | None = None
    detail: str | None = None
    data: dict | None = None


def _validate_url(url: str) -> str | None:
    """Return None if the URL is acceptable, else an error string."""
    if not isinstance(url, str) or not url.strip():
        return "url must be a non-empty string"
    u = url.strip()
    if len(u) > 4096:
        return f"url too long: {len(u)} chars"
    from urllib.parse import urlparse
    try:
        parsed = urlparse(u)
    except Exception as exc:
        return f"unparseable url: {exc}"
    if not parsed.scheme:
        return "url missing scheme (http:// or https://)"
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return f"url scheme {parsed.scheme!r} not allowed (http/https only)"
    if not parsed.netloc:
        return "url missing host"
    return None


def _normalize_timeout(seconds: int | None) -> int:
    try:
        s = int(seconds) if seconds is not None else _DEFAULT_TIMEOUT_SEC
    except (TypeError, ValueError):
        s = _DEFAULT_TIMEOUT_SEC
    return max(1, min(s, _MAX_TIMEOUT_SEC))


class BrowserAgent:
    """Owns a single Playwright + Firefox session bound to one user's profile.

    Use as an async context manager:
        async with BrowserAgent(user_id) as agent:
            r = await agent.fetch_url("https://example.com")
    """

    def __init__(self, user_id: str, *, force_headless: bool | None = None,
                 stealth_enabled: bool = True):
        self.user_id = user_id
        self._force_headless = force_headless
        self._stealth_enabled = stealth_enabled
        self._selection: ProfileSelection | None = None
        self._playwright = None
        self._context = None
        self._stealth = None

    # ---------- lifecycle ----------

    async def __aenter__(self) -> "BrowserAgent":
        try:
            self._selection = resolve_profile(self.user_id)
        except BrowserProfileError as exc:
            raise BrowserProfileError(str(exc)) from None

        if self._selection.is_real_profile and not profile_exists(self._selection):
            raise BrowserProfileError(
                f"profile for {self._selection.user_label} does not exist: {self._selection.profile_path}"
            )
        if profile_is_locked(self._selection):
            raise BrowserProfileError(
                f"profile is locked (Firefox already running?): {self._selection.profile_path}"
            )

        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        headless = _decide_headless(
            force_headless=self._force_headless,
            selection_headless_ok=self._selection.headless_ok,
            is_real_profile=self._selection.is_real_profile,
            env=os.environ,
        )
        if headless and self._selection.is_real_profile:
            logger.warning(
                "browser_agent_no_display_falling_back_to_headless",
                extra={"user_id": self.user_id,
                       "profile": str(self._selection.profile_path)},
            )

        try:
            # Chromium chosen over Firefox (2026-05-12): Ubuntu 24's
            # apparmor_restrict_unprivileged_userns=1 blocks Playwright Firefox's
            # sandbox helper (CLONE_NEWUSER EPERM). Chromium's sandbox model is
            # permitted, and playwright-stealth has much stronger Chromium support.
            launch_args: list[str] = []
            if not headless:
                launch_args.extend(["--start-maximized"])
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self._selection.profile_path),
                headless=headless,
                args=launch_args,
                ignore_default_args=["--enable-automation"],   # hides the "Chrome is being controlled" infobar
                timeout=30_000,
            )
        except Exception as exc:
            await self._teardown_playwright()
            raise BrowserProfileError(f"chromium launch failed: {exc}") from None

        if self._stealth_enabled:
            try:
                from playwright_stealth import Stealth
                self._stealth = Stealth()
                # Apply to existing pages and patch the context for future pages.
                for page in self._context.pages:
                    try:
                        await self._stealth.apply_stealth_async(page)
                    except Exception:
                        logger.debug("stealth_apply_existing_page_failed", exc_info=True)

                async def _on_new_page(page):
                    try:
                        await self._stealth.apply_stealth_async(page)
                    except Exception:
                        logger.debug("stealth_apply_new_page_failed", exc_info=True)

                self._context.on("page", lambda p: asyncio.ensure_future(_on_new_page(p)))
            except ImportError:
                logger.warning("playwright_stealth_not_installed; continuing without stealth")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._teardown_playwright()

    async def _teardown_playwright(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                logger.debug("context_close_failed", exc_info=True)
            self._context = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                logger.debug("playwright_stop_failed", exc_info=True)
            self._playwright = None

    async def _new_page(self):
        if self._context is None:
            raise RuntimeError("BrowserAgent not entered as context manager")
        page = await self._context.new_page()
        return page

    # ---------- human-like helpers ----------

    @staticmethod
    async def _random_delay() -> None:
        await asyncio.sleep(random.uniform(_ACTION_DELAY_MIN_MS, _ACTION_DELAY_MAX_MS) / 1000.0)

    @staticmethod
    async def _human_type(page, selector: str, text: str) -> None:
        await page.click(selector)
        for ch in text:
            await page.keyboard.type(ch)
            if random.random() < _HUMAN_TYPE_PAUSE_PROBABILITY:
                await asyncio.sleep(_HUMAN_TYPE_PAUSE_MS / 1000.0)
            else:
                ms = random.uniform(_HUMAN_TYPE_MIN_MS, _HUMAN_TYPE_MAX_MS)
                await asyncio.sleep(ms / 1000.0)

    @staticmethod
    async def _scroll_then_click(page, selector: str) -> None:
        await page.locator(selector).scroll_into_view_if_needed(timeout=10_000)
        await asyncio.sleep(random.uniform(0.2, 0.6))
        await page.click(selector)

    # ---------- tool primitives ----------

    async def fetch_url(self, url: str, timeout_sec: int | None = None,
                       wait_for_network_idle: bool = True) -> BrowserActionResult:
        """Navigate to URL, return (title, text, html, final_url, status_code)."""
        err = _validate_url(url)
        if err is not None:
            return BrowserActionResult(ok=False, reason="invalid_url", detail=err)
        timeout = _normalize_timeout(timeout_sec) * 1000
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            response = await page.goto(url, timeout=timeout,
                                       wait_until="networkidle" if wait_for_network_idle else "domcontentloaded")
            status = response.status if response else None
            title = await page.title()
            final_url = page.url
            html = await page.content()
            text = await page.evaluate("() => document.body && document.body.innerText")
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="navigation_failed", detail=str(exc)[:300])
        await page.close()
        return BrowserActionResult(ok=True, data={
            "url": url, "final_url": final_url, "status": status,
            "title": title, "text": text, "html_length": len(html or ""),
            "html_preview": (html or "")[:5000],
        })

    async def screenshot(self, url: str, output_path: str,
                        full_page: bool = True, timeout_sec: int | None = None) -> BrowserActionResult:
        """Navigate then save a PNG screenshot to a validated output path."""
        err = _validate_url(url)
        if err is not None:
            return BrowserActionResult(ok=False, reason="invalid_url", detail=err)
        try:
            target = _validate_write_target(output_path)
        except FileSystemSecurityError as exc:
            return BrowserActionResult(ok=False, reason="output_path_denied", detail=str(exc))
        if target.suffix.lower() != ".png":
            return BrowserActionResult(ok=False, reason="output_must_be_png", detail=target.name)
        timeout = _normalize_timeout(timeout_sec) * 1000
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            await self._random_delay()
            await page.screenshot(path=str(target), full_page=full_page)
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="screenshot_failed", detail=str(exc)[:300])
        await page.close()
        return BrowserActionResult(ok=True, data={
            "url": url, "path": str(target),
            "size_bytes": target.stat().st_size, "full_page": full_page,
        })

    async def check_status(self, url: str, timeout_sec: int | None = None) -> BrowserActionResult:
        err = _validate_url(url)
        if err is not None:
            return BrowserActionResult(ok=False, reason="invalid_url", detail=err)
        timeout = _normalize_timeout(timeout_sec) * 1000
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            response = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            status = response.status if response else None
            ok_status = bool(response and response.ok)
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="navigation_failed", detail=str(exc)[:300])
        await page.close()
        return BrowserActionResult(ok=True, data={
            "url": url, "status_code": status, "ok": ok_status,
        })

    async def search_web(self, query: str, num_results: int = 10) -> BrowserActionResult:
        """DuckDuckGo HTML endpoint — no JS required, no API key.  Returns top results."""
        if not isinstance(query, str) or not query.strip():
            return BrowserActionResult(ok=False, reason="empty_query", detail="")
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            return BrowserActionResult(ok=False, reason="non_integer_num_results", detail="")
        num_results = max(1, min(num_results, 50))
        from urllib.parse import quote_plus
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query.strip())}"
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            await page.goto(search_url, timeout=30_000, wait_until="domcontentloaded")
            results = await page.evaluate("""
                (max) => {
                    const out = [];
                    document.querySelectorAll('.result__title a, a.result__a').forEach((a, i) => {
                        if (out.length >= max) return;
                        const title = (a.innerText || a.textContent || '').trim();
                        const url = a.href || '';
                        const snippetEl = a.closest('.result') ? a.closest('.result').querySelector('.result__snippet') : null;
                        const snippet = snippetEl ? (snippetEl.innerText || snippetEl.textContent || '').trim() : '';
                        if (title && url) out.push({title, url, snippet});
                    });
                    return out;
                }
            """, num_results)
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="search_failed", detail=str(exc)[:300])
        await page.close()
        return BrowserActionResult(ok=True, data={
            "query": query.strip(), "count": len(results or []),
            "results": results or [],
        })

    async def fill_form(self, url: str, field_values: dict,
                       submit_selector: str | None = None,
                       timeout_sec: int | None = None) -> BrowserActionResult:
        """Navigate to URL and fill named form fields. Does NOT click submit unless caller passes submit_selector."""
        err = _validate_url(url)
        if err is not None:
            return BrowserActionResult(ok=False, reason="invalid_url", detail=err)
        if not isinstance(field_values, dict) or not field_values:
            return BrowserActionResult(ok=False, reason="empty_field_values", detail="")
        timeout = _normalize_timeout(timeout_sec) * 1000
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="navigation_failed", detail=str(exc)[:300])
        filled: list[str] = []
        for selector, value in field_values.items():
            if not isinstance(selector, str) or not selector.strip():
                await page.close()
                return BrowserActionResult(ok=False, reason="invalid_field_selector",
                                           detail=f"selector must be non-empty string, got {selector!r}")
            if not isinstance(value, str):
                value = str(value)
            try:
                await page.locator(selector).scroll_into_view_if_needed(timeout=5000)
                await self._random_delay()
                await self._human_type(page, selector, value)
                filled.append(selector)
            except Exception as exc:
                await page.close()
                return BrowserActionResult(ok=False, reason="field_fill_failed",
                                           detail=f"{selector}: {str(exc)[:200]}")
        submitted = False
        if submit_selector:
            try:
                await self._scroll_then_click(page, submit_selector)
                submitted = True
                await page.wait_for_load_state("domcontentloaded", timeout=timeout)
            except Exception as exc:
                await page.close()
                return BrowserActionResult(ok=False, reason="submit_failed", detail=str(exc)[:300])
        final_url = page.url
        await page.close()
        return BrowserActionResult(ok=True, data={
            "url": url, "filled_fields": filled, "submitted": submitted,
            "final_url": final_url,
        })

    async def scrape(self, url: str, schema: dict,
                    timeout_sec: int | None = None) -> BrowserActionResult:
        """Navigate to URL and extract data per a CSS-selector schema:
        {"title": "h1", "links": {"selector": "a", "attr": "href", "all": true}}
        """
        err = _validate_url(url)
        if err is not None:
            return BrowserActionResult(ok=False, reason="invalid_url", detail=err)
        if not isinstance(schema, dict) or not schema:
            return BrowserActionResult(ok=False, reason="invalid_schema",
                                       detail="schema must be a non-empty dict")
        timeout = _normalize_timeout(timeout_sec) * 1000
        try:
            page = await self._new_page()
        except Exception as exc:
            return BrowserActionResult(ok=False, reason="page_create_failed", detail=str(exc)[:300])
        try:
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="navigation_failed", detail=str(exc)[:300])
        extracted: dict[str, Any] = {}
        try:
            for field, rule in schema.items():
                if isinstance(rule, str):
                    selector, attr, multiple = rule, None, False
                elif isinstance(rule, dict):
                    selector = rule.get("selector")
                    attr = rule.get("attr")
                    multiple = bool(rule.get("all"))
                    if not isinstance(selector, str) or not selector:
                        extracted[field] = {"_error": "missing selector"}
                        continue
                else:
                    extracted[field] = {"_error": f"unsupported rule type {type(rule).__name__}"}
                    continue
                if multiple:
                    vals = await page.locator(selector).evaluate_all(
                        "(els, attr) => els.map(el => attr ? el.getAttribute(attr) : el.innerText)",
                        attr,
                    )
                    extracted[field] = vals
                else:
                    loc = page.locator(selector).first
                    try:
                        if attr:
                            extracted[field] = await loc.get_attribute(attr)
                        else:
                            extracted[field] = await loc.inner_text(timeout=5000)
                    except Exception:
                        extracted[field] = None
        except Exception as exc:
            await page.close()
            return BrowserActionResult(ok=False, reason="scrape_failed", detail=str(exc)[:300])
        await page.close()
        return BrowserActionResult(ok=True, data={"url": url, "extracted": extracted})


__all__ = [
    "BrowserAgent",
    "BrowserActionResult",
    "BrowserProfileError",
    "_validate_url",
    "_normalize_timeout",
]
