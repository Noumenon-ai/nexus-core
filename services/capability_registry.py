from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from models import User
from services.google_auth_service import GoogleAuthService
from services.tool_registry import ToolRegistry


CapabilityState = str


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    name: str
    state: CapabilityState
    reason: str
    service: str
    tool_name: str
    safe_to_attempt: bool
    manual_fix: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'state': self.state,
            'reason': self.reason,
            'service': self.service,
            'tool_name': self.tool_name,
            'safe_to_attempt': self.safe_to_attempt,
            'manual_fix': self.manual_fix,
            'details': dict(self.details),
        }


class CapabilityRegistry:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        google_auth_service: GoogleAuthService | None = None,
        whatsapp_status_provider=None,
        rentals_reader=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.google_auth_service = google_auth_service
        self._whatsapp_status_provider = whatsapp_status_provider or _default_whatsapp_bridge_status
        self._rentals_reader = rentals_reader or _default_rentals_reader

    def get_capability(
        self,
        name: str,
        *,
        user: User | None = None,
        registry: ToolRegistry | None = None,
    ) -> CapabilityStatus:
        if name == 'rentals_read':
            return self._rentals_read_status(registry=registry)
        if name == 'rentals_write':
            return self._rentals_write_status(registry=registry)
        if name == 'contacts_read':
            return self._contacts_read_status(user=user, registry=registry)
        if name == 'contacts_write':
            return self._contacts_write_status(user=user, registry=registry)
        if name == 'reminders_read':
            return self._simple_tool_status(
                name='reminders_read',
                service='nexus.db',
                tool_name='list_active_reminders',
                registry=registry,
                reason='dispatcher reminder-read tool available',
            )
        if name == 'reminders_write':
            return self._simple_tool_status(
                name='reminders_write',
                service='nexus.db',
                tool_name='create_reminder',
                registry=registry,
                reason='dispatcher reminder-write tool available',
            )
        if name == 'whatsapp_send':
            return self._whatsapp_send_status()
        if name == 'calendar_read':
            return self._calendar_status(user=user, registry=registry, write=False)
        if name == 'calendar_write':
            return self._calendar_status(user=user, registry=registry, write=True)
        if name == 'gmail_read':
            return self._gmail_read_status(registry=registry, user=user)
        if name == 'gmail_send':
            return self._gmail_send_status(user=user)
        if name == 'dashboard_status':
            return self._dashboard_status()
        return CapabilityStatus(
            name=name,
            state='unknown',
            reason='capability is not classified yet',
            service='unknown',
            tool_name='',
            safe_to_attempt=False,
            manual_fix='Add a capability probe for this surface.',
        )

    def list_capabilities(
        self,
        *,
        user: User | None = None,
        registry: ToolRegistry | None = None,
    ) -> list[CapabilityStatus]:
        names = (
            'rentals_read',
            'rentals_write',
            'contacts_read',
            'contacts_write',
            'reminders_read',
            'reminders_write',
            'whatsapp_send',
            'calendar_read',
            'calendar_write',
            'gmail_read',
            'gmail_send',
            'dashboard_status',
        )
        return [
            self.get_capability(name, user=user, registry=registry)
            for name in names
        ]

    def render_status_text(
        self,
        *,
        user: User | None = None,
        registry: ToolRegistry | None = None,
    ) -> str:
        statuses = {
            status.name: status
            for status in self.list_capabilities(user=user, registry=registry)
        }
        ordered = (
            'reminders_read',
            'reminders_write',
            'contacts_read',
            'contacts_write',
            'whatsapp_send',
            'rentals_read',
            'rentals_write',
            'calendar_read',
            'calendar_write',
            'gmail_read',
            'gmail_send',
            'dashboard_status',
        )
        labels = {
            'reminders_read': 'reminders read',
            'reminders_write': 'reminders write',
            'contacts_read': 'contacts read',
            'contacts_write': 'contacts write',
            'whatsapp_send': 'WhatsApp send',
            'rentals_read': 'rentals read',
            'rentals_write': 'rentals write',
            'calendar_read': 'calendar read',
            'calendar_write': 'calendar write',
            'gmail_read': 'Gmail read',
            'gmail_send': 'Gmail send',
            'dashboard_status': 'dashboard status',
        }
        lines = ['Capabilities:']
        for name in ordered:
            status = statuses[name]
            label = labels.get(name, name.replace('_', ' '))
            reason = _render_status_reason(status)
            if reason:
                lines.append(f'- {label}: {status.state} ({reason})')
            else:
                lines.append(f'- {label}: {status.state}')
        return '\n'.join(lines)

    def _simple_tool_status(
        self,
        *,
        name: str,
        service: str,
        tool_name: str,
        registry: ToolRegistry | None,
        reason: str,
    ) -> CapabilityStatus:
        if _registry_has_tool(registry, tool_name):
            return CapabilityStatus(
                name=name,
                state='available',
                reason=reason,
                service=service,
                tool_name=tool_name,
                safe_to_attempt=True,
                manual_fix='',
            )
        return CapabilityStatus(
            name=name,
            state='not_wired',
            reason=f'{tool_name} is not registered in this dispatcher',
            service=service,
            tool_name=tool_name,
            safe_to_attempt=False,
            manual_fix=f'Register {tool_name} in the dispatcher registry.',
        )

    def _rentals_read_status(self, *, registry: ToolRegistry | None) -> CapabilityStatus:
        tool_name = _first_registered_tool(
            registry,
            (
                'read_rentals',
                'list_rental_records',
                'get_rental_status',
            ),
        )
        try:
            records = list(self._rentals_reader())
        except Exception as exc:
            return CapabilityStatus(
                name='rentals_read',
                state='service_down',
                reason='rentals dashboard storage could not be read',
                service='nexus-dashboard.service',
                tool_name=tool_name or 'dashboard.storage.list_units',
                safe_to_attempt=False,
                manual_fix='Fix dashboard storage access before checking rentals from chat.',
                details={'error_class': type(exc).__name__},
            )
        effective_tool = tool_name or 'dashboard.storage.list_units'
        reason = 'rentals dashboard storage is accessible'
        if tool_name is not None:
            reason = 'dispatcher rentals-read tool available'
        return CapabilityStatus(
            name='rentals_read',
            state='available',
            reason=reason,
            service='nexus-dashboard.service',
            tool_name=effective_tool,
            safe_to_attempt=True,
            manual_fix='',
            details={
                'record_count': len(records),
                'fallback_source': 'dashboard_storage' if tool_name is None else 'dispatcher_tool',
            },
        )

    def _rentals_write_status(self, *, registry: ToolRegistry | None) -> CapabilityStatus:
        tool_name = _first_registered_tool(
            registry,
            (
                'update_unit',
                'add_payment',
                'delete_unit',
            ),
        )
        if tool_name is not None:
            return CapabilityStatus(
                name='rentals_write',
                state='available',
                reason='dispatcher rentals-write tool available',
                service='nexus-dashboard.service',
                tool_name=tool_name,
                safe_to_attempt=True,
                manual_fix='',
            )
        try:
            self._rentals_reader()
        except Exception as exc:
            return CapabilityStatus(
                name='rentals_write',
                state='service_down',
                reason='rentals dashboard storage could not be read',
                service='nexus-dashboard.service',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Fix dashboard storage access before wiring rentals updates.',
                details={'error_class': type(exc).__name__},
            )
        return CapabilityStatus(
            name='rentals_write',
            state='not_wired',
            reason='the rentals dashboard exists, but Telegram has no rentals-write tool',
            service='nexus-dashboard.service',
            tool_name='',
            safe_to_attempt=False,
            manual_fix='Add a rentals-write tool before editing rental records from chat.',
        )

    def _contacts_read_status(
        self,
        *,
        user: User | None,
        registry: ToolRegistry | None,
    ) -> CapabilityStatus:
        local_tool = _first_registered_tool(registry, ('resolve_contact_alias', 'list_contact_aliases'))
        if local_tool is not None:
            return CapabilityStatus(
                name='contacts_read',
                state='available',
                reason='local contact alias tools are available',
                service='nexus.db',
                tool_name=local_tool,
                safe_to_attempt=True,
                manual_fix='',
            )
        google_tool = _first_registered_tool(registry, ('lookup_contact',))
        if google_tool is None:
            return CapabilityStatus(
                name='contacts_read',
                state='not_wired',
                reason='no contacts-read tool is registered in this dispatcher',
                service='Google Contacts',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Register a contacts-read tool for Telegram.',
            )
        google_auth = self._google_auth_state(user=user)
        if google_auth != 'ready':
            return CapabilityStatus(
                name='contacts_read',
                state='auth_required',
                reason='Google Contacts needs auth before contact lookup can run',
                service='Google Contacts',
                tool_name=google_tool,
                safe_to_attempt=False,
                manual_fix='Reconnect Google before looking up Google Contacts.',
            )
        return CapabilityStatus(
            name='contacts_read',
            state='available',
            reason='Google Contacts lookup is available',
            service='Google Contacts',
            tool_name=google_tool,
            safe_to_attempt=True,
            manual_fix='',
        )

    def _contacts_write_status(
        self,
        *,
        user: User | None,
        registry: ToolRegistry | None,
    ) -> CapabilityStatus:
        tool_name = _first_registered_tool(registry, ('create_contact',))
        if tool_name is None:
            return CapabilityStatus(
                name='contacts_write',
                state='not_wired',
                reason='no contact-create tool is registered in this dispatcher',
                service='Google Contacts',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Register create_contact before writing contacts from chat.',
            )
        google_auth = self._google_auth_state(user=user)
        if google_auth != 'ready':
            return CapabilityStatus(
                name='contacts_write',
                state='auth_required',
                reason='Google Contacts auth is required before creating contacts',
                service='Google Contacts',
                tool_name=tool_name,
                safe_to_attempt=False,
                manual_fix='Reconnect Google before creating contacts.',
            )
        return CapabilityStatus(
            name='contacts_write',
            state='available',
            reason='Google Contacts write path is available',
            service='Google Contacts',
            tool_name=tool_name,
            safe_to_attempt=True,
            manual_fix='',
        )

    def _whatsapp_send_status(self) -> CapabilityStatus:
        status = self._whatsapp_status_provider()
        state = str(status.get('state') or 'unknown').strip().lower()
        detail = str(status.get('detail') or '').strip()
        if state == 'connected':
            return CapabilityStatus(
                name='whatsapp_send',
                state='available',
                reason=detail or 'WhatsApp bridge is paired',
                service='whatsapp-bridge.service',
                tool_name='services.contact_reminder_dispatcher.dispatch',
                safe_to_attempt=True,
                manual_fix='',
            )
        if state == 'qr':
            return CapabilityStatus(
                name='whatsapp_send',
                state='auth_required',
                reason=detail or 'WhatsApp bridge is waiting for QR pairing',
                service='whatsapp-bridge.service',
                tool_name='services.contact_reminder_dispatcher.dispatch',
                safe_to_attempt=False,
                manual_fix='Scan the WhatsApp QR in the dashboard to pair the bridge.',
            )
        if state in {'starting', 'closed', 'unreachable', 'error'}:
            return CapabilityStatus(
                name='whatsapp_send',
                state='service_down',
                reason=detail or f'WhatsApp bridge state is {state}',
                service='whatsapp-bridge.service',
                tool_name='services.contact_reminder_dispatcher.dispatch',
                safe_to_attempt=False,
                manual_fix='Start or restart whatsapp-bridge.service before sending WhatsApp messages.',
            )
        return CapabilityStatus(
            name='whatsapp_send',
            state='unknown',
            reason=detail or 'WhatsApp bridge state is unknown',
            service='whatsapp-bridge.service',
            tool_name='services.contact_reminder_dispatcher.dispatch',
            safe_to_attempt=False,
            manual_fix='Check whatsapp-bridge.service and its /status endpoint.',
        )

    def _calendar_status(
        self,
        *,
        user: User | None,
        registry: ToolRegistry | None,
        write: bool,
    ) -> CapabilityStatus:
        name = 'calendar_write' if write else 'calendar_read'
        tools = ('create_calendar_event', 'update_calendar_event', 'delete_calendar_event') if write else ('list_calendar_events', 'check_freebusy')
        tool_name = _first_registered_tool(registry, tools)
        if not self.settings.google.enabled:
            return CapabilityStatus(
                name=name,
                state='unavailable',
                reason='Google integration is disabled in this runtime',
                service='Google Calendar',
                tool_name=tool_name or '',
                safe_to_attempt=False,
                manual_fix='Enable GOOGLE_INTEGRATION_ENABLED before using Google Calendar.',
            )
        if tool_name is None:
            return CapabilityStatus(
                name=name,
                state='not_wired',
                reason='calendar tool is not registered in this dispatcher',
                service='Google Calendar',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Register the calendar tool in the dispatcher registry.',
            )
        google_auth = self._google_auth_state(user=user)
        if google_auth != 'ready':
            return CapabilityStatus(
                name=name,
                state='auth_required',
                reason='Google Calendar auth is required for this runtime',
                service='Google Calendar',
                tool_name=tool_name,
                safe_to_attempt=False,
                manual_fix='Reconnect Google before using Calendar from chat.',
            )
        return CapabilityStatus(
            name=name,
            state='available',
            reason='Google Calendar tools are available',
            service='Google Calendar',
            tool_name=tool_name,
            safe_to_attempt=True,
            manual_fix='',
        )

    def _gmail_read_status(
        self,
        *,
        registry: ToolRegistry | None,
        user: User | None,
    ) -> CapabilityStatus:
        tool_name = _first_registered_tool(registry, ('get_email_summary',))
        if not self.settings.gmail.gmail_enabled:
            return CapabilityStatus(
                name='gmail_read',
                state='unavailable',
                reason='Gmail integration is disabled in this runtime',
                service='Gmail',
                tool_name=tool_name or '',
                safe_to_attempt=False,
                manual_fix='Enable GMAIL_ENABLED before reading Gmail from chat.',
            )
        if tool_name is None:
            return CapabilityStatus(
                name='gmail_read',
                state='not_wired',
                reason='Gmail read tool is not registered in this dispatcher',
                service='Gmail',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Register get_email_summary in the dispatcher registry.',
            )
        if user is None or not self._gmail_token_exists(user.id):
            return CapabilityStatus(
                name='gmail_read',
                state='auth_required',
                reason='Gmail auth is required before reading email',
                service='Gmail',
                tool_name=tool_name,
                safe_to_attempt=False,
                manual_fix='Connect Gmail before reading email from chat.',
            )
        return CapabilityStatus(
            name='gmail_read',
            state='available',
            reason='Gmail read path is available',
            service='Gmail',
            tool_name=tool_name,
            safe_to_attempt=True,
            manual_fix='',
        )

    def _gmail_send_status(self, *, user: User | None) -> CapabilityStatus:
        if not self.settings.gmail.gmail_enabled:
            return CapabilityStatus(
                name='gmail_send',
                state='unavailable',
                reason='Gmail integration is disabled in this runtime',
                service='Gmail',
                tool_name='',
                safe_to_attempt=False,
                manual_fix='Enable GMAIL_ENABLED before sending Gmail from chat.',
            )
        token_exists = bool(user and self._gmail_token_exists(user.id))
        reason = 'Gmail MCP exists, but Telegram has no send-email tool wired yet'
        if not token_exists:
            reason = 'Gmail send is not wired in Telegram, and no Gmail token is present for this user'
        return CapabilityStatus(
            name='gmail_send',
            state='not_wired',
            reason=reason,
            service='Gmail',
            tool_name='mcp_servers.nexus_email.send_email',
            safe_to_attempt=False,
            manual_fix='Wire the Gmail send path into the dispatcher before sending email from chat.',
        )

    def _dashboard_status(self) -> CapabilityStatus:
        try:
            self._rentals_reader()
        except Exception as exc:
            return CapabilityStatus(
                name='dashboard_status',
                state='service_down',
                reason='dashboard storage is not readable',
                service='nexus-dashboard.service',
                tool_name='dashboard.storage.get_session_factory',
                safe_to_attempt=False,
                manual_fix='Fix dashboard storage before relying on dashboard-backed checks.',
                details={'error_class': type(exc).__name__},
            )
        return CapabilityStatus(
            name='dashboard_status',
            state='available',
            reason='dashboard storage is readable',
            service='nexus-dashboard.service',
            tool_name='dashboard.storage.get_session_factory',
            safe_to_attempt=True,
            manual_fix='',
        )

    def _google_auth_state(self, *, user: User | None) -> str:
        if user is None:
            return 'missing_user'
        if self.google_auth_service is not None:
            try:
                if self.google_auth_service.has_pending(user.id):
                    return 'pending'
            except Exception:
                pass
            try:
                if self.google_auth_service.is_connected(user.id) and self._google_token_exists(user.id):
                    return 'ready'
            except Exception:
                pass
        if getattr(user, 'google_connected', False) and self._google_token_exists(user.id):
            return 'ready'
        return 'auth_required'

    def _google_token_exists(self, user_id: str) -> bool:
        return (self.settings.google.token_dir / f'{user_id}.json').exists()

    def _gmail_token_exists(self, user_id: str) -> bool:
        return (self.settings.gmail.gmail_token_dir / f'{user_id}.json').exists()


def _registry_has_tool(registry: ToolRegistry | None, tool_name: str) -> bool:
    return registry is not None and registry.get(tool_name) is not None


def _first_registered_tool(
    registry: ToolRegistry | None,
    tool_names: tuple[str, ...],
) -> str | None:
    if registry is None:
        return None
    for tool_name in tool_names:
        if registry.get(tool_name) is not None:
            return tool_name
    return None


def _default_rentals_reader() -> list[dict[str, Any]]:
    from dashboard import storage as dashboard_storage

    return dashboard_storage.list_units()


def _default_whatsapp_bridge_status() -> dict[str, Any]:
    bridge_url = os.environ.get('WA_BRIDGE_URL', 'http://127.0.0.1:3744').rstrip('/')
    url = f'{bridge_url}/status'
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        detail = str(exc)[:80]
        if 'Operation not permitted' in detail or 'Permission denied' in detail:
            return {'state': 'unknown', 'detail': f'bridge status probe blocked: {detail}'}
        return {'state': 'unreachable', 'detail': f'bridge offline: {str(exc)[:80]}'}
    except Exception as exc:
        return {'state': 'error', 'detail': str(exc)[:80]}

    state = str(body.get('state') or 'unknown').strip().lower()
    if state == 'connected':
        me = body.get('me') or {}
        name = me.get('name') or (me.get('jid') or '').split('@')[0]
        detail = f'paired as {name}' if name else 'paired'
        return {'state': 'connected', 'detail': detail}
    if state == 'qr':
        return {'state': 'qr', 'detail': 'scan QR to pair'}
    if state in {'starting', 'closed'}:
        return {'state': 'starting', 'detail': f'bridge {state} — wait a few seconds'}
    return {'state': 'unknown', 'detail': f'bridge state: {state}'}


def _render_status_reason(status: CapabilityStatus) -> str:
    if status.state == 'available':
        return ''
    if status.manual_fix:
        return status.manual_fix
    return status.reason


# ── Step 4 (2026-05-27) — flat capability map for reasoning prompts ─────────
#
# NEXUS_ARCHITECTURE_REFACTOR.md step 4 specifies a flat
# `CAPABILITIES: dict[str, bool]` that is_available() and available_list()
# read from. The rich CapabilityRegistry above stays — it answers
# "WHY isn't this available?" for diagnostics and approval gates. The
# flat map below answers the simpler question Claude needs at reasoning
# time: "can I claim this in a natural-language reply?". Pure data, no
# I/O, no Google subprocess polling. Defaults match the spec.

CAPABILITIES: dict[str, bool] = {
    # WIRED AND WORKING
    'email_read':         True,
    'email_send':         True,
    'reminder_create':    True,
    'reminder_list':      True,
    'reminder_update':    True,
    'rental_read':        True,
    'business_read':      True,
    'knowledge_read':     True,
    'knowledge_write':    True,
    'square_read':        True,
    'messaging_whatsapp': False,   # not logged in
    'messaging_sms':      False,   # google voice not configured
    'trading':            False,   # disabled by env flag
    'smart_home':         False,   # HA not configured
    'car':                False,   # KIA disabled
    'osint':              False,   # disabled
    'voice_input':        True,    # whisper
    'voice_output':       False,   # elevenlabs disabled
}


def is_available(capability: str) -> bool:
    """Return True iff this capability is wired and Claude may claim it."""
    return bool(CAPABILITIES.get(capability, False))


def available_list() -> list[str]:
    """Sorted list of available capability names — what reason_with_fallback
    passes to Claude as the `capabilities` arg.

    Sorted so the system prompt is stable across runs (cheaper cache hits
    on Claude CLI, easier diff in audit logs).
    """
    return sorted(name for name, enabled in CAPABILITIES.items() if enabled)


def set_capability(capability: str, enabled: bool) -> None:
    """Toggle a capability at runtime. Used by feature-flag tests and the
    operational override hook. The change is process-local — config
    edits still need a service restart to take effect across processes.
    """
    CAPABILITIES[capability] = bool(enabled)
