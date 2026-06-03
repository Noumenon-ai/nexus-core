"""V3.2.5.5 — create_contact tool-level tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.auto_write_tools import make_auto_write_tools
from services.google_auth_service import GoogleAuthError
from services.google_people_service import GooglePeopleService
from services.google_types import GoogleContact


def _build_tools(*, people_service=None):
    return {meta['name']: fn for fn, meta in make_auto_write_tools(
        reminders_repository=MagicMock(),
        tasks_repository=MagicMock(),
        memories_repository=MagicMock(),
        google_people_service=people_service,
    )}


@pytest.mark.asyncio
async def test_create_contact_calls_service_with_kwargs():
    fake = MagicMock(spec=GooglePeopleService)
    fake.create_contact = AsyncMock(return_value=GoogleContact(
        resource_name='people/c1', display_name='Alice', emails=['a@x.com'], phones=[],
    ))
    tools = _build_tools(people_service=fake)
    result = await tools['create_contact'](user_id='user-1', name='Alice', email='a@x.com')
    assert result.success is True
    assert result.data['created'] is True
    assert result.data['display_name'] == 'Alice'
    fake.create_contact.assert_awaited_once_with('user-1', name='Alice', email='a@x.com', phone=None)


@pytest.mark.asyncio
async def test_create_contact_returns_not_configured_when_service_missing():
    tools = _build_tools(people_service=None)
    result = await tools['create_contact'](user_id='user-1', name='Alice')
    assert result.success is True
    assert result.data['created'] is False
    assert result.data['reason'] == 'people_not_configured'


@pytest.mark.asyncio
async def test_create_contact_handles_value_error():
    fake = MagicMock(spec=GooglePeopleService)
    fake.create_contact = AsyncMock(side_effect=ValueError('name must be non-empty'))
    tools = _build_tools(people_service=fake)
    result = await tools['create_contact'](user_id='user-1', name='Alice')
    assert result.data['created'] is False
    assert result.data['reason'] == 'invalid_input'


@pytest.mark.asyncio
async def test_create_contact_handles_auth_error():
    fake = MagicMock(spec=GooglePeopleService)
    fake.create_contact = AsyncMock(side_effect=GoogleAuthError('expired'))
    tools = _build_tools(people_service=fake)
    result = await tools['create_contact'](user_id='user-1', name='Alice')
    assert result.data['created'] is False
    assert result.data['reason'] == 'auth_error'


@pytest.mark.asyncio
async def test_create_contact_passes_user_id_through():
    fake = MagicMock(spec=GooglePeopleService)
    fake.create_contact = AsyncMock(return_value=GoogleContact(
        resource_name='people/c1', display_name='X', emails=[], phones=[],
    ))
    tools = _build_tools(people_service=fake)
    await tools['create_contact'](user_id='user-A', name='X')
    fake.create_contact.assert_awaited_once_with('user-A', name='X', email=None, phone=None)
