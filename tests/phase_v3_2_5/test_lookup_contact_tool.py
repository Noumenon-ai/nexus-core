"""V3.2.5.5 — lookup_contact tool-level tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.google_auth_service import GoogleAuthError
from services.google_people_service import GooglePeopleService
from services.google_types import GoogleContact
from services.read_tools import make_lookup_contact


@pytest.mark.asyncio
async def test_lookup_contact_returns_real_contacts():
    fake = MagicMock(spec=GooglePeopleService)
    fake.lookup_contact = AsyncMock(return_value=[
        GoogleContact(resource_name='people/c1', display_name='Alice', emails=['a@x.com'], phones=[]),
    ])
    tool = make_lookup_contact(fake)
    result = await tool(user_id='user-1', query='alice')
    assert result.success is True
    assert result.data['count'] == 1
    assert result.data['contacts'][0]['display_name'] == 'Alice'


@pytest.mark.asyncio
async def test_lookup_contact_passes_user_id_and_query():
    fake = MagicMock(spec=GooglePeopleService)
    fake.lookup_contact = AsyncMock(return_value=[])
    tool = make_lookup_contact(fake)
    await tool(user_id='user-A', query='bob')
    fake.lookup_contact.assert_awaited_once_with('user-A', 'bob')


@pytest.mark.asyncio
async def test_lookup_contact_rejects_empty_query():
    fake = MagicMock(spec=GooglePeopleService)
    fake.lookup_contact = AsyncMock()
    tool = make_lookup_contact(fake)
    result = await tool(user_id='user-1', query='   ')
    assert result.success is False
    fake.lookup_contact.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_contact_handles_auth_error():
    fake = MagicMock(spec=GooglePeopleService)
    fake.lookup_contact = AsyncMock(side_effect=GoogleAuthError('expired'))
    tool = make_lookup_contact(fake)
    result = await tool(user_id='user-1', query='alice')
    assert result.success is False
    assert 'reconnect' in result.error.lower()
