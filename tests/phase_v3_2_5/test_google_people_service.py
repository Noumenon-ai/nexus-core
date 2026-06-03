"""V3.2.5.5 — GooglePeopleService unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from services.google_auth_service import GoogleAuthError
from services.google_people_service import GooglePeopleService
from services.google_types import GoogleContact


def _http_error(status):
    return HttpError(resp=MagicMock(status=status), content=b'{}')


def _search_stack(response):
    request = MagicMock()
    request.execute = MagicMock(return_value=response)
    people_resource = MagicMock()
    people_resource.searchContacts.return_value = request
    service = MagicMock()
    service.people.return_value = people_resource
    return service, people_resource, request


def _create_stack(response):
    request = MagicMock()
    request.execute = MagicMock(return_value=response)
    people_resource = MagicMock()
    people_resource.createContact.return_value = request
    service = MagicMock()
    service.people.return_value = people_resource
    return service, people_resource, request


# ---------- lookup_contact ----------

@pytest.mark.asyncio
async def test_lookup_contact_passes_user_credentials():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, _, _ = _search_stack({'results': []})
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        await svc.lookup_contact('user-1', 'alice')
    auth.get_credentials.assert_called_with('user-1')


@pytest.mark.asyncio
async def test_lookup_contact_calls_people_v1_searchContacts():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, people_resource, _ = _search_stack({'results': []})
    with patch('services.google_people_service.discovery.build', return_value=api_service) as build_mock:
        await svc.lookup_contact('user-1', 'alice')
    build_mock.assert_called_with('people', 'v1', credentials=auth.get_credentials.return_value, cache_discovery=False)
    search_calls = [c for c in people_resource.searchContacts.call_args_list if c.kwargs]
    assert any(c.kwargs.get('query') == 'alice' for c in search_calls)


@pytest.mark.asyncio
async def test_lookup_contact_parses_results():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, _, _ = _search_stack({
        'results': [
            {'person': {
                'resourceName': 'people/c1',
                'names': [{'displayName': 'Alice Smith', 'givenName': 'Alice'}],
                'emailAddresses': [{'value': 'alice@example.com'}],
                'phoneNumbers': [{'value': '+15551234567'}],
            }},
        ]
    })
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        contacts = await svc.lookup_contact('user-1', 'alice')
    assert len(contacts) == 1
    assert isinstance(contacts[0], GoogleContact)
    assert contacts[0].display_name == 'Alice Smith'
    assert contacts[0].emails == ['alice@example.com']
    assert contacts[0].phones == ['+15551234567']


@pytest.mark.asyncio
async def test_lookup_contact_returns_empty_list_when_no_results():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, _, _ = _search_stack({})
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        contacts = await svc.lookup_contact('user-1', 'alice')
    assert contacts == []


@pytest.mark.asyncio
async def test_lookup_contact_rejects_empty_query():
    svc = GooglePeopleService(MagicMock())
    with pytest.raises(ValueError, match='query'):
        await svc.lookup_contact('user-1', '   ')


@pytest.mark.asyncio
async def test_lookup_contact_no_credentials_raises_auth_error():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=None)
    svc = GooglePeopleService(auth)
    with pytest.raises(GoogleAuthError):
        await svc.lookup_contact('user-1', 'alice')


@pytest.mark.asyncio
async def test_lookup_contact_maps_403_to_google_auth_error():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service = MagicMock()
    api_service.people().searchContacts().execute.side_effect = _http_error(403)
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        with pytest.raises(GoogleAuthError):
            await svc.lookup_contact('user-1', 'alice')


# ---------- create_contact ----------

@pytest.mark.asyncio
async def test_create_contact_passes_user_credentials():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, _, _ = _create_stack({'resourceName': 'people/c1', 'names': [{'givenName': 'Alice'}]})
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        await svc.create_contact('user-1', name='Alice')
    auth.get_credentials.assert_called_with('user-1')


@pytest.mark.asyncio
async def test_create_contact_passes_correct_body():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, people_resource, _ = _create_stack({'resourceName': 'people/c1', 'names': [{'givenName': 'Alice'}]})
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        await svc.create_contact('user-1', name='Alice Smith', email='alice@example.com', phone='+15551234567')
    create_calls = [c for c in people_resource.createContact.call_args_list if c.kwargs]
    assert len(create_calls) >= 1
    body = create_calls[-1].kwargs['body']
    assert body['names'] == [{'givenName': 'Alice Smith'}]
    assert body['emailAddresses'] == [{'value': 'alice@example.com'}]
    assert body['phoneNumbers'] == [{'value': '+15551234567'}]


@pytest.mark.asyncio
async def test_create_contact_omits_optional_fields_when_not_set():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, people_resource, _ = _create_stack({'resourceName': 'people/c1', 'names': [{'givenName': 'Bob'}]})
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        await svc.create_contact('user-1', name='Bob')
    body = [c for c in people_resource.createContact.call_args_list if c.kwargs][-1].kwargs['body']
    assert 'emailAddresses' not in body
    assert 'phoneNumbers' not in body


@pytest.mark.asyncio
async def test_create_contact_rejects_empty_name():
    svc = GooglePeopleService(MagicMock())
    with pytest.raises(ValueError, match='name'):
        await svc.create_contact('user-1', name='   ')


@pytest.mark.asyncio
async def test_create_contact_returns_parsed_contact():
    auth = MagicMock(); auth.get_credentials = MagicMock(return_value=object())
    svc = GooglePeopleService(auth)
    api_service, _, _ = _create_stack({
        'resourceName': 'people/c42',
        'names': [{'displayName': 'Bob Jones', 'givenName': 'Bob'}],
        'emailAddresses': [{'value': 'bob@example.com'}],
    })
    with patch('services.google_people_service.discovery.build', return_value=api_service):
        contact = await svc.create_contact('user-1', name='Bob Jones', email='bob@example.com')
    assert contact.resource_name == 'people/c42'
    assert contact.display_name == 'Bob Jones'
    assert contact.emails == ['bob@example.com']
