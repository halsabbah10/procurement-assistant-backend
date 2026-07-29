"""Real MongoDB + real FastAPI app (httpx ASGITransport), not mocks — this
app's own testing standard. Verifies the anonymous client_id scoping added
after this project's audit found GET /api/conversations returned every
conversation from every visitor with no filter, and rename/delete/read had
no ownership check at all."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.client import get_checkpoint_database
from app.db.conversations import touch_conversation
from app.main import app


@pytest.fixture(autouse=True)
async def _clean_conversations():
    db = get_checkpoint_database()
    yield
    await db.conversations.delete_many({})


async def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_list_conversations_only_returns_the_caller_s_own():
    owner_a = str(uuid.uuid4())
    owner_b = str(uuid.uuid4())
    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())
    await touch_conversation(conv_a, "question from A", owner_a)
    await touch_conversation(conv_b, "question from B", owner_b)

    async with await _client() as client:
        resp_a = await client.get("/api/conversations", headers={"X-Client-Id": owner_a})
        resp_b = await client.get("/api/conversations", headers={"X-Client-Id": owner_b})

    ids_a = {c["id"] for c in resp_a.json()}
    ids_b = {c["id"] for c in resp_b.json()}
    assert ids_a == {conv_a}
    assert ids_b == {conv_b}


@pytest.mark.asyncio
async def test_list_conversations_without_header_returns_empty_not_everything():
    owner_a = str(uuid.uuid4())
    await touch_conversation(str(uuid.uuid4()), "question from A", owner_a)

    async with await _client() as client:
        resp = await client.get("/api/conversations")

    assert resp.json() == []


@pytest.mark.asyncio
async def test_rename_by_a_different_client_id_is_404_not_a_silent_success():
    owner = str(uuid.uuid4())
    attacker = str(uuid.uuid4())
    conv = str(uuid.uuid4())
    await touch_conversation(conv, "original question", owner)

    async with await _client() as client:
        resp = await client.patch(
            f"/api/conversations/{conv}",
            json={"title": "hijacked"},
            headers={"X-Client-Id": attacker},
        )
    assert resp.status_code == 404

    db = get_checkpoint_database()
    doc = await db.conversations.find_one({"_id": conv})
    assert doc["title"] != "hijacked"


@pytest.mark.asyncio
async def test_delete_by_a_different_client_id_is_404_and_does_not_delete():
    owner = str(uuid.uuid4())
    attacker = str(uuid.uuid4())
    conv = str(uuid.uuid4())
    await touch_conversation(conv, "original question", owner)

    async with await _client() as client:
        resp = await client.delete(f"/api/conversations/{conv}", headers={"X-Client-Id": attacker})
    assert resp.status_code == 404

    db = get_checkpoint_database()
    doc = await db.conversations.find_one({"_id": conv})
    assert doc["is_deleted"] is False


@pytest.mark.asyncio
async def test_owner_can_rename_and_delete_their_own_conversation():
    owner = str(uuid.uuid4())
    conv = str(uuid.uuid4())
    await touch_conversation(conv, "original question", owner)

    async with await _client() as client:
        rename_resp = await client.patch(
            f"/api/conversations/{conv}",
            json={"title": "renamed by owner"},
            headers={"X-Client-Id": owner},
        )
        assert rename_resp.status_code == 200

        delete_resp = await client.delete(
            f"/api/conversations/{conv}", headers={"X-Client-Id": owner}
        )
        assert delete_resp.status_code == 204

    db = get_checkpoint_database()
    doc = await db.conversations.find_one({"_id": conv})
    assert doc["title"] == "renamed by owner"
    assert doc["is_deleted"] is True


@pytest.mark.asyncio
async def test_a_headerless_request_cannot_touch_a_legacy_ownerless_conversation():
    # A conversation created before this scoping existed has no client_id
    # field at all. Mongo's equality match treats {"client_id": None} as
    # "field is null OR absent" — a header-less request must still be
    # rejected, not silently match every such legacy conversation.
    conv = str(uuid.uuid4())
    await touch_conversation(conv, "a legacy conversation", client_id=None)

    async with await _client() as client:
        resp = await client.delete(f"/api/conversations/{conv}")
    assert resp.status_code == 404

    db = get_checkpoint_database()
    doc = await db.conversations.find_one({"_id": conv})
    assert doc["is_deleted"] is False


@pytest.mark.asyncio
async def test_malformed_client_id_header_behaves_like_no_header():
    conv = str(uuid.uuid4())
    await touch_conversation(conv, "a legacy conversation", client_id=None)

    async with await _client() as client:
        resp = await client.get("/api/conversations", headers={"X-Client-Id": "not-a-uuid"})
    assert resp.json() == []
