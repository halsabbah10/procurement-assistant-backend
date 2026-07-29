"""Anonymous, client-generated conversation ownership.

This app has no login system (a deliberate scope decision — see
project memory). Before this module existed, GET /api/conversations
returned every conversation from every visitor with no filter at all, and
rename/delete/read had no ownership check — any client could browse,
rename, or delete anyone else's chat history by ID. This isn't real auth
(no passwords, no accounts); it's the minimal anonymous scoping needed so
one visitor's browser only sees its own conversations, via a random UUID
the frontend generates once and sends back on every request.
"""

from __future__ import annotations

import uuid

from fastapi import Header

CLIENT_ID_HEADER = "X-Client-Id"


def get_client_id(
    x_client_id: str | None = Header(default=None, alias=CLIENT_ID_HEADER)
) -> str | None:
    """Best-effort: returns None (never raises) on a missing or malformed
    header, so callers that only need scoping-if-available (chat.py) don't
    have to special-case validation. Callers that require ownership
    (conversations.py) treat None the same as any other non-match."""
    if not x_client_id:
        return None
    try:
        uuid.UUID(x_client_id)
    except ValueError:
        return None
    return x_client_id
