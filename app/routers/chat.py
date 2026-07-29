import json

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.agent.graph import run_agent
from app.core.client_id import get_client_id
from app.core.config import get_settings
from app.core.rate_limit import RateLimiter
from app.db.conversations import touch_conversation
from app.schemas.chat import ChatRequest

log = structlog.get_logger()
router = APIRouter()
_settings = get_settings()
limiter = RateLimiter(
    per_minute=_settings.rate_limit_per_minute, daily_cap=_settings.daily_request_cap
)


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest, client_id: str | None = Depends(get_client_id)):
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.check(client_ip):
        return StreamingResponse(
            iter(
                [
                    f"data: {json.dumps({'type': 'error', 'text': 'Demo limit reached — try again shortly.'})}\n\n"
                ]
            ),
            media_type="text/event-stream",
            status_code=429,
        )
    limiter.record(client_ip)
    await touch_conversation(body.conversation_id, body.message, client_id)

    async def event_stream():
        try:
            async for chunk in run_agent(body.message, thread_id=body.conversation_id):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:  # noqa: BLE001 — last-resort guard around the whole
            # stream; run_agent already handles its own model-level failures,
            # so reaching here means something outside that (e.g. a dropped
            # DB connection) — log it server-side rather than swallowing it.
            log.error("chat_stream_failure", conversation_id=body.conversation_id, error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'text': 'An error occurred while processing your request.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
