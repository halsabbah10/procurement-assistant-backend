import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.agent.graph import run_agent
from app.core.rate_limit import RateLimiter
from app.db.conversations import touch_conversation
from app.schemas.chat import ChatRequest

router = APIRouter()
limiter = RateLimiter(per_minute=20, daily_cap=200)


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.check(client_ip):
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'text': 'Demo limit reached — try again shortly.'})}\n\n"]),
            media_type="text/event-stream",
            status_code=429,
        )
    limiter.record(client_ip)
    await touch_conversation(body.conversation_id, body.message)

    async def event_stream():
        try:
            async for chunk in run_agent(body.message, thread_id=body.conversation_id):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': 'An error occurred while processing your request.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
