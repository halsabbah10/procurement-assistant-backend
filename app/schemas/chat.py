import uuid

from pydantic import BaseModel, Field, field_validator

# Generous enough for any legitimate detailed question (~800-1000 tokens of
# natural language), small enough to bound worst-case cost on a request
# that fans out to Haiku -> Sonnet -> possibly Opus -> Haiku enrichment, all
# priced per input token. A count-based rate limiter (RateLimiter) does
# nothing to cap the size of any individual request within its quota.
MAX_MESSAGE_LENGTH = 4_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: str

    @field_validator("conversation_id")
    @classmethod
    def _conversation_id_must_be_uuid(cls, value: str) -> str:
        # Every conversation_id is generated client-side via
        # crypto.randomUUID() (see frontend/src/hooks/useConversation.ts) and
        # used directly as both a LangGraph thread_id and a Mongo _id —
        # enforcing the format here rejects malformed/oversized values with
        # a clean 422 instead of a generic 500 surfacing from deep inside
        # the checkpointer or a Mongo driver error.
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("conversation_id must be a UUID") from exc
        return value
