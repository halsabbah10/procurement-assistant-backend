from typing import Literal

from pydantic import BaseModel, Field

# A full aggregation pipeline as JSON-ish text can legitimately run a few
# thousand characters for a query with several stages; this bounds it well
# above any real pipeline while still capping worst-case request cost.
MAX_QUERY_LENGTH = 20_000


class ExportRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    format: Literal["csv", "json"] = "csv"
