from typing import Literal

from pydantic import BaseModel


class ExportRequest(BaseModel):
    query: str
    format: Literal["csv", "json"] = "csv"
