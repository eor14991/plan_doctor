"""API Schemas: Data routes. do_reset: bool (not int) — explicit, no conversion needed."""

from typing import Optional

from pydantic import BaseModel, Field


class ProcessFileRequest(BaseModel):
    file_id: str = Field(..., min_length=1)
    chunk_size: int = Field(default=450, gt=0, le=4096)
    overlap_size: int = Field(default=40, ge=0)
    do_reset: bool = Field(default=False)


class ProcessAllRequest(BaseModel):
    chunk_size: int = Field(default=450, gt=0, le=4096)
    overlap_size: int = Field(default=40, ge=0)
    do_reset: bool = Field(default=False)
