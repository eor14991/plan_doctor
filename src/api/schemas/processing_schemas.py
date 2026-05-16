from pydantic import BaseModel,Field


class ProcessingRequest(BaseModel):
    chunk_size: int = Field(default=450, gt=0, le=4096)
    overlap_size: int = Field(default=40, ge=0)
    do_reset: bool = Field(default=False)