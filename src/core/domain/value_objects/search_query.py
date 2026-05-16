"""Value Object: SearchQuery — immutable, bounded limit enforced."""
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    text: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    model_config = {"frozen": True}
