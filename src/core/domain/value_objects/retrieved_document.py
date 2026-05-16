"""
Value Object: RetrievedDocument
================================
A single search result returned by IVectorStore.search_by_vector().

Using a dedicated value object rather than exposing Qdrant's ScoredPoint
type keeps the vector store implementation detail confined to the adapter.
The use case and router work with this type regardless of which vector
database backs the search.

The score field represents cosine or dot-product similarity as reported by
the vector store, normalised to the range [0.0, 1.0].
"""

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    text: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)

    model_config = {"frozen": True}
