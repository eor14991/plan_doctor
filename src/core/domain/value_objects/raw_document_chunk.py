"""
Value Object: RawDocumentChunk
================================
An immutable, framework-free representation of a single text segment
produced by the document chunking adapter.

The IDocumentChunker port returns this type rather than the DataChunk domain
entity. This separation ensures that the adapter layer does not assign domain
identifiers (doc_id, chunk_id) or make persistence decisions. The use case
receives RawDocumentChunk objects and constructs DataChunk entities with the
correct identifiers before delegating to the chunk repository.
"""

from pydantic import BaseModel, Field


class RawDocumentChunk(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)
    order: int = Field(..., ge=0)

    model_config = {"frozen": True}
