"""
Domain Entity: DataChunk
=========================
Represents a single text segment extracted from a Document during the
chunking stage of the ingestion pipeline. DataChunk is the atom of
information in the RAG system: embedding, vector indexing, and retrieval
all operate on individual chunks rather than on entire documents.

The chunk_order field records the position of this chunk within the source
document. It is used to reconstruct surrounding context when needed and to
detect whether re-processing a document produced the same segmentation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DataChunk(BaseModel):
    chunk_id: Optional[str] = Field(default=None)
    doc_id: str = Field(...)
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_order: int = Field(..., ge=0)

    model_config = {"frozen": False}
