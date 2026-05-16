"""
Domain Entity: Document
========================
Represents a knowledge file uploaded through the web portal.

The status field tracks the document through the processing pipeline:
    uploaded   - File received and recorded. Processing has not started.
    processing - Chunking and embedding are in progress.
    processed  - Pipeline completed. Chunks are indexed in Qdrant.
    failed     - Pipeline encountered an unrecoverable error.

The user_id field records who uploaded the document. It is used for
access control in the upload and processing routers to ensure that
users can only manage their own documents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: Optional[str] = Field(default=None)
    user_id: str = Field(...)
    file_name: str = Field(..., min_length=1)
    file_size: int = Field(default=0, ge=0)
    file_path: Optional[str] = Field(default=None)
    status: Literal["uploaded", "processing", "processed", "failed"] = "uploaded"
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}
