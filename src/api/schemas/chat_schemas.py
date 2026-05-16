"""
API Schemas: Chat routes.
"""
from typing import Optional

from pydantic import BaseModel, Field


class StartChatRequest(BaseModel):
    title: Optional[str] = Field(default="New Diagnosis", max_length=100)
    chat_id: Optional[str] = Field(default=None)


class MessageRequest(BaseModel):
    """
    Request body for sending a message to the RAG pipeline.

    At least one of prompt or label must be present. The router validates
    this and returns HTTP 400 if both are absent.
    """
    prompt: str | None = Field(default=None, max_length=1000)
    label: str | None = Field(default=None, description="Plant disease label from the mobile ML model.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    image_url: str | None = Field(default=None)


class MessageResponse(BaseModel):
    answer: str
    sources_count: int
    chat_id: str
