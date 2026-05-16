"""
Domain Entity: Chat
====================
Represents a single conversation session between a user and the RAG system.

The summary field is central to the history management strategy. Rather than
sending the full message history on every turn, the summarization use case
compresses past messages into this field. ChatConversationUseCase reads it
and injects it as a single context block, reducing token usage while
preserving conversational continuity.

The needs_summarization property allows the router to determine whether to
schedule a background summarization task after each message, without
containing any of that logic itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class Chat(BaseModel):
    chat_id: Optional[str] = Field(default=None)
    user_id: str = Field(...)
    title: str = Field(default="New Diagnosis", min_length=1)
    summary: Optional[str] = Field(default=None)
    message_count: int = Field(default=0, ge=0)
    summarize_every_n_messages: int = Field(default=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    @property
    def needs_summarization(self) -> bool:
        """
        Return True if the chat has reached the summarization threshold.

        The threshold is checked as a modulo condition so that summarization
        triggers every N messages rather than only at the first multiple of N.
        """
        return (
            self.message_count > 0
            and self.message_count % self.summarize_every_n_messages == 0
        )
