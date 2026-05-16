"""
Domain Entity: Message
========================
Represents a single turn in a chat conversation.

Field mapping to the Flutter mobile client schema:
    prompt       - Free-text question typed by the user. May be None when
                   the user sends only an image or a label.
    rag_response - The assistant's generated answer. None for user messages.
    label        - Plant disease classification produced by the mobile ML model
                   (e.g. "Tomato___Bacterial_spot"). May be None.
    confidence   - The model's confidence in the label, in the range [0, 1].
    image_url    - Remote URL of the uploaded plant image. Stored for audit
                   purposes; the RAG pipeline does not process images directly.

The display_text property composes a single query string from the label and
prompt. This string is what gets embedded and used for Qdrant retrieval,
allowing the system to search the knowledge base using the full context of
what the user communicated, including the ML detection result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    message_id: Optional[str] = Field(default=None)
    chat_id: str = Field(...)
    role: Literal["user", "assistant", "system"]
    prompt: Optional[str] = Field(default=None)
    rag_response: Optional[str] = Field(default=None)
    image_url: Optional[str] = Field(default=None)
    label: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    @property
    def display_text(self) -> str:
        """
        Compose the text string used for embedding and retrieval.

        When a label is present, it is prepended with its confidence score so
        that the embedding captures the disease context alongside the user's
        question. Returns an empty string if neither label nor prompt is set.
        """
        parts = []
        if self.label:
            confidence_str = f" ({self.confidence:.0%})" if self.confidence else ""
            parts.append(f"[Detected: {self.label}{confidence_str}]")
        if self.prompt:
            parts.append(self.prompt)
        return " ".join(parts)

    @property
    def content(self) -> str:
        """
        Dynamically return the correct text body based on the role
        so the generation service can read 'msg.content'.
        """
        if self.role == "user":
            return self.display_text
        elif self.role == "assistant":
            return self.rag_response or ""
        else:  # For the injected system prompt
            return self.prompt or ""
