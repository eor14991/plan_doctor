"""
Value Object: ChatResponse
===========================
The structured output of ChatConversationUseCase returned to the router.

The sources list contains the knowledge chunks retrieved from Qdrant that
were used to ground the generated answer. Exposing them allows the API
to optionally surface citations to the client for transparency.

The needs_summarization flag is set by the use case based on the atomically
incremented message count. The router reads this flag to decide whether to
schedule a background summarization task, eliminating the need for a second
Firestore read after execute() returns.
"""
from pydantic import BaseModel, Field

from .retrieved_document import RetrievedDocument


class ChatResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[RetrievedDocument] = Field(default_factory=list)
    needs_summarization: bool = Field(default=False)

    model_config = {"frozen": True}