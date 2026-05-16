"""
Value Object: SummarizationResult
===================================
The structured output of ChatSummarizationUseCase.

Carries the generated summary text, the number of messages that were
included in the input, and a success flag indicating whether the summary
was successfully persisted to the Chat document in Firestore.
"""
from pydantic import BaseModel, Field


class SummarizationResult(BaseModel):
    summary: str = Field(default="")
    messages_summarized: int = Field(default=0, ge=0)
    success: bool = Field(default=False)

    model_config = {"frozen": True}
