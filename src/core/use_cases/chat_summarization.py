"""
Use Case: ChatSummarizationUseCase
=====================================
Condenses the recent message history of a chat session into a single summary
string and persists it to the Chat document in Firestore.

Purpose:
    Sending the full message history to the LLM on every turn is expensive in
    tokens and becomes impractical as conversations grow. The summary replaces
    the raw history: on the next user turn, ChatConversationUseCase detects
    that a summary exists and injects it as a single compressed context block
    instead of fetching and sending N individual messages.

Trigger:
    Scheduled by the chat router as a BackgroundTask after a message is sent,
    when chat.needs_summarization evaluates to True. It runs after the HTTP
    response has been returned, so the user never waits for it.

Summarization model:
    The HuggingFaceSummarizationService (BART) is used rather than the main
    LLM (Grok/Cohere). BART is trained specifically for abstractive
    summarization, is faster for this task, and incurs no per-token API cost.
"""

from __future__ import annotations

import logging

from ..domain.value_objects.summarization_result import SummarizationResult
from ..ports.repositories.i_chat_repository import IChatRepository
from ..ports.repositories.i_message_repository import IMessageRepository
from ..ports.services.i_summarization_service import ISummarizationService

logger = logging.getLogger(__name__)


class ChatSummarizationUseCase:

    def __init__(
        self,
        chat_repo: IChatRepository,
        message_repo: IMessageRepository,
        summarization_service: ISummarizationService,
        messages_to_summarize: int = 10,
    ) -> None:
        self._chat_repo = chat_repo
        self._message_repo = message_repo
        self._summarizer = summarization_service
        self._n = messages_to_summarize

    async def execute(self, chat_id: str) -> SummarizationResult:
        """
        Summarise the most recent messages in a chat and persist the result.

        Args:
            chat_id: Firestore chat document ID.

        Returns:
            SummarizationResult with the generated summary, the count of
            messages that were included, and a success flag.
        """
        messages = await self._message_repo.get_recent_n(chat_id, self._n)

        if not messages:
            logger.warning("No messages available to summarise.", extra={"chat_id": chat_id})
            return SummarizationResult(summary="", messages_summarized=0, success=False)

        conversation_text = self._format_conversation(messages)

        summary = self._summarizer.summarize(
            text=conversation_text,
            max_length=600,
            min_length=50,
        )

        updated = await self._chat_repo.update_summary(chat_id, summary)

        if updated:
            logger.info(
                "Chat summary updated.",
                extra={"chat_id": chat_id, "messages_count": len(messages)},
            )
        else:
            logger.error("Failed to persist chat summary.", extra={"chat_id": chat_id})

        return SummarizationResult(
            summary=summary,
            messages_summarized=len(messages),
            success=updated,
        )

    @staticmethod
    def _format_conversation(messages) -> str:
        """
        Convert a list of Message entities into a plain text conversation
        suitable for input to the summarization model.

        Only non-empty user and assistant turns are included.
        """
        lines = []
        for msg in messages:
            if msg.role == "user":
                text = msg.display_text
                if text:
                    lines.append(f"User: {text}")
            elif msg.role == "assistant" and msg.rag_response:
                lines.append(f"Assistant: {msg.rag_response}")
        return "\n".join(lines)
