"""
Port: IMessageRepository
==========================
Abstract interface for Message persistence and retrieval operations.

get_recent_n() and get_by_chat_id() serve different use cases:
    get_recent_n()     - Returns the last N messages for summarization and
                         history injection. Always ordered oldest to newest.
    get_by_chat_id()   - Returns a paginated list for the history endpoint.
                         Intended for client display rather than LLM context.
"""

from abc import ABC, abstractmethod

from ...domain.entities.message import Message


class IMessageRepository(ABC):

    @abstractmethod
    async def save(self, message: Message) -> Message:
        """
        Persist a Message entity and return it with message_id populated.
        """
        ...

    @abstractmethod
    async def get_by_chat_id(
        self,
        chat_id: str,
        page_number: int = 1,
        page_size: int = 20,
    ) -> list[Message]:
        """Return a paginated list of messages for a chat session."""
        ...

    @abstractmethod
    async def get_recent_n(self, chat_id: str, n: int) -> list[Message]:
        """
        Return the most recent n messages in chronological order.

        Used by ChatConversationUseCase to build the LLM context window
        and by ChatSummarizationUseCase to select messages to condense.
        """
        ...

    @abstractmethod
    async def delete_by_chat_id(self, chat_id: str) -> int:
        """Delete all messages for a chat session and return the count deleted."""
        ...
