"""
Port: IChatRepository
=======================
Abstract interface for Chat session persistence operations.

All methods accept and return the Chat domain entity or Python primitives.
No Firestore types, ObjectId, or SDK-specific constructs appear in this
interface. The FirebaseChatRepository adapter handles all translation.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ...domain.entities.chat import Chat


class IChatRepository(ABC):

    @abstractmethod
    async def get_by_id(self, chat_id: str) -> Optional[Chat]:
        """Return a Chat by its identifier, or None if not found."""
        ...

    @abstractmethod
    async def save(self, chat: Chat) -> Chat:
        """
        Persist a Chat entity and return it with chat_id populated.

        If chat.chat_id is set, the existing document is overwritten.
        If chat.chat_id is None, a new document is created and the
        generated identifier is written back to the returned entity.
        """
        ...

    @abstractmethod
    async def get_by_user_id(
        self, user_id: str, page_number: int = 1, page_size: int = 20
    ) -> list[Chat]:
        """Return a page of chat sessions belonging to a user."""
        ...

    @abstractmethod
    async def update_summary(self, chat_id: str, summary: str) -> bool:
        """
        Update only the summary field of an existing chat document.

        Returns True on success, False if the update failed.
        """
        ...

    @abstractmethod
    async def increment_message_count(self, chat_id: str) -> int:
        """
        Atomically increment the message counter by 1.

        Returns the updated count after the increment.
        """
        ...

    @abstractmethod
    async def delete_by_id(self, chat_id: str) -> bool:
        """Delete a chat session and return True on success."""
        ...
