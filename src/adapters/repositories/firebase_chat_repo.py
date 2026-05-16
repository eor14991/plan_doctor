"""
Adapter: FirebaseChatRepository
================================
Implements IChatRepository using Firestore as the persistence backend.

Firestore collection: chats/{chatId}
Document fields:
    userId:       str   - Firebase UID of the owning user.
    title:        str   - Display title for the chat session.
    summary:      str   - Condensed history produced by the summarization use case.
    messageCount: int   - Running total of messages, maintained atomically.
    createdAt:    ts    - UTC timestamp set on first save.
    updatedAt:    ts    - UTC timestamp updated on every write.

Field naming convention:
    Firestore documents use camelCase to match the Flutter mobile client
    that reads them directly. The adapter performs the conversion between
    snake_case domain entities and camelCase Firestore documents so that
    no other layer is aware of the naming difference.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

from ...core.domain.entities.chat import Chat
from ...core.ports.repositories.i_chat_repository import IChatRepository

logger = logging.getLogger(__name__)

_COLLECTION = "chats"


class FirebaseChatRepository(IChatRepository):

    def __init__(self, db: firestore.AsyncClient) -> None:
        self._db = db
        self._col = db.collection(_COLLECTION)

    async def get_by_user_id(
        self, user_id: str, page_number: int = 1, page_size: int = 20
    ) -> list[Chat]:
        offset = (page_number - 1) * page_size
        docs = (
            self._col.where("userId", "==", user_id)
            .order_by("updatedAt", direction=firestore.Query.DESCENDING)
            .offset(offset)
            .limit(page_size)
            .stream()
        )

        chats = []
        async for doc in docs:
            chats.append(self._to_entity(doc.id, doc.to_dict()))

        return chats

    async def delete_by_id(self, chat_id: str) -> bool:
        try:
            await self._col.document(chat_id).delete()
            return True
        except Exception:
            logger.error("Failed to delete chat.", extra={"chat_id": chat_id}, exc_info=True)
            return False

    async def save(self, chat: Chat) -> Chat:
        """
        Persist a Chat entity to Firestore.

        If chat.chat_id is already set the document is overwritten at that
        identifier. If it is None, Firestore generates a new document ID
        which is then written back to chat.chat_id before returning.

        Args:
            chat: The Chat entity to persist.

        Returns:
            The same Chat instance with chat_id populated.
        """
        data = self._to_firestore(chat)
        if chat.chat_id:
            await self._col.document(chat.chat_id).set(data)
        else:
            doc_ref = self._col.document()
            await doc_ref.set(data)
            chat.chat_id = doc_ref.id
        return chat

    async def get_by_id(self, chat_id: str) -> Optional[Chat]:
        """
        Retrieve a Chat entity by its Firestore document ID.

        Returns None if no document exists with that identifier.
        """
        doc = await self._col.document(chat_id).get()
        if not doc.exists:
            return None
        return self._to_entity(doc.id, doc.to_dict())

    async def update_summary(self, chat_id: str, summary: str) -> bool:
        """
        Update only the summary field of an existing chat document.

        Uses a partial update rather than a full document write to avoid
        overwriting fields that may have changed concurrently.

        Returns:
            True on success. False if the update raised an exception.
        """
        try:
            await self._col.document(chat_id).update(
                {
                    "summary": summary,
                    "updatedAt": datetime.now(timezone.utc),
                }
            )
            return True
        except Exception:
            logger.error(
                "Failed to update chat summary.",
                extra={"chat_id": chat_id},
                exc_info=True,
            )
            return False

    async def increment_message_count(self, chat_id: str) -> int:
        """
        Atomically increment the messageCount field by 1 and return the new value.

        Uses Firestore's server-side Increment transform so that concurrent
        increments from multiple requests are safe without requiring a
        read-modify-write transaction.

        Returns:
            The updated message count, or 0 if the document does not exist.
        """
        ref = self._col.document(chat_id)
        await ref.update(
            {
                "messageCount": firestore.Increment(1),
                "updatedAt": datetime.now(timezone.utc),
            }
        )
        doc = await ref.get(field_paths=["messageCount"])
        return doc.to_dict().get("messageCount", 0) if doc.exists else 0

    @staticmethod
    def _to_firestore(chat: Chat) -> dict:
        """Convert a Chat domain entity to a Firestore document dictionary."""

        return {
            "userId": chat.user_id,
            "title": chat.title,
            "summary": chat.summary,
            "messageCount": chat.message_count,
            "createdAt": chat.created_at,
            "updatedAt": chat.updated_at,
        }

    @staticmethod
    def _to_entity(doc_id: str, data: dict) -> Chat:
        """Convert a Firestore document dictionary to a Chat domain entity."""
        return Chat(
            chat_id=doc_id,
            user_id=data.get("userId", ""),
            title=data.get("title", "New Diagnosis"),
            summary=data.get("summary"),
            message_count=data.get("messageCount", 0),
        )
