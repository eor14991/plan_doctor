"""
Adapter: FirebaseMessageRepository
=====================================
Implements IMessageRepository using Firestore sub-collections.

Firestore path: chats/{chatId}/messages/{messageId}
Document fields:
    role:        str   - "user" or "assistant".
    prompt:      str   - The text typed by the user. May be None if only a
                         label was sent.
    ragResponse: str   - The assistant's generated answer. None for user messages.
    imageURL:    str   - Remote URL of an uploaded plant image. May be None.
    label:       str   - Disease label produced by the mobile ML model. May be None.
    confidence:  float - Model confidence in the label, in the range [0, 1].
    timestamp:   ts    - UTC timestamp of the message.
    isSynced:    bool  - Always True when written by this service. The Flutter
                         client uses this field to track local-only drafts.

Sub-collection rationale:
    The Flutter client queries messages as a sub-collection of chats. Storing
    messages in a top-level collection with a chatId filter would also work, but
    using sub-collections aligns with the mobile client's data model and keeps
    related data co-located in Firestore.

Retrieval ordering:
    Messages are fetched in DESCENDING timestamp order and then reversed before
    being returned. This is necessary because Firestore's index-based ordering
    is more efficient with the newest documents first, while the LLM context
    window requires chronological (oldest first) ordering.
"""
from __future__ import annotations

import logging

from google.cloud import firestore

from ...core.domain.entities.message import Message
from ...core.ports.repositories.i_message_repository import IMessageRepository

logger = logging.getLogger(__name__)

_CHATS_COLLECTION = "chats"
_MESSAGES_SUBCOLLECTION = "messages"


class FirebaseMessageRepository(IMessageRepository):

    def __init__(self, db: firestore.AsyncClient) -> None:
        self._db = db

    def _messages_col(self, chat_id: str):
        """Return a reference to the messages sub-collection for a given chat."""
        return (
            self._db
            .collection(_CHATS_COLLECTION)
            .document(chat_id)
            .collection(_MESSAGES_SUBCOLLECTION)
        )

    async def save(self, message: Message) -> Message:
        """
        Persist a Message entity to the messages sub-collection.

        If message.message_id is set, the document at that identifier is
        overwritten. Otherwise Firestore generates a new ID which is written
        back to message.message_id before returning.

        Args:
            message: The Message entity to persist.

        Returns:
            The same Message instance with message_id populated.
        """
        col = self._messages_col(message.chat_id)
        data = self._to_firestore(message)

        if message.message_id:
            await col.document(message.message_id).set(data)
        else:
            doc_ref = col.document()
            await doc_ref.set(data)
            message.message_id = doc_ref.id

        return message

    async def get_recent_n(self, chat_id: str, n: int) -> list[Message]:
        """
        Return the most recent n messages for a chat in chronological order.

        Documents are fetched newest-first to use the Firestore descending
        index efficiently, then reversed so the caller receives them in the
        order they were sent.

        Args:
            chat_id: The Firestore chat document ID.
            n:       Maximum number of messages to return.

        Returns:
            A list of Message entities ordered from oldest to newest.
        """
        col = self._messages_col(chat_id)
        docs = (
            col
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(n)
            .stream()
        )
        messages = []
        async for doc in docs:
            messages.append(self._to_entity(doc.id, chat_id, doc.to_dict()))

        messages.reverse()
        return messages

    async def get_by_chat_id(self, chat_id: str, page_number: int = 1, page_size: int = 20) -> list[Message]:
        col = self._messages_col(chat_id)
        offset = (page_number - 1) * page_size
        docs = (
            col.order_by("timestamp", direction=firestore.Query.ASCENDING)
            .offset(offset)
            .limit(page_size)
            .stream()
        )
        messages = []
        async for doc in docs:
            messages.append(self._to_entity(doc.id, chat_id, doc.to_dict()))
        return messages

    async def delete_by_chat_id(self, chat_id: str) -> int:
        col = self._messages_col(chat_id)
        count = 0
        async for doc in col.stream():
            await doc.reference.delete()
            count += 1
        return count

    @staticmethod
    def _to_firestore(message: Message) -> dict:
        """
        Convert a Message domain entity to a Firestore document dictionary.

        Field names use camelCase to match the Flutter mobile client schema.
        """
        return {
            "role": message.role,
            "prompt": message.prompt,
            "ragResponse": message.rag_response,
            "imageURL": message.image_url,
            "label": message.label,
            "confidence": message.confidence,
            "timestamp": message.created_at,
            "isSynced": True,
        }

    @staticmethod
    def _to_entity(doc_id: str, chat_id: str, data: dict) -> Message:
        """Convert a Firestore document dictionary to a Message domain entity."""
        return Message(
            message_id=doc_id,
            chat_id=chat_id,
            role=data.get("role", "user"),
            prompt=data.get("prompt"),
            rag_response=data.get("ragResponse"),
            image_url=data.get("imageURL"),
            label=data.get("label"),
            confidence=data.get("confidence"),
        )