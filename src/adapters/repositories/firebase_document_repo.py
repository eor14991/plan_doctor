"""
Adapter: FirebaseDocumentRepository
======================================
Implements IDocumentRepository using Firestore as the persistence backend.

Firestore collection: documents/{docId}
Document fields:
    userId:     str  - Firebase UID of the user who uploaded the file.
    fileName:   str  - Unique sanitised filename as stored on disk.
    fileSize:   int  - Size in bytes.
    filePath:   str  - Absolute path to the file on the local storage volume.
    status:     str  - Lifecycle status: uploaded | processing | processed | failed.
    uploadedAt: ts   - UTC timestamp set when the document was first saved.

The status field is the primary signal used by the web portal to track
progress. The processing use case updates it via update_status() at each
stage of the pipeline rather than rewriting the entire document.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from google.cloud import firestore

from ...core.domain.entities.document import Document
from ...core.ports.repositories.i_document_repository import IDocumentRepository

logger = logging.getLogger(__name__)

_COLLECTION = "documents"


class FirebaseDocumentRepository(IDocumentRepository):

    def __init__(self, db: firestore.AsyncClient) -> None:
        self._db = db
        self._col = db.collection(_COLLECTION)

    async def get_all(self, user_id: Optional[str] = None) -> List[Document]:
        query = self._col.where("userId", "==", user_id) if user_id else self._col
        result = []
        async for doc in query.stream():
            result.append(self._to_entity(doc.id, doc.to_dict()))
        return result

    async def delete_by_id(self, doc_id: str) -> bool:
        try:
            await self._col.document(doc_id).delete()
            return True
        except Exception:
            logger.error("Failed to delete document.", extra={"doc_id": doc_id}, exc_info=True)
            return False

    async def save(self, document: Document) -> Document:
        """
        Persist a Document entity to Firestore.

        If document.doc_id is set, the document at that identifier is
        overwritten. Otherwise Firestore generates a new ID.

        Returns:
            The same Document instance with doc_id populated.
        """
        data = self._to_firestore(document)
        if document.doc_id:
            await self._col.document(document.doc_id).set(data)
        else:
            doc_ref = self._col.document()
            await doc_ref.set(data)
            document.doc_id = doc_ref.id
        return document

    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a Document entity by its Firestore document ID.

        Returns None if no document exists with that identifier.
        """
        doc = await self._col.document(doc_id).get()
        if not doc.exists:
            return None
        return self._to_entity(doc.id, doc.to_dict())

    async def update_status(
        self, doc_id: str, status: Literal["uploaded", "processing", "processed", "failed"]
    ) -> bool:
        """
        Update only the status field of an existing document record.

        Uses a partial update to avoid overwriting unrelated fields that may
        have changed concurrently.

        Returns:
            True on success. False if the Firestore update raised an exception.
        """
        try:
            await self._col.document(doc_id).update({"status": status})
            return True
        except Exception:
            logger.error(
                "Failed to update document status.", extra={"doc_id": doc_id}, exc_info=True
            )
            return False

    async def find_all_by_user(self, user_id: str) -> list[Document]:
        """
        Return all document records belonging to a given user.

        Returns an empty list if no documents are found.
        """
        docs = self._col.where("userId", "==", user_id).stream()
        result = []
        async for doc in docs:
            result.append(self._to_entity(doc.id, doc.to_dict()))
        return result

    @staticmethod
    def _to_firestore(doc: Document) -> dict:
        """Convert a Document domain entity to a Firestore document dictionary."""
        return {
            "userId": doc.user_id,
            "fileName": doc.file_name,
            "fileSize": doc.file_size,
            "filePath": doc.file_path,
            "status": doc.status,
            "uploadedAt": doc.uploaded_at,
        }

    @staticmethod
    def _to_entity(doc_id: str, data: dict) -> Document:
        """Convert a Firestore document dictionary to a Document domain entity."""
        return Document(
            doc_id=doc_id,
            user_id=data.get("userId", ""),
            file_name=data.get("fileName", ""),
            file_size=data.get("fileSize", 0),
            file_path=data.get("filePath"),
            status=data.get("status", "uploaded"),
        )
