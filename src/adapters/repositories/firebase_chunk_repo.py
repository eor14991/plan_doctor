"""
Adapter: FirebaseChunkRepository
===================================
Implements IChunkRepository using Firestore as the persistence backend.

Firestore collection: chunks/{chunkId}
Document fields:
    docId:         str  - Firestore document ID of the parent Document record.
    chunkText:     str  - The extracted text content of this chunk.
    chunkOrder:    int  - Position of this chunk within the source document.
    chunkMetadata: dict - Arbitrary metadata from the document splitter
                          (e.g. header hierarchy, file_id).
    createdAt:     ts   - UTC timestamp set on insertion.

Bulk write strategy:
    Firestore does not support bulk inserts equivalent to MongoDB's bulk_write.
    The WriteBatch API is used instead. A single batch accepts a maximum of 500
    operations. The constant _FIRESTORE_BATCH_LIMIT is set to 450 to stay below
    that ceiling and account for potential internal Firestore overhead.

Bulk delete strategy:
    Firestore has no DELETE WHERE equivalent. To delete all chunks for a given
    document, the repository first streams the matching document references and
    then deletes them in batches using the same WriteBatch mechanism.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from google.cloud import firestore

from ...core.domain.entities.chunk import DataChunk
from ...core.ports.repositories.i_chunk_repository import IChunkRepository

logger = logging.getLogger(__name__)

_COLLECTION = "chunks"
_FIRESTORE_BATCH_LIMIT = 450


class FirebaseChunkRepository(IChunkRepository):

    def __init__(self, db: firestore.AsyncClient) -> None:
        self._db = db
        self._col = db.collection(_COLLECTION)

    async def get_by_doc_id(
        self, doc_id: str, page_number: int = 1, page_size: int = 50
    ) -> List[DataChunk]:
        """
        Retrieve chunks for a given document with pagination, ordered by their
        chunk order so they can be reassembled or read chronologically.
        """
        offset = (page_number - 1) * page_size
        docs = (
            self._col.where("docId", "==", doc_id)
            .order_by("chunkOrder", direction=firestore.Query.ASCENDING)
            .offset(offset)
            .limit(page_size)
            .stream()
        )

        chunks = []
        async for doc in docs:
            chunks.append(self._to_entity(doc.id, doc.to_dict()))

        return chunks

    async def insert_bulk(
        self, chunks: list[DataChunk], batch_size: int = _FIRESTORE_BATCH_LIMIT
    ) -> int:
        """
        Insert a list of DataChunk entities in batches using Firestore WriteBatch.

        Each batch is committed atomically. If a batch fails, the error is logged
        and the method continues with the next batch so that a partial failure
        does not stop the entire ingestion pipeline.

        Args:
            chunks:     The list of DataChunk entities to insert.
            batch_size: Maximum number of operations per Firestore batch.
                        Must not exceed 500.

        Returns:
            The total number of chunks successfully committed across all batches.
        """
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch = self._db.batch()

            for chunk in batch_chunks:
                doc_ref = self._col.document()
                batch.set(
                    doc_ref,
                    {
                        "docId": chunk.doc_id,
                        "chunkText": chunk.chunk_text,
                        "chunkOrder": chunk.chunk_order,
                        "chunkMetadata": chunk.chunk_metadata,
                        "createdAt": datetime.now(timezone.utc),
                    },
                )

            try:
                await batch.commit()
                total += len(batch_chunks)
            except Exception:
                logger.error(
                    "Firestore batch write failed.",
                    extra={"batch_start": i, "batch_size": len(batch_chunks)},
                    exc_info=True,
                )

        return total

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunk records associated with a given document ID.

        Documents are streamed and deleted in batches. A new WriteBatch is
        started after every _FIRESTORE_BATCH_LIMIT deletions to stay within
        Firestore's per-batch operation limit.

        Args:
            doc_id: The Firestore document ID of the parent Document record.

        Returns:
            The total number of chunk records deleted.
        """
        docs = self._col.where("docId", "==", doc_id).stream()
        batch = self._db.batch()
        count = 0

        async for doc in docs:
            batch.delete(doc.reference)
            count += 1
            if count % _FIRESTORE_BATCH_LIMIT == 0:
                await batch.commit()
                batch = self._db.batch()

        if count % _FIRESTORE_BATCH_LIMIT != 0:
            await batch.commit()

        return count

    async def count_by_doc_id(self, doc_id: str) -> int:
        """
        Return the number of chunk records associated with a given document ID.

        Used by the processing use case to verify that indexing produced at
        least one chunk before updating the document status to 'processed'.
        """
        docs = self._col.where("docId", "==", doc_id).stream()
        count = 0
        async for _ in docs:
            count += 1
        return count

    @staticmethod
    def _to_entity(chunk_id: str, data: dict) -> DataChunk:
        """Convert a Firestore document dictionary to a DataChunk domain entity."""
        # Adjust the kwargs below if your DataChunk entity uses different field names (e.g., chunk_id vs id)
        return DataChunk(
            chunk_id=chunk_id,
            doc_id=data.get("docId", ""),
            chunk_text=data.get("chunkText", ""),
            chunk_order=data.get("chunkOrder", 0),
            chunk_metadata=data.get("chunkMetadata", {}),
        )
