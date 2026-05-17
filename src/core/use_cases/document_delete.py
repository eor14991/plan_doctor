"""
Use Case: DocumentUploadUseCase
=================================
Orchestrates the file upload pipeline: validate, persist to disk, register
in Firestore.

Validation rules applied here (not in the router):
    - MIME type must be in the allowed types list.
    - File size must not exceed the configured maximum.

These rules are business rules and belong in the use case, not the delivery
layer. The router's only responsibility is translating the HTTP request into
arguments and the FileValidationResult back into an HTTP response.

Atomicity note:
    The file is written to disk before the Firestore document is created.
    If the Firestore write fails, the file is deleted from disk so that
    orphaned files do not accumulate. The reverse case (disk write failure)
    returns early before any Firestore interaction.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_community.chains.pebblo_retrieval.models import VectorDB

from ..domain.entities.document import Document
from ..domain.value_objects.file_upload_result import FileValidationResult
from ..ports.repositories import IChunkRepository
from ..ports.repositories.i_document_repository import IDocumentRepository
from ..ports.services import IVectorStore
from ..ports.storage.i_file_storage import IFileStorage

_KNOWLEDGE_COLLECTION = "plant_knowledge"

logger = logging.getLogger(__name__)



class DocumentDeleteUseCase:

    def __init__(
        self,
        document_repo: IDocumentRepository,
        file_storage: IFileStorage,
        chunk_repository: IChunkRepository,
        vector_store: IVectorStore,
    ) -> None:
        self._document_repo = document_repo
        self._file_storage = file_storage
        self._chunk_repository = chunk_repository
        self._vector_store = vector_store


    async def delete_document(self, doc_id):
        """ Delete a document completely:
            1. Verify the document exists.
            2. Delete chunks from Firestore.
            3. Delete vectors from Qdrant by doc_id filter.
            4. Delete the file from disk.
            5. Delete the document record from Firestore.

        Returns a dict with 'signal' and optional 'error' keys."""

        document = await self._document_repo.get_by_id(doc_id=doc_id)
        if not document:
            return { "signal":"DOCUMENT_NOT_EXIST"}

        await self._chunk_repository.delete_by_doc_id(doc_id=doc_id)

        self._vector_store.delete_by_filter(
            collection_name=_KNOWLEDGE_COLLECTION,
            filter_key="metadata.doc_id",
            filter_value=doc_id,
        )

        if document.file_path:
            await self._file_storage.delete_file(document.file_path)

        await self._document_repo.delete_by_id(doc_id=doc_id)

        logger.info("Document deleted.", extra={"doc_id": doc_id})
        return { "signal":"DOCUMENT_DELETED"}


    async def delete_documents_batch(self, doc_ids: list[str]) -> dict:
        """ Delete multiple documents concurrently.
            1. Fetch all documents in parallel.
            2. Filter out non-existent records.
            3. Execute chunk, vector, file, and repo deletions concurrently.
        """
        documents = await asyncio.gather(
            *(self._document_repo.get_by_id(doc_id=d_id) for d_id in doc_ids)
        )

        valid_pairs = [(doc_ids[i], doc) for i, doc in enumerate(documents) if doc]
        if not valid_pairs:
            return {"signal": "NO_DOCUMENTS_EXIST"}

        valid_ids = [p[0] for p in valid_pairs]
        valid_docs = [p[1] for p in valid_pairs]

        await asyncio.gather(
            *(self._chunk_repository.delete_by_doc_id(doc_id=d_id) for d_id in valid_ids)
        )

        for d_id in valid_ids:
            self._vector_store.delete_by_filter(
                collection_name=_KNOWLEDGE_COLLECTION,
                filter_key="metadata.doc_id",
                filter_value=d_id,
            )

        file_tasks = [self._file_storage.delete_file(doc.file_path) for doc in valid_docs if doc.file_path]
        if file_tasks:
            await asyncio.gather(*file_tasks)

        await asyncio.gather(
            *(self._document_repo.delete_by_id(doc_id=d_id) for d_id in valid_ids)
        )

        logger.info("Batch documents deleted.", extra={"doc_ids": valid_ids})
        return {
            "signal": "BATCH_DOCUMENTS_DELETED",
            "deleted_count": len(valid_ids),
            "requested_count": len(doc_ids)
        }