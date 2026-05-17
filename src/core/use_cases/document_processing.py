"""
Use Case: DocumentProcessingUseCase
======================================
Orchestrates the full document ingestion pipeline: load, chunk, embed, index.

Pipeline steps:
    1. Retrieve the Document record and update its status to 'processing'.
    2. If do_reset is True, delete existing chunks from Firestore and Qdrant.
    3. Ensure the Qdrant knowledge collection exists.
    4. Load the file from disk and split it into RawDocumentChunk value objects.
    5. Embed each chunk's text using the embedding service.
    6. Insert vectors and metadata into Qdrant.
    7. Persist chunk records to Firestore.
    8. Update the Document status to 'processed' or 'failed'.

Dependency audit:
    This use case contains no imports from firebase_admin, qdrant_client,
    transformers, langchain, or fastapi. All infrastructure interactions are
    delegated to abstract ports. The use case can be tested with in-memory
    fakes without any external services.
"""

from __future__ import annotations

import logging

import asyncio

from ..domain.entities.chunk import DataChunk
from ..ports.repositories.i_chunk_repository import IChunkRepository
from ..ports.repositories.i_document_repository import IDocumentRepository
from ..ports.services.i_document_chunker import IDocumentChunker
from ..ports.services.i_embedding_service import DocumentType, IEmbeddingService
from ..ports.services.i_vector_store import IVectorStore

logger = logging.getLogger(__name__)

_KNOWLEDGE_COLLECTION = "plant_knowledge"


class DocumentProcessingUseCase:

    def __init__(
        self,
        document_repo: IDocumentRepository,
        chunk_repo: IChunkRepository,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        chunker: IDocumentChunker,
    ) -> None:
        self._document_repo = document_repo
        self._chunk_repo = chunk_repo
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunker = chunker

    async def execute(
        self,
        doc_id: str,
        file_path: str,
        chunk_size: int = 450,
        overlap_size: int = 40,
        do_reset: bool = False,
    ) -> dict:
        """
        Run the full processing pipeline for a single document.

        Args:
            doc_id:       Firestore document ID of the Document record.
            file_path:    Absolute path to the file on the local storage volume.
            chunk_size:   Maximum token count per chunk.
            overlap_size: Token overlap between adjacent chunks.
            do_reset:     If True, delete existing chunks and vectors before
                          reprocessing.

        Returns:
            A dictionary with 'signal' and 'chunks_count' keys.
        """
        document = await self._document_repo.get_by_id(doc_id)
        if not document:
            return {"signal": "DOCUMENT_NOT_FOUND", "chunks_count": 0}

        await self._document_repo.update_status(doc_id, "processing")

        if do_reset:
            await self._chunk_repo.delete_by_doc_id(doc_id)
            self._vector_store.delete_by_doc_id(_KNOWLEDGE_COLLECTION, doc_id)
            logger.info("Existing chunks and vectors deleted for reset.", extra={"doc_id": doc_id})

        # Ensure the Qdrant collection exists before inserting vectors.
        self._vector_store.create_collection(
            collection_name=_KNOWLEDGE_COLLECTION,
            embedding_size=self._embedding_service.embedding_size,
            do_reset=False,
        )

        # Load the file and split into RawDocumentChunk value objects.
        raw_chunks = self._chunker.load_and_split(
            file_path=file_path,
            file_id=doc_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        if not raw_chunks:
            await self._document_repo.update_status(doc_id, "failed")
            return {"signal": "CHUNKING_FAILED", "chunks_count": 0}

        # Build DataChunk domain entities from the raw chunk value objects.
        chunk_entities = [
            DataChunk(
                doc_id=doc_id,
                chunk_text=rc.text,
                chunk_metadata={**rc.metadata, "doc_id": doc_id},
                chunk_order=i,
            )
            for i, rc in enumerate(raw_chunks)
        ]

        # Embed all chunk texts.
        texts = [c.chunk_text for c in chunk_entities]

        loop = asyncio.get_running_loop()

        vectors = await loop.run_in_executor(
            None,
            lambda: self._embedding_service.batch_embed_text(
                texts, DocumentType.DOCUMENT, batch_size=32
            )
        )

        metadata_list = [{"doc_id": c.doc_id, "order": c.chunk_order} for c in chunk_entities]
        await loop.run_in_executor(
            None,
            lambda: self._vector_store.insert_many(
                collection_name=_KNOWLEDGE_COLLECTION,
                texts=texts,
                vectors=vectors,
                metadata=metadata_list,
            )
        )

        # Persist chunk records to Firestore.
        inserted = await self._chunk_repo.insert_bulk(chunk_entities)

        await self._document_repo.update_status(doc_id, "processed")

        logger.info(
            "Document processing complete.",
            extra={"doc_id": doc_id, "chunks_inserted": inserted},
        )
        return {"signal": "PROCESSING_SUCCESSFUL", "chunks_count": inserted}

    async def get_document(self, doc_id: str):
        """
        Return a Document entity by its identifier, or None if not found.

        Exposes the repository lookup as a named use case method so that
        routers never access _document_repo directly.
        """
        return await self._document_repo.get_by_id(doc_id)


    async def execute_batch(
            self,
            documents: list[dict],
            chunk_size: int = 450,
            overlap_size: int = 40,
            do_reset: bool = False,
    ) -> dict:
        """
        Run the processing pipeline for multiple documents concurrently.

        Args:
            documents: List of dicts containing 'doc_id' and 'file_path'.
                       Example: [{"doc_id": "123", "file_path": "/tmp/a.pdf"}]
        """
        tasks = [
             self.execute(
                doc_id=doc["doc_id"],
                file_path=doc["file_path"],
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                do_reset=do_reset,
            )
            for doc in documents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        total_chunks = 0
        errors = []

        for i, result in enumerate(results):
            doc_id = documents[i]["doc_id"]
            if isinstance(result, Exception):
                logger.error("Batch processing exception.", extra={"doc_id": doc_id, "error": str(result)})
                errors.append({"doc_id": doc_id, "error": str(result)})
            elif result.get("signal") == "PROCESSING_SUCCESSFUL":
                success_count += 1
                total_chunks += result.get("chunks_count", 0)
            else:
                errors.append({"doc_id": doc_id, "signal": result.get("signal")})

        return {
            "signal": "BATCH_PROCESSING_COMPLETE",
            "processed_count": success_count,
            "failed_count": len(errors),
            "total_chunks_inserted": total_chunks,
            "errors": errors,
        }

    async def get_documents_batch(self, doc_ids: list[str]) -> list:
        """
        Return multiple Document entities concurrently.
        """
        tasks = [self._document_repo.get_by_id(doc_id) for doc_id in doc_ids]
        documents = await asyncio.gather(*tasks, return_exceptions=True)

        return [doc for doc in documents if doc and not isinstance(doc, Exception)]