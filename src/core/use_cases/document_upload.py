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

import logging

from ..domain.entities.document import Document
from ..domain.value_objects.file_upload_result import FileValidationResult
from ..ports.repositories.i_document_repository import IDocumentRepository
from ..ports.storage.i_file_storage import IFileStorage

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


class DocumentUploadUseCase:

    def __init__(
        self,
        document_repo: IDocumentRepository,
        file_storage: IFileStorage,
        allowed_types: list[str] | None = None,
        max_size_mb: int = 20,
    ) -> None:
        self._document_repo = document_repo
        self._file_storage = file_storage
        self._allowed_types = allowed_types or ["application/pdf", "text/plain"]
        self._max_size_bytes = max_size_mb * _MB

    async def execute(
        self,
        user_id: str,
        file_name: str,
        file_content: bytes,
        content_type: str,
    ) -> FileValidationResult:
        """
        Validate and persist an uploaded file.

        Args:
            user_id:      Firebase UID from the verified auth token.
            file_name:    Original filename from the upload request.
            file_content: Raw file bytes.
            content_type: MIME type string.

        Returns:
            FileValidationResult with is_valid=True and the document ID on
            success, or is_valid=False with a descriptive signal on failure.
        """
        # Validate MIME type against the configured allow-list.
        if content_type not in self._allowed_types:
            logger.warning(
                "File upload rejected: type not allowed.",
                extra={"content_type": content_type, "user_id": user_id},
            )
            return FileValidationResult(is_valid=False, signal="FILE_TYPE_NOT_ALLOWED")

        # Validate file size against the configured maximum.
        if len(file_content) > self._max_size_bytes:
            logger.warning(
                "File upload rejected: size limit exceeded.",
                extra={"size_bytes": len(file_content), "max_bytes": self._max_size_bytes},
            )
            return FileValidationResult(is_valid=False, signal="FILE_SIZE_EXCEEDED")

        # Write the file to disk.
        try:
            file_path, stored_name = await self._file_storage.save_bytes(
                user_id=user_id,
                original_name=file_name,
                content=file_content,
            )
        except Exception:
            logger.error("File storage write failed.", exc_info=True)
            return FileValidationResult(is_valid=False, signal="FILE_STORAGE_FAILED")

        # Register the document record in Firestore.
        document = Document(
            user_id=user_id,
            file_name=stored_name,
            file_size=len(file_content),
            file_path=file_path,
            status="uploaded",
        )
        try:
            saved_doc = await self._document_repo.save(document)
        except Exception:
            logger.error("Document registration in Firestore failed.", exc_info=True)
            # Clean up the file from disk to prevent orphaned storage.
            await self._file_storage.delete_file(file_path)
            return FileValidationResult(is_valid=False, signal="DOCUMENT_REGISTRATION_FAILED")

        logger.info(
            "Document uploaded and registered.",
            extra={"doc_id": saved_doc.doc_id, "user_id": user_id},
        )
        return FileValidationResult(
            is_valid=True,
            signal="FILE_UPLOAD_SUCCESSFUL",
            file_id=stored_name,
            asset_db_id=saved_doc.doc_id,
        )
