"""
API Router: Upload
===================
Handles knowledge document upload requests from the web portal.

Endpoints:
    POST /upload/document   - Upload a PDF or plain text file.

Authentication:
    The user_id is sourced from the verified Firebase ID token, not from the
    request body. This guarantees that a document is always attributed to the
    authenticated user, preventing any client from forging ownership.

File validation:
    MIME type and size constraints are enforced inside DocumentUploadUseCase,
    not here. The router's only responsibility is reading the file bytes from
    the UploadFile object and forwarding them to the use case.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse

from ...core.use_cases.document_upload import DocumentUploadUseCase
from ..dependencies import get_current_user_id, get_document_upload_use_case

logger = logging.getLogger(__name__)

upload_router = APIRouter(prefix="/upload", tags=["upload"])


@upload_router.post("/document", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    user_id: str = Depends(get_current_user_id),
    upload_use_case: DocumentUploadUseCase = Depends(get_document_upload_use_case),
) -> JSONResponse:
    """
    Upload a knowledge document to the RAG knowledge base.

    Accepted MIME types and maximum file size are defined in application
    settings. The response includes the Firestore document ID (doc_id) which
    the client uses to trigger processing via the /processing router.
    """
    file_content = await file.read()

    result = await upload_use_case.execute(
        user_id=user_id,
        file_name=file.filename or "upload",
        file_content=file_content,
        content_type=file.content_type or "",
    )

    if not result.is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result.signal},
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "signal": result.signal,
            "doc_id": result.asset_db_id,
            "file_id": result.file_id,
        },
    )

@upload_router.get("/documents", status_code=status.HTTP_200_OK)
async def upload_document(
            upload_use_case: DocumentUploadUseCase = Depends(get_document_upload_use_case),
    ) -> JSONResponse:
    docs = await upload_use_case.list_documents()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"docs": [doc.model_dump(mode='json') for doc in docs ]},

        )