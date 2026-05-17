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

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status, Response
from fastapi.responses import JSONResponse

from ...core.use_cases.document_delete import DocumentDeleteUseCase
from ..schemas.delete_schemas import DeleteRequest
from ..dependencies import get_current_user_id, get_document_delete_use_case

logger = logging.getLogger(__name__)

delete_router = APIRouter(prefix="/delete", tags=["delete"])


@delete_router.delete("/document/batch", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
        body: DeleteRequest,
        user_id: str = Depends(get_current_user_id),
        delete_use_case: DocumentDeleteUseCase = Depends(get_document_delete_use_case),
):
    result = await delete_use_case.delete_documents_batch(body.doc_ids)
    if result["signal"] == "NO_DOCUMENTS_EXIST":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching documents found to delete."
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@delete_router.delete("/document/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
        doc_id: str,
        user_id: str = Depends(get_current_user_id),
        delete_use_case: DocumentDeleteUseCase = Depends(get_document_delete_use_case),
):
    signal = await delete_use_case.delete_document(doc_id)
    signal = signal["signal"]
    if signal == "DOCUMENT_DELETED":
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_404_NOT_FOUND)

