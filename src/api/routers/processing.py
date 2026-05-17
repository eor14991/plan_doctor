"""
API Router: Processing
=======================
Handles document processing requests from the web portal.

Processing pipeline:
    chunking -> embedding -> Qdrant indexing -> Firestore chunk storage

Endpoints:
    POST /processing/{doc_id}          - Start processing a document.
    GET  /processing/{doc_id}/status   - Poll the current processing status.

HTTP 202 Accepted:
    The processing endpoint returns 202, not 200. This signals to the client
    that the request has been accepted and work has begun, but is not yet
    complete. The client should poll the status endpoint to determine when
    processing finishes. This is the correct HTTP semantic for long-running
    background operations.

Background tasks:
    Processing is dispatched as a FastAPI BackgroundTask so the 202 response
    is returned immediately without making the client wait for chunking and
    embedding to complete.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ...core.use_cases.document_processing import DocumentProcessingUseCase
from ..dependencies import get_current_user_id, get_document_processing_use_case
from ..schemas import ProcessingRequest,ProcessingBatchRequest

logger = logging.getLogger(__name__)

processing_router = APIRouter(prefix="/processing", tags=["processing"])

@processing_router.post("/batch", status_code=status.HTTP_202_ACCEPTED)
async def process_batch_document(
    body: ProcessingBatchRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    processing_use_case: DocumentProcessingUseCase = Depends(get_document_processing_use_case),
) -> JSONResponse:
    documents = await processing_use_case.get_documents_batch(body.doc_ids)
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching documents found."
        )
    batch_payload = [{
        "doc_id":document.doc_id,
        "file_path":document.file_path
    } for document in documents]

    if not batch_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FILE_PATH_MISSING: None of the requested documents have uploaded files."
        )

    background_tasks.add_task(
        processing_use_case.execute_batch,
        documents=batch_payload,
        chunk_size=body.chunk_size,
        overlap_size=body.overlap_size,
        do_reset=body.do_reset,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
        "signal": "BATCH_PROCESSING_STARTED",
        "doc_ids": [item["doc_id"] for item in batch_payload],
        "message": "Processing started in background.",
    })


@processing_router.get("/batch/status")
async def get_batch_processing_status(
        body: ProcessingBatchRequest,
        user_id: str = Depends(get_current_user_id),
        processing_use_case: DocumentProcessingUseCase = Depends(get_document_processing_use_case),
) -> JSONResponse:
    """
    Return the current processing status for a document.

    Possible status values:
        uploaded    - File received, processing not yet started.
        processing  - Chunking and indexing are in progress.
        processed   - Pipeline completed successfully.
        failed      - Pipeline encountered an unrecoverable error.
    """
    documents = await processing_use_case.get_documents_batch(body.doc_ids)
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching documents found."
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "results": [
                {
                    "doc_id": document.doc_id,
                    "status": document.status,
                    "file_name": document.file_name,
                }
                for document in documents
            ]
        }
    )




@processing_router.post("/{doc_id}", status_code=status.HTTP_202_ACCEPTED)
async def process_document(
    doc_id: str,
    body: ProcessingRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    processing_use_case: DocumentProcessingUseCase = Depends(get_document_processing_use_case),
) -> JSONResponse:
    """
    Initiate chunking, embedding, and vector indexing for a previously uploaded
    document. Returns 202 immediately; poll /status to track completion.
    """
    document = await processing_use_case.get_document(doc_id)
    if not document:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "DOCUMENT_NOT_FOUND"},
        )

    if not document.file_path:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "FILE_PATH_MISSING"},
        )

    background_tasks.add_task(
        processing_use_case.execute,
        doc_id=doc_id,
        file_path=document.file_path,
        chunk_size=body.chunk_size,
        overlap_size=body.overlap_size,
        do_reset=body.do_reset,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "signal": "PROCESSING_STARTED",
            "doc_id": doc_id,
            "message": "Processing started in background. Poll the status endpoint to track progress.",
        },
    )


@processing_router.get("/{doc_id}/status")
async def get_single_processing_status(
    doc_id: str,
    user_id: str = Depends(get_current_user_id),
    processing_use_case: DocumentProcessingUseCase = Depends(get_document_processing_use_case),
) -> JSONResponse:
    """
    Return the current processing status for a document.

    Possible status values:
        uploaded    - File received, processing not yet started.
        processing  - Chunking and indexing are in progress.
        processed   - Pipeline completed successfully.
        failed      - Pipeline encountered an unrecoverable error.
    """
    document = await processing_use_case.get_document(doc_id)
    if not document:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "DOCUMENT_NOT_FOUND"},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "doc_id": doc_id,
            "status": document.status,
            "file_name": document.file_name,
        },
    )
