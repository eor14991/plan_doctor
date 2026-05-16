"""
API: Dependency Providers
==========================
Provides typed use case instances to FastAPI route handlers via Depends().

Design rationale:
    Route handlers should depend on use cases, not on the Container.
    Providing a fine-grained Depends() function per use case enforces this
    boundary. The route handler's function signature becomes a precise
    declaration of what it needs, and tests can override a single provider
    without affecting unrelated routes.

    Example test override:
        app.dependency_overrides[get_chat_use_case] = lambda: MockChatUseCase()

Authentication:
    get_current_user() verifies the Firebase ID token from the Authorization
    header. It raises HTTP 401 for missing, expired, or invalid tokens.
    get_current_user_id() is a convenience wrapper that extracts only the uid.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.use_cases.chat_conversation import ChatConversationUseCase
from ..core.use_cases.chat_summarization import ChatSummarizationUseCase
from ..core.use_cases.document_processing import DocumentProcessingUseCase
from ..core.use_cases.document_upload import DocumentUploadUseCase
from ..infrastructure.container import Container
from ..infrastructure.firebase_client import verify_firebase_token

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """Return the singleton Container attached to app.state during startup."""
    return request.app.state.container


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_security),
) -> dict:
    """
    Verify the Firebase ID token from the Authorization header and return
    the decoded token claims.

    The decoded claims dictionary contains at minimum:
        uid:   str  - The Firebase user identifier.
        email: str  - The user's email address.

    Raises:
        HTTPException 401: If the Authorization header is absent, the token
            has expired, or the token signature is invalid.
    """

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required.",
        )
    try:
        return verify_firebase_token(credentials.credentials)
    except Exception as exc:
        detail = (
            "Token expired. Please log in again."
            if "expired" in str(exc).lower()
            else "Invalid authentication token."
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def get_current_user_id(user: dict = Depends(get_current_user)) -> str:
    """Return only the uid from the verified token claims."""
    return user["uid"]


def get_chat_use_case(
    container: Container = Depends(get_container),
) -> ChatConversationUseCase:
    assert container.chat_conversation is not None
    return container.chat_conversation


def get_summarization_use_case(
    container: Container = Depends(get_container),
) -> ChatSummarizationUseCase:
    assert container.chat_summarization is not None
    return container.chat_summarization


def get_document_upload_use_case(
    container: Container = Depends(get_container),
) -> DocumentUploadUseCase:
    assert container.document_upload is not None
    return container.document_upload


def get_document_processing_use_case(
    container: Container = Depends(get_container),
) -> DocumentProcessingUseCase:
    assert container.document_processing is not None
    return container.document_processing
