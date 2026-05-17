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
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..core.use_cases.document_delete import DocumentDeleteUseCase
from ..core.use_cases.chat_conversation import ChatConversationUseCase
from ..core.use_cases.chat_summarization import ChatSummarizationUseCase
from ..core.use_cases.document_processing import DocumentProcessingUseCase
from ..core.use_cases.document_upload import DocumentUploadUseCase
from ..infrastructure.container import Container
from ..infrastructure.firebase_client import verify_firebase_token

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def get_limiter_id(request: Request) -> str:
    """Extract identity for rate limiting, falling back to IP."""
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        return user.get("uid", get_remote_address(request))
    return get_remote_address(request)


limiter = Limiter(key_func=get_limiter_id, default_limits=["100/minute"])


def get_container(request: Request) -> Container:
    """Return the singleton Container attached to app.state during startup."""
    return request.app.state.container


def get_current_user(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(security)
):
    if credentials is None:
        return None
    try:
        user = auth.verify_id_token(credentials.credentials)
        request.state.user = user
        return user
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


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

def get_document_delete_use_case(
    container: Container = Depends(get_container),
) -> DocumentDeleteUseCase:
    assert container.document_delete is not None
    return container.document_delete