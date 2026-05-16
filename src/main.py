"""
Application Entry Point
========================
Creates and configures the FastAPI application instance.

The create_app() factory function is used instead of a module-level
app = FastAPI(...) to allow tests to create isolated application instances
with independent dependency overrides.

Routers:
    /chat        - Mobile app conversation endpoints.
    /upload      - Web portal file upload endpoints.
    /processing  - Web portal document processing endpoints.
"""
from __future__ import annotations

from fastapi import FastAPI
from .api.dependencies import get_current_user_id

from .api.lifespan import lifespan
from .api.routers.chat import chat_router
from .api.routers.upload import upload_router
from .api.routers.processing import processing_router


def create_app() -> FastAPI:
    application = FastAPI(
        title="PlantDoctor RAG",
        version="1.0.0",
        description=(
            "RAG backend for PlantDoctor mobile app. "
            "Provides plant disease Q&A powered by a curated knowledge base."
        ),
        lifespan=lifespan,
    )

    application.dependency_overrides[get_current_user_id] = lambda: "test_user_123"
    application.include_router(chat_router)
    application.include_router(upload_router)
    application.include_router(processing_router)

    return application


# Module-level instance for Uvicorn: uvicorn src.main:app --reload
app = create_app()
