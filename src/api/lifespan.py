"""
API: Application Lifespan
==========================
Manages startup and shutdown of the FastAPI application using the
asynccontextmanager lifespan protocol introduced in FastAPI 0.93.

Startup sequence:
    1. Initialise structured logging so all subsequent startup steps are
       captured in structured format.
    2. Load and validate settings from the environment.
    3. Build the DI Container, which initialises Firebase, loads all ML
       models into memory, and connects to Qdrant.
    4. Attach the container to app.state so that dependency providers can
       retrieve it from the request object.

Shutdown sequence:
    5. Call Container.shutdown() to disconnect from Qdrant and release any
       other resources held by the container.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ..infrastructure.config import get_settings
from ..infrastructure.container import Container
from ..infrastructure.logging_setup import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(log_level="INFO", service_name=settings.APP_NAME)

    container = Container.build(settings)
    app.state.container = container

    yield

    container.shutdown()
