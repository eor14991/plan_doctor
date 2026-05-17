"""
Infrastructure: Application Settings
=====================================
Centralised, validated configuration loaded from environment variables
and an optional .env file. Every field is type-coerced and validated at
startup, causing the application to fail immediately with a descriptive
error rather than at the first usage of a missing value.

Firebase credential resolution:
    FIREBASE_SERVICE_ACCOUNT_FILE takes priority over FIREBASE_SERVICE_ACCOUNT.
    At least one of the two must be set. The container validates this during
    the build step and raises a clear error if neither is present.

The @lru_cache decorator on get_settings() ensures the .env file is read
exactly once for the lifetime of the process.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    """
    Application-wide settings resolved from environment variables.

    Fields with no default are required. The application will not start
    if they are absent from the environment or the .env file.
    """

    APP_NAME: str = "plan-doctor-rag"
    APP_VERSION: str = "1.0.0"

    # Firebase credentials — at least one must be set.
    FIREBASE_SERVICE_ACCOUNT_FILE: Optional[str] = Field(default=None)
    FIREBASE_SERVICE_ACCOUNT: Optional[str] = Field(default=None)

    # File upload constraints.
    FILE_ALLOWED_TYPES: list[str] = Field(default=["application/pdf", "text/plain"])
    FILE_MAX_SIZE: int = Field(default=20, description="Maximum upload size in megabytes.")
    FILE_DEFAULT_CHUNK_SIZE: int = Field(default=1024 * 1024)
    UPLOAD_BASE_DIR: str = "./storage/uploads"

    # ── Generation ──────────────────────────────────────────────────────────
    ACTIVE_GENERATION_BACKEND: Literal["GROQ", "COHERE"] = "GROQ"  # ← was "GROK"
    GROQ_API_KEY: str
    GROQ_API_URL: str = "https://api.groq.com/openai/v1"
    GROQ_GENERATION_MODEL: str = "openai/gpt-oss-120b"
    COHERE_API_KEY: str = ""
    COHERE_GENERATION_MODEL: str = "command-r-plus"
    INPUT_DEFAULT_MAX_CHARACTERS: int = 4096
    GENERATION_DEFAULT_MAX_TOKENS: int = 1000
    GENERATION_DEFAULT_TEMPERATURE: float = 0.1

    # Embedding model.
    ACTIVE_EMBEDDING_BACKEND: Literal["SENTENCE_TRANSFORMER"] = "SENTENCE_TRANSFORMER"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_SIZE: int = 1024

    # Summarization model.
    SUMMARIZATION_MODEL_ID: str = "facebook/bart-large-cnn"
    SUMMARIZE_EVERY_N_MESSAGES: int = Field(default=10, ge=2)

    # Qdrant vector store.
    VECTOR_DB_BACKEND: Literal["QDRANT"] = "QDRANT"
    VECTOR_DB_PATH: str = "./storage/qdrant"
    VECTOR_DB_DISTANCE_METHOD: Literal["cosine", "dot"] = "cosine"
    KNOWLEDGE_COLLECTION: str = "plant_knowledge"

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"
    # RAG retrieval.
    RAG_TOP_K: int = Field(default=5, ge=1, le=20)
    VECTOR_DB_HOST: Optional[str] = Field(default=None)
    VECTOR_DB_PORT: int = Field(default=6333)

    @model_validator(mode="after")
    def firebase_credentials_must_be_present(self) -> "Settings":
        """Fail fast if neither Firebase credential source is configured."""
        if not self.FIREBASE_SERVICE_ACCOUNT_FILE and not self.FIREBASE_SERVICE_ACCOUNT:
            raise ValueError(
                "Firebase credentials are required. Set either "
                "FIREBASE_SERVICE_ACCOUNT_FILE (path to JSON key file) or "
                "FIREBASE_SERVICE_ACCOUNT (JSON content as string)."
            )
        return self

    @field_validator("EMBEDDING_MODEL_SIZE")
    @classmethod
    def embedding_size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("EMBEDDING_MODEL_SIZE must be a positive integer.")
        return v

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    To override in tests:
        app.dependency_overrides[get_settings] = lambda: Settings(
            FIREBASE_SERVICE_ACCOUNT_FILE="./test-credentials.json", ...
        )
    """
    return Settings()
