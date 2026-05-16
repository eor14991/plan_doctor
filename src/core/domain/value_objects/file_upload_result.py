"""Value Object: FileValidationResult — command result object for DataIngestionUseCase."""
from typing import Optional
from pydantic import BaseModel, Field


class FileValidationResult(BaseModel):
    is_valid: bool
    signal: str
    file_id: Optional[str] = Field(default=None)
    asset_db_id: Optional[str] = Field(default=None)
    model_config = {"frozen": True}
