"""
Port: IEmbeddingService — ISP fix: embedding-only, separated from IGenerationService.
Defines the contract for transforming text into high-dimensional vectors.
"""
from abc import ABC, abstractmethod
from enum import Enum

class DocumentType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"

class IEmbeddingService(ABC):
    @property
    @abstractmethod
    def embedding_size(self) -> int: ...

    @abstractmethod
    def embed_text(self, text: str, document_type: DocumentType) -> list[float]: ...

    @abstractmethod
    def batch_embed_text(self, texts: list[str], document_type: DocumentType, batch_size:int = 32) -> list[list[float]]: ...