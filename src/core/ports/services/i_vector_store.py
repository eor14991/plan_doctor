"""
Port: IVectorStore
===================
Abstract interface for all vector store operations.

The port uses only Python primitives and domain value objects.
No Qdrant types, collection handles, or SDK constructs appear here.
The QdrantAdapter in adapters/services/vector_store/ is the concrete
implementation. All methods are synchronous because the Qdrant local-mode
client is synchronous and wrapping it in async without an executor would
be misleading.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ...domain.value_objects.retrieved_document import RetrievedDocument


class IVectorStore(ABC):

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False) -> bool: ...

    @abstractmethod
    def insert_many(self, collection_name: str, texts: list[str], vectors: list[list[float]],
                    metadata: Optional[list[dict]] = None, batch_size: int = 50) -> bool: ...

    @abstractmethod
    def search_by_vector(self, collection_name: str, query_vector: list[float], limit: int) -> list[
        RetrievedDocument]: ...

    @abstractmethod
    def delete_by_filter(self, collection_name: str, filter_key: str, filter_value: str) -> bool: ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool: ...

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Optional[dict]: ...