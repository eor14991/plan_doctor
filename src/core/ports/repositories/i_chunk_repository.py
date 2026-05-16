"""
Port: IChunkRepository
========================
Abstract interface for DataChunk bulk persistence and retrieval operations.

insert_bulk() returns the actual number of records written to the database,
not the length of the input list. If a batch fails partially, the returned
count will be less than len(chunks), allowing the use case to detect and
log partial failures accurately.
"""

from abc import ABC, abstractmethod
from typing import List

from ...domain.entities.chunk import DataChunk


class IChunkRepository(ABC):

    @abstractmethod
    async def insert_bulk(self, chunks: List[DataChunk]) -> int:
        """
        Persist a list of DataChunk entities in batches.

        Returns:
            The number of records successfully written to the database.
        """
        ...

    @abstractmethod
    async def get_by_doc_id(
        self, doc_id: str, page_number: int = 1, page_size: int = 50
    ) -> List[DataChunk]:
        """Return a paginated list of chunks for a given document."""
        ...

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks associated with a document.

        Returns the number of records deleted.
        """
        ...
