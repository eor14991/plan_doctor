"""
Port: IDocumentChunker
========================
Abstract interface for the document loading and chunking pipeline step.

Implementations receive a file path and splitting parameters and return
a list of RawDocumentChunk value objects. The use case assigns domain
identifiers (doc_id, chunk_order) and constructs DataChunk entities from
the returned value objects. Implementations return an empty list on
failure rather than raising exceptions.
"""

from abc import ABC, abstractmethod

from ...domain.value_objects.raw_document_chunk import RawDocumentChunk


class IDocumentChunker(ABC):

    @abstractmethod
    def load_and_split(
        self,
        file_path: str,
        file_id: str,
        chunk_size: int = 450,
        overlap_size: int = 40,
    ) -> list[RawDocumentChunk]:
        """
        Load a file from disk and split it into text chunks.

        Args:
            file_path:    Absolute path to the file.
            file_id:      Identifier embedded in each chunk's metadata.
            chunk_size:   Maximum token count per chunk.
            overlap_size: Token overlap between adjacent chunks.

        Returns:
            A list of RawDocumentChunk value objects, or an empty list on failure.
        """
        ...
