"""
Port: IDocumentRepository
===========================
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ...domain.entities.document import Document


class IDocumentRepository(ABC):

    @abstractmethod
    async def save(self, document: Document) -> Document: ...

    @abstractmethod
    async def get_by_id(self, doc_id: str) -> Optional[Document]: ...

    @abstractmethod
    async def get_all(self, user_id: Optional[str] = None) -> List[Document]: ...

    @abstractmethod
    async def update_status(self, doc_id: str, status: str) -> bool: ...

    @abstractmethod
    async def delete_by_id(self, doc_id: str) -> bool: ...
