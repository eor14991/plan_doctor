"""
Port: IFileStorage
===================
Abstract interface for file persistence operations.

All use cases that need to read or write files depend on this interface.
The concrete implementation (local disk, Firebase Storage, S3, etc.) is
determined by the Container at startup and injected via the constructor.

Design decision: save_bytes() vs save_stream():
    FastAPI's UploadFile.read() returns bytes directly, which makes bytes the
    natural currency for upload handling. Using bytes also simplifies unit
    testing because tests can pass a literal bytes value without constructing
    a BinaryIO mock.
"""
from abc import ABC, abstractmethod


class IFileStorage(ABC):

    @abstractmethod
    async def save_bytes(
        self,
        user_id: str,
        original_name: str,
        content: bytes,
    ) -> tuple[str, str]:
        """
        Persist the provided bytes and return a stable reference to the file.

        Args:
            user_id:       Identifier of the owning user. Used to namespace
                           storage so that each user's files are isolated.
            original_name: The filename supplied by the caller. Implementations
                           are expected to sanitise this value before use.
            content:       Raw file bytes to persist.

        Returns:
            A tuple of (full_path, stored_name).
            full_path:   The complete path or URI that can be used to retrieve
                         or delete the file later.
            stored_name: The unique filename that should be recorded in the
                         database as the file's identifier.
        """
        ...

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete the file at the given path.

        Args:
            file_path: The full_path value returned by save_bytes().

        Returns:
            True if the file was deleted. False if it did not exist or if
            the deletion failed.
        """
        ...

    @abstractmethod
    def get_full_path(self, user_id: str, stored_name: str) -> str:
        """
        Reconstruct the full path for a file that was previously saved.

        Args:
            user_id:     The owning user's identifier.
            stored_name: The stored_name value returned by save_bytes().

        Returns:
            The full path or URI of the stored file.
        """
        ...
