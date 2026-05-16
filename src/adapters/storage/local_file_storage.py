"""
Adapter: LocalFileStorageAdapter
==================================
Implements IFileStorage by writing files to the local filesystem.

Directory layout:
    {base_upload_dir}/{user_id}/{random_prefix}_{sanitised_original_name}

Per-user subdirectories isolate each user's files and make bulk deletion
of a user's data straightforward. The random prefix prevents filename
collisions when the same file is uploaded multiple times.

Replaceability:
    This adapter can be replaced with a FirebaseStorageAdapter or an
    S3StorageAdapter in the Container without changing any use case code,
    provided the new adapter implements IFileStorage.
"""
from __future__ import annotations

import logging
import os
import random
import re
import string

import aiofiles

from ...core.ports.storage.i_file_storage import IFileStorage

logger = logging.getLogger(__name__)


class LocalFileStorageAdapter(IFileStorage):

    def __init__(self, base_upload_dir: str) -> None:
        self._base_dir = os.path.abspath(base_upload_dir)
        os.makedirs(self._base_dir, exist_ok=True)

    async def save_bytes(
        self,
        user_id: str,
        original_name: str,
        content: bytes,
    ) -> tuple[str, str]:
        """
        Write the provided bytes to a new file under the user's directory.

        The original filename is sanitised to remove unsafe characters and
        prefixed with a 10-character random alphanumeric string to guarantee
        uniqueness even when the same filename is uploaded multiple times.

        Args:
            user_id:       Firebase UID used as the subdirectory name.
            original_name: The original filename from the upload request.
            content:       Raw file bytes to write.

        Returns:
            A tuple of (full_path, stored_name) where full_path is the
            absolute path on disk and stored_name is the value to record
            in the database.
        """
        user_dir = os.path.join(self._base_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)

        clean_name = self._sanitise(original_name)
        stored_name = self._unique_name(clean_name, user_dir)
        full_path = os.path.join(user_dir, stored_name)

        async with aiofiles.open(full_path, "wb") as f:
            await f.write(content)

        logger.debug("File written to disk.", extra={"path": full_path, "bytes": len(content)})
        return full_path, stored_name

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the local filesystem.

        Returns True if the file was deleted. Returns False if the file did
        not exist. Returns False and logs the error if an exception occurs.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            logger.error("Failed to delete file.", extra={"path": file_path}, exc_info=True)
            return False

    def get_full_path(self, user_id: str, stored_name: str) -> str:
        """Return the absolute path for an existing stored file."""
        return os.path.join(self._base_dir, user_id, stored_name)

    @staticmethod
    def _sanitise(name: str) -> str:
        """
        Remove characters that are unsafe in filenames.

        Replaces anything that is not a word character, dot, or hyphen with
        an underscore. Returns 'file' if the result is empty.
        """
        clean = re.sub(r"[^\w.\-]", "_", name.strip())
        return clean or "file"

    @staticmethod
    def _unique_name(clean_name: str, directory: str) -> str:
        """
        Generate a filename that does not already exist in the target directory.

        Prepends a 10-character random alphanumeric string to the cleaned
        name and retries until a non-colliding candidate is found.
        """
        while True:
            prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
            candidate = f"{prefix}_{clean_name}"
            if not os.path.exists(os.path.join(directory, candidate)):
                return candidate
