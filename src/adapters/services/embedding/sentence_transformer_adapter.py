"""
Adapter: SentenceTransformerAdapter
======================================
Implements IEmbeddingService using a local HuggingFace SentenceTransformer model.

The model is loaded once in __init__ and reused for every call.
No disk I/O occurs after startup.
"""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from ....core.ports.services.i_embedding_service import DocumentType, IEmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerAdapter(IEmbeddingService):

    def __init__(self, model_id: str, embedding_size: int) -> None:
        self._model_id = model_id
        self._embedding_size = embedding_size
        self._model: Optional[SentenceTransformer] = None
        logger.info("Loading embedding model.", extra={"model_id": model_id})
        self._model = SentenceTransformer(model_id)
        self._model.max_seq_length = 8192
        logger.info("Embedding model loaded.", extra={"embedding_size": embedding_size})

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def embed_text(
        self, text: str, document_type: DocumentType = DocumentType.DOCUMENT
    ) -> list[float]:
        """Encode a single text string into a dense float vector."""
        if self._model is None:
            return []
        try:
            return self._model.encode(text).tolist()
        except Exception:
            logger.error("embed_text failed.", exc_info=True)
            return []

    def batch_embed_text(
        self,
        texts: list[str],
        document_type: DocumentType,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Encode a list of texts in batches.

        SentenceTransformer.encode() handles batching internally when passed
        a list, but the batch_size parameter controls peak memory usage.
        This method is synchronous — SentenceTransformer is CPU/GPU bound.
        """
        if self._model is None or not texts:
            return []
        try:
            embeddings = self._model.encode(texts, batch_size=batch_size)
            return embeddings.tolist()
        except Exception:
            logger.error("batch_embed_text failed.", exc_info=True)
            return []
