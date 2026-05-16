"""
Adapter: QdrantAdapter — IVectorStore implementation.
DEFECT #10 FIX: No `from pymongo import MongoClient` (was in original QdrantDBProvider).
DEFECT #3 FIX: db_path passed in by Container, not pulled from BaseController.
search_by_vector returns list[RetrievedDocument] — domain type, not ScoredPoint.
"""
from __future__ import annotations
import logging
import uuid
from typing import Optional
from qdrant_client import QdrantClient, models
from ....core.ports.services.i_vector_store import IVectorStore
from ....core.domain.value_objects.retrieved_document import RetrievedDocument


logger = logging.getLogger(__name__)
_DISTANCE_MAP = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT}


class QdrantAdapter(IVectorStore):
    def __init__(self, db_path: str, distance_method: str = "cosine") -> None:
        self._db_path = db_path
        self._distance = _DISTANCE_MAP.get(distance_method.lower(), models.Distance.COSINE)
        self._client: Optional[QdrantClient] = None

    def connect(self) -> None:
        self._client = QdrantClient(path=self._db_path)
        logger.info("Qdrant connected", extra={"db_path": self._db_path})

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def delete_by_filter(self, collection_name: str, filter_key: str, filter_value: str) -> bool:
        assert self._client
        try:
            self._client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=filter_key,
                            match=models.MatchValue(value=filter_value)
                        )
                    ]
                )
            )
            return True
        except Exception:
            logger.error("delete_by_filter failed", exc_info=True)
            return False

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False) -> bool:
        assert self._client
        try:
            if do_reset:
                self.delete_collection(collection_name)
            if not self.collection_exists(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=embedding_size, distance=self._distance),
                )
            return True
        except Exception:
            logger.error("create_collection failed", extra={"collection": collection_name}, exc_info=True)
            return False

    def delete_collection(self, collection_name: str) -> bool:
        assert self._client
        if not self.collection_exists(collection_name):
            return False
        try:
            self._client.delete_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def collection_exists(self, collection_name: str) -> bool:
        assert self._client
        return self._client.collection_exists(collection_name=collection_name)

    def insert_many(self, collection_name: str, texts: list[str], vectors: list[list[float]],
                    metadata: Optional[list[dict]] = None, batch_size: int = 50) -> bool:
        assert self._client
        if not self.collection_exists(collection_name):
            return False
        meta = metadata or [{} for _ in texts]
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vectors[i],
                               payload={"text": texts[i], "metadata": meta[i]})
            for i in range(len(texts))
        ]
        try:
            self._client.upload_points(collection_name=collection_name, points=points, batch_size=batch_size)
            return True
        except Exception:
            logger.error("insert_many failed", exc_info=True)
            return False

    def search_by_vector(self, collection_name: str, query_vector: list[float], limit: int) -> list[RetrievedDocument]:
        assert self._client
        try:
            pts = self._client.query_points(collection_name=collection_name, query=query_vector, limit=limit).points
            return [RetrievedDocument(text=p.payload.get("text", ""), score=float(p.score))
                    for p in pts if p.payload]
        except Exception:
            logger.error("search_by_vector failed", exc_info=True)
            return []

    def get_collection_info(self, collection_name: str) -> Optional[dict]:
        assert self._client
        try:
            info = self._client.get_collection(collection_name=collection_name)
            return info.model_dump() if hasattr(info, "model_dump") else dict(info)
        except Exception:
            return None