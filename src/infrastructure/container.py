"""
Infrastructure: Dependency Injection Container
===============================================
This is the composition root of the application. It is the only module
that imports concrete adapter classes directly. Every other module depends
exclusively on abstract ports from core/.

Startup order is deterministic. Each step depends only on what was
constructed in previous steps, so the order can be reasoned about by
reading build() top to bottom.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ..core.use_cases.document_delete import DocumentDeleteUseCase
from ..adapters.repositories.firebase_chat_repo import FirebaseChatRepository
from ..adapters.repositories.firebase_chunk_repo import FirebaseChunkRepository
from ..adapters.repositories.firebase_document_repo import FirebaseDocumentRepository
from ..adapters.repositories.firebase_message_repo import FirebaseMessageRepository
from ..adapters.services.chunking.tokenizer_chunker import TokenizerDocumentChunker
from ..adapters.services.embedding.sentence_transformer_adapter import SentenceTransformerAdapter
from ..adapters.services.generation.cohere_adapter import CohereGenerationAdapter
from ..adapters.services.generation.groq_adapter import GroqGenerationAdapter
from ..adapters.services.summarization.huggingface_summarizer import HuggingFaceSummarizationService
from ..adapters.services.vector_store.qdrant_adapter import QdrantAdapter
from ..adapters.storage.local_file_storage import LocalFileStorageAdapter
from ..core.ports.services.i_generation_service import IGenerationService
from ..core.use_cases.chat_conversation import ChatConversationUseCase
from ..core.use_cases.chat_summarization import ChatSummarizationUseCase
from ..core.use_cases.document_processing import DocumentProcessingUseCase
from ..core.use_cases.document_upload import DocumentUploadUseCase
from .config import Settings
from .firebase_client import get_firestore_client, initialize_firebase
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """
    Typed holder for all application singletons.

    All attributes start as None and are populated by build(). After build()
    returns successfully, every attribute is guaranteed to be set.
    """

    db: Optional[object] = field(default=None, repr=False)

    chat_repo: Optional[FirebaseChatRepository] = field(default=None)
    message_repo: Optional[FirebaseMessageRepository] = field(default=None)
    document_repo: Optional[FirebaseDocumentRepository] = field(default=None)
    chunk_repo: Optional[FirebaseChunkRepository] = field(default=None)

    prompt_builder: Optional[PromptBuilder] = field(default=None)

    embedder: Optional[SentenceTransformerAdapter] = field(default=None)
    generator: Optional[IGenerationService] = field(default=None)
    vector_store: Optional[QdrantAdapter] = field(default=None)
    chunker: Optional[TokenizerDocumentChunker] = field(default=None)
    summarizer: Optional[HuggingFaceSummarizationService] = field(default=None)

    file_storage: Optional[LocalFileStorageAdapter] = field(default=None)

    document_upload: Optional[DocumentUploadUseCase] = field(default=None)
    document_processing: Optional[DocumentProcessingUseCase] = field(default=None)
    document_delete: Optional[DocumentDeleteUseCase] = field(default=None)
    chat_conversation: Optional[ChatConversationUseCase] = field(default=None)
    chat_summarization: Optional[ChatSummarizationUseCase] = field(default=None)

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        """
        Construct and return a fully wired container.

        Args:
            settings: The validated Settings instance from get_settings().

        Returns:
            A Container with every attribute populated and ready to serve.
        """
        c = cls()

        # Step 1: Initialise Firebase.
        # FIREBASE_SERVICE_ACCOUNT_FILE takes priority over the inline JSON string.
        initialize_firebase(
            service_account_json=settings.FIREBASE_SERVICE_ACCOUNT,
            service_account_file=settings.FIREBASE_SERVICE_ACCOUNT_FILE,
        )
        c.db = get_firestore_client()
        logger.info("Firestore client ready.")

        # Step 2: Repository adapters.
        c.chat_repo = FirebaseChatRepository(db=c.db)
        c.message_repo = FirebaseMessageRepository(db=c.db)
        c.document_repo = FirebaseDocumentRepository(db=c.db)
        c.chunk_repo = FirebaseChunkRepository(db=c.db)
        logger.info("Firebase repository adapters ready.")

        # Step 3: Embedding model — loaded once, never per request.
        logger.info("Loading embedding model.", extra={"model_id": settings.EMBEDDING_MODEL_ID})
        c.embedder = SentenceTransformerAdapter(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE,
        )

        # Step 4: Tokenizer for document chunking — same model ID as embedder.
        c.chunker = TokenizerDocumentChunker(model_id=settings.EMBEDDING_MODEL_ID)

        c.prompt_builder = PromptBuilder(
            language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG
        )

        # Step 5: Summarization model — loaded once, never per request.
        logger.info(
            "Loading summarization model.", extra={"model_id": settings.SUMMARIZATION_MODEL_ID}
        )
        c.summarizer = HuggingFaceSummarizationService(model_id=settings.SUMMARIZATION_MODEL_ID)

        # Step 6: Generation service adapter.
        c.generator = cls._build_generator(settings)

        # Step 7: Qdrant vector store.
        if settings.VECTOR_DB_HOST:
            c.vector_store = QdrantAdapter(
                host=settings.VECTOR_DB_HOST,
                port=settings.VECTOR_DB_PORT,
                distance_method=settings.VECTOR_DB_DISTANCE_METHOD,
            )
            c.vector_store.connect()
            logger.info("Qdrant connected (remote).", extra={"host": settings.VECTOR_DB_HOST})
        else:
            db_path = os.path.abspath(
                os.path.join(settings.VECTOR_DB_PATH, settings.VECTOR_DB_BACKEND)
            )
            os.makedirs(db_path, exist_ok=True)
            c.vector_store = QdrantAdapter(
                db_path=db_path,
                distance_method=settings.VECTOR_DB_DISTANCE_METHOD,
            )
            c.vector_store.connect()
            logger.info("Qdrant connected (local).", extra={"db_path": db_path})

        # Step 8: File storage adapter.
        c.file_storage = LocalFileStorageAdapter(base_upload_dir=settings.UPLOAD_BASE_DIR)

        # Step 9: Use cases — each receives abstract port interfaces only.
        c.document_upload = DocumentUploadUseCase(
            document_repo=c.document_repo,
            file_storage=c.file_storage,
            allowed_types=settings.FILE_ALLOWED_TYPES,
            max_size_mb=settings.FILE_MAX_SIZE
        )
        c.document_processing = DocumentProcessingUseCase(
            document_repo=c.document_repo,
            chunk_repo=c.chunk_repo,
            embedding_service=c.embedder,
            vector_store=c.vector_store,
            chunker=c.chunker,
        )

        c.document_delete = DocumentDeleteUseCase(
            document_repo=c.document_repo,
            file_storage=c.file_storage,
            chunk_repository=c.chunk_repo,
            vector_store=c.vector_store,
        )

        c.chat_conversation = ChatConversationUseCase(
            chat_repo=c.chat_repo,
            message_repo=c.message_repo,
            embedding_service=c.embedder,
            vector_store=c.vector_store,
            generation_service=c.generator,
            knowledge_collection=settings.KNOWLEDGE_COLLECTION,
            rag_top_k=settings.RAG_TOP_K,
            prompt_builder=c.prompt_builder,
        )
        c.chat_summarization = ChatSummarizationUseCase(
            chat_repo=c.chat_repo,
            message_repo=c.message_repo,
            summarization_service=c.summarizer,
            messages_to_summarize=settings.SUMMARIZE_EVERY_N_MESSAGES,
        )

        logger.info("DI container fully built.")
        return c

    @staticmethod
    def _build_generator(settings: Settings) -> IGenerationService:
        """
        Select and construct the generation service adapter from settings.

        Raises:
            ValueError: If ACTIVE_GENERATION_BACKEND is not a known value.
        """
        if settings.ACTIVE_GENERATION_BACKEND == "GROQ":
            return GroqGenerationAdapter(
                api_key=settings.GROQ_API_KEY,
                api_url=settings.GROQ_API_URL,
                model_id=settings.GROQ_GENERATION_MODEL,
                default_input_max_characters=settings.INPUT_DEFAULT_MAX_CHARACTERS,
                default_max_output_tokens=settings.GENERATION_DEFAULT_MAX_TOKENS,
                default_temperature=settings.GENERATION_DEFAULT_TEMPERATURE,
            )
        if settings.ACTIVE_GENERATION_BACKEND == "COHERE":
            return CohereGenerationAdapter(
                api_key=settings.COHERE_API_KEY,
                model_id=settings.COHERE_GENERATION_MODEL,
                default_input_max_characters=settings.INPUT_DEFAULT_MAX_CHARACTERS,
                default_max_output_tokens=settings.GENERATION_DEFAULT_MAX_TOKENS,
                default_temperature=settings.GENERATION_DEFAULT_TEMPERATURE,
            )
        raise ValueError(
            f"Unknown ACTIVE_GENERATION_BACKEND: '{settings.ACTIVE_GENERATION_BACKEND}'. "
            f"Valid values are: 'GROQ', 'COHERE'."
        )

    def shutdown(self) -> None:
        """
        Release all resources held by the container.

        Called by the lifespan context manager during application shutdown.
        """
        if self.vector_store:
            self.vector_store.disconnect()
            logger.info("Qdrant disconnected.")
        logger.info("Container shut down.")
