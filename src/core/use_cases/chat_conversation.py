from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ...infrastructure.prompt_builder import PromptBuilder
from ..domain.entities.chat import Chat
from ..domain.entities.message import Message
from ..domain.value_objects import ChatResponse
from ..ports.repositories.i_chat_repository import IChatRepository
from ..ports.repositories.i_message_repository import IMessageRepository
from ..ports.services.i_embedding_service import DocumentType, IEmbeddingService
from ..ports.services.i_generation_service import ChatMessage, ChatRole, IGenerationService
from ..ports.services.i_vector_store import IVectorStore

logger = logging.getLogger(__name__)


class ChatConversationUseCase:

    def __init__(
        self,
        chat_repo: IChatRepository,
        message_repo: IMessageRepository,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        generation_service: IGenerationService,
        prompt_builder: PromptBuilder,
        knowledge_collection: str = "plant_knowledge",
        rag_top_k: int = 5,
    ) -> None:
        self._chat_repo = chat_repo
        self._message_repo = message_repo
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._generation_service = generation_service
        self._prompt_builder = prompt_builder
        self._knowledge_collection = knowledge_collection
        self._rag_top_k = rag_top_k

    async def execute(
        self,
        chat_id: str,
        prompt: Optional[str] = None,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        image_url: Optional[str] = None,
    ) -> tuple[ChatResponse, float]:
        """
        Orchestrate the RAG conversation pipeline.
        Returns a tuple of (ChatResponse, total_execution_time).
        """
        overall_start = time.perf_counter()

        # --- 1. DB Initialisation ---
        t_start = time.perf_counter()
        chat = await self._chat_repo.get_by_id(chat_id)
        if not chat:
            logger.warning("Chat session not found.", extra={"chat_id": chat_id})
            return (
                ChatResponse(answer="Session not found. Please start a new chat.", sources=[]),
                time.perf_counter() - overall_start,
            )

        user_message = Message(
            chat_id=chat_id,
            role="user",
            prompt=prompt,
            label=label,
            confidence=confidence,
            image_url=image_url,
        )

        # Save message and increment count in parallel
        # Note: increment_message_count returns the NEW count from the DB
        _, new_count = await asyncio.gather(
            self._message_repo.save(user_message),
            self._chat_repo.increment_message_count(chat_id),
        )

        needs_summarization = new_count > 0 and new_count % chat.summarize_every_n_messages == 0
        time_db_init = time.perf_counter() - t_start

        query_text = user_message.display_text
        if not query_text:
            return (
                ChatResponse(answer="Please send a question or an image.", sources=[]),
                time.perf_counter() - overall_start,
            )

        # --- 2. Local Query Enrichment (no LLM call) ---
        t_start = time.perf_counter()
        search_query = self._enrich_query(query_text, label)
        time_enrichment = time.perf_counter() - t_start

        # --- 3. Embedding + Vector Search (offloaded to thread pool) ---
        t_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        query_vector = await loop.run_in_executor(
            None,
            lambda: self._embedding_service.embed_text(search_query, DocumentType.QUERY),
        )
        retrieved_docs = await loop.run_in_executor(
            None,
            lambda: self._vector_store.search_by_vector(
                self._knowledge_collection, query_vector, self._rag_top_k
            ),
        )
        time_vector = time.perf_counter() - t_start

        # Render prompts
        doc_ctx = self._prompt_builder.render(
            "rag_document_context",
            {"documents": retrieved_docs},
        )
        system_prompt_text = self._prompt_builder.render(
            "rag_system_prompt",
            {"documents_context": doc_ctx},
        )
        user_prompt = self._prompt_builder.render(
            "rag_user_query",
            {"query": query_text, "label": label, "confidence": confidence},
        )

        # --- 4. LLM Generation ---
        chat_history = await self._build_chat_history(chat, system_prompt_text)

        t_start = time.perf_counter()
        answer = await self._generation_service.generate_text(
            prompt=user_prompt,
            chat_history=chat_history,
        )
        time_generation = time.perf_counter() - t_start

        if not answer:
            logger.error("Generation returned None.", extra={"chat_id": chat_id})
            return (
                ChatResponse(
                    answer="Sorry, I could not generate a response. Please try again.",
                    sources=[],
                ),
                time.perf_counter() - overall_start,
            )

        # --- 5. Persist Assistant Response ---
        t_start = time.perf_counter()
        await self._message_repo.save(
            Message(chat_id=chat_id, role="assistant", rag_response=answer)
        )
        time_db_save = time.perf_counter() - t_start

        total_time = time.perf_counter() - overall_start

        logger.info(
            "Response generated.",
            extra={
                "chat_id": chat_id,
                "sources": len(retrieved_docs),
                "needs_summarization": needs_summarization,
                "time_db_init": round(time_db_init, 3),
                "time_enrichment": round(time_enrichment, 3),
                "time_vector": round(time_vector, 3),
                "time_generation": round(time_generation, 3),
                "time_db_save": round(time_db_save, 3),
                "time_total": round(total_time, 3),
            },
        )

        return (
            ChatResponse(
                answer=answer,
                sources=retrieved_docs,
                needs_summarization=needs_summarization,
            ),
            total_time,
        )

    async def _build_chat_history(self, chat: Chat, system_prompt_text: str) -> list[ChatMessage]:
        """
        Build the message list sent to the LLM.

        [0] SYSTEM  — retrieved documents + instructions (rebuilt fresh every turn).
        [1] ASSISTANT — compressed summary if one exists, otherwise last 5 Q&A pairs.

        Documents are injected into the system message only and are never stored
        in Firestore, so they never pollute the history window of future requests.
        """
        system_msg = self._generation_service.build_system_message(system_prompt_text)
        history = [system_msg]

        if chat.summary:
            history.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=f"[Summary of previous conversation]: {chat.summary}",
                )
            )
        else:
            recent = await self._message_repo.get_recent_n(chat.chat_id, n=5)
            for msg in recent:
                if msg.role == "user" and msg.display_text:
                    history.append(ChatMessage(role=ChatRole.USER, content=msg.display_text))
                elif msg.role == "assistant" and msg.rag_response:
                    history.append(ChatMessage(role=ChatRole.ASSISTANT, content=msg.rag_response))

        return history

    @staticmethod
    def _enrich_query(query_text: str, label: Optional[str]) -> str:
        """
        Enrich the search query locally without an LLM call.

        When a disease label is present it is appended in both its original
        form and as a readable phrase. BAAI/bge-m3 handles the rest.
        """
        parts = [query_text]
        if label:
            readable = label.replace("___", " ").replace("_", " ")
            parts.append(readable)
        return " ".join(parts)

    async def create_chat(self, user_id: str, chat_id: str, title: str = "New Diagnosis") -> Chat:
        chat = Chat(user_id=user_id, title=title, chat_id=chat_id)
        saved = await self._chat_repo.save(chat)
        logger.info(
            "Chat session created.",
            extra={"chat_id": saved.chat_id, "user_id": user_id},
        )
        return saved

    async def get_history(self, chat_id: str, limit: int = 5) -> list[Message]:
        return await self._message_repo.get_recent_n(chat_id, n=limit)
