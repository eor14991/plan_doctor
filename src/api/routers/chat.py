"""
API Router: Chat
=================
Handles all conversation endpoints consumed by the PlantDoctor mobile app.

Endpoints:
    POST /chat/start               - Create a new chat session.
    POST /chat/{chat_id}/message   - Send a message and receive a RAG response.
    GET  /chat/{chat_id}/history   - Retrieve recent messages for a session.

Authentication:
    All endpoints require a valid Firebase ID token in the Authorization header.
    The user_id is extracted from the verified token, not from the request body.
    This prevents a client from impersonating another user by supplying an
    arbitrary user_id in the payload.

Summarization:
    After each message, the route checks whether the chat has reached the
    summarization threshold. If it has, a background task is scheduled so
    that the response is returned to the client immediately without waiting
    for the summarization model to finish.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ..schemas.chat_schemas import MessageRequest, MessageResponse, StartChatRequest
from ...core.use_cases.chat_conversation import ChatConversationUseCase
from ...core.use_cases.chat_summarization import ChatSummarizationUseCase
from ..dependencies import get_chat_use_case, get_current_user_id, get_summarization_use_case
import time



logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post("/start", status_code=status.HTTP_201_CREATED)
async def start_chat(
    body: StartChatRequest,
    user_id: str = Depends(get_current_user_id),
    chat_use_case: ChatConversationUseCase = Depends(get_chat_use_case),
) -> JSONResponse:
    """
    Create a new chat session and return its identifier.

    The mobile client stores this chat_id and includes it in all subsequent
    message requests for this conversation.
    """
    saved_chat = await chat_use_case.create_chat(user_id=user_id, title=body.title, chat_id=body.chat_id)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"chat_id": saved_chat.chat_id, "title": saved_chat.title},
    )


@chat_router.post("/{chat_id}/message")
async def send_message(
    chat_id: str,
    body: MessageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    chat_use_case: ChatConversationUseCase = Depends(get_chat_use_case),
    summarization_use_case: ChatSummarizationUseCase = Depends(get_summarization_use_case),
) -> JSONResponse:
    """
    Accept a user message and return a RAG-grounded response.

    The request body must include at least one of: prompt (text typed by
    the user) or label (a plant disease label from the mobile ML model).

    After the response is sent, the route schedules a background summarization
    task if the chat has reached the configured message threshold.
    """
    start = time.perf_counter()
    if not body.prompt and not body.label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'prompt' or 'label' must be provided.",
        )

    response, total_time1 = await chat_use_case.execute(
        chat_id=chat_id,
        prompt=body.prompt,
        label=body.label,
        confidence=body.confidence,
        image_url=body.image_url,
    )

    if response.needs_summarization:
        background_tasks.add_task(summarization_use_case.execute, chat_id=chat_id)
        logger.info("Summarization task scheduled.", extra={"chat_id": chat_id})


    # Execute process
    total_time = time.perf_counter() - start
    return JSONResponse(
    status_code=status.HTTP_200_OK,
    content={
        "message": MessageResponse(
            answer=response.answer,
            sources_count=len(response.sources),
            chat_id=chat_id
        ).model_dump(),
        "total_time":total_time,
        "total_time_for pipeline":total_time1
    }
)

@chat_router.get("/{chat_id}/history")
async def get_chat_history(
    chat_id: str,
    limit: int = 20,
    user_id: str = Depends(get_current_user_id),
    chat_use_case: ChatConversationUseCase = Depends(get_chat_use_case),
) -> JSONResponse:
    """
    Return the most recent messages in a chat session.

    Messages are returned in chronological order, oldest first.
    """
    messages = await chat_use_case.get_history(chat_id, limit=limit)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "chat_id": chat_id,
            "messages": [
                {
                    "role": m.role,
                    "prompt": m.prompt,
                    "rag_response": m.rag_response,
                    "label": m.label,
                    "confidence": m.confidence,
                    "timestamp": m.created_at.isoformat(),
                }
                for m in messages
            ],
        },
    )
