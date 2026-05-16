from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from openai import AsyncOpenAI, RateLimitError

from ....core.ports.services.i_generation_service import ChatMessage, ChatRole, IGenerationService

logger = logging.getLogger(__name__)


class GroqGenerationAdapter(IGenerationService):

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_id: str,
        default_input_max_characters: int = 4096,
        default_max_output_tokens: int = 1000,
        default_temperature: float = 0.1,
    ) -> None:
        self._model_id = model_id
        self._max_chars = default_input_max_characters
        self._max_tokens = default_max_output_tokens
        self._temperature = default_temperature
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_url,max_retries=0)
        logger.info("GroqGenerationAdapter ready.", extra={"model_id": model_id})

    async def _call_api(self, **kwargs) -> object:
        """
        Internal helper for all Groq API calls.
        Uses asyncio.sleep during rate-limit backoff so the event loop
        remains unblocked and other requests can continue.
        """
        max_retries = 4
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                delay = (base_delay * 2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Groq rate limit hit, backing off.",
                    extra={"attempt": attempt + 1, "delay_seconds": round(delay, 2)},
                )
                await asyncio.sleep(delay)

    async def generate_text(
        self,
        prompt: str,
        chat_history: Optional[list[ChatMessage]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        history = list(chat_history) if chat_history else []
        history.append(ChatMessage(role=ChatRole.USER, content=prompt[: self._max_chars].strip()))

        messages = [
            {
                "role": m.role.value if hasattr(m.role, "value") else m.role,
                "content": m.content,
            }
            for m in history
        ]

        try:
            resp = await self._call_api(
                model=self._model_id,
                messages=messages,
                max_tokens=max_output_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._temperature,
            )
            return resp.choices[0].message.content if resp.choices else None
        except Exception:
            logger.error("Groq generate_text failed.", exc_info=True)
            return None

    def build_system_message(self, system_prompt: str) -> ChatMessage:
        return ChatMessage(
            role=ChatRole.SYSTEM,
            content=system_prompt[: self._max_chars].strip(),
        )
