"""
Adapter: CohereGenerationAdapter
"""

from __future__ import annotations

import logging
from typing import Optional

from ....core.ports.services.i_generation_service import ChatMessage, ChatRole, IGenerationService

logger = logging.getLogger(__name__)


class CohereGenerationAdapter(IGenerationService):

    def __init__(
        self,
        api_key: str,
        model_id: str = "command-r-plus",
        default_input_max_characters: int = 4096,
        default_max_output_tokens: int = 1000,
        default_temperature: float = 0.1,
    ) -> None:
        self._model_id = model_id
        self._max_chars = default_input_max_characters
        self._max_tokens = default_max_output_tokens
        self._temperature = default_temperature
        import cohere

        # Using AsyncClient for asynchronous operations
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def generate_text(
        self,
        prompt: str,
        chat_history: Optional[list[ChatMessage]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        history = list(chat_history) if chat_history else []
        preamble: Optional[str] = None
        messages: list[dict] = []

        for msg in history:
            if msg.role == ChatRole.SYSTEM and preamble is None:
                preamble = msg.content
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": ChatRole.USER.value, "content": prompt[: self._max_chars].strip()})

        try:
            kwargs: dict = {
                "model": self._model_id,
                "messages": messages,
                "max_tokens": max_output_tokens or self._max_tokens,
                "temperature": temperature if temperature is not None else self._temperature,
            }
            if preamble:
                kwargs["system"] = preamble

            resp = await self._client.chat(**kwargs)
            content = resp.message.content if resp and hasattr(resp, "message") else None

            if not content:
                return None

            return content[0].text if isinstance(content, list) else str(content)
        except Exception:
            logger.error("Cohere API call failed", exc_info=True)
            return None

    def build_system_message(self, system_prompt: str) -> ChatMessage:
        return ChatMessage(role=ChatRole.SYSTEM, content=system_prompt[: self._max_chars].strip())
