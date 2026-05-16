"""
Port: IGenerationService
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


class IGenerationService(ABC):

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        chat_history: Optional[list[ChatMessage]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]: ...

    @abstractmethod
    def build_system_message(self, system_prompt: str) -> ChatMessage: ...
