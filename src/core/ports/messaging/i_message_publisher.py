from abc import ABC, abstractmethod
from typing import Any


class IMessagePublisher(ABC):
    @abstractmethod
    async def publish(self, topic: str, message: dict[str, Any]) -> None: ...
