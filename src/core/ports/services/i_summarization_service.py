"""
Port: ISummarizationService
=============================
Abstract interface for text summarization operations.

Kept separate from IGenerationService for two reasons aligned with the
Interface Segregation Principle:
    - IGenerationService is backed by a large general-purpose LLM (Grok).
    - ISummarizationService is backed by a smaller task-specific model
      (facebook/bart-large-cnn) that is faster and cheaper for condensing
      conversation history.

Merging them into one interface would force GrokAdapter to stub out
summarize() and HuggingFaceAdapter to stub out generate_text(), both of
which are ISP violations.
"""
from abc import ABC, abstractmethod


class ISummarizationService(ABC):

    @abstractmethod
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> str:
        """
        Condense a long text into a shorter summary.

        Args:
            text:       The full text to summarise.
            max_length: Upper bound on the summary length in tokens.
            min_length: Lower bound on the summary length in tokens.

        Returns:
            The summarised string. Never returns None; falls back to a
            truncated version of the input if the model fails.
        """
        ...
