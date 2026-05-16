"""
Adapter: HuggingFaceSummarizationService
==========================================
Implements ISummarizationService using a local HuggingFace summarization
pipeline backed by facebook/bart-large-cnn by default.

Model selection rationale:
    BART is trained specifically for abstractive summarization, making it
    more accurate than a general-purpose LLM for condensing conversation
    history. Running it locally avoids per-token API costs and network
    latency on every summarization trigger.

    Recommended alternatives by server size:
        facebook/bart-large-cnn   (~1.6 GB)  - default, highest quality.
        sshleifer/distilbart-cnn-12-6 (~250 MB) - suitable for small servers.

Startup behaviour:
    The model is loaded once in __init__, which is called once by
    Container.build() at application startup. Subsequent calls to summarize()
    use the in-memory pipeline with no disk I/O.
"""
from __future__ import annotations

import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ....core.ports.services.i_summarization_service import ISummarizationService

logger = logging.getLogger(__name__)


class HuggingFaceSummarizationService:
    def __init__(self,model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def summarize(self, text,max_length,
            min_length):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        output_ids = self.model.generate(
            **inputs,
            min_length=min_length,
            max_length=max_length
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _load_model(self) -> None:
        """
        Load the HuggingFace summarization pipeline into memory.

        Raises the underlying transformers exception if the model cannot be
        loaded, because a missing summarization model is a fatal startup error.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            logger.info("Loading summarization model.", extra={"model_id": self._model_id})
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)
            logger.info("Summarization model loaded.")
        except Exception:
            logger.error("Failed to load summarization model.", exc_info=True)
            raise


