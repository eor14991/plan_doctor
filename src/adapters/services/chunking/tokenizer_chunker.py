"""
Adapter: TokenizerDocumentChunker — implements IDocumentChunker.
ONLY file in the production system that imports transformers and langchain.
Defect #8 fix: tokenizer loaded once in __init__, never per-request.
Returns RawDocumentChunk (domain value objects), never langchain Documents.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from ....core.domain.value_objects.raw_document_chunk import RawDocumentChunk
from ....core.ports.services.i_document_chunker import IDocumentChunker

logger = logging.getLogger(__name__)
_MD_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]


class TokenizerDocumentChunker(IDocumentChunker):
    def __init__(self, model_id: str) -> None:
        logger.info("Loading tokenizer", extra={"model_id": model_id})
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer.model_max_length = 8192
        logger.info("Tokenizer ready")

    def load_and_split(
        self, file_path: str, file_id: str, chunk_size: int, overlap_size: int
    ) -> list[RawDocumentChunk]:
        loader = self._build_loader(file_path)
        if loader is None:
            return []
        try:
            raw_docs: list[Document] = loader.load()
        except Exception:
            logger.error("Load failed", extra={"file_path": file_path}, exc_info=True)
            return []
        if not raw_docs:
            return []
        lc_chunks = self._split(raw_docs, file_id, chunk_size, overlap_size)
        return [
            RawDocumentChunk(text=d.page_content, metadata=d.metadata, order=i)
            for i, d in enumerate(lc_chunks)
            if d.page_content.strip()
        ]

    def _split(
        self, docs: list[Document], file_id: str, chunk_size: int, overlap_size: int
    ) -> list[Document]:
        text = "\n\n".join(d.page_content for d in docs)
        md_split = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS, strip_headers=False)
        txt_split = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=["\n\n", "\n", " ", ""],
        )
        try:
            sections = md_split.split_text(text)
            chunks: list[Document] = []
            for s in sections:
                s.metadata["file_id"] = file_id
                chunks.extend(txt_split.split_documents([s]))
            return chunks
        except Exception:
            logger.error("Split failed", exc_info=True)
            return []

    @staticmethod
    def _build_loader(file_path: str) -> Optional[BaseLoader]:
        if not os.path.exists(file_path):
            logger.warning("File not found", extra={"file_path": file_path})
            return None
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        if ext == ".pdf":
            try:
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from langchain_docling.loader import DoclingLoader

                opts = PdfPipelineOptions()
                opts.do_ocr = True
                opts.do_table_structure = True
                return DoclingLoader(
                    file_path,
                    converter=DocumentConverter(
                        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
                    ),
                )
            except ImportError:
                logger.error("langchain-docling not installed")
                return None
        logger.warning("Unsupported extension", extra={"ext": ext})
        return None
