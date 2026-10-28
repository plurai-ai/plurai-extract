"""Wrappers for LangChain and LlamaIndex document loaders."""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

from plurai_extract.sources.base import Document, chunk_tokens

logger = logging.getLogger(__name__)


@dataclass
class LangChainSource:
    """Wraps any LangChain DocumentLoader to produce plurai-extract Documents."""

    loader: t.Any  # langchain_core.document_loaders.BaseLoader
    chunk_size: int = 1024
    chunk_overlap: int = 50

    def load(self, **kwargs: t.Any) -> list[Document]:
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        lc_docs = self.loader.load()
        documents: list[Document] = []

        for lc_doc in lc_docs:
            name = lc_doc.metadata.get("source", "unknown")
            meta = {k: v for k, v in lc_doc.metadata.items() if k != "source"}
            chunks = chunk_tokens(lc_doc.page_content, chunk_size, chunk_overlap)
            for idx, text in enumerate(chunks):
                documents.append(Document(name=name, text=text, chunk_idx=idx, metadata=meta))

        logger.info("Loaded %d chunks from %d LangChain documents", len(documents), len(lc_docs))
        return documents


@dataclass
class LlamaIndexSource:
    """Wraps any LlamaIndex Reader to produce plurai-extract Documents."""

    reader: t.Any  # llama_index.core.readers.base.BaseReader
    chunk_size: int = 1024
    chunk_overlap: int = 50

    def load(self, **kwargs: t.Any) -> list[Document]:
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)

        li_docs = self.reader.load_data()
        documents: list[Document] = []

        for li_doc in li_docs:
            meta = li_doc.metadata or {}
            name = meta.get("file_name", meta.get("source", "unknown"))
            chunks = chunk_tokens(li_doc.text, chunk_size, chunk_overlap)
            for idx, text in enumerate(chunks):
                documents.append(
                    Document(
                        name=name,
                        text=text,
                        chunk_idx=idx,
                        metadata={k: v for k, v in meta.items() if k not in ("file_name", "source")},
                    )
                )

        logger.info("Loaded %d chunks from %d LlamaIndex documents", len(documents), len(li_docs))
        return documents
