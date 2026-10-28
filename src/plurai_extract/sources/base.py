"""Base types, protocol, and shared utilities for data source adapters."""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass, field


@dataclass
class Document:
    """A chunk of text, the unit a consumer indexes."""

    name: str
    text: str
    chunk_idx: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable form. Stable wire format for handing
        extracted documents to another process (uploads, caches)."""
        return {
            "name": self.name,
            "text": self.text,
            "chunk_idx": self.chunk_idx,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            name=data["name"],
            text=data["text"],
            chunk_idx=data.get("chunk_idx", 0),
            metadata=data.get("metadata", {}) or {},
        )


@dataclass
class SeedQA:
    """A seed question-answer pair for coverage and terminology extraction."""

    question: str
    answer: str
    evidence_texts: list[str] = field(default_factory=list)
    referenced_doc_keys: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence_texts": list(self.evidence_texts),
            # tuples → lists for JSON; from_dict restores the tuple shape.
            "referenced_doc_keys": [list(k) for k in self.referenced_doc_keys],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedQA":
        return cls(
            question=data["question"],
            answer=data["answer"],
            evidence_texts=list(data.get("evidence_texts", [])),
            referenced_doc_keys=[(k[0], k[1]) for k in data.get("referenced_doc_keys", [])],
        )


@dataclass
class InitialRelationship:
    """A pre-built edge from structural data (e.g. foreign keys)."""

    source_doc_name: str
    source_chunk_idx: int
    target_doc_name: str
    target_chunk_idx: int
    type: str
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_doc_name": self.source_doc_name,
            "source_chunk_idx": self.source_chunk_idx,
            "target_doc_name": self.target_doc_name,
            "target_chunk_idx": self.target_chunk_idx,
            "type": self.type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InitialRelationship":
        return cls(
            source_doc_name=data["source_doc_name"],
            source_chunk_idx=data["source_chunk_idx"],
            target_doc_name=data["target_doc_name"],
            target_chunk_idx=data["target_chunk_idx"],
            type=data["type"],
            properties=data.get("properties", {}) or {},
        )


class DataSource(t.Protocol):
    """Protocol for data source adapters."""

    def load(self, **kwargs: t.Any) -> list[Document]: ...


def chunk_tokens(text: str, chunk_size: int = 1024, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping token-based chunks (tiktoken cl100k_base)."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        chunk = enc.decode(tokens[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


FINGERPRINT_LEN = 60


def fingerprint_match_seed_qa(
    examples: list[dict],
    documents: list[Document],
) -> list[SeedQA]:
    """Build SeedQA pairs by fingerprint-matching evidence text to document chunks."""

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    doc_index: dict[str, list[tuple[int, str]]] = {}
    for doc in documents:
        doc_index.setdefault(doc.name, []).append((doc.chunk_idx, norm(doc.text)))

    results: list[SeedQA] = []
    for ex in examples:
        question, answer = ex.get("question", ""), ex.get("answer", "")
        if not question or not answer:
            continue

        evidence_entries = ex.get("evidence", [])
        if isinstance(evidence_entries, str):
            evidence_entries = [{"evidence_text": evidence_entries, "doc_name": "unknown"}]

        evidence_texts = [ev["evidence_text"] for ev in evidence_entries if ev.get("evidence_text", "").strip()]

        # Match evidence fingerprints to chunks
        refs: list[tuple[str, int]] = []
        for ev in evidence_entries:
            ev_text, doc_name = ev.get("evidence_text", ""), ev.get("doc_name", "")
            if not ev_text or not doc_name:
                continue
            normed = norm(ev_text)
            fp = normed[len(normed) // 4 : len(normed) // 4 + FINGERPRINT_LEN]
            if fp:
                refs.extend((doc_name, idx) for idx, content in doc_index.get(doc_name, []) if fp in content)

        results.append(SeedQA(question=question, answer=str(answer), evidence_texts=evidence_texts, referenced_doc_keys=refs))
    return results
