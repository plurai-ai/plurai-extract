"""Cloud RAG and vectorstore data source adapters.

These adapters fetch documents/chunks from existing cloud RAG services and
vectorstores, so a corpus can be built from what is already indexed, with no manual
data preparation.

Cloud services:
    AzureAISearchSource, VertexAISearchSource, BedrockKnowledgeBaseSource

Vectorstore adapters:
    LangChainVectorStoreSource, LlamaIndexVectorStoreSource
"""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from plurai_extract.sources.base import Document

logger = logging.getLogger(__name__)


# ── Azure AI Search ──────────────────────────────────────────────────────────


@dataclass
class AzureAISearchSource:
    """Fetch all documents from an Azure AI Search index.

    Scrolls through the entire index using the search API with ``*`` query
    and returns each result as a Document.

    Usage::

        source = AzureAISearchSource(
            endpoint="https://my-search.search.windows.net",
            index_name="my-index",
            api_key="...",
            content_field="content",
        )
        documents, relationships = load_only(source)
    """

    endpoint: str
    index_name: str
    api_key: str = ""
    content_field: str = "content"
    title_field: str = "title"
    top: int = 1000
    metadata_fields: list[str] = field(default_factory=list)

    def load(self, **kwargs: t.Any) -> list[Document]:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.api_key),
        )

        documents: list[Document] = []
        try:
            results = client.search(
                search_text="*",
                top=self.top,
                select=[self.content_field, self.title_field] + self.metadata_fields,
            )
        except Exception as exc:
            logger.error(
                "Azure AI Search query failed for index %r: %s",
                self.index_name,
                exc,
            )
            raise RuntimeError(f"Failed to search Azure AI Search index {self.index_name!r}: {exc}") from exc

        for idx, result in enumerate(results):
            text = result.get(self.content_field, "")
            if not text:
                continue
            name = result.get(self.title_field, f"doc_{idx}")
            meta = {k: result.get(k) for k in self.metadata_fields if result.get(k)}
            documents.append(Document(name=name, text=text, chunk_idx=idx, metadata=meta))

        logger.info("Loaded %d documents from Azure AI Search index %r", len(documents), self.index_name)
        return documents


# ── Google Vertex AI Search ──────────────────────────────────────────────────


@dataclass
class VertexAISearchSource:
    """Fetch documents from a Google Vertex AI Search data store.

    Lists documents in the data store and extracts their content.

    Usage::

        source = VertexAISearchSource(
            project_id="my-project",
            data_store_id="my-datastore",
        )
    """

    project_id: str
    data_store_id: str
    location: str = "global"
    content_field: str = "snippet"
    max_documents: int = 1000

    def load(self, **kwargs: t.Any) -> list[Document]:
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.DocumentServiceClient()
        parent = f"projects/{self.project_id}/locations/{self.location}/dataStores/{self.data_store_id}/branches/default_branch"

        documents: list[Document] = []
        try:
            request = discoveryengine.ListDocumentsRequest(
                parent=parent,
                page_size=min(self.max_documents, 1000),
            )
            doc_iter = client.list_documents(request=request)
        except Exception as exc:
            logger.error(
                "Vertex AI Search list failed for data store %r: %s",
                self.data_store_id,
                exc,
            )
            raise RuntimeError(f"Failed to list documents in Vertex AI Search data store {self.data_store_id!r}: {exc}") from exc

        for idx, doc in enumerate(doc_iter):
            if idx >= self.max_documents:
                break

            text = self._extract_text(doc)
            if not text:
                continue

            name = doc.name.rsplit("/", 1)[-1] if getattr(doc, "name", None) else f"doc_{idx}"
            documents.append(Document(name=name, text=text, chunk_idx=idx))

        logger.info(
            "Loaded %d documents from Vertex AI Search data store %r",
            len(documents),
            self.data_store_id,
        )
        return documents

    @staticmethod
    def _extract_text(doc: t.Any) -> str:
        """Safely extract text from a Vertex AI document object."""
        # Try derived_struct_data first
        struct = getattr(doc, "derived_struct_data", None)
        if struct and isinstance(struct, dict):
            text = struct.get("snippet", "")
            if text:
                return text
            # Try page content
            pages = struct.get("pages", [])
            if pages:
                text = " ".join(p.get("content", "") for p in pages if isinstance(p, dict))
                if text.strip():
                    return text.strip()

        # Try raw bytes content
        content = getattr(doc, "content", None)
        if content:
            raw_bytes = getattr(content, "raw_bytes", None)
            if raw_bytes:
                try:
                    return raw_bytes.decode("utf-8", errors="replace")
                except Exception:
                    pass

        return ""


# ── AWS Bedrock Knowledge Bases ──────────────────────────────────────────────


@dataclass
class BedrockKnowledgeBaseSource:
    """Fetch chunks from an AWS Bedrock Knowledge Base.

    Uses the ``retrieve`` API with a broad query to pull indexed chunks.

    Usage::

        source = BedrockKnowledgeBaseSource(
            knowledge_base_id="KBXXXXXXXX",
        )
    """

    knowledge_base_id: str
    region_name: str = "us-east-1"
    retrieval_query: str = "summarize all topics"
    max_results: int = 100

    def load(self, **kwargs: t.Any) -> list[Document]:
        import boto3

        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region_name,
        )

        documents: list[Document] = []
        next_token: str | None = None
        retrieved = 0
        empty_pages = 0

        while retrieved < self.max_results:
            req: dict[str, t.Any] = {
                "knowledgeBaseId": self.knowledge_base_id,
                "retrievalQuery": {"text": self.retrieval_query},
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": min(self.max_results - retrieved, 100),
                    },
                },
            }
            if next_token:
                req["nextToken"] = next_token

            try:
                response = client.retrieve(**req)
            except Exception as exc:
                logger.error(
                    "Bedrock retrieve failed for KB %r: %s",
                    self.knowledge_base_id,
                    exc,
                )
                if documents:
                    logger.warning("Returning %d partial results", len(documents))
                    break
                raise RuntimeError(f"Failed to retrieve from Bedrock Knowledge Base {self.knowledge_base_id!r}: {exc}") from exc

            page_count = 0
            for result in response.get("retrievalResults", []):
                text = result.get("content", {}).get("text", "")
                if not text:
                    continue

                location = result.get("location", {})
                uri = location.get("s3Location", {}).get("uri", "")
                name = uri.rsplit("/", 1)[-1] if uri else f"chunk_{retrieved}"

                meta: dict[str, t.Any] = {}
                if uri:
                    meta["s3_uri"] = uri
                score = result.get("score")
                if score is not None:
                    meta["score"] = score

                documents.append(Document(name=name, text=text, chunk_idx=retrieved, metadata=meta))
                retrieved += 1
                page_count += 1

            next_token = response.get("nextToken")
            if not next_token:
                break

            # Guard against infinite loops on empty pages
            if page_count == 0:
                empty_pages += 1
                if empty_pages >= 3:
                    logger.warning("3 consecutive empty pages — stopping pagination")
                    break
            else:
                empty_pages = 0

        logger.info(
            "Retrieved %d chunks from Bedrock Knowledge Base %r",
            len(documents),
            self.knowledge_base_id,
        )
        return documents


# ── LangChain VectorStore ────────────────────────────────────────────────────


@dataclass
class LangChainVectorStoreSource:
    """Fetch documents from a LangChain vectorstore.

    Uses similarity search to retrieve indexed chunks.  Works with any
    LangChain vectorstore (FAISS, Chroma, Pinecone, Weaviate, etc.).

    .. note::

        For best coverage, provide a domain-relevant ``search_query``
        (e.g. ``"financial reports and revenue analysis"``).  The default
        broad query may not surface all documents depending on the
        vectorstore backend.

    Usage::

        from langchain_community.vectorstores import FAISS
        vs = FAISS.load_local("my_index", embeddings)
        source = LangChainVectorStoreSource(vectorstore=vs, search_query="all documents")
    """

    vectorstore: t.Any  # langchain_core.vectorstores.VectorStore
    search_query: str = "summarize all content"
    k: int = 500

    def load(self, **kwargs: t.Any) -> list[Document]:
        query = kwargs.get("search_query", self.search_query)
        k = kwargs.get("k", self.k)

        try:
            lc_docs = self.vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            logger.error("LangChain vectorstore search failed: %s", exc)
            raise RuntimeError(f"Failed to search LangChain vectorstore: {exc}") from exc

        documents: list[Document] = []
        for idx, doc in enumerate(lc_docs):
            text = doc.page_content
            if not text:
                continue
            meta = doc.metadata or {}
            name = meta.get("source", meta.get("title", f"doc_{idx}"))
            filtered_meta = {k_: v for k_, v in meta.items() if k_ not in ("source", "title")}
            documents.append(Document(name=name, text=text, chunk_idx=idx, metadata=filtered_meta))

        logger.info("Loaded %d documents from LangChain vectorstore", len(documents))
        return documents


# ── LlamaIndex VectorStore ───────────────────────────────────────────────────


@dataclass
class LlamaIndexVectorStoreSource:
    """Fetch documents from a LlamaIndex index.

    Uses the retriever to pull indexed chunks.

    .. note::

        For best coverage, provide a domain-relevant ``query_str``
        (e.g. ``"financial reports and revenue analysis"``).

    Usage::

        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents(my_docs)
        source = LlamaIndexVectorStoreSource(index=index, query_str="all documents")
    """

    index: t.Any  # llama_index.core.indices.base.BaseIndex
    similarity_top_k: int = 500
    query_str: str = "summarize all content"

    def load(self, **kwargs: t.Any) -> list[Document]:
        retriever = self.index.as_retriever(
            similarity_top_k=kwargs.get("similarity_top_k", self.similarity_top_k),
        )
        query = kwargs.get("query_str", self.query_str)

        try:
            nodes = retriever.retrieve(query)
        except Exception as exc:
            logger.error("LlamaIndex retriever failed: %s", exc)
            raise RuntimeError(f"Failed to retrieve from LlamaIndex index: {exc}") from exc

        documents: list[Document] = []
        for idx, node_with_score in enumerate(nodes):
            node = node_with_score.node if hasattr(node_with_score, "node") else node_with_score

            # Only accept nodes with proper get_content() — skip objects
            # whose str() would produce garbage like "<Node object at 0x...>"
            if not hasattr(node, "get_content"):
                continue
            text = node.get_content()
            if not text:
                continue

            meta = getattr(node, "metadata", {}) or {}
            name = meta.get("file_name", meta.get("source", f"node_{idx}"))
            documents.append(
                Document(
                    name=name,
                    text=text,
                    chunk_idx=idx,
                    metadata={k: v for k, v in meta.items() if k not in ("file_name", "source")},
                )
            )

        logger.info("Loaded %d documents from LlamaIndex index", len(documents))
        return documents
