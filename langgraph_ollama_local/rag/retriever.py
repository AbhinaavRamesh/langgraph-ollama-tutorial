"""
Document retrieval from ChromaDB for RAG applications.

This module provides retrieval functionality to query documents
from ChromaDB using similarity search.

Example:
    >>> from langgraph_ollama_local.rag import LocalRetriever
    >>> retriever = LocalRetriever()
    >>> results = retriever.retrieve("What is Self-RAG?", k=3)
    >>> for doc, score in results:
    ...     print(f"Score: {score:.3f} - {doc.page_content[:100]}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


class RetrieverConfig(BaseSettings):
    """
    Configuration for document retrieval.

    Attributes:
        collection_name: ChromaDB collection to query.
        persist_directory: ChromaDB persistence directory.
        embedding_model: Model for query embeddings.
        default_k: Default number of results to return.
        score_threshold: Minimum similarity score (0-1).
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    collection_name: str = Field(
        default="documents",
        description="ChromaDB collection name",
    )
    persist_directory: str = Field(
        default=".chromadb",
        description="ChromaDB persistence directory",
    )
    embedding_model: str = Field(
        default="all-mpnet-base-v2",
        description="Embedding model name",
    )
    default_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Default number of results",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score",
    )


class LocalRetriever:
    """
    Local document retriever using ChromaDB.

    This class provides similarity search over indexed documents.

    Attributes:
        config: Retriever configuration.
        embeddings: Embedding model for queries.

    Example:
        >>> retriever = LocalRetriever()
        >>> docs = retriever.retrieve("What is RAG?")
        >>> print(f"Found {len(docs)} relevant documents")
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        embeddings: LocalEmbeddings | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            config: Retriever configuration.
            embeddings: Embedding model for queries.
        """
        self.config = config or RetrieverConfig()
        self.embeddings = embeddings or LocalEmbeddings(
            model_name=self.config.embedding_model
        )

        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> "chromadb.Client":
        """Get the ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb is required for retrieval. "
                    "Install with: pip install langgraph-ollama-local[rag]"
                )

            persist_dir = Path(self.config.persist_directory)
            if not persist_dir.exists():
                raise FileNotFoundError(
                    f"ChromaDB directory not found: {persist_dir}. "
                    "Run DocumentIndexer first to create the index."
                )

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

        return self._client

    @property
    def collection(self) -> "chromadb.Collection":
        """Get the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_collection(
                name=self.config.collection_name,
            )

        return self._collection

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents similar to the query.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_metadata: Metadata filters (e.g., {"source": "paper.pdf"}).
            score_threshold: Minimum similarity score to include.

        Returns:
            List of (Document, score) tuples, sorted by relevance.

        Example:
            >>> results = retriever.retrieve("self-reflective RAG", k=5)
            >>> for doc, score in results:
            ...     print(f"{score:.3f}: {doc.metadata['source']}")
        """
        k = k or self.config.default_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Build query params
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Execute query
        results = self.collection.query(**query_params)

        # Process results
        documents_with_scores = []

        if results["documents"] and results["documents"][0]:
            for i, (doc_text, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
                    results["distances"][0] if results["distances"] else [0] * len(results["documents"][0]),
                )
            ):
                # Convert distance to similarity score (cosine distance -> similarity)
                # ChromaDB returns L2 distance by default, or cosine distance if configured
                score = 1 - distance if distance <= 1 else 1 / (1 + distance)

                if score >= threshold:
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata or {},
                    )
                    documents_with_scores.append((doc, score))

        # Sort by score (highest first)
        documents_with_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Retrieved {len(documents_with_scores)} documents for query: {query[:50]}...")
        return documents_with_scores

    def retrieve_documents(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve documents without scores (simpler interface).

        Args:
            query: The search query.
            k: Number of results to return.
            filter_metadata: Metadata filters.

        Returns:
            List of Document objects.

        Example:
            >>> docs = retriever.retrieve_documents("What is CRAG?")
            >>> context = "\\n".join([d.page_content for d in docs])
        """
        results = self.retrieve(query, k=k, filter_metadata=filter_metadata)
        return [doc for doc, _ in results]

    def retrieve_with_sources(
        self,
        query: str,
        k: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve documents with source information for citations.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            Dictionary with documents and formatted sources.

        Example:
            >>> result = retriever.retrieve_with_sources("What is Self-RAG?")
            >>> print(result["sources"])
        """
        results = self.retrieve(query, k=k)

        # Group by source
        sources: dict[str, list[dict]] = {}
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append({
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
                "score": score,
            })

        # Format source citations
        formatted_sources = []
        for i, (source, chunks) in enumerate(sources.items(), start=1):
            filename = Path(source).name
            pages = sorted(set(c["page"] for c in chunks if c["page"]))
            page_str = f"pages {', '.join(map(str, pages))}" if pages else ""
            avg_score = sum(c["score"] for c in chunks) / len(chunks)

            formatted_sources.append({
                "index": i,
                "filename": filename,
                "path": source,
                "pages": pages,
                "page_string": page_str,
                "relevance_score": avg_score,
                "chunk_count": len(chunks),
            })

        return {
            "documents": [doc for doc, _ in results],
            "scores": [score for _, score in results],
            "sources": formatted_sources,
            "query": query,
        }

    def get_document_by_id(self, doc_id: str) -> Document | None:
        """
        Get a specific document by its ID.

        Args:
            doc_id: The document ID.

        Returns:
            Document if found, None otherwise.
        """
        results = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])

        if results["documents"]:
            return Document(
                page_content=results["documents"][0],
                metadata=results["metadatas"][0] if results["metadatas"] else {},
            )
        return None

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        LangChain-compatible interface for similarity search.

        Args:
            query: The search query.
            k: Number of results.

        Returns:
            List of (Document, relevance_score) tuples.
        """
        return self.retrieve(query, k=k)


def create_retriever(
    collection_name: str = "documents",
    persist_directory: str = ".chromadb",
    embedding_model: str = "all-mpnet-base-v2",
) -> LocalRetriever:
    """
    Create a configured retriever instance.

    Args:
        collection_name: ChromaDB collection name.
        persist_directory: ChromaDB persistence directory.
        embedding_model: Embedding model to use.

    Returns:
        Configured LocalRetriever instance.

    Example:
        >>> retriever = create_retriever()
        >>> docs = retriever.retrieve_documents("query")
    """
    config = RetrieverConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
    )
    return LocalRetriever(config=config)
