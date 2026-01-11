"""
Document indexing pipeline for RAG applications.

This module provides document chunking and indexing into ChromaDB
for efficient similarity search.

Example:
    >>> from langgraph_ollama_local.rag import DocumentIndexer
    >>> indexer = DocumentIndexer()
    >>> indexer.index_directory("sources/")
    >>> # Documents are now searchable in ChromaDB
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langgraph_ollama_local.rag.document_loader import DocumentLoader
from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


class IndexerConfig(BaseSettings):
    """
    Configuration for document indexing.

    Attributes:
        chunk_size: Maximum size of text chunks in characters.
        chunk_overlap: Overlap between consecutive chunks.
        collection_name: ChromaDB collection name.
        persist_directory: Directory for ChromaDB persistence.
        embedding_model: Sentence-transformers model for embeddings.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chunk_size: int = Field(
        default=1000,
        gt=100,
        description="Maximum chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks",
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


class DocumentIndexer:
    """
    Document indexing pipeline using ChromaDB.

    This class handles the complete indexing pipeline:
    1. Load documents from files
    2. Split into chunks with overlap
    3. Generate embeddings
    4. Store in ChromaDB

    Attributes:
        config: Indexer configuration.
        embeddings: Embedding model instance.
        collection: ChromaDB collection.

    Example:
        >>> indexer = DocumentIndexer()
        >>> indexer.index_directory("sources/")
        >>> stats = indexer.get_stats()
        >>> print(f"Indexed {stats['document_count']} chunks")
    """

    def __init__(
        self,
        config: IndexerConfig | None = None,
        embeddings: LocalEmbeddings | None = None,
    ):
        """
        Initialize the document indexer.

        Args:
            config: Indexer configuration. Uses defaults if not provided.
            embeddings: Embedding model. Creates one if not provided.
        """
        self.config = config or IndexerConfig()
        self.embeddings = embeddings or LocalEmbeddings(
            model_name=self.config.embedding_model
        )

        # Lazy-load ChromaDB
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> "chromadb.Client":
        """Get or create the ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError:
                raise ImportError(
                    "chromadb is required for document indexing. "
                    "Install with: pip install langgraph-ollama-local[rag]"
                )

            persist_dir = Path(self.config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB client initialized at {persist_dir}")

        return self._client

    @property
    def collection(self) -> "chromadb.Collection":
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(f"Using collection: {self.config.collection_name}")

        return self._collection

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Split a document into chunks with overlap.

        Args:
            document: Document to split.

        Returns:
            List of chunked documents with updated metadata.
        """
        text = document.page_content
        chunks = []

        # Simple character-based chunking with overlap
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", ".\n", "? ", "!\n", "\n\n"]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.config.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "chunk_start": start,
                    "chunk_end": end,
                })

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )
                chunk_idx += 1

            # Move start with overlap
            start = end - self.config.chunk_overlap
            if start <= chunks[-1].metadata.get("chunk_start", 0) if chunks else 0:
                start = end  # Prevent infinite loop

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of all chunked documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _generate_doc_id(self, document: Document) -> str:
        """Generate a unique ID for a document chunk."""
        content_hash = hashlib.md5(document.page_content.encode()).hexdigest()[:12]
        source = document.metadata.get("source", "unknown")
        chunk_idx = document.metadata.get("chunk_index", 0)
        return f"{Path(source).stem}_{chunk_idx}_{content_hash}"

    def index_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Index documents into ChromaDB.

        Args:
            documents: Documents to index (already chunked).
            batch_size: Number of documents to process at once.

        Returns:
            Number of documents indexed.
        """
        if not documents:
            logger.warning("No documents to index")
            return 0

        total_indexed = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Generate IDs
            ids = [self._generate_doc_id(doc) for doc in batch]

            # Get texts and metadata
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Add to collection (upsert to handle duplicates)
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_indexed += len(batch)
            logger.info(f"Indexed {total_indexed}/{len(documents)} documents")

        return total_indexed

    def index_file(self, file_path: str | Path) -> int:
        """
        Load, chunk, and index a single file.

        Args:
            file_path: Path to the file to index.

        Returns:
            Number of chunks indexed.
        """
        loader = DocumentLoader()
        documents = loader.load_file(file_path)
        chunks = self.chunk_documents(documents)
        return self.index_documents(chunks)

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> int:
        """
        Load, chunk, and index all documents in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.

        Returns:
            Number of chunks indexed.

        Example:
            >>> indexer = DocumentIndexer()
            >>> count = indexer.index_directory("sources/")
            >>> print(f"Indexed {count} chunks")
        """
        loader = DocumentLoader()
        documents = loader.load_directory(directory, recursive=recursive)
        chunks = self.chunk_documents(documents)
        return self.index_documents(chunks)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the indexed documents.

        Returns:
            Dictionary with collection statistics.
        """
        return {
            "collection_name": self.config.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        # Get all IDs and delete them
        results = self.collection.get()
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Cleared {len(results['ids'])} documents from collection")

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source file.

        Args:
            source: The source file path to delete.

        Returns:
            Number of documents deleted.
        """
        results = self.collection.get(
            where={"source": source},
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks from {source}")
            return len(results["ids"])
        return 0


def create_index_from_sources(
    sources_dir: str = "sources",
    collection_name: str = "documents",
    embedding_model: str = "all-mpnet-base-v2",
) -> DocumentIndexer:
    """
    Convenience function to create an index from the sources directory.

    Args:
        sources_dir: Path to the sources directory.
        collection_name: ChromaDB collection name.
        embedding_model: Embedding model to use.

    Returns:
        Configured DocumentIndexer instance.

    Example:
        >>> indexer = create_index_from_sources()
        >>> stats = indexer.get_stats()
    """
    config = IndexerConfig(
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    indexer = DocumentIndexer(config=config)
    indexer.index_directory(sources_dir)
    return indexer
