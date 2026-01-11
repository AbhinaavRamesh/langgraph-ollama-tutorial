"""
Embedding models for RAG applications.

This module provides local embedding models using sentence-transformers,
keeping everything running locally without external API calls.

Supported models:
    - all-mpnet-base-v2: High quality, 768 dimensions (default)
    - all-MiniLM-L6-v2: Fast, 384 dimensions
    - nomic-embed-text: Ollama native embeddings

Example:
    >>> from langgraph_ollama_local.rag import LocalEmbeddings
    >>> embeddings = LocalEmbeddings()
    >>> vectors = embeddings.embed_documents(["Hello world", "RAG is great"])
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Supported embedding models
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "description": "High quality, best for semantic search",
        "size_mb": 420,
    },
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Fast, good balance of speed and quality",
        "size_mb": 90,
    },
    "paraphrase-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Optimized for paraphrase detection",
        "size_mb": 90,
    },
}

EmbeddingModelName = Literal[
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
]


class EmbeddingConfig(BaseSettings):
    """
    Configuration for embedding models.

    Attributes:
        model_name: The sentence-transformers model to use.
        device: Device to run on ('cpu', 'cuda', 'mps').
        normalize: Whether to normalize embeddings to unit length.
        cache_folder: Directory to cache downloaded models.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence-transformers model name",
    )
    device: str = Field(
        default="cpu",
        description="Device for inference (cpu, cuda, mps)",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings to unit length",
    )
    cache_folder: str | None = Field(
        default=None,
        description="Cache folder for model downloads",
    )


class LocalEmbeddings:
    """
    Local embedding model using sentence-transformers.

    This class provides a LangChain-compatible interface for generating
    embeddings using sentence-transformers models that run locally.

    Attributes:
        model_name: The model being used.
        dimensions: Embedding vector dimensions.

    Example:
        >>> embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
        >>> vector = embeddings.embed_query("What is RAG?")
        >>> print(len(vector))  # 768
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        normalize: bool = True,
        cache_folder: str | None = None,
    ):
        """
        Initialize the local embedding model.

        Args:
            model_name: The sentence-transformers model to use.
            device: Device for inference ('cpu', 'cuda', 'mps').
            normalize: Whether to normalize embeddings.
            cache_folder: Directory to cache models.
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.cache_folder = cache_folder

        # Get model info
        model_info = EMBEDDING_MODELS.get(model_name, {})
        self.dimensions = model_info.get("dimensions", 768)

        # Lazy load the model
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install langgraph-ollama-local[rag]"
                )

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
        )
        return embedding.tolist()

    def __repr__(self) -> str:
        return f"LocalEmbeddings(model_name='{self.model_name}', dimensions={self.dimensions})"


class OllamaEmbeddings:
    """
    Embedding model using Ollama's native embeddings.

    Use this if you want to keep everything within Ollama ecosystem.

    Example:
        >>> embeddings = OllamaEmbeddings(model="nomic-embed-text")
        >>> vector = embeddings.embed_query("What is RAG?")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "127.0.0.1",
        port: int = 11434,
    ):
        """
        Initialize Ollama embeddings.

        Args:
            model: Ollama embedding model name.
            host: Ollama server host.
            port: Ollama server port.
        """
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.dimensions = 768  # nomic-embed-text default

    def _embed(self, text: str) -> list[float]:
        """Make embedding request to Ollama."""
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed(text)


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = "all-mpnet-base-v2",
    use_ollama: bool = False,
) -> "Embeddings":
    """
    Get a cached embedding model instance.

    This function caches embedding models to avoid reloading them.

    Args:
        model_name: The model name to use.
        use_ollama: Use Ollama embeddings instead of sentence-transformers.

    Returns:
        An embedding model instance.

    Example:
        >>> embeddings = get_embedding_model("all-mpnet-base-v2")
        >>> # Same instance returned on subsequent calls
        >>> embeddings2 = get_embedding_model("all-mpnet-base-v2")
        >>> assert embeddings is embeddings2
    """
    if use_ollama:
        return OllamaEmbeddings(model=model_name)
    return LocalEmbeddings(model_name=model_name)


def get_embedding_dimensions(model_name: str) -> int:
    """
    Get the embedding dimensions for a model.

    Args:
        model_name: The model name.

    Returns:
        Number of dimensions in the embedding vector.
    """
    return EMBEDDING_MODELS.get(model_name, {}).get("dimensions", 768)
