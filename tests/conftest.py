"""
Pytest configuration and shared fixtures for langgraph-ollama-local tests.

This module provides:
- Configuration fixtures for testing
- Mock Ollama client fixtures
- Sample tools for agent testing
- Integration test markers
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file at import time for integration tests
from dotenv import load_dotenv

load_dotenv()
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage


# =============================================================================
# Test Environment Configuration
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def test_env() -> None:
    """Set up test environment variables.

    Note: OLLAMA_HOST and OLLAMA_PORT are NOT set here to allow .env
    configuration for integration tests against LAN servers.
    """
    # Only set test-specific settings that don't affect connectivity
    os.environ.setdefault("OLLAMA_TIMEOUT", "30")
    os.environ.setdefault("LANGGRAPH_RECURSION_LIMIT", "10")


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def ollama_config() -> "OllamaConfig":
    """Create a test OllamaConfig instance."""
    from langgraph_ollama_local.config import OllamaConfig

    return OllamaConfig(
        host="127.0.0.1",
        port=11434,
        model="llama3.2:1b",
        timeout=30,
        max_retries=1,
        temperature=0.0,
    )


@pytest.fixture
def langgraph_config(temp_checkpoint_dir: Path) -> "LangGraphConfig":
    """Create a test LangGraphConfig instance."""
    from langgraph_ollama_local.config import LangGraphConfig

    return LangGraphConfig(
        recursion_limit=10,
        checkpoint_dir=temp_checkpoint_dir,
        enable_streaming=False,
    )


@pytest.fixture
def local_agent_config(
    ollama_config: "OllamaConfig",
    langgraph_config: "LangGraphConfig",
) -> "LocalAgentConfig":
    """Create a test LocalAgentConfig instance."""
    from langgraph_ollama_local.config import LocalAgentConfig

    return LocalAgentConfig(
        ollama=ollama_config,
        langgraph=langgraph_config,
    )


# =============================================================================
# Mock LLM Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for unit testing."""
    from langchain_core.messages import AIMessage

    mock = MagicMock(spec=["invoke", "ainvoke", "stream", "astream"])

    # Default response
    default_response = AIMessage(content="This is a mock response.")

    mock.invoke.return_value = default_response
    mock.ainvoke = AsyncMock(return_value=default_response)

    # Streaming mock
    def mock_stream(*args: Any, **kwargs: Any) -> list:
        return [AIMessage(content="Mock"), AIMessage(content=" response")]

    mock.stream.return_value = mock_stream()

    return mock


@pytest.fixture
def mock_chat_client() -> MagicMock:
    """Create a mock ChatOllama client."""
    from langchain_core.messages import AIMessage

    mock = MagicMock()

    def invoke_side_effect(messages: list, *args: Any, **kwargs: Any) -> AIMessage:
        """Simulate a simple response based on input."""
        if isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", str(last_msg))
            return AIMessage(content=f"Response to: {content[:50]}")
        return AIMessage(content="Hello!")

    mock.invoke.side_effect = invoke_side_effect
    mock.ainvoke = AsyncMock(side_effect=invoke_side_effect)

    return mock


# =============================================================================
# Sample Tools for Testing
# =============================================================================


@pytest.fixture
def sample_tools() -> list:
    """Create sample tools for agent testing."""
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    @tool
    def get_length(text: str) -> int:
        """Get the length of a text string."""
        return len(text)

    return [add, multiply, get_length]


@pytest.fixture
def math_tools() -> list:
    """Create math-focused tools for agent testing."""
    from langchain_core.tools import tool

    @tool
    def square(x: float) -> float:
        """Calculate the square of a number."""
        return x * x

    @tool
    def cube(x: float) -> float:
        """Calculate the cube of a number."""
        return x * x * x

    @tool
    def sqrt(x: float) -> float:
        """Calculate the square root of a number."""
        import math

        return math.sqrt(x)

    return [square, cube, sqrt]


# =============================================================================
# Integration Test Helpers
# =============================================================================


@pytest.fixture
def ollama_available() -> bool:
    """Check if Ollama is available for integration tests."""
    try:
        import httpx
        from langgraph_ollama_local.config import LocalAgentConfig

        config = LocalAgentConfig()
        response = httpx.get(f"{config.ollama.base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require running Ollama)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify test collection to handle integration tests."""
    # Check if we should skip integration tests
    skip_integration = pytest.mark.skip(
        reason="Ollama not available (run with --run-integration to force)"
    )

    for item in items:
        if "integration" in item.keywords:
            # Check if Ollama is available using configured host
            try:
                import httpx
                from langgraph_ollama_local.config import LocalAgentConfig

                agent_config = LocalAgentConfig()
                response = httpx.get(
                    f"{agent_config.ollama.base_url}/api/tags", timeout=5
                )
                if response.status_code != 200:
                    item.add_marker(skip_integration)
            except Exception:
                item.add_marker(skip_integration)


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def event_loop_policy():
    """Use default event loop policy for async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


# =============================================================================
# RAG Test Fixtures
# =============================================================================


@pytest.fixture
def sample_documents():
    """Create sample documents for RAG testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="RAG combines retrieval with generation for better answers. "
            "It first retrieves relevant documents, then uses them as context.",
            metadata={"source": "rag_intro.pdf", "page": 1, "filename": "rag_intro.pdf"}
        ),
        Document(
            page_content="Self-RAG adds reflection to grade document relevance and detect hallucinations. "
            "It uses the LLM to evaluate its own outputs.",
            metadata={"source": "self_rag.pdf", "page": 3, "filename": "self_rag.pdf"}
        ),
        Document(
            page_content="CRAG uses web search as a fallback mechanism when local retrieval fails. "
            "This ensures comprehensive coverage of topics.",
            metadata={"source": "crag.pdf", "page": 5, "filename": "crag.pdf"}
        ),
        Document(
            page_content="Adaptive RAG routes queries to different strategies based on query type. "
            "It can use vectorstore, web search, or direct LLM responses.",
            metadata={"source": "adaptive_rag.pdf", "page": 2, "filename": "adaptive_rag.pdf"}
        ),
    ]


@pytest.fixture
def temp_chromadb_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for ChromaDB."""
    chromadb_dir = tmp_path / "chromadb"
    chromadb_dir.mkdir()
    return chromadb_dir


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    import numpy as np

    mock = MagicMock()

    def embed_documents(texts: list) -> list:
        # Return random but consistent embeddings
        return [np.random.rand(768).tolist() for _ in texts]

    def embed_query(text: str) -> list:
        return np.random.rand(768).tolist()

    mock.embed_documents = embed_documents
    mock.embed_query = embed_query
    mock.dimensions = 768
    mock.model_name = "mock-embeddings"

    return mock


@pytest.fixture
def indexer_config(temp_chromadb_dir: Path):
    """Create test indexer config."""
    from langgraph_ollama_local.rag.indexer import IndexerConfig

    return IndexerConfig(
        chunk_size=200,
        chunk_overlap=50,
        collection_name="test_collection",
        persist_directory=str(temp_chromadb_dir),
        embedding_model="all-MiniLM-L6-v2",
    )


@pytest.fixture
def retriever_config(temp_chromadb_dir: Path):
    """Create test retriever config."""
    from langgraph_ollama_local.rag.retriever import RetrieverConfig

    return RetrieverConfig(
        collection_name="test_collection",
        persist_directory=str(temp_chromadb_dir),
        default_k=4,
    )
