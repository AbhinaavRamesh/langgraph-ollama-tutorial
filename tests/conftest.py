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
    """Set up test environment variables."""
    # Use test-specific settings
    os.environ.setdefault("OLLAMA_HOST", "127.0.0.1")
    os.environ.setdefault("OLLAMA_PORT", "11434")
    os.environ.setdefault("OLLAMA_MODEL", "llama3.2:1b")
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

        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2)
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
            # Check if Ollama is available
            try:
                import httpx

                response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2)
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
