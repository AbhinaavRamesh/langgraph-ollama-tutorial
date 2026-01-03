"""
Integration tests for Ollama connection.

These tests require a running Ollama instance and are skipped if unavailable.
"""

from __future__ import annotations

import pytest

from langgraph_ollama_local.config import LocalAgentConfig


@pytest.mark.integration
class TestOllamaConnection:
    """Integration tests for Ollama connectivity."""

    def test_connection_check(self, ollama_available: bool) -> None:
        """Test basic connection to Ollama."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        import httpx

        config = LocalAgentConfig()
        response = httpx.get(f"{config.ollama.base_url}/api/tags", timeout=5)
        assert response.status_code == 200

    def test_create_chat_client(self, ollama_available: bool) -> None:
        """Test creating a chat client that can connect."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        client = config.create_chat_client()
        assert client is not None

    @pytest.mark.slow
    def test_simple_inference(self, ollama_available: bool) -> None:
        """Test a simple inference call."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        # Use the configured model (from .env or defaults)
        client = config.create_chat_client()

        response = client.invoke("Say 'hello' and nothing else.")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
