"""
Tests for the configuration module.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langgraph_ollama_local.config import (
    LangGraphConfig,
    LocalAgentConfig,
    OllamaConfig,
    create_quick_client,
    get_default_config,
)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # Clear env vars to test true defaults
        with patch.dict(os.environ, {}, clear=True):
            config = OllamaConfig(_env_file=None)
            assert config.host == "127.0.0.1"
            assert config.port == 11434
            assert config.model == "llama3.2:3b"
            assert config.timeout == 120
            assert config.max_retries == 3
            assert config.temperature == 0.0
            assert config.num_ctx == 4096

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OllamaConfig(
            host="192.168.1.100",
            port=11435,
            model="llama3.2:7b",
            timeout=60,
            max_retries=5,
            temperature=0.7,
            num_ctx=8192,
        )
        assert config.host == "192.168.1.100"
        assert config.port == 11435
        assert config.model == "llama3.2:7b"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.temperature == 0.7
        assert config.num_ctx == 8192

    def test_base_url_computed_field(self) -> None:
        """Test base_url computed field."""
        config = OllamaConfig(host="192.168.1.100", port=11434)
        assert config.base_url == "http://192.168.1.100:11434"

    def test_to_network_config(self) -> None:
        """Test conversion to NetworkConfig."""
        config = OllamaConfig(
            host="192.168.1.100",
            port=11435,
            timeout=60,
            max_retries=5,
        )
        network_config = config.to_network_config()
        assert network_config.host == "192.168.1.100"
        assert network_config.port == 11435
        assert network_config.timeout == 60
        assert network_config.max_retries == 5

    def test_environment_variable_loading(self) -> None:
        """Test loading from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OLLAMA_HOST": "env-host",
                "OLLAMA_PORT": "12345",
                "OLLAMA_MODEL": "env-model",
            },
        ):
            config = OllamaConfig()
            assert config.host == "env-host"
            assert config.port == 12345
            assert config.model == "env-model"

    def test_port_validation(self) -> None:
        """Test port number validation."""
        with pytest.raises(ValueError):
            OllamaConfig(port=0)
        with pytest.raises(ValueError):
            OllamaConfig(port=70000)

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValueError):
            OllamaConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            OllamaConfig(temperature=2.1)


class TestLangGraphConfig:
    """Tests for LangGraphConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        # Clear env vars to test true defaults
        with patch.dict(os.environ, {}, clear=True):
            config = LangGraphConfig(_env_file=None)
            assert config.recursion_limit == 25
            assert config.enable_streaming is True
            assert config.thread_id_prefix == "thread"

    def test_checkpoint_dir_creation(self, tmp_path: Path) -> None:
        """Test that checkpoint directory is created."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        config = LangGraphConfig(checkpoint_dir=checkpoint_dir)
        assert config.checkpoint_dir.exists()

    def test_recursion_limit_validation(self) -> None:
        """Test recursion limit validation."""
        with pytest.raises(ValueError):
            LangGraphConfig(recursion_limit=0)
        with pytest.raises(ValueError):
            LangGraphConfig(recursion_limit=101)


class TestLocalAgentConfig:
    """Tests for LocalAgentConfig."""

    def test_nested_config_creation(self) -> None:
        """Test nested configuration creation."""
        config = LocalAgentConfig()
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.langgraph, LangGraphConfig)

    def test_custom_nested_configs(self) -> None:
        """Test custom nested configurations."""
        ollama = OllamaConfig(host="custom-host")
        langgraph = LangGraphConfig(recursion_limit=50)
        config = LocalAgentConfig(ollama=ollama, langgraph=langgraph)
        assert config.ollama.host == "custom-host"
        assert config.langgraph.recursion_limit == 50

    def test_get_graph_config(self) -> None:
        """Test graph configuration generation."""
        config = LocalAgentConfig(
            langgraph=LangGraphConfig(recursion_limit=30)
        )
        graph_config = config.get_graph_config()
        assert graph_config["recursion_limit"] == 30

    @patch("langgraph_ollama_local.config.create_langchain_chat_client")
    def test_create_chat_client(self, mock_create: MagicMock) -> None:
        """Test chat client creation."""
        mock_create.return_value = MagicMock()
        config = LocalAgentConfig()
        client = config.create_chat_client()

        mock_create.assert_called_once()
        assert client is not None

    @patch("langgraph_ollama_local.config.create_langchain_chat_client")
    def test_create_chat_client_with_overrides(self, mock_create: MagicMock) -> None:
        """Test chat client creation with parameter overrides."""
        mock_create.return_value = MagicMock()
        config = LocalAgentConfig()
        config.create_chat_client(model="llama3.2:7b", temperature=0.5)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "llama3.2:7b"
        assert call_kwargs["temperature"] == 0.5

    def test_create_checkpointer_memory(self) -> None:
        """Test memory checkpointer creation."""
        config = LocalAgentConfig()
        checkpointer = config.create_checkpointer(backend="memory")
        assert checkpointer is not None

    def test_create_checkpointer_unknown_backend(self) -> None:
        """Test error on unknown backend."""
        config = LocalAgentConfig()
        with pytest.raises(ValueError, match="Unknown backend"):
            config.create_checkpointer(backend="unknown")  # type: ignore


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_default_config(self) -> None:
        """Test get_default_config function."""
        config = get_default_config()
        assert isinstance(config, LocalAgentConfig)

    @patch("langgraph_ollama_local.config.create_langchain_chat_client")
    def test_create_quick_client(self, mock_create: MagicMock) -> None:
        """Test create_quick_client function."""
        mock_create.return_value = MagicMock()
        client = create_quick_client(
            model="llama3.2:1b",
            host="192.168.1.100",
            port=11435,
        )
        assert client is not None
        mock_create.assert_called_once()
