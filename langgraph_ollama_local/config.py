"""
Configuration management for LangGraph Ollama Local.

This module provides a unified configuration system that wraps ollama-local-serve's
NetworkConfig with additional LangGraph-specific settings. All settings can be
loaded from environment variables and .env files.

Configuration Hierarchy (highest to lowest priority):
    1. Programmatic configuration (explicit arguments)
    2. Environment variables
    3. .env file
    4. Default values

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>> agent = create_react_agent(llm, tools)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)


class OllamaConfig(BaseSettings):
    """
    Ollama server connection configuration.

    This configuration is used to connect to a local or remote Ollama server.
    All settings support environment variables with the OLLAMA_ prefix.

    Attributes:
        host: Ollama server hostname. Use IP for LAN access (e.g., '192.168.1.100').
        port: Ollama server port. Default is 11434.
        model: Default model to use for agents. Smaller models like 'llama3.2:1b'
               are recommended for quick iteration.
        timeout: Request timeout in seconds. Increase for slower hardware.
        max_retries: Maximum retry attempts for failed requests.
        temperature: Default temperature for model responses (0.0-2.0).
        num_ctx: Context window size. Larger values use more memory.

    Example:
        >>> config = OllamaConfig(host="192.168.1.100", model="llama3.2:7b")
        >>> print(config.base_url)
        'http://192.168.1.100:11434'
    """

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Ollama server hostname or IP address",
    )
    port: int = Field(
        default=11434,
        ge=1,
        le=65535,
        description="Ollama server port",
    )
    model: str = Field(
        default="llama3.1:8b",
        description="Default model for agents",
    )
    timeout: int = Field(
        default=120,
        gt=0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default temperature for responses",
    )
    num_ctx: int = Field(
        default=4096,
        gt=0,
        description="Context window size",
    )

    @computed_field
    @property
    def base_url(self) -> str:
        """Get the base URL for the Ollama service."""
        return f"http://{self.host}:{self.port}"


class LangGraphConfig(BaseSettings):
    """
    LangGraph-specific configuration settings.

    These settings control the behavior of LangGraph agents and graphs.

    Attributes:
        recursion_limit: Maximum recursion depth for agent loops. Prevents infinite loops.
        checkpoint_dir: Directory for storing agent state checkpoints.
        enable_streaming: Whether to enable token streaming by default.
        thread_id_prefix: Prefix for auto-generated thread IDs.

    Example:
        >>> config = LangGraphConfig(recursion_limit=50)
        >>> graph = create_graph(..., config={"recursion_limit": config.recursion_limit})
    """

    model_config = SettingsConfigDict(
        env_prefix="LANGGRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    recursion_limit: int = Field(
        default=25,
        gt=0,
        le=100,
        description="Maximum recursion depth for agent loops",
    )
    checkpoint_dir: Path = Field(
        default=Path(".checkpoints"),
        description="Directory for state checkpoints",
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable token streaming by default",
    )
    thread_id_prefix: str = Field(
        default="thread",
        description="Prefix for auto-generated thread IDs",
    )

    @model_validator(mode="after")
    def ensure_checkpoint_dir(self) -> "LangGraphConfig":
        """Ensure the checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self


class LocalAgentConfig(BaseSettings):
    """
    Combined configuration for local agent development.

    This is the main configuration class that combines Ollama connection settings
    with LangGraph-specific settings. Use this class for most use cases.

    Attributes:
        ollama: Ollama server connection settings.
        langgraph: LangGraph behavior settings.

    Example:
        >>> from langgraph_ollama_local import LocalAgentConfig
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def add(a: int, b: int) -> int:
        ...     \"\"\"Add two numbers.\"\"\"
        ...     return a + b
        >>>
        >>> config = LocalAgentConfig()
        >>> llm = config.create_chat_client()
        >>> # Use with LangGraph
        >>> from langgraph.prebuilt import create_react_agent
        >>> agent = create_react_agent(llm, [add])
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)

    def create_chat_client(
        self,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> "BaseChatModel":
        """
        Create a LangChain chat client configured for this setup.

        Args:
            model: Override the default model. If None, uses config.ollama.model.
            temperature: Override the default temperature. If None, uses config.ollama.temperature.
            **kwargs: Additional arguments passed to ChatOllama.

        Returns:
            A configured ChatOllama instance ready for use with LangGraph.

        Example:
            >>> config = LocalAgentConfig()
            >>> llm = config.create_chat_client(model="llama3.2:7b")
            >>> response = llm.invoke("Hello!")
        """
        from langchain_ollama import ChatOllama

        effective_model = model or self.ollama.model
        effective_temp = temperature if temperature is not None else self.ollama.temperature

        logger.info(
            f"Creating chat client: {self.ollama.base_url} "
            f"model={effective_model} temp={effective_temp}"
        )

        return ChatOllama(
            base_url=self.ollama.base_url,
            model=effective_model,
            temperature=effective_temp,
            num_ctx=self.ollama.num_ctx,
            **kwargs,
        )

    def create_checkpointer(
        self,
        backend: Literal["memory", "sqlite", "postgres", "redis"] = "memory",
        **kwargs: Any,
    ) -> "BaseCheckpointSaver":
        """
        Create a checkpoint saver for agent state persistence.

        Args:
            backend: The persistence backend to use.
                - 'memory': In-memory (default, lost on restart)
                - 'sqlite': SQLite file-based
                - 'postgres': PostgreSQL database
                - 'redis': Redis key-value store
            **kwargs: Backend-specific configuration.

        Returns:
            A configured checkpoint saver.

        Raises:
            ImportError: If the required backend package is not installed.
            ValueError: If an unknown backend is specified.

        Example:
            >>> config = LocalAgentConfig()
            >>> checkpointer = config.create_checkpointer(backend="sqlite")
            >>> agent = create_react_agent(llm, tools, checkpointer=checkpointer)
        """
        if backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()

        elif backend == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError:
                raise ImportError(
                    "SQLite checkpointer requires aiosqlite. "
                    "Install with: pip install langgraph-ollama-local[persistence]"
                )
            db_path = kwargs.get("db_path", self.langgraph.checkpoint_dir / "checkpoints.db")
            return SqliteSaver.from_conn_string(str(db_path))

        elif backend == "postgres":
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
            except ImportError:
                raise ImportError(
                    "PostgreSQL checkpointer requires asyncpg. "
                    "Install with: pip install langgraph-ollama-local[persistence]"
                )
            conn_string = kwargs.get("conn_string")
            if not conn_string:
                raise ValueError("PostgreSQL backend requires 'conn_string' argument")
            return PostgresSaver.from_conn_string(conn_string)

        elif backend == "redis":
            try:
                from langgraph.checkpoint.redis import RedisSaver
            except ImportError:
                raise ImportError(
                    "Redis checkpointer requires redis. "
                    "Install with: pip install langgraph-ollama-local[persistence]"
                )
            url = kwargs.get("url", "redis://localhost:6379")
            return RedisSaver.from_conn_string(url)

        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Supported: memory, sqlite, postgres, redis"
            )

    def get_graph_config(self) -> dict[str, Any]:
        """
        Get configuration dictionary for LangGraph graph execution.

        Returns:
            Dictionary with recursion_limit and other graph settings.

        Example:
            >>> config = LocalAgentConfig()
            >>> graph_config = config.get_graph_config()
            >>> result = graph.invoke(input, config=graph_config)
        """
        return {
            "recursion_limit": self.langgraph.recursion_limit,
        }


# Convenience functions


def pull_model(
    model: str,
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: int = 600,
) -> bool:
    """
    Pull a model from Ollama registry.

    This function pulls a model to the Ollama server, which is required
    before using the model in tutorials.

    Args:
        model: The model name to pull (e.g., 'llama3.2:3b', 'llama3.1:8b').
        host: The Ollama server host.
        port: The Ollama server port.
        timeout: Timeout in seconds for the pull operation.

    Returns:
        True if the model was pulled successfully, False otherwise.

    Example:
        >>> from langgraph_ollama_local import pull_model
        >>> pull_model("llama3.2:3b")
        Pulling model: llama3.2:3b...
        Model llama3.2:3b pulled successfully!
        True
    """
    import httpx

    url = f"http://{host}:{port}/api/pull"
    print(f"Pulling model: {model}...")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                json={"name": model},
            )
            response.raise_for_status()
            print(f"Model {model} pulled successfully!")
            return True
    except httpx.TimeoutException:
        print(f"Timeout pulling model {model}. Try increasing timeout.")
        return False
    except httpx.HTTPStatusError as e:
        print(f"Error pulling model {model}: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"Error pulling model {model}: {e}")
        return False


def list_models(
    host: str = "127.0.0.1",
    port: int = 11434,
) -> list[str]:
    """
    List available models on the Ollama server.

    Args:
        host: The Ollama server host.
        port: The Ollama server port.

    Returns:
        List of model names available on the server.

    Example:
        >>> from langgraph_ollama_local import list_models
        >>> list_models()
        ['llama3.2:3b', 'mistral:7b']
    """
    import httpx

    url = f"http://{host}:{port}/api/tags"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def ensure_model(
    model: str,
    host: str = "127.0.0.1",
    port: int = 11434,
) -> bool:
    """
    Ensure a model is available, pulling it if necessary.

    This is useful at the start of tutorials to make sure the required
    model is available before running.

    Args:
        model: The model name to ensure is available.
        host: The Ollama server host.
        port: The Ollama server port.

    Returns:
        True if the model is available (was already present or pulled).

    Example:
        >>> from langgraph_ollama_local import ensure_model
        >>> ensure_model("llama3.2:3b")
        Model llama3.2:3b is already available.
        True
    """
    available = list_models(host=host, port=port)
    if model in available:
        print(f"Model {model} is already available.")
        return True

    # Try without tag if not found
    base_model = model.split(":")[0]
    matching = [m for m in available if m.startswith(base_model)]
    if matching:
        print(f"Model {model} not found, but found: {matching}")
        return True

    # Pull the model
    return pull_model(model, host=host, port=port)


def get_default_config() -> LocalAgentConfig:
    """
    Get the default LocalAgentConfig instance.

    This function creates a new config instance with settings loaded from
    environment variables and .env file.

    Returns:
        LocalAgentConfig with default/environment settings.
    """
    return LocalAgentConfig()


def create_quick_client(
    model: str = "llama3.2:3b",
    host: str = "127.0.0.1",
    port: int = 11434,
) -> "BaseChatModel":
    """
    Quickly create a chat client with minimal configuration.

    This is a convenience function for quick prototyping when you don't
    need the full configuration system.

    Args:
        model: The Ollama model to use.
        host: The Ollama server host.
        port: The Ollama server port.

    Returns:
        A configured ChatOllama instance.

    Example:
        >>> llm = create_quick_client(model="llama3.2:1b")
        >>> response = llm.invoke("Hello!")
    """
    config = LocalAgentConfig(
        ollama=OllamaConfig(host=host, port=port, model=model)
    )
    return config.create_chat_client()
