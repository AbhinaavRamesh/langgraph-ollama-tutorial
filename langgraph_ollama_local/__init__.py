"""
LangGraph Ollama Local - Local agent building at scale using Ollama.

This package provides comprehensive LangGraph patterns adapted for local LLM
deployment using Ollama, powered by ollama-local-serve infrastructure.

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig, create_react_agent_local
    >>> config = LocalAgentConfig()
    >>> agent = create_react_agent_local(tools=[...], config=config)
    >>> result = agent.invoke({"messages": [("user", "Hello!")]})
"""

from langgraph_ollama_local.config import (
    LangGraphConfig,
    LocalAgentConfig,
    OllamaConfig,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "OllamaConfig",
    "LangGraphConfig",
    "LocalAgentConfig",
    # Version
    "__version__",
]
