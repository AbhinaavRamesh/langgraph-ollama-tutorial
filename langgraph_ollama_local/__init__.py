"""
LangGraph Ollama Local - LangGraph tutorials for local Ollama deployment.

This package provides LangGraph patterns and tutorials adapted for local LLM
deployment using Ollama. Learn to build agents that run entirely on your hardware.

For LAN server setup, see the companion package: ollama-local-serve
https://github.com/abhinaavramesh/ollama-local-serve

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>> response = llm.invoke("Hello!")

Multi-Agent Patterns (Phase 4):
    >>> from langgraph_ollama_local.agents import create_multi_agent_graph
    >>> graph = create_multi_agent_graph(llm)
    >>> result = graph.invoke({"task": "Build a calculator", ...})
"""

from langgraph_ollama_local.config import (
    LangGraphConfig,
    LocalAgentConfig,
    OllamaConfig,
    ensure_model,
    list_models,
    pull_model,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "OllamaConfig",
    "LangGraphConfig",
    "LocalAgentConfig",
    # Model management
    "pull_model",
    "list_models",
    "ensure_model",
    # Version
    "__version__",
]
