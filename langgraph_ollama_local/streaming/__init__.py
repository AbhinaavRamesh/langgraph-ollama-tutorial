"""
Streaming utilities for local Ollama deployment.

This module contains streaming helpers for real-time agent output:
- Token streaming: Character-by-character output
- Event streaming: Tool calls and intermediate steps
- Content streaming: Final node output
- Subgraph streaming: Nested graph events

Example:
    >>> from langgraph_ollama_local.streaming import stream_tokens
    >>> async for token in stream_tokens(agent, query):
    ...     print(token, end="", flush=True)
"""

__all__: list[str] = []
