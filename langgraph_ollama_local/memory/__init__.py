"""
Memory and persistence patterns for local Ollama deployment.

This module contains conversation memory and state persistence utilities:
- Buffer memory: Full conversation history
- Summary memory: LLM-summarized history
- Window memory: Last N messages
- Persistence backends: SQLite, PostgreSQL, Redis

Example:
    >>> from langgraph_ollama_local.memory import create_memory_saver
    >>> checkpointer = create_memory_saver(backend="sqlite")
    >>> agent = create_react_agent_local(tools=[...], checkpointer=checkpointer)
"""

__all__: list[str] = []
