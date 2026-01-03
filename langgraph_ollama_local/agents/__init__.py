"""
Agent implementations for local Ollama deployment.

This module contains various agent patterns including:
- ReAct agents (reasoning + acting)
- ReAct with memory
- ReAct with human-in-the-loop
- Multi-agent collaboration
- Hierarchical agent teams

Example:
    >>> from langgraph_ollama_local.agents import create_react_agent_local
    >>> agent = create_react_agent_local(tools=[multiply, add])
    >>> result = agent.invoke({"messages": [("user", "What is 5 * 3?")]})
"""

__all__: list[str] = []
