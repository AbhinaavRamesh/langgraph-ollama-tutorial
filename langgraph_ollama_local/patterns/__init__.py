"""
Advanced reasoning patterns for local Ollama deployment.

This module contains sophisticated agent patterns including:
- Plan-and-Execute: Two-phase planning then execution
- Reflection: Self-critique and improvement
- Reflexion: Learning from execution failures
- LATS: Language Agent Tree Search
- ReWOO: Reasoning Without Observations
- Subgraphs: Composable agent graphs

Example:
    >>> from langgraph_ollama_local.patterns import create_plan_execute_agent
    >>> agent = create_plan_execute_agent(tools=[...])
    >>> result = agent.invoke({"task": "Build a web scraper"})
"""

__all__: list[str] = []
