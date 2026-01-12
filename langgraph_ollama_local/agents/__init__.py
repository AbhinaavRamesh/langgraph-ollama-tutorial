"""
Multi-Agent Patterns Module.

This module provides implementations for multi-agent collaboration patterns
including supervisor-based coordination, hierarchical teams, and agent
communication utilities.

Example:
    >>> from langgraph_ollama_local.agents import create_multi_agent_graph
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>>
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>> graph = create_multi_agent_graph(llm)
    >>> result = graph.invoke({"task": "Build a calculator", ...})

Hierarchical Teams:
    >>> from langgraph_ollama_local.agents import create_team_graph, create_hierarchical_graph
    >>> research_team = create_team_graph(llm, "research", members=[...])
    >>> graph = create_hierarchical_graph(llm, {"research": research_team})
"""

from langgraph_ollama_local.agents.multi_agent import (
    MultiAgentState,
    SupervisorDecision,
    create_agent_node,
    create_multi_agent_graph,
    create_supervisor_node,
    route_supervisor,
    run_multi_agent_task,
    synthesize_node,
)

from langgraph_ollama_local.agents.hierarchical import (
    HierarchicalState,
    TeamState,
    create_hierarchical_graph,
    create_team_graph,
    run_hierarchical_task,
)

__all__ = [
    # Multi-Agent State
    "MultiAgentState",
    "SupervisorDecision",
    # Hierarchical State
    "TeamState",
    "HierarchicalState",
    # Multi-Agent Node creators
    "create_supervisor_node",
    "create_agent_node",
    "synthesize_node",
    # Routing
    "route_supervisor",
    # Multi-Agent Graph builders
    "create_multi_agent_graph",
    # Hierarchical Graph builders
    "create_team_graph",
    "create_hierarchical_graph",
    # Utilities
    "run_multi_agent_task",
    "run_hierarchical_task",
]
