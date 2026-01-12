"""
Agent Swarm/Network Patterns Module.

This module provides a decentralized agent swarm pattern where agents
communicate peer-to-peer without a central supervisor. Agents can broadcast
messages, share context, and collaborate dynamically based on their connections.

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                     Agent Network                           │
    │                                                             │
    │         ┌─────────┐         ┌─────────┐                   │
    │         │ Agent A │◄───────►│ Agent B │                   │
    │         └────┬────┘         └────┬────┘                   │
    │              │                   │                         │
    │              │    ┌─────────┐   │                         │
    │              └───►│ Agent C │◄──┘                         │
    │                   └────┬────┘                             │
    │                        │                                   │
    │                   ┌────▼────┐                             │
    │                   │ Agent D │                             │
    │                   └─────────┘                             │
    │                                                             │
    │  • No central supervisor                                   │
    │  • Peer-to-peer communication                              │
    │  • Dynamic routing based on agent responses                │
    │  • Shared context across network                           │
    └─────────────────────────────────────────────────────────────┘
    ```

Key Concepts:
- **Decentralized**: No central coordinator, agents decide independently
- **Network Topology**: Define which agents can communicate (fully/partially connected)
- **Shared Context**: Agents build on each other's work via shared state
- **Dynamic Routing**: Next agent is determined by current agent's output
- **Broadcasting**: Messages can be shared with all connected agents

Example:
    >>> from langgraph_ollama_local.patterns.swarm import (
    ...     SwarmAgent,
    ...     create_swarm_graph,
    ... )
    >>>
    >>> # Define swarm agents
    >>> agents = [
    ...     SwarmAgent(
    ...         name="researcher",
    ...         system_prompt="Research and gather information",
    ...         connections=["analyst", "writer"],
    ...     ),
    ...     SwarmAgent(
    ...         name="analyst",
    ...         system_prompt="Analyze findings",
    ...         connections=["writer"],
    ...     ),
    ...     SwarmAgent(
    ...         name="writer",
    ...         system_prompt="Write final report",
    ...         connections=[],  # Terminal node
    ...     ),
    ... ]
    >>>
    >>> graph = create_swarm_graph(llm, agents)
    >>> result = graph.invoke({
    ...     "messages": [],
    ...     "task": "Research quantum computing",
    ...     "agents_state": {},
    ...     "shared_context": [],
    ...     "current_agent": "",
    ...     "iteration": 0,
    ...     "max_iterations": 10,
    ...     "final_result": "",
    ... })
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === State Definition ===

class SwarmState(TypedDict):
    """
    State schema for agent swarm/network.

    This state enables peer-to-peer agent communication with shared context
    and no central supervisor.

    Attributes:
        messages: Conversation history (accumulates via add_messages reducer)
        task: The overall task for the swarm
        agents_state: Per-agent state dict (agent_name -> agent_data)
        shared_context: Accumulated shared findings/outputs from all agents
        current_agent: Name of the currently active agent
        iteration: Current iteration count (for all agents)
        max_iterations: Maximum iterations before forcing completion
        final_result: The synthesized final result
    """

    messages: Annotated[list, add_messages]
    task: str
    agents_state: dict[str, dict[str, Any]]
    shared_context: Annotated[list[dict], operator.add]
    current_agent: str
    iteration: int
    max_iterations: int
    final_result: str


# === Agent Configuration ===

class SwarmAgent(BaseModel):
    """
    Configuration for a swarm agent.

    Each agent has a name, role description, and list of agents it can
    communicate with (connections).

    Attributes:
        name: Unique identifier for the agent
        system_prompt: System prompt defining the agent's role and behavior
        connections: List of agent names this agent can hand off to
        tools: Optional list of tools available to this agent
    """

    name: str = Field(description="Unique name for the agent")
    system_prompt: str = Field(description="System prompt defining agent's role")
    connections: list[str] = Field(
        default_factory=list,
        description="List of agent names this agent can hand off to"
    )
    tools: list | None = Field(
        default=None,
        description="Optional tools for this agent"
    )


# === Structured Output for Routing ===

class SwarmRouting(BaseModel):
    """
    Routing decision from a swarm agent.

    After completing its work, an agent decides:
    1. Which agent should work next (from its connections)
    2. Whether the task is complete
    3. What context to share
    """

    next_agent: str = Field(
        description="Name of the next agent to work, or 'DONE' if task is complete"
    )
    reasoning: str = Field(
        description="Brief explanation for this routing decision"
    )
    share_context: bool = Field(
        default=True,
        description="Whether to share this agent's output with the swarm"
    )


# === Node Creation ===

def create_swarm_node(
    llm: "BaseChatModel",
    agent_config: SwarmAgent,
):
    """
    Create a node for a swarm agent.

    The agent:
    1. Sees the overall task and shared context from other agents
    2. Performs its specialized work
    3. Decides which connected agent should work next
    4. Optionally shares its findings with the swarm

    Args:
        llm: Language model for the agent
        agent_config: Configuration for this agent

    Returns:
        Node function for the swarm agent
    """
    # Try to use structured output for routing
    try:
        structured_llm = llm.with_structured_output(SwarmRouting)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    # Bind tools if provided
    agent_llm = llm.bind_tools(agent_config.tools) if agent_config.tools else llm

    def swarm_agent_node(state: SwarmState) -> dict:
        """Execute the swarm agent's work and routing decision."""
        # Build context from shared findings
        shared_context = state.get("shared_context", [])
        if shared_context:
            context_parts = []
            for ctx in shared_context:
                agent_name = ctx.get("agent", "unknown")
                content = ctx.get("content", "")
                # Include full context (swarm agents need rich context)
                context_parts.append(f"**{agent_name}**: {content}")
            context_str = "\n\n".join(context_parts)
        else:
            context_str = "No shared context yet. You are the first agent."

        # Get this agent's previous state (if any)
        agents_state = state.get("agents_state", {})
        agent_state = agents_state.get(agent_config.name, {})
        previous_work = agent_state.get("last_output", "")

        # Build work prompt
        work_messages = [
            SystemMessage(content=agent_config.system_prompt),
            HumanMessage(content=f"""Task: {state['task']}

Shared Context from Swarm:
{context_str}

Your previous work (if any):
{previous_work if previous_work else "This is your first contribution."}

Provide your contribution to the task."""),
        ]

        # Get agent's work output
        work_response = agent_llm.invoke(work_messages)
        agent_output = work_response.content

        # Build routing prompt
        connections_str = ", ".join(agent_config.connections) if agent_config.connections else "none"
        routing_messages = [
            SystemMessage(content=f"""You are {agent_config.name} in a swarm network.

Your connections: {connections_str}

Based on your work and the task, decide:
1. Which connected agent should work next (or 'DONE' if task is complete)
2. Whether to share your output with the swarm

If you have no connections or the task is complete, respond with 'DONE'."""),
            HumanMessage(content=f"""Your work:
{agent_output}

Task: {state['task']}

Iteration: {state['iteration'] + 1}/{state['max_iterations']}

Which agent should work next?"""),
        ]

        # Get routing decision
        if use_structured:
            routing_decision = structured_llm.invoke(routing_messages)
            next_agent = routing_decision.next_agent
            reasoning = routing_decision.reasoning
            share_context = routing_decision.share_context
        else:
            # Fallback: parse from text
            response = structured_llm.invoke(routing_messages)
            content = response.content.lower()
            if "done" in content or not agent_config.connections:
                next_agent = "DONE"
            else:
                # Try to find a connection name
                next_agent = "DONE"
                for conn in agent_config.connections:
                    if conn.lower() in content:
                        next_agent = conn
                        break
            reasoning = response.content[:200]
            share_context = True

        # Update agent's state
        new_agents_state = agents_state.copy()
        new_agents_state[agent_config.name] = {
            "last_output": agent_output,
            "routing_decision": next_agent,
            "work_count": agent_state.get("work_count", 0) + 1,
        }

        # Prepare updates
        updates: dict[str, Any] = {
            "agents_state": new_agents_state,
            "current_agent": next_agent,
            "iteration": state["iteration"] + 1,
            "messages": [AIMessage(
                content=f"[{agent_config.name}] {agent_output[:300]}... | Next: {next_agent}. {reasoning}"
            )],
        }

        # Add to shared context if requested
        if share_context:
            updates["shared_context"] = [{
                "agent": agent_config.name,
                "content": agent_output,
                "iteration": state["iteration"] + 1,
            }]

        return updates

    return swarm_agent_node


# === Swarm Routing ===

def route_swarm(state: SwarmState) -> str:
    """
    Route to the next agent in the swarm network.

    This reads the current_agent field (set by the previous agent)
    and routes accordingly. Unlike supervisor pattern, there's no
    central decision-maker - each agent decides its successor.

    Args:
        state: Current swarm state

    Returns:
        Name of the next node to execute
    """
    current_agent = state.get("current_agent", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Force completion at max iterations
    if iteration >= max_iterations:
        return "aggregate"

    # Route based on agent's decision
    if current_agent == "DONE" or not current_agent:
        return "aggregate"

    # Route to the named agent
    return current_agent


# === Aggregation ===

def create_aggregate_node():
    """
    Create a node that aggregates all swarm outputs.

    This node combines work from all agents in the swarm into
    a final result, organized by agent and iteration.

    Returns:
        Aggregate node function
    """
    def aggregate_node(state: SwarmState) -> dict:
        """Aggregate all swarm agent outputs."""
        shared_context = state.get("shared_context", [])

        if not shared_context:
            return {
                "final_result": "No work was completed by the swarm.",
                "messages": [AIMessage(content="[System] Swarm completed with no output.")],
            }

        # Group by agent
        by_agent: dict[str, list[dict]] = {}
        for ctx in shared_context:
            agent = ctx.get("agent", "unknown")
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(ctx)

        # Build final result
        result_parts = ["# Agent Swarm Results\n"]

        for agent_name, contexts in by_agent.items():
            result_parts.append(f"## {agent_name.title()}")
            for i, ctx in enumerate(contexts, 1):
                iteration = ctx.get("iteration", 0)
                content = ctx.get("content", "")
                if len(contexts) > 1:
                    result_parts.append(f"### Iteration {iteration}")
                result_parts.append(content)
                result_parts.append("")  # Empty line

        final_result = "\n\n".join(result_parts).strip()

        return {
            "final_result": final_result,
            "messages": [AIMessage(content="[System] Swarm work aggregated.")],
        }

    return aggregate_node


# === Broadcast Utilities ===

def broadcast_message(
    state: SwarmState,
    message: str,
    sender: str,
) -> dict:
    """
    Broadcast a message to all agents in the swarm.

    This utility function adds a message to the shared context
    that all agents can see.

    Args:
        state: Current swarm state
        message: Message to broadcast
        sender: Name of the sending agent

    Returns:
        State update with broadcasted message
    """
    return {
        "shared_context": [{
            "agent": sender,
            "content": message,
            "broadcast": True,
            "iteration": state.get("iteration", 0),
        }],
    }


# === Graph Builder ===

def create_swarm_graph(
    llm: "BaseChatModel",
    agents: list[SwarmAgent],
    entry_agent: str | None = None,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create an agent swarm/network graph.

    This creates a decentralized graph where agents communicate peer-to-peer
    without a central supervisor. Each agent decides which connected agent
    should work next.

    Args:
        llm: Language model for all agents (can be ChatOllama or any LangChain LLM)
        agents: List of SwarmAgent configurations
        entry_agent: Name of the first agent to run (defaults to first in list)
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> agents = [
        ...     SwarmAgent(name="researcher", system_prompt="Research", connections=["analyst"]),
        ...     SwarmAgent(name="analyst", system_prompt="Analyze", connections=["writer"]),
        ...     SwarmAgent(name="writer", system_prompt="Write", connections=[]),
        ... ]
        >>> graph = create_swarm_graph(llm, agents)
    """
    if not agents:
        raise ValueError("Must provide at least one agent")

    # Create agent name -> config mapping
    agent_map = {agent.name: agent for agent in agents}

    # Validate connections
    for agent in agents:
        for conn in agent.connections:
            if conn not in agent_map:
                raise ValueError(
                    f"Agent '{agent.name}' has connection to unknown agent '{conn}'"
                )

    # Determine entry agent
    entry = entry_agent or agents[0].name
    if entry not in agent_map:
        raise ValueError(f"Entry agent '{entry}' not found in agent list")

    workflow = StateGraph(SwarmState)

    # Add agent nodes
    for agent in agents:
        node = create_swarm_node(llm, agent)
        workflow.add_node(agent.name, node)

    # Add aggregate node
    workflow.add_node("aggregate", create_aggregate_node())

    # Entry point: start at entry agent
    workflow.add_edge(START, entry)

    # Build routing map (all agents + aggregate)
    routing_map = {agent.name: agent.name for agent in agents}
    routing_map["aggregate"] = "aggregate"

    # All agents route dynamically via route_swarm
    for agent in agents:
        workflow.add_conditional_edges(
            agent.name,
            route_swarm,
            routing_map,
        )

    # Aggregate leads to end
    workflow.add_edge("aggregate", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_swarm_task(
    graph: "CompiledStateGraph",
    task: str,
    max_iterations: int = 10,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through the agent swarm.

    This is a convenience function that sets up the initial state
    and invokes the graph with proper configuration.

    Args:
        graph: Compiled swarm graph
        task: Task description for the swarm
        max_iterations: Maximum iterations (default: 10)
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - final_result: Aggregated output from all agents
        - shared_context: List of all agent outputs
        - agents_state: Per-agent state information
        - iteration: Final iteration count

    Example:
        >>> result = run_swarm_task(
        ...     graph,
        ...     "Research and write a report on AI safety",
        ...     max_iterations=8
        ... )
        >>> print(result["final_result"])
    """
    initial_state: SwarmState = {
        "messages": [HumanMessage(content=f"Task: {task}")],
        "task": task,
        "agents_state": {},
        "shared_context": [],
        "current_agent": "",  # Will be set by entry edge
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_result": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===

__all__ = [
    # State
    "SwarmState",
    # Configuration
    "SwarmAgent",
    "SwarmRouting",
    # Node creators
    "create_swarm_node",
    "create_aggregate_node",
    # Routing
    "route_swarm",
    # Utilities
    "broadcast_message",
    # Graph builders
    "create_swarm_graph",
    # Convenience
    "run_swarm_task",
]
