"""
Agent Handoffs Pattern Module.

This module provides utilities for implementing agent handoff patterns where
agents explicitly transfer control to other agents, enabling peer-to-peer
collaboration without a central supervisor.

Key concepts:
- **Explicit Handoffs**: Agents decide when and to whom to hand off control
- **Peer-to-Peer**: Direct agent-to-agent communication vs supervisor routing
- **Command Pattern**: Using tools to signal handoff intentions
- **Context Preservation**: Handoffs maintain conversation and work context

Architecture:
    ```
    ┌─────────────┐         handoff_to_support         ┌─────────────┐
    │   Sales     │────────────────────────────────────▶│   Support   │
    │   Agent     │                                     │   Agent     │
    └─────────────┘                                     └─────────────┘
          ▲                                                     │
          │                                                     │
          │              handoff_to_billing                     │
          │         ┌─────────────────────────────────────────┘
          │         │
          │         ▼
          │    ┌─────────────┐
          └────│   Billing   │
               │   Agent     │
               └─────────────┘
    ```

Example:
    >>> from langgraph_ollama_local.patterns.handoffs import (
    ...     create_handoff_graph,
    ...     create_handoff_tool,
    ... )
    >>>
    >>> # Create handoff tools
    >>> handoff_to_support = create_handoff_tool(
    ...     target_agent="support",
    ...     description="Transfer to support for technical issues",
    ... )
    >>>
    >>> # Build handoff graph
    >>> graph = create_handoff_graph(
    ...     llm,
    ...     agents={
    ...         "sales": ("Handle sales inquiries", [handoff_to_support]),
    ...         "support": ("Handle technical support", []),
    ...     }
    ... )
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


# === State Definition ===

class HandoffState(TypedDict):
    """
    State schema for agent handoff pattern.

    Attributes:
        messages: Conversation history (accumulates via add_messages reducer)
        task: The original task or query
        current_agent: Currently active agent name
        handoff_target: Agent to hand off to (empty if no handoff)
        context: Shared context accumulating across handoffs
        handoff_history: List of handoff events for tracking
        iteration: Number of handoffs that have occurred
        max_iterations: Maximum allowed handoffs before forcing completion
        final_result: The final response to the user
    """

    messages: Annotated[list, add_messages]
    task: str
    current_agent: str
    handoff_target: str
    context: Annotated[list[dict], operator.add]
    handoff_history: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    final_result: str


# === Handoff Tool Creation ===

def create_handoff_tool(
    target_agent: str,
    description: str | None = None,
) -> BaseTool:
    """
    Create a tool that allows an agent to hand off to another agent.

    This tool enables explicit handoffs where an agent decides to transfer
    control to another agent based on the conversation context.

    Args:
        target_agent: Name of the agent to hand off to
        description: Description of when to use this handoff (for LLM guidance)

    Returns:
        Tool that signals a handoff to the target agent

    Example:
        >>> handoff_to_support = create_handoff_tool(
        ...     target_agent="support",
        ...     description="Transfer to support agent for technical issues or bugs",
        ... )
        >>>
        >>> handoff_to_billing = create_handoff_tool(
        ...     target_agent="billing",
        ...     description="Transfer to billing agent for payment or pricing questions",
        ... )
    """
    tool_description = description or f"Hand off the conversation to the {target_agent} agent"

    @tool(description=tool_description)
    def handoff_tool(
        reason: str = Field(
            description="Brief explanation for why you're handing off to this agent"
        ),
    ) -> str:
        """Execute the handoff to another agent."""
        return f"Handing off to {target_agent} agent. Reason: {reason}"

    # Manually set the name after creation
    handoff_tool.name = f"handoff_to_{target_agent}"

    return handoff_tool


# === Agent Node Creation ===

HANDOFF_AGENT_PROMPT_TEMPLATE = """You are the {agent_name} agent in a collaborative customer service team.

Your role: {agent_role}

IMPORTANT INSTRUCTIONS:
1. If you can fully handle the user's request, provide a complete response and DO NOT use any handoff tools
2. Only use a handoff tool if the request is outside your area of expertise
3. When handing off, explain to the user that you're transferring them and why

Previous work from other agents:
{previous_context}

Available handoff tools: {handoff_tools}

Current user request: Handle this professionally and completely if it's within your role."""


def create_handoff_agent_node(
    llm: BaseChatModel,
    agent_name: str,
    agent_role: str,
    handoff_tools: list[BaseTool] | None = None,
) -> callable:
    """
    Create an agent node that can process work and initiate handoffs.

    The agent can either:
    1. Complete the task and return a final response
    2. Use a handoff tool to transfer to another agent

    Args:
        llm: Language model for the agent
        agent_name: Name of this agent
        agent_role: Description of the agent's role and responsibilities
        handoff_tools: List of handoff tools this agent can use

    Returns:
        Node function for the agent

    Example:
        >>> sales_agent = create_handoff_agent_node(
        ...     llm,
        ...     "sales",
        ...     "Handle sales inquiries, product questions, and pricing",
        ...     handoff_tools=[handoff_to_support, handoff_to_billing],
        ... )
    """
    # Bind tools to LLM if provided
    tools = handoff_tools or []
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def agent_node(state: HandoffState) -> dict:
        """Execute the agent's work and potentially handoff."""
        # Build context from previous agents
        context_items = state.get("context", [])
        if context_items:
            context_parts = []
            for item in context_items:
                agent = item.get("agent", "unknown")
                work = item.get("work", "")
                context_parts.append(f"**{agent}**: {work}")
            previous_context = "\n\n".join(context_parts)
        else:
            previous_context = "This is the first agent handling the request."

        # List available handoff tools
        if tools:
            tool_names = [t.name for t in tools]
            handoff_tools_desc = ", ".join(tool_names)
        else:
            handoff_tools_desc = "None (you must complete the task)"

        # Build prompt
        system_prompt = HANDOFF_AGENT_PROMPT_TEMPLATE.format(
            agent_name=agent_name,
            agent_role=agent_role,
            previous_context=previous_context,
            handoff_tools=handoff_tools_desc,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["task"]),
        ]

        # Invoke LLM
        response = llm_with_tools.invoke(messages)

        # Check if tool was called (handoff)
        handoff_target = ""
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Agent initiated a handoff
            tool_call = response.tool_calls[0]
            # Extract target agent from tool name (format: "handoff_to_<agent>")
            if tool_call["name"].startswith("handoff_to_"):
                handoff_target = tool_call["name"].replace("handoff_to_", "")

        # Record this agent's work
        context_entry = {
            "agent": agent_name,
            "work": response.content,
        }

        updates = {
            "context": [context_entry],
            "messages": [AIMessage(
                content=f"[{agent_name.title()}] {response.content}",
                tool_calls=response.tool_calls if hasattr(response, "tool_calls") else [],
            )],
        }

        # If handoff occurred, record it
        if handoff_target:
            updates["handoff_target"] = handoff_target
            updates["handoff_history"] = [f"{agent_name} -> {handoff_target}"]
        else:
            # No handoff means this agent completed the task
            updates["handoff_target"] = ""

        return updates

    return agent_node


# === Routing ===

def route_handoffs(state: HandoffState) -> str:
    """
    Route based on handoff decisions.

    This function determines the next agent to activate or whether to
    complete the conversation.

    Args:
        state: Current handoff state

    Returns:
        Name of the next node to execute

    Routing logic:
    - If handoff_target is set, route to that agent
    - If no handoff and not at max iterations, go to completion
    - If at max iterations, force completion
    """
    handoff_target = state.get("handoff_target", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Check iteration limit
    if iteration >= max_iterations:
        return "complete"

    # If there's a handoff target, route to it
    if handoff_target:
        return handoff_target

    # No handoff means the current agent completed the task
    return "complete"


def update_current_agent(state: HandoffState) -> dict:
    """
    Update tracking when a handoff occurs.

    This is used as an intermediate node to update state between
    agent handoffs.

    Args:
        state: Current state

    Returns:
        State updates to track the handoff
    """
    handoff_target = state.get("handoff_target", "")

    if handoff_target:
        return {
            "current_agent": handoff_target,
            "iteration": state.get("iteration", 0) + 1,
        }

    return {}


# === Completion Node ===

def create_completion_node() -> callable:
    """
    Create a node that finalizes the conversation.

    This node synthesizes all agent contributions into a final response.

    Returns:
        Completion node function

    Example:
        >>> complete = create_completion_node()
        >>> workflow.add_node("complete", complete)
    """
    def complete_node(state: HandoffState) -> dict:
        """Synthesize final result from all agent work."""
        context_items = state.get("context", [])

        if not context_items:
            return {
                "final_result": "No agent was able to handle the request.",
                "messages": [AIMessage(content="[System] No response generated.")],
            }

        # Build final result from all agent contributions
        parts = []
        for item in context_items:
            agent = item.get("agent", "unknown")
            work = item.get("work", "")
            parts.append(f"**{agent.title()} Agent**:\n{work}")

        final_result = "\n\n".join(parts)

        return {
            "final_result": final_result,
            "messages": [AIMessage(content="[System] Conversation completed.")],
        }

    return complete_node


# === Graph Builder ===

def create_handoff_graph(
    llm: BaseChatModel,
    agents: dict[str, tuple[str, list[BaseTool]]],
    entry_agent: str | None = None,
    max_iterations: int = 10,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """
    Create a graph where agents can hand off control to each other.

    This pattern enables peer-to-peer agent collaboration where agents
    explicitly decide when to transfer control based on their expertise.

    Args:
        llm: Language model for all agents
        agents: Dict mapping agent names to (role_description, handoff_tools)
        entry_agent: Which agent handles requests first (default: first in dict)
        max_iterations: Maximum handoffs allowed before forcing completion
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled handoff graph

    Example:
        >>> # Define handoff tools
        >>> handoff_to_support = create_handoff_tool("support", "Transfer for tech issues")
        >>> handoff_to_billing = create_handoff_tool("billing", "Transfer for payments")
        >>>
        >>> # Create graph
        >>> graph = create_handoff_graph(
        ...     llm,
        ...     agents={
        ...         "sales": ("Handle sales and product questions", [handoff_to_support, handoff_to_billing]),
        ...         "support": ("Handle technical issues", [handoff_to_billing]),
        ...         "billing": ("Handle payments and invoices", []),
        ...     },
        ...     entry_agent="sales",
        ... )
        >>>
        >>> # Run a conversation
        >>> result = graph.invoke({
        ...     "messages": [],
        ...     "task": "I need help with my invoice",
        ...     "current_agent": "sales",
        ...     "handoff_target": "",
        ...     "context": [],
        ...     "handoff_history": [],
        ...     "iteration": 0,
        ...     "max_iterations": 5,
        ...     "final_result": "",
        ... })
    """
    workflow = StateGraph(HandoffState)

    # Determine entry agent
    first_agent = entry_agent or list(agents.keys())[0]

    # Add agent nodes
    for agent_name, (role_desc, handoff_tools) in agents.items():
        agent_node = create_handoff_agent_node(
            llm,
            agent_name,
            role_desc,
            handoff_tools,
        )
        workflow.add_node(agent_name, agent_node)

    # Add completion node
    workflow.add_node("complete", create_completion_node())

    # Entry point goes to first agent
    workflow.add_edge(START, first_agent)

    # Build routing map for conditional edges
    routing_map = {name: name for name in agents.keys()}
    routing_map["complete"] = "complete"

    # Each agent routes based on handoff decision
    for agent_name in agents.keys():
        workflow.add_conditional_edges(
            agent_name,
            route_handoffs,
            routing_map,
        )

    # Completion ends the graph
    workflow.add_edge("complete", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_handoff_conversation(
    graph: CompiledStateGraph,
    task: str,
    entry_agent: str,
    max_iterations: int = 10,
    thread_id: str = "default",
) -> dict:
    """
    Run a conversation through the handoff system.

    Args:
        graph: Compiled handoff graph
        task: User's request or query
        entry_agent: Which agent to start with
        max_iterations: Maximum handoffs allowed
        thread_id: Thread ID for checkpointing

    Returns:
        Final state dict containing:
        - final_result: Complete conversation with all agent contributions
        - handoff_history: List of handoffs that occurred
        - context: All agent work items
        - iteration: Total number of handoffs

    Example:
        >>> result = run_handoff_conversation(
        ...     graph,
        ...     "I'm having trouble logging into my account",
        ...     entry_agent="sales",
        ...     max_iterations=5,
        ... )
        >>> print(result["final_result"])
        >>> print("Handoff chain:", " -> ".join(result["handoff_history"]))
    """
    initial_state: HandoffState = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "current_agent": entry_agent,
        "handoff_target": "",
        "context": [],
        "handoff_history": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_result": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===

__all__ = [
    # State
    "HandoffState",
    # Tool creation
    "create_handoff_tool",
    # Node creators
    "create_handoff_agent_node",
    "create_completion_node",
    # Routing
    "route_handoffs",
    "update_current_agent",
    # Graph builder
    "create_handoff_graph",
    # Utilities
    "run_handoff_conversation",
]
