"""
Hierarchical Agent Teams Module.

This module provides patterns for creating nested agent teams where
team supervisors coordinate sub-teams of specialized agents. This
enables complex organizational structures for sophisticated tasks.

Architecture:
    ```
                         ┌─────────────────┐
                         │   Top-Level     │
                         │   Supervisor    │
                         └────────┬────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
       ┌─────────────────┐                     ┌─────────────────┐
       │  Research Team  │                     │  Development    │
       │    Supervisor   │                     │  Team Supervisor│
       └────────┬────────┘                     └────────┬────────┘
                │                                       │
        ┌───────┴───────┐                       ┌───────┴───────┐
        │               │                       │               │
        ▼               ▼                       ▼               ▼
   ┌─────────┐    ┌─────────┐             ┌─────────┐    ┌─────────┐
   │ Web     │    │ Doc     │             │ Frontend│    │ Backend │
   │ Searcher│    │ Analyst │             │ Dev     │    │ Dev     │
   └─────────┘    └─────────┘             └─────────┘    └─────────┘
    ```

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> from langgraph_ollama_local.agents.hierarchical import (
    ...     create_team_graph,
    ...     create_hierarchical_graph,
    ... )
    >>>
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>>
    >>> # Create individual teams
    >>> research_team = create_team_graph(
    ...     llm, "research",
    ...     members=[("searcher", "Search for information", None),
    ...              ("analyst", "Analyze findings", None)]
    ... )
    >>>
    >>> # Create hierarchical structure
    >>> graph = create_hierarchical_graph(llm, {"research": research_team})
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === Team State Definition ===

class TeamState(TypedDict):
    """
    State schema for a single team.

    Attributes:
        messages: Team conversation history
        task: Task assigned to the team
        team_name: Name of this team
        next_member: Which team member should work next
        member_outputs: Accumulated outputs from team members
        iteration: Current iteration within the team
        max_iterations: Maximum iterations for this team
        team_result: Final synthesized result from the team
    """

    messages: Annotated[list, add_messages]
    task: str
    team_name: str
    next_member: str
    member_outputs: Annotated[list[dict], operator.add]
    iteration: int
    max_iterations: int
    team_result: str


class HierarchicalState(TypedDict):
    """
    State schema for hierarchical agent teams.

    Attributes:
        messages: Top-level conversation history
        task: Overall task description
        active_team: Which team is currently working
        team_results: Results from each team (dict: team_name -> result)
        iteration: Top-level iteration count
        max_iterations: Maximum top-level iterations
        final_result: Final synthesized result from all teams
    """

    messages: Annotated[list, add_messages]
    task: str
    active_team: str
    team_results: dict[str, str]
    iteration: int
    max_iterations: int
    final_result: str


# === Structured Outputs ===

class TeamSupervisorDecision(BaseModel):
    """Decision from a team supervisor."""

    next_member: str = Field(
        description="Name of the team member to work next, or 'DONE' if team task is complete"
    )
    reasoning: str = Field(
        description="Brief explanation for this routing decision"
    )


class TopSupervisorDecision(BaseModel):
    """Decision from the top-level supervisor."""

    next_team: str = Field(
        description="Name of the team to activate, or 'FINISH' if task is complete"
    )
    reasoning: str = Field(
        description="Brief explanation for this routing decision"
    )


# === Team Building ===

TEAM_SUPERVISOR_PROMPT = """You are the supervisor of the {team_name} team.

Your team members:
{members_description}

Based on the task and progress, decide which team member should work next.
When your team's part is complete, respond with 'DONE'.

Be efficient - coordinate your team to complete the task quickly."""


def create_team_supervisor_node(
    llm: "BaseChatModel",
    team_name: str,
    member_names: list[str],
):
    """
    Create a team supervisor node.

    Args:
        llm: Language model for the supervisor
        team_name: Name of the team
        member_names: List of team member names

    Returns:
        Team supervisor node function
    """
    members_desc = "\n".join([f"- {name}" for name in member_names])

    # Create dynamic decision model with valid member names
    valid_choices = member_names + ["DONE"]

    try:
        structured_llm = llm.with_structured_output(TeamSupervisorDecision)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def team_supervisor(state: TeamState) -> dict:
        """Team supervisor decides which member works next."""
        outputs = state.get("member_outputs", [])
        if outputs:
            progress = "\n".join([
                f"**{o['member']}**: {o['output'][:300]}..."
                for o in outputs
            ])
        else:
            progress = "No work completed yet."

        messages = [
            SystemMessage(content=TEAM_SUPERVISOR_PROMPT.format(
                team_name=team_name,
                members_description=members_desc,
            )),
            HumanMessage(content=f"""Task: {state['task']}

Progress:
{progress}

Iteration {state['iteration'] + 1}/{state['max_iterations']}

Which team member should work next?"""),
        ]

        if use_structured:
            decision = structured_llm.invoke(messages)
            next_member = decision.next_member
            reasoning = decision.reasoning
        else:
            response = structured_llm.invoke(messages)
            content = response.content.lower()
            if "done" in content:
                next_member = "DONE"
            else:
                # Try to find a member name
                next_member = "DONE"
                for name in member_names:
                    if name.lower() in content:
                        next_member = name
                        break
            reasoning = response.content[:200]

        return {
            "next_member": next_member,
            "iteration": state["iteration"] + 1,
            "messages": [AIMessage(
                content=f"[{team_name} Supervisor] Next: {next_member}. {reasoning}"
            )],
        }

    return team_supervisor


def create_team_member_node(
    llm: "BaseChatModel",
    member_name: str,
    member_prompt: str,
    tools: list | None = None,
):
    """
    Create a team member node.

    Args:
        llm: Language model for the member
        member_name: Name of the team member
        member_prompt: System prompt describing the member's role
        tools: Optional tools for the member

    Returns:
        Team member node function
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def team_member(state: TeamState) -> dict:
        """Execute the team member's task."""
        outputs = state.get("member_outputs", [])
        if outputs:
            context = "\n".join([
                f"**{o['member']}**: {o['output']}"
                for o in outputs
            ])
        else:
            context = "No previous work from team."

        messages = [
            SystemMessage(content=member_prompt),
            HumanMessage(content=f"""Team task: {state['task']}

Previous team work:
{context}

Provide your contribution."""),
        ]

        response = llm_with_tools.invoke(messages)

        return {
            "member_outputs": [{
                "member": member_name,
                "output": response.content,
            }],
            "messages": [AIMessage(
                content=f"[{member_name}] {response.content}"
            )],
        }

    return team_member


def create_team_finalize_node(team_name: str):
    """
    Create a node that finalizes team output.

    Args:
        team_name: Name of the team

    Returns:
        Finalize node function
    """
    def finalize(state: TeamState) -> dict:
        """Combine team member outputs into team result."""
        outputs = state.get("member_outputs", [])

        if not outputs:
            return {"team_result": "No work completed by team."}

        parts = []
        for output in outputs:
            parts.append(f"### {output['member']}\n{output['output']}")

        team_result = f"## {team_name.title()} Team Results\n\n" + "\n\n".join(parts)

        return {
            "team_result": team_result,
            "messages": [AIMessage(content=f"[{team_name} Team] Work complete.")],
        }

    return finalize


def create_team_graph(
    llm: "BaseChatModel",
    team_name: str,
    members: list[tuple[str, str, list | None]],
    max_iterations: int = 5,
) -> "CompiledStateGraph":
    """
    Create a team subgraph with supervisor and members.

    Args:
        llm: Language model for all team agents
        team_name: Name of the team
        members: List of (name, prompt, tools) tuples for each member
        max_iterations: Maximum iterations for the team

    Returns:
        Compiled team graph

    Example:
        >>> research_team = create_team_graph(
        ...     llm,
        ...     "research",
        ...     members=[
        ...         ("searcher", "Search the web for information.", [search_tool]),
        ...         ("analyst", "Analyze and summarize findings.", None),
        ...     ]
        ... )
    """
    member_names = [name for name, _, _ in members]

    workflow = StateGraph(TeamState)

    # Add team supervisor
    workflow.add_node(
        "team_supervisor",
        create_team_supervisor_node(llm, team_name, member_names)
    )

    # Add team members
    for name, prompt, tools in members:
        workflow.add_node(
            name,
            create_team_member_node(llm, name, prompt, tools)
        )

    # Add finalize node
    workflow.add_node("finalize", create_team_finalize_node(team_name))

    # Entry point
    workflow.add_edge(START, "team_supervisor")

    # Routing from team supervisor
    def route_team(state: TeamState) -> str:
        """Route within team."""
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if state["next_member"] == "DONE":
            return "finalize"
        if state["next_member"] in member_names:
            return state["next_member"]
        return "finalize"

    routing_map = {name: name for name in member_names}
    routing_map["finalize"] = "finalize"

    workflow.add_conditional_edges("team_supervisor", route_team, routing_map)

    # Members return to team supervisor
    for name, _, _ in members:
        workflow.add_edge(name, "team_supervisor")

    # Finalize ends the team graph
    workflow.add_edge("finalize", END)

    return workflow.compile()


# === Hierarchical Graph Building ===

TOP_SUPERVISOR_PROMPT = """You are the top-level supervisor coordinating multiple teams.

Available teams:
{teams_description}

Based on the overall task and progress from teams, decide which team should work next.
When the overall task is complete, respond with 'FINISH'.

Coordinate teams efficiently to complete the task."""


def create_top_supervisor_node(
    llm: "BaseChatModel",
    team_names: list[str],
):
    """
    Create the top-level supervisor node.

    Args:
        llm: Language model for the supervisor
        team_names: List of available team names

    Returns:
        Top supervisor node function
    """
    teams_desc = "\n".join([f"- {name} team" for name in team_names])

    try:
        structured_llm = llm.with_structured_output(TopSupervisorDecision)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def top_supervisor(state: HierarchicalState) -> dict:
        """Top supervisor decides which team works next."""
        team_results = state.get("team_results", {})
        if team_results:
            progress = "\n".join([
                f"**{team}**: {result[:300]}..."
                for team, result in team_results.items()
            ])
        else:
            progress = "No team has reported yet."

        messages = [
            SystemMessage(content=TOP_SUPERVISOR_PROMPT.format(
                teams_description=teams_desc,
            )),
            HumanMessage(content=f"""Overall task: {state['task']}

Team progress:
{progress}

Iteration {state['iteration'] + 1}/{state['max_iterations']}

Which team should work next?"""),
        ]

        if use_structured:
            decision = structured_llm.invoke(messages)
            next_team = decision.next_team
            reasoning = decision.reasoning
        else:
            response = structured_llm.invoke(messages)
            content = response.content.lower()
            if "finish" in content:
                next_team = "FINISH"
            else:
                next_team = "FINISH"
                for name in team_names:
                    if name.lower() in content:
                        next_team = name
                        break
            reasoning = response.content[:200]

        return {
            "active_team": next_team,
            "iteration": state["iteration"] + 1,
            "messages": [AIMessage(
                content=f"[Top Supervisor] Next team: {next_team}. {reasoning}"
            )],
        }

    return top_supervisor


def create_team_node_wrapper(
    team_graph: "CompiledStateGraph",
    team_name: str,
):
    """
    Wrap a team graph as a node in the hierarchical graph.

    This handles state transformation between hierarchical and team states.

    Args:
        team_graph: Compiled team graph
        team_name: Name of the team

    Returns:
        Node function that runs the team graph
    """
    def team_node(state: HierarchicalState) -> dict:
        """Run the team graph and collect results."""
        # Transform hierarchical state to team state
        team_input: TeamState = {
            "messages": [],
            "task": state["task"],
            "team_name": team_name,
            "next_member": "",
            "member_outputs": [],
            "iteration": 0,
            "max_iterations": 5,
            "team_result": "",
        }

        # Run the team
        team_output = team_graph.invoke(team_input)

        # Update hierarchical state with team result
        new_results = state.get("team_results", {}).copy()
        new_results[team_name] = team_output.get("team_result", "")

        return {
            "team_results": new_results,
            "messages": [AIMessage(
                content=f"[{team_name} Team] Completed work."
            )],
        }

    return team_node


def create_aggregate_node():
    """
    Create a node that aggregates all team results.

    Returns:
        Aggregate node function
    """
    def aggregate(state: HierarchicalState) -> dict:
        """Combine all team results into final output."""
        team_results = state.get("team_results", {})

        if not team_results:
            return {"final_result": "No teams completed work."}

        parts = []
        for team_name, result in team_results.items():
            parts.append(f"# {team_name.title()} Team\n\n{result}")

        final_result = "\n\n---\n\n".join(parts)

        return {
            "final_result": final_result,
            "messages": [AIMessage(content="[System] All teams complete. Results aggregated.")],
        }

    return aggregate


def create_hierarchical_graph(
    llm: "BaseChatModel",
    teams: dict[str, "CompiledStateGraph"],
    max_iterations: int = 10,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a hierarchical graph with multiple teams.

    Args:
        llm: Language model for the top supervisor
        teams: Dict mapping team names to compiled team graphs
        max_iterations: Maximum top-level iterations
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled hierarchical graph

    Example:
        >>> research_team = create_team_graph(llm, "research", [...])
        >>> dev_team = create_team_graph(llm, "development", [...])
        >>>
        >>> graph = create_hierarchical_graph(
        ...     llm,
        ...     {"research": research_team, "development": dev_team}
        ... )
    """
    team_names = list(teams.keys())

    workflow = StateGraph(HierarchicalState)

    # Add top supervisor
    workflow.add_node(
        "top_supervisor",
        create_top_supervisor_node(llm, team_names)
    )

    # Add team nodes (wrapped subgraphs)
    for team_name, team_graph in teams.items():
        workflow.add_node(
            team_name,
            create_team_node_wrapper(team_graph, team_name)
        )

    # Add aggregate node
    workflow.add_node("aggregate", create_aggregate_node())

    # Entry point
    workflow.add_edge(START, "top_supervisor")

    # Routing from top supervisor
    def route_top(state: HierarchicalState) -> str:
        """Route at top level."""
        if state["iteration"] >= state["max_iterations"]:
            return "aggregate"
        if state["active_team"] == "FINISH":
            return "aggregate"
        if state["active_team"] in team_names:
            return state["active_team"]
        return "aggregate"

    routing_map = {name: name for name in team_names}
    routing_map["aggregate"] = "aggregate"

    workflow.add_conditional_edges("top_supervisor", route_top, routing_map)

    # Teams return to top supervisor
    for team_name in team_names:
        workflow.add_edge(team_name, "top_supervisor")

    # Aggregate ends the graph
    workflow.add_edge("aggregate", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_hierarchical_task(
    graph: "CompiledStateGraph",
    task: str,
    max_iterations: int = 10,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through the hierarchical agent system.

    Args:
        graph: Compiled hierarchical graph
        task: Task description
        max_iterations: Maximum top-level iterations
        thread_id: Thread ID for checkpointing

    Returns:
        Final state dict with results
    """
    initial_state: HierarchicalState = {
        "messages": [HumanMessage(content=f"Task: {task}")],
        "task": task,
        "active_team": "",
        "team_results": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_result": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===

__all__ = [
    # State types
    "TeamState",
    "HierarchicalState",
    # Decision types
    "TeamSupervisorDecision",
    "TopSupervisorDecision",
    # Team building
    "create_team_supervisor_node",
    "create_team_member_node",
    "create_team_finalize_node",
    "create_team_graph",
    # Hierarchical building
    "create_top_supervisor_node",
    "create_team_node_wrapper",
    "create_aggregate_node",
    "create_hierarchical_graph",
    # Utilities
    "run_hierarchical_task",
]
