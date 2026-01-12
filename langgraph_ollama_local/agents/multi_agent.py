"""
Multi-Agent Collaboration Module.

This module provides the supervisor pattern for coordinating multiple
specialized agents working on complex tasks. The supervisor agent
routes work to specialized agents (researcher, coder, reviewer) and
synthesizes their outputs into a final result.

Architecture:
    ```
                         ┌─────────────────┐
                         │   Supervisor    │
                         │     Agent       │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
       │  Researcher │     │   Coder     │     │  Reviewer   │
       │    Agent    │     │   Agent     │     │   Agent     │
       └─────────────┘     └─────────────┘     └─────────────┘
              │                   │                   │
              └───────────────────┴───────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Synthesize    │
                         └─────────────────┘
    ```

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> from langgraph_ollama_local.agents import create_multi_agent_graph
    >>>
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>> graph = create_multi_agent_graph(llm)
    >>>
    >>> result = graph.invoke({
    ...     "messages": [],
    ...     "task": "Create a Python function to calculate fibonacci numbers",
    ...     "next_agent": "",
    ...     "agent_outputs": [],
    ...     "iteration": 0,
    ...     "max_iterations": 10,
    ...     "final_result": "",
    ... })
    >>> print(result["final_result"])
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === Constants ===

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a team of specialized agents to complete a task.

Your team consists of:
- **researcher**: Gathers information, analyzes requirements, searches for solutions and best practices
- **coder**: Writes code, implements solutions, handles technical implementation tasks
- **reviewer**: Reviews work quality, checks for issues, provides constructive feedback

Your job is to coordinate the team by deciding which agent should work next based on:
1. The current task requirements
2. What work has been completed so far
3. What still needs to be done

**Decision Guidelines:**
- Start with 'researcher' if the task needs analysis or information gathering
- Use 'coder' when requirements are clear and implementation is needed
- Use 'reviewer' after code is written to check quality
- Respond with 'FINISH' when the task is complete and all necessary work is done

Be efficient - don't over-iterate. Most tasks need 2-4 agent interactions."""

SUPERVISOR_HUMAN_PROMPT = """Current task: {task}

Work completed so far:
{progress}

Iteration: {iteration}/{max_iterations}

Based on the task and progress, which agent should work next?
If the task is complete, respond with FINISH."""

AGENT_SYSTEM_PROMPTS = {
    "researcher": """You are a research agent on a collaborative team.

Your responsibilities:
- Analyze task requirements and break them down
- Research best practices and solutions
- Gather relevant information
- Identify potential challenges and considerations

Provide clear, structured analysis that helps the team understand what needs to be built.
Be concise but thorough. Focus on actionable insights.""",

    "coder": """You are a coding agent on a collaborative team.

Your responsibilities:
- Write clean, well-documented code
- Implement solutions based on research and requirements
- Follow best practices and coding standards
- Include helpful comments explaining the code

Write production-quality code. If you need to make assumptions, state them clearly.
Include example usage where appropriate.""",

    "reviewer": """You are a review agent on a collaborative team.

Your responsibilities:
- Review code for correctness, efficiency, and style
- Check if requirements are met
- Identify bugs, edge cases, or improvements
- Provide constructive, actionable feedback

Be specific in your feedback. Point out both strengths and areas for improvement.
If everything looks good, say so clearly.""",
}


# === State Definition ===

class MultiAgentState(TypedDict):
    """
    State schema for multi-agent collaboration.

    Attributes:
        messages: Conversation history (accumulates via add_messages reducer)
        task: The current task description
        next_agent: Which agent should run next (set by supervisor)
        agent_outputs: Accumulated outputs from all agents (list of dicts)
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations before forcing completion
        final_result: The synthesized final result
    """

    messages: Annotated[list, add_messages]
    task: str
    next_agent: str
    agent_outputs: Annotated[list[dict], operator.add]
    iteration: int
    max_iterations: int
    final_result: str


# === Structured Output ===

class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decisions."""

    next_agent: Literal["researcher", "coder", "reviewer", "FINISH"] = Field(
        description="The next agent to work on the task, or FINISH if complete"
    )
    reasoning: str = Field(
        description="Brief explanation for this routing decision (1-2 sentences)"
    )


# === Node Functions ===

def create_supervisor_node(llm: "BaseChatModel"):
    """
    Create a supervisor node that routes work to specialized agents.

    The supervisor uses structured output to decide which agent should
    work next, based on the current task and progress made so far.

    Args:
        llm: Language model for the supervisor

    Returns:
        Node function for the supervisor
    """
    # Try to use structured output, fall back to parsing if not supported
    try:
        structured_llm = llm.with_structured_output(SupervisorDecision)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def supervisor_node(state: MultiAgentState) -> dict:
        """Supervisor decides which agent should act next."""
        # Build progress summary from agent outputs
        agent_outputs = state.get("agent_outputs", [])
        if agent_outputs:
            progress_parts = []
            for output in agent_outputs:
                agent_name = output.get("agent", "unknown")
                content = output.get("output", "")
                # Truncate long outputs for the supervisor prompt
                if len(content) > 500:
                    content = content[:500] + "..."
                progress_parts.append(f"**{agent_name}**:\n{content}")
            progress = "\n\n".join(progress_parts)
        else:
            progress = "No work completed yet."

        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)

        # Build the prompt
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=SUPERVISOR_HUMAN_PROMPT.format(
                task=state["task"],
                progress=progress,
                iteration=iteration + 1,
                max_iterations=max_iterations,
            )),
        ]

        if use_structured:
            decision = structured_llm.invoke(messages)
            next_agent = decision.next_agent
            reasoning = decision.reasoning
        else:
            # Fallback: parse from text response
            response = structured_llm.invoke(messages)
            content = response.content.lower()
            if "finish" in content:
                next_agent = "FINISH"
            elif "researcher" in content:
                next_agent = "researcher"
            elif "coder" in content:
                next_agent = "coder"
            elif "reviewer" in content:
                next_agent = "reviewer"
            else:
                next_agent = "FINISH"  # Default to finish if unclear
            reasoning = response.content[:200]

        return {
            "next_agent": next_agent,
            "iteration": iteration + 1,
            "messages": [AIMessage(
                content=f"[Supervisor] Next: {next_agent}. Reason: {reasoning}"
            )],
        }

    return supervisor_node


def create_agent_node(
    llm: "BaseChatModel",
    agent_name: str,
    tools: list | None = None,
):
    """
    Create an agent node for a specialized role.

    Args:
        llm: Language model for the agent
        agent_name: Name of the agent (researcher, coder, reviewer)
        tools: Optional list of tools for the agent

    Returns:
        Node function for the agent
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    system_prompt = AGENT_SYSTEM_PROMPTS.get(
        agent_name,
        "You are a helpful assistant working on a collaborative team."
    )

    def agent_node(state: MultiAgentState) -> dict:
        """Execute the agent's task."""
        # Build context from previous work
        agent_outputs = state.get("agent_outputs", [])
        if agent_outputs:
            context_parts = []
            for output in agent_outputs:
                context_parts.append(
                    f"**{output['agent']}**: {output['output']}"
                )
            previous_work = "\n\n".join(context_parts)
        else:
            previous_work = "No previous work."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Task: {state['task']}

Previous work from the team:
{previous_work}

Now it's your turn. Provide your contribution to help complete this task."""),
        ]

        response = llm_with_tools.invoke(messages)

        return {
            "agent_outputs": [{
                "agent": agent_name,
                "output": response.content,
            }],
            "messages": [AIMessage(
                content=f"[{agent_name.title()}] {response.content}"
            )],
        }

    return agent_node


def synthesize_node(state: MultiAgentState) -> dict:
    """
    Synthesize all agent outputs into a final result.

    This node combines the work from all agents into a coherent
    final output, organizing by agent role.

    Args:
        state: Current multi-agent state

    Returns:
        State update with final_result
    """
    agent_outputs = state.get("agent_outputs", [])

    if not agent_outputs:
        return {"final_result": "No work was completed."}

    # Group outputs by agent
    by_agent: dict[str, list[str]] = {}
    for output in agent_outputs:
        agent = output.get("agent", "unknown")
        content = output.get("output", "")
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(content)

    # Build final result
    result_parts = []
    for agent_name in ["researcher", "coder", "reviewer"]:
        if agent_name in by_agent:
            outputs = by_agent[agent_name]
            result_parts.append(f"## {agent_name.title()} Output")
            for i, output in enumerate(outputs, 1):
                if len(outputs) > 1:
                    result_parts.append(f"### Iteration {i}")
                result_parts.append(output)
            result_parts.append("")  # Empty line between sections

    # Include any other agents not in the standard list
    for agent_name, outputs in by_agent.items():
        if agent_name not in ["researcher", "coder", "reviewer"]:
            result_parts.append(f"## {agent_name.title()} Output")
            for output in outputs:
                result_parts.append(output)
            result_parts.append("")

    final_result = "\n\n".join(result_parts).strip()

    return {
        "final_result": final_result,
        "messages": [AIMessage(content="[System] Task completed. Results synthesized.")],
    }


# === Routing ===

def route_supervisor(state: MultiAgentState) -> str:
    """
    Route based on supervisor's decision.

    Args:
        state: Current state with next_agent set by supervisor

    Returns:
        Name of the next node to execute
    """
    next_agent = state.get("next_agent", "")
    max_iterations = state.get("max_iterations", 10)
    iteration = state.get("iteration", 0)

    # Force completion if at max iterations
    if iteration >= max_iterations:
        return "synthesize"

    # Route based on supervisor decision
    if next_agent == "FINISH":
        return "synthesize"

    if next_agent.lower() in ["researcher", "coder", "reviewer"]:
        return next_agent.lower()

    # Default to synthesize if unknown
    return "synthesize"


# === Graph Builder ===

def create_multi_agent_graph(
    llm: "BaseChatModel",
    researcher_tools: list | None = None,
    coder_tools: list | None = None,
    reviewer_tools: list | None = None,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a multi-agent collaboration graph.

    This creates a graph with a supervisor that coordinates three
    specialized agents: researcher, coder, and reviewer. The supervisor
    routes work to agents based on the task requirements and progress.

    Args:
        llm: Language model for all agents (can be ChatOllama or any LangChain LLM)
        researcher_tools: Optional tools for the researcher agent
        coder_tools: Optional tools for the coder agent
        reviewer_tools: Optional tools for the reviewer agent
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> from langgraph_ollama_local import LocalAgentConfig
        >>> config = LocalAgentConfig()
        >>> llm = config.create_chat_client()
        >>> graph = create_multi_agent_graph(llm)
        >>>
        >>> result = run_multi_agent_task(graph, "Build a calculator")
        >>> print(result["final_result"])
    """
    workflow = StateGraph(MultiAgentState)

    # Create nodes
    supervisor = create_supervisor_node(llm)
    researcher = create_agent_node(llm, "researcher", researcher_tools)
    coder = create_agent_node(llm, "coder", coder_tools)
    reviewer = create_agent_node(llm, "reviewer", reviewer_tools)

    # Add nodes to graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("coder", coder)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("synthesize", synthesize_node)

    # Entry point: start at supervisor
    workflow.add_edge(START, "supervisor")

    # Supervisor routes to agents or synthesize
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "reviewer": "reviewer",
            "synthesize": "synthesize",
        }
    )

    # All agents return to supervisor for next decision
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("reviewer", "supervisor")

    # Synthesize leads to end
    workflow.add_edge("synthesize", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_multi_agent_task(
    graph: "CompiledStateGraph",
    task: str,
    max_iterations: int = 10,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through the multi-agent system.

    This is a convenience function that sets up the initial state
    and invokes the graph with proper configuration.

    Args:
        graph: Compiled multi-agent graph
        task: Task description for the team
        max_iterations: Maximum supervisor iterations (default: 10)
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - final_result: Synthesized output from all agents
        - agent_outputs: List of individual agent outputs
        - messages: Full conversation history
        - iteration: Final iteration count

    Example:
        >>> result = run_multi_agent_task(
        ...     graph,
        ...     "Create a Python function to validate email addresses",
        ...     max_iterations=5
        ... )
        >>> print(result["final_result"])
    """
    initial_state: MultiAgentState = {
        "messages": [HumanMessage(content=f"Task: {task}")],
        "task": task,
        "next_agent": "",
        "agent_outputs": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_result": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Example Tools ===

@tool
def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: Search query string

    Returns:
        Search results as text
    """
    # Placeholder - in real usage, integrate with a search API
    return f"[Search results for: {query}] No real search configured."


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the result.

    Args:
        code: Python code to execute

    Returns:
        Output from code execution
    """
    # Placeholder - in real usage, use a sandboxed executor
    return f"[Code execution not available in this environment]"


# === Module Exports ===

__all__ = [
    # State
    "MultiAgentState",
    "SupervisorDecision",
    # Node creators
    "create_supervisor_node",
    "create_agent_node",
    "synthesize_node",
    # Routing
    "route_supervisor",
    # Graph builders
    "create_multi_agent_graph",
    # Utilities
    "run_multi_agent_task",
    # Example tools
    "search_web",
    "execute_python",
]
