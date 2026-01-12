"""
ReWOO (Reasoning WithOut Observation) Pattern Module.

This module implements the ReWOO pattern which decouples planning from execution
to achieve token efficiency. Instead of interleaving reasoning with tool calls
(like ReAct), ReWOO generates a complete plan upfront, executes all tools
sequentially with variable substitution, and synthesizes a final answer.

Key concepts:
- **Single Planning Call**: Generate complete plan with evidence variables (#E1, #E2, ...)
- **Variable Substitution**: Replace variables with actual tool results during execution
- **Sequential Tool Execution**: Execute tools one by one, building evidence map
- **Single Synthesis Call**: Combine all evidence into final answer

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    ReWOO Pattern                         │
    │                                                           │
    │  ┌──────────┐       ┌──────────┐       ┌──────────┐    │
    │  │ Planner  │──────►│ Tool     │──────►│ Solver   │    │
    │  │ (1 call) │       │ Executor │       │ (1 call) │    │
    │  └──────────┘       └────┬─────┘       └──────────┘    │
    │       │                  │                               │
    │       │                  └──────┐                        │
    │       │                         ▼                        │
    │       │                   ┌──────────┐                  │
    │       │                   │  Loop    │                  │
    │       │                   │ Until    │                  │
    │       │                   │  Done    │                  │
    │       │                   └──────────┘                  │
    │       │                                                  │
    │       ▼                                                  │
    │  Plan: Search for info                                  │
    │  #E1 = Google[query]                                    │
    │  Plan: Analyze results                                  │
    │  #E2 = LLM[given #E1, what is answer?]                  │
    └─────────────────────────────────────────────────────────┘
    ```

Token Efficiency Comparison:
    ReAct Pattern:
    - ~10 LLM calls (interleaved reasoning and tool calls)
    - High token usage due to repeated context

    ReWOO Pattern:
    - 2 LLM calls (planner + solver)
    - Low token usage with upfront planning

Example:
    >>> from langgraph_ollama_local.patterns.rewoo import (
    ...     create_rewoo_graph,
    ...     run_rewoo_task,
    ... )
    >>> from langchain_community.tools import DuckDuckGoSearchRun
    >>>
    >>> # Create tools
    >>> tools = {"Google": DuckDuckGoSearchRun()}
    >>>
    >>> # Create graph
    >>> graph = create_rewoo_graph(llm, tools)
    >>>
    >>> # Run task
    >>> result = run_rewoo_task(
    ...     graph,
    ...     "Who won the 2024 NBA championship and who was the finals MVP?"
    ... )
    >>> print(result["result"])

Reference:
    Paper: https://arxiv.org/abs/2305.18323
    ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


# === State Definition ===


class ReWOOState(TypedDict):
    """
    State schema for ReWOO pattern.

    The ReWOO pattern maintains a plan with variable references, executes
    tools to populate evidence, and synthesizes a final answer.

    Attributes:
        task: Original user task/query
        plan_string: Raw plan text from planner
        steps: Parsed plan steps as list of (reasoning, var_name, tool, args)
        results: Evidence map with variable substitution (e.g., {"#E1": "...", "#E2": "..."})
        result: Final synthesized answer
    """

    task: str
    plan_string: str
    steps: list[tuple[str, str, str, str]]
    results: dict[str, str]
    result: str


# === Plan Parsing ===

# Regex pattern to parse ReWOO plan format
# Matches lines like:
#   Plan: Search for information about X
#   #E1 = Google[search query]
PLAN_REGEX = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"


def parse_plan(plan_string: str) -> list[tuple[str, str, str, str]]:
    """
    Parse a ReWOO plan string into structured steps.

    Args:
        plan_string: Raw plan text from planner containing Plan/Evidence lines

    Returns:
        List of tuples: (reasoning, var_name, tool_name, tool_args)

    Example:
        >>> plan = '''
        ... Plan: Search for recent NBA champions
        ... #E1 = Google[2024 NBA championship winner]
        ...
        ... Plan: Analyze the search results
        ... #E2 = LLM[Who won according to #E1?]
        ... '''
        >>> steps = parse_plan(plan)
        >>> print(steps)
        [
            ('Search for recent NBA champions', '#E1', 'Google', '2024 NBA championship winner'),
            ('Analyze the search results', '#E2', 'LLM', 'Who won according to #E1?')
        ]
    """
    matches = re.findall(PLAN_REGEX, plan_string, re.MULTILINE)
    return [(reasoning.strip(), var, tool, args.strip()) for reasoning, var, tool, args in matches]


# === Prompts ===

PLANNER_PROMPT = """For the following task, make plans that can solve the problem step by step.
For each plan, indicate which external tool together with tool input to retrieve evidence.
You can store the evidence into a variable #E that can be called by later tools.

Available tools:
{tool_descriptions}

The output MUST follow this format for each step:
Plan: [reasoning for this step]
#E[number] = [ToolName][input]

You can reference previous evidence variables in later steps (e.g., use #E1 in the input for #E2).

Example:
Plan: Search for information about the topic
#E1 = Google[search query here]

Plan: Analyze the search results to answer the question
#E2 = LLM[Based on #E1, what is the answer?]

Now create a plan for this task:
Task: {task}

Plan:"""

SOLVER_PROMPT = """Solve the following task using the provided evidence.

Task: {task}

Evidence and Plan:
{plan_with_evidence}

Based on the evidence above, provide a comprehensive answer to the task.

Answer:"""

# LLM tool prompt for self-reasoning steps
LLM_TOOL_PROMPT = """You are a helpful assistant. Answer the following question:

{input}"""


# === Node Functions ===


def create_planner_node(llm: "BaseChatModel"):
    """
    Create a planner node that generates a complete plan upfront.

    The planner generates all steps with evidence variables (#E1, #E2, etc.)
    before any tool execution begins. This is the key difference from ReAct
    which interleaves planning and execution.

    Args:
        llm: Language model for planning

    Returns:
        Node function that generates a plan

    Example:
        >>> planner = create_planner_node(llm)
        >>> state = {"task": "Who won the 2024 NBA championship?"}
        >>> result = planner(state)
        >>> print(result["plan_string"])
        Plan: Search for 2024 NBA championship winner
        #E1 = Google[2024 NBA championship winner]
        ...
    """

    def planner(state: ReWOOState) -> dict:
        """Generate complete plan with evidence variables."""
        task = state["task"]

        # Get tool descriptions from state if available, otherwise use defaults
        tool_descriptions = state.get(
            "tool_descriptions",
            "(1) Google[input]: Search engine for finding information.\n(2) LLM[input]: Language model for reasoning and analysis.",
        )

        # Format prompt
        prompt = PLANNER_PROMPT.format(
            tool_descriptions=tool_descriptions,
            task=task,
        )

        # Generate plan
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        plan_string = response.content

        # Parse plan into structured steps
        steps = parse_plan(plan_string)

        return {
            "plan_string": plan_string,
            "steps": steps,
        }

    return planner


def _get_current_step(state: ReWOOState) -> Optional[int]:
    """
    Get the index of the next step to execute.

    Args:
        state: Current ReWOO state

    Returns:
        Index of next step, or None if all steps are complete

    Example:
        >>> state = {
        ...     "steps": [("#E1", ...), ("#E2", ...)],
        ...     "results": {"#E1": "result 1"}
        ... }
        >>> _get_current_step(state)
        1  # Next step is index 1 (#E2)
    """
    results = state.get("results", {})
    steps = state.get("steps", [])

    if len(results) >= len(steps):
        return None  # All steps complete

    return len(results)  # Next step index


def create_tool_executor(tools: dict[str, "BaseTool"], llm: Optional["BaseChatModel"] = None):
    """
    Create a tool executor node with variable substitution.

    The executor processes one step at a time, substituting evidence
    variables (#E1, #E2, etc.) with their actual values before calling tools.

    Args:
        tools: Dictionary mapping tool names to tool instances
        llm: Optional language model for LLM tool (for reasoning steps)

    Returns:
        Node function that executes one tool step

    Example:
        >>> tools = {"Google": search_tool, "LLM": None}
        >>> executor = create_tool_executor(tools, llm)
        >>> state = {
        ...     "steps": [("reasoning", "#E1", "Google", "query"), ("#E2", "LLM", "analyze #E1")],
        ...     "results": {}
        ... }
        >>> result = executor(state)
        >>> print(result["results"])
        {"#E1": "search results..."}
    """

    def executor(state: ReWOOState) -> dict:
        """Execute one tool step with variable substitution."""
        current_step = _get_current_step(state)

        if current_step is None:
            # All steps complete
            return {}

        reasoning, var_name, tool_name, tool_input = state["steps"][current_step]
        results = state.get("results", {})

        # Variable substitution: replace #E1, #E2, etc. with actual values
        substituted_input = tool_input
        for var, value in results.items():
            substituted_input = substituted_input.replace(var, value)

        # Execute tool
        if tool_name == "LLM":
            # Use LLM for reasoning steps
            if llm is None:
                result = f"LLM tool not available. Input was: {substituted_input}"
            else:
                prompt = LLM_TOOL_PROMPT.format(input=substituted_input)
                response = llm.invoke([HumanMessage(content=prompt)])
                result = response.content
        elif tool_name in tools:
            # Use external tool
            tool = tools[tool_name]
            try:
                result = tool.invoke(substituted_input)
                # Convert to string if needed
                if not isinstance(result, str):
                    result = str(result)
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
        else:
            result = f"Tool {tool_name} not found. Available tools: {list(tools.keys())}"

        # Store result with variable name
        new_results = results.copy()
        new_results[var_name] = result

        return {"results": new_results}

    return executor


def create_solver_node(llm: "BaseChatModel"):
    """
    Create a solver node that synthesizes the final answer.

    The solver takes all evidence collected during tool execution and
    generates a comprehensive final answer. This is a single LLM call
    that completes the ReWOO workflow.

    Args:
        llm: Language model for synthesis

    Returns:
        Node function that generates final answer

    Example:
        >>> solver = create_solver_node(llm)
        >>> state = {
        ...     "task": "Who won?",
        ...     "steps": [...],
        ...     "results": {"#E1": "Team A won", "#E2": "Player B was MVP"}
        ... }
        >>> result = solver(state)
        >>> print(result["result"])
        "Team A won the championship with Player B as MVP..."
    """

    def solver(state: ReWOOState) -> dict:
        """Synthesize final answer from all evidence."""
        task = state["task"]
        steps = state.get("steps", [])
        results = state.get("results", {})

        # Reconstruct plan with actual evidence values
        plan_with_evidence = ""

        for reasoning, var_name, tool_name, tool_input in steps:
            # Substitute variables in the input
            substituted_input = tool_input
            for var, value in results.items():
                substituted_input = substituted_input.replace(var, value)

            # Get the result for this step
            result_value = results.get(var_name, "[not executed]")

            # Format as: Plan: ... \n #E1 = Tool[input] \n Result: ...
            plan_with_evidence += f"Plan: {reasoning}\n"
            plan_with_evidence += f"{var_name} = {tool_name}[{substituted_input}]\n"
            plan_with_evidence += f"Result: {result_value}\n\n"

        # Generate final answer
        prompt = SOLVER_PROMPT.format(
            task=task,
            plan_with_evidence=plan_with_evidence,
        )

        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        return {"result": response.content}

    return solver


# === Routing ===


def route_rewoo(state: ReWOOState) -> Literal["executor", "solver"]:
    """
    Route between tool executor and solver.

    If there are more steps to execute, route to executor.
    If all steps are complete, route to solver for final synthesis.

    Args:
        state: Current ReWOO state

    Returns:
        "executor" if more steps remain, "solver" if ready to synthesize

    Example:
        >>> state = {"steps": [(...), (...)], "results": {"#E1": "..."}}
        >>> route_rewoo(state)
        "executor"  # More steps remaining
        >>>
        >>> state = {"steps": [(...), (...)], "results": {"#E1": "...", "#E2": "..."}}
        >>> route_rewoo(state)
        "solver"  # All steps complete
    """
    current_step = _get_current_step(state)

    if current_step is None:
        # All steps complete, go to solver
        return "solver"
    else:
        # More steps to execute
        return "executor"


# === Graph Builder ===


def create_rewoo_graph(
    llm: "BaseChatModel",
    tools: dict[str, "BaseTool"],
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a ReWOO graph with planner, executor, and solver.

    The graph implements the ReWOO pattern:
    1. Planner generates complete plan with evidence variables
    2. Executor runs tools sequentially with variable substitution
    3. Solver synthesizes final answer from all evidence

    Args:
        llm: Language model for planner and solver (also used for LLM tool steps)
        tools: Dictionary mapping tool names to tool instances
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled ReWOO graph

    Example:
        >>> from langchain_community.tools import DuckDuckGoSearchRun
        >>>
        >>> tools = {
        ...     "Google": DuckDuckGoSearchRun(),
        ... }
        >>>
        >>> graph = create_rewoo_graph(llm, tools)
        >>> result = graph.invoke({"task": "What is the capital of France?"})
        >>> print(result["result"])
    """
    workflow = StateGraph(ReWOOState)

    # Add nodes
    workflow.add_node("planner", create_planner_node(llm))
    workflow.add_node("executor", create_tool_executor(tools, llm))
    workflow.add_node("solver", create_solver_node(llm))

    # Entry point: planner
    workflow.add_edge(START, "planner")

    # Planner -> executor (start executing first step)
    workflow.add_edge("planner", "executor")

    # Conditional routing from executor
    workflow.add_conditional_edges(
        "executor",
        route_rewoo,
        {
            "executor": "executor",  # Loop back for next step
            "solver": "solver",  # All steps done, synthesize
        },
    )

    # Solver ends
    workflow.add_edge("solver", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Runner ===


def run_rewoo_task(
    graph: "CompiledStateGraph",
    task: str,
    tool_descriptions: str | None = None,
    thread_id: str = "default",
) -> dict:
    """
    Run a ReWOO task from start to finish.

    Args:
        graph: Compiled ReWOO graph
        task: User task/query to solve
        tool_descriptions: Optional custom tool descriptions for planner
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - task: Original task
        - plan_string: Generated plan
        - steps: Parsed plan steps
        - results: Evidence map with all tool results
        - result: Final synthesized answer

    Example:
        >>> graph = create_rewoo_graph(llm, tools)
        >>> result = run_rewoo_task(
        ...     graph,
        ...     "Who won the 2024 NBA championship?"
        ... )
        >>> print(result["result"])
        "The Boston Celtics won the 2024 NBA championship..."
    """
    initial_state: ReWOOState = {
        "task": task,
        "plan_string": "",
        "steps": [],
        "results": {},
        "result": "",
    }

    # Add tool descriptions if provided
    if tool_descriptions:
        initial_state["tool_descriptions"] = tool_descriptions  # type: ignore

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


def format_tool_descriptions(tools: dict[str, "BaseTool"]) -> str:
    """
    Format tool descriptions for the planner prompt.

    Args:
        tools: Dictionary mapping tool names to tool instances

    Returns:
        Formatted string describing available tools

    Example:
        >>> tools = {"Google": search_tool, "Calculator": calc_tool}
        >>> print(format_tool_descriptions(tools))
        (1) Google[input]: Searches the web for information.
        (2) Calculator[input]: Performs mathematical calculations.
    """
    descriptions = []

    for i, (name, tool) in enumerate(tools.items(), 1):
        # Get tool description
        if hasattr(tool, "description"):
            desc = tool.description
        elif hasattr(tool, "__doc__") and tool.__doc__:
            desc = tool.__doc__.strip().split("\n")[0]
        else:
            desc = f"Tool for {name}"

        descriptions.append(f"({i}) {name}[input]: {desc}")

    # Always include LLM tool
    descriptions.append(f"({len(descriptions) + 1}) LLM[input]: A language model for reasoning and analysis.")

    return "\n".join(descriptions)


# === Module Exports ===

__all__ = [
    # State
    "ReWOOState",
    # Plan parsing
    "PLAN_REGEX",
    "parse_plan",
    # Node creators
    "create_planner_node",
    "create_tool_executor",
    "create_solver_node",
    # Routing
    "route_rewoo",
    # Graph builder
    "create_rewoo_graph",
    # Runners
    "run_rewoo_task",
    # Utilities
    "format_tool_descriptions",
]
