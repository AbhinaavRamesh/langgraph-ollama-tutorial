"""
Plan-and-Execute Pattern Module.

This module provides the plan-and-execute pattern for breaking complex tasks
into steps, executing them sequentially, and optionally replanning based on
results. This pattern is ideal for multi-step problems that benefit from
upfront planning.

Key concepts:
- **Planning**: Create a step-by-step plan upfront
- **Execution**: Execute steps sequentially using a ReAct agent
- **Replanning**: Adaptively adjust the plan based on execution results
- **Two-Phase Approach**: Plan with one model, execute with another

Architecture:
    ```
    ┌─────────┐     ┌──────────┐     ┌───────────┐
    │ Planner │────►│ Executor │────►│ Replanner │
    └─────────┘     └──────────┘     └─────┬─────┘
                          ↑                 │
                          │    Replan      │
                          └─────────────────┤
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │   Finalize   │
                                    └──────────────┘
    ```

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> from langgraph_ollama_local.patterns import create_plan_execute_graph
    >>> from langgraph.prebuilt import create_react_agent
    >>>
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>> tools = [search_tool, calculator_tool]
    >>>
    >>> # Create plan-execute graph
    >>> graph = create_plan_execute_graph(llm, tools)
    >>>
    >>> result = graph.invoke({
    ...     "task": "Research the GDP of France and Germany, then compare them",
    ...     "plan": [],
    ...     "past_steps": [],
    ...     "current_step": 0,
    ...     "response": "",
    ... })
    >>> print(result["response"])
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === Constants ===

PLANNER_SYSTEM_PROMPT = """You are a strategic planner that breaks down complex tasks into clear, actionable steps.

Your responsibilities:
1. Analyze the task thoroughly
2. Create a step-by-step plan with 3-7 concrete steps
3. Ensure each step is specific and actionable
4. Order steps logically for efficient execution
5. Make steps independent where possible

Guidelines:
- Keep steps simple and focused
- Avoid vague or abstract steps
- Each step should have a clear completion criterion
- Consider dependencies between steps
- Aim for the minimum number of steps needed"""

EXECUTOR_SYSTEM_PROMPT = """You are a task executor working through a plan step by step.

Your responsibilities:
- Execute the current step thoroughly
- Use available tools when needed
- Provide clear, specific results
- Build on previous step results when relevant

Focus on completing your assigned step effectively."""

REPLANNER_SYSTEM_PROMPT = """You are a replanner that decides whether to continue with the plan or finalize the response.

Your responsibilities:
1. Review completed steps and their results
2. Determine if the original task is accomplished
3. Decide to either:
   - Respond with the final answer if the task is complete
   - Create a new plan if more work is needed

Guidelines:
- Only finalize if the task is truly complete
- If replanning, create steps that build on what's been done
- Be efficient - don't add unnecessary steps"""


# === State Definition ===


class PlanExecuteState(TypedDict):
    """
    State schema for plan-and-execute pattern.

    Attributes:
        task: The original task/objective to accomplish
        plan: List of step descriptions to execute in order
        past_steps: Accumulated history of (step, result) pairs
        current_step: Index of the next step to execute (0-based)
        response: Final response when task is complete
    """

    task: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    current_step: int
    response: str


# === Structured Outputs ===


class Plan(BaseModel):
    """Plan output from the planner."""

    steps: list[str] = Field(
        description="List of 3-7 actionable steps to accomplish the task"
    )


class Response(BaseModel):
    """Final response when task is complete."""

    response: str = Field(
        description="Comprehensive response addressing the original task"
    )


class Act(BaseModel):
    """Action decision from replanner - either respond or create new plan."""

    action: Union[Response, Plan] = Field(
        description="Either a final Response or a new Plan for continued execution"
    )


# === Node Functions ===


def create_planner_node(llm: "BaseChatModel") -> callable:
    """
    Create a planner node that generates a step-by-step plan.

    The planner analyzes the task and creates a structured plan with
    3-7 actionable steps. Uses structured output if supported by the LLM.

    Args:
        llm: Language model for planning

    Returns:
        Node function that creates a plan from the task

    Example:
        >>> planner = create_planner_node(llm)
        >>> state = {"task": "Research and compare GDP of France and Germany", ...}
        >>> result = planner(state)
        >>> print(result["plan"])  # List of steps
    """
    # Try to use structured output, fall back to parsing if not supported
    try:
        structured_llm = llm.with_structured_output(Plan)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def planner(state: PlanExecuteState) -> dict:
        """Generate initial plan from task."""
        task = state["task"]

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Task: {task}

Create a detailed step-by-step plan to accomplish this task.
Break it into 3-7 clear, actionable steps."""
            ),
        ]

        if use_structured:
            output = structured_llm.invoke(messages)
            steps = output.steps
        else:
            # Fallback: parse from text response
            response = structured_llm.invoke(messages)
            content = response.content

            # Parse numbered or bulleted lists
            lines = content.split("\n")
            steps = []
            for line in lines:
                line = line.strip()
                # Match patterns like "1.", "Step 1:", "- ", "* "
                if line and (
                    line[0].isdigit()
                    or line.lower().startswith("step")
                    or line.startswith("-")
                    or line.startswith("*")
                ):
                    # Clean the line
                    cleaned = line
                    # Remove numbering and prefixes
                    for prefix in ["step", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "-", "*"]:
                        if cleaned.lower().startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                            cleaned = cleaned.lstrip("0123456789.:) ").strip()
                            break
                    if cleaned and len(cleaned) > 5:  # Avoid empty or too-short steps
                        steps.append(cleaned)

            # Ensure we have at least one step
            if not steps:
                steps = ["Complete the task: " + task]

        return {"plan": steps, "current_step": 0}

    return planner


def create_executor_node(
    llm: "BaseChatModel",
    tools: list[Any] | None = None,
) -> callable:
    """
    Create an executor node that executes one step at a time.

    The executor uses the LLM (optionally with tools in ReAct mode) to
    complete the current step. It builds on results from previous steps.

    Args:
        llm: Language model for execution
        tools: Optional list of tools for the executor to use

    Returns:
        Node function that executes the current step

    Example:
        >>> executor = create_executor_node(llm, tools=[search_tool])
        >>> state = {"plan": ["Search for X", "Analyze Y"], "current_step": 0, ...}
        >>> result = executor(state)
        >>> # Returns updated past_steps and current_step
    """
    # If tools provided, create ReAct agent for execution
    if tools:
        from langgraph.prebuilt import create_react_agent

        react_agent = create_react_agent(llm, tools)
        use_tools = True
    else:
        use_tools = False

    def executor(state: PlanExecuteState) -> dict:
        """Execute the current step."""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        past_steps = state.get("past_steps", [])
        task = state["task"]

        # Check if we're done
        if current_step >= len(plan):
            return {}

        step = plan[current_step]

        # Build context from past steps
        context_parts = [f"Original task: {task}\n"]

        if past_steps:
            context_parts.append("Steps completed so far:")
            for i, (prev_step, prev_result) in enumerate(past_steps, 1):
                context_parts.append(f"\n{i}. {prev_step}")
                # Truncate long results
                result_preview = (
                    prev_result[:200] + "..."
                    if len(prev_result) > 200
                    else prev_result
                )
                context_parts.append(f"   Result: {result_preview}")
            context_parts.append("\n")

        context = "".join(context_parts)

        # Execute the step
        if use_tools:
            # Use ReAct agent with tools
            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"""{context}
Now execute this step: {step}

Use tools if needed to complete this step thoroughly."""
                    )
                ]
            }
            agent_result = react_agent.invoke(agent_input)
            # Extract the final message content
            messages = agent_result.get("messages", [])
            if messages:
                result = messages[-1].content
            else:
                result = "Step executed"
        else:
            # Use LLM without tools
            messages = [
                SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""{context}
Now execute this step: {step}

Provide a clear, specific result for this step."""
                ),
            ]
            response = llm.invoke(messages)
            result = response.content

        return {
            "past_steps": [(step, result)],
            "current_step": current_step + 1,
        }

    return executor


def create_replanner_node(llm: "BaseChatModel") -> callable:
    """
    Create a replanner node that decides to finalize or create a new plan.

    The replanner reviews completed steps and determines whether the task
    is complete (returns Response) or needs more work (returns new Plan).

    Args:
        llm: Language model for replanning decisions

    Returns:
        Node function that decides next action

    Example:
        >>> replanner = create_replanner_node(llm)
        >>> state = {"task": "...", "past_steps": [...], ...}
        >>> result = replanner(state)
        >>> # Returns either {"response": "..."} or {"plan": [...]}
    """
    # Try to use structured output
    try:
        structured_llm = llm.with_structured_output(Act)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def replanner(state: PlanExecuteState) -> dict:
        """Decide whether to respond or replan."""
        task = state["task"]
        past_steps = state.get("past_steps", [])

        # Format past steps for review
        steps_summary = []
        for i, (step, result) in enumerate(past_steps, 1):
            result_preview = (
                result[:300] + "..." if len(result) > 300 else result
            )
            steps_summary.append(f"{i}. {step}\n   Result: {result_preview}")

        steps_text = "\n\n".join(steps_summary)

        messages = [
            SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""Original task: {task}

Steps completed:
{steps_text}

Based on these results, decide:
1. If the task is complete, provide a final Response synthesizing the results
2. If more work is needed, provide a new Plan with additional steps

What should we do next?"""
            ),
        ]

        if use_structured:
            output = structured_llm.invoke(messages)

            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:  # New Plan
                return {
                    "plan": output.action.steps,
                    "current_step": 0,
                }
        else:
            # Fallback: simple heuristic
            response = structured_llm.invoke(messages)
            content = response.content

            # Simple check: if response seems conclusive, finalize
            conclusive_indicators = [
                "in conclusion",
                "to summarize",
                "final answer",
                "completed",
                "accomplished",
            ]

            if any(indicator in content.lower() for indicator in conclusive_indicators):
                return {"response": content}
            else:
                # Otherwise treat as needing more steps
                # Parse potential steps from content
                lines = content.split("\n")
                steps = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        cleaned = line.lstrip("0123456789.-* :").strip()
                        if cleaned and len(cleaned) > 5:
                            steps.append(cleaned)

                if steps:
                    return {"plan": steps, "current_step": 0}
                else:
                    # No clear steps, finalize with content
                    return {"response": content}

    return replanner


# === Routing Functions ===


def route_after_executor(state: PlanExecuteState) -> str:
    """
    Route after executor: decide to continue executing or move to replanner.

    Args:
        state: Current plan-execute state

    Returns:
        Next node name: "executor" if more steps, "replanner" if plan complete
    """
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current_step < len(plan):
        # More steps to execute
        return "executor"
    else:
        # Plan complete, go to replanner
        return "replanner"


def route_after_replanner(state: PlanExecuteState) -> str:
    """
    Route after replanner: decide to finalize or continue with new plan.

    Args:
        state: Current plan-execute state

    Returns:
        Next node name: "executor" if new plan, END if response provided
    """
    response = state.get("response", "")

    if response:
        # Response provided, we're done
        return END
    else:
        # New plan provided, continue executing
        return "executor"


# === Graph Builder ===


def create_plan_execute_graph(
    llm: "BaseChatModel",
    tools: list[Any] | None = None,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a plan-and-execute graph for complex task execution.

    This creates a graph that:
    1. Planner creates initial step-by-step plan
    2. Executor processes steps sequentially (optionally with tools)
    3. Replanner decides to finalize or create new plan based on results

    Args:
        llm: Language model for all nodes (planner, executor, replanner)
        tools: Optional tools for executor to use in ReAct mode
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> from langgraph_ollama_local import LocalAgentConfig
        >>> config = LocalAgentConfig()
        >>> llm = config.create_chat_client()
        >>>
        >>> graph = create_plan_execute_graph(llm, tools=[search_tool])
        >>> result = run_plan_execute_task(graph, "Research topic X and summarize")
        >>> print(result["response"])
    """
    workflow = StateGraph(PlanExecuteState)

    # Create and add nodes
    workflow.add_node("planner", create_planner_node(llm))
    workflow.add_node("executor", create_executor_node(llm, tools))
    workflow.add_node("replanner", create_replanner_node(llm))

    # Build graph structure
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")

    # After executor: loop back to executor or go to replanner
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor": "executor",
            "replanner": "replanner",
        },
    )

    # After replanner: execute new plan or end
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            END: END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===


def run_plan_execute_task(
    graph: "CompiledStateGraph",
    task: str,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through the plan-and-execute system.

    This is a convenience function that sets up the initial state
    and invokes the graph with proper configuration.

    Args:
        graph: Compiled plan-execute graph
        task: Task description to accomplish
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - response: Final answer to the task
        - plan: Final plan (may be replanned)
        - past_steps: All executed (step, result) pairs

    Example:
        >>> result = run_plan_execute_task(
        ...     graph,
        ...     "Find the population of Tokyo and compare it to New York"
        ... )
        >>> print(result["response"])
        >>> print(f"Executed {len(result['past_steps'])} steps")
    """
    initial_state: PlanExecuteState = {
        "task": task,
        "plan": [],
        "past_steps": [],
        "current_step": 0,
        "response": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===

__all__ = [
    # State
    "PlanExecuteState",
    "Plan",
    "Response",
    "Act",
    # Node creators
    "create_planner_node",
    "create_executor_node",
    "create_replanner_node",
    # Routing functions
    "route_after_executor",
    "route_after_replanner",
    # Graph builder
    "create_plan_execute_graph",
    # Utilities
    "run_plan_execute_task",
]
