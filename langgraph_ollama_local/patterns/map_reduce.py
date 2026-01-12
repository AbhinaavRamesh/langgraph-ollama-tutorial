"""
Map-Reduce Agents Pattern Module.

This module provides the map-reduce pattern for parallel agent execution
with aggregation. The pattern splits tasks into subtasks, processes them
in parallel with worker agents, and aggregates results into a final output.

Key concepts:
- **Fan-Out**: Distribute work to multiple parallel workers
- **Parallel Execution**: Multiple agents work simultaneously
- **Fan-In**: Collect and aggregate results from all workers
- **Scalability**: Add more workers to handle larger workloads

Architecture:
    ```
                         ┌─────────────────┐
                         │     Mapper      │
                         │  (Split Task)   │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
       │  Worker 1   │     │  Worker 2   │     │  Worker 3   │
       │  (Parallel) │     │  (Parallel) │     │  (Parallel) │
       └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Reducer      │
                         │  (Aggregate)    │
                         └─────────────────┘
    ```

Example:
    >>> from langgraph_ollama_local import LocalAgentConfig
    >>> from langgraph_ollama_local.patterns import create_map_reduce_graph
    >>>
    >>> config = LocalAgentConfig()
    >>> llm = config.create_chat_client()
    >>>
    >>> # Create map-reduce graph with 3 workers
    >>> graph = create_map_reduce_graph(
    ...     llm,
    ...     num_workers=3,
    ...     worker_prompt="Analyze this document section carefully."
    ... )
    >>>
    >>> result = graph.invoke({
    ...     "task": "Analyze this research paper",
    ...     "subtasks": [],
    ...     "worker_results": [],
    ...     "final_result": "",
    ... })
    >>> print(result["final_result"])
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === Constants ===

MAPPER_SYSTEM_PROMPT = """You are a mapper agent that splits complex tasks into parallel subtasks.

Your job is to:
1. Analyze the main task
2. Break it into independent, parallel subtasks
3. Ensure each subtask can be completed independently
4. Aim for balanced workload across subtasks

Guidelines:
- Create subtasks that don't depend on each other
- Make subtasks clear and specific
- Aim for {num_workers} subtasks (one per worker)
- Each subtask should be roughly equal in complexity"""

WORKER_SYSTEM_PROMPT = """You are a worker agent processing a subtask as part of a larger job.

Your responsibilities:
- Complete your assigned subtask thoroughly
- Work independently without knowledge of other workers
- Provide clear, structured output
- Focus on your specific subtask only

{custom_instructions}"""

REDUCER_SYSTEM_PROMPT = """You are a reducer agent that synthesizes results from multiple workers.

Your job is to:
1. Review all worker outputs
2. Identify common themes and patterns
3. Resolve any conflicts or inconsistencies
4. Synthesize a coherent final result

Guidelines:
- Ensure all important points from workers are included
- Remove redundancy while preserving key information
- Organize the final output logically
- Provide a comprehensive synthesis, not just concatenation"""


# === State Definition ===

class MapReduceState(TypedDict):
    """
    State schema for map-reduce pattern.

    Attributes:
        task: The main task to be processed
        subtasks: List of subtasks created by mapper (one per worker)
        worker_results: Accumulated results from all workers
        final_result: The aggregated final result from reducer
    """

    task: str
    subtasks: list[str]
    worker_results: Annotated[list[dict], operator.add]
    final_result: str


# === Structured Outputs ===

class MapperOutput(BaseModel):
    """Structured output from mapper for task decomposition."""

    subtasks: list[str] = Field(
        description="List of independent subtasks, one for each worker"
    )
    reasoning: str = Field(
        description="Brief explanation of how the task was split"
    )


class ReducerOutput(BaseModel):
    """Structured output from reducer for result aggregation."""

    final_result: str = Field(
        description="Synthesized result combining all worker outputs"
    )
    summary: str = Field(
        description="Brief summary of key findings across all workers"
    )


# === Node Functions ===

def create_mapper_node(
    llm: "BaseChatModel",
    num_workers: int = 3,
) -> callable:
    """
    Create a mapper node that splits tasks into subtasks.

    The mapper analyzes the main task and breaks it into independent
    subtasks that can be processed in parallel by workers.

    Args:
        llm: Language model for the mapper
        num_workers: Number of workers (and subtasks to create)

    Returns:
        Node function for the mapper

    Example:
        >>> mapper = create_mapper_node(llm, num_workers=3)
        >>> state = {"task": "Analyze research paper", "subtasks": [], ...}
        >>> result = mapper(state)
        >>> print(result["subtasks"])  # List of 3 subtasks
    """
    # Try to use structured output, fall back to parsing if not supported
    try:
        structured_llm = llm.with_structured_output(MapperOutput)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def mapper_node(state: MapReduceState) -> dict:
        """Split the task into parallel subtasks."""
        task = state["task"]

        messages = [
            SystemMessage(content=MAPPER_SYSTEM_PROMPT.format(num_workers=num_workers)),
            HumanMessage(content=f"""Task: {task}

Please break this task into {num_workers} independent subtasks that can be processed in parallel.
Each subtask should be complete and self-contained."""),
        ]

        if use_structured:
            output = structured_llm.invoke(messages)
            subtasks = output.subtasks[:num_workers]  # Ensure correct count
            reasoning = output.reasoning
        else:
            # Fallback: parse from text response
            response = structured_llm.invoke(messages)
            content = response.content

            # Simple parsing: look for numbered or bulleted lists
            lines = content.split("\n")
            subtasks = []
            for line in lines:
                line = line.strip()
                # Match numbered (1., 2.) or bulleted (-, *) lists
                if line and (
                    line[0].isdigit() or
                    line.startswith("-") or
                    line.startswith("*")
                ):
                    # Clean up the line
                    cleaned = line.lstrip("0123456789.-* ").strip()
                    if cleaned:
                        subtasks.append(cleaned)

            # Ensure we have the right number of subtasks
            if len(subtasks) < num_workers:
                # Pad with generic subtasks
                for i in range(len(subtasks), num_workers):
                    subtasks.append(f"Process part {i + 1} of: {task}")
            elif len(subtasks) > num_workers:
                subtasks = subtasks[:num_workers]

            reasoning = "Task split into parallel subtasks"

        return {
            "subtasks": subtasks,
        }

    return mapper_node


def create_worker_node(
    llm: "BaseChatModel",
    worker_id: int,
    worker_prompt: str = "",
) -> callable:
    """
    Create a worker node that processes a single subtask.

    Workers operate independently and in parallel, each processing
    their assigned subtask without knowledge of other workers.

    Args:
        llm: Language model for the worker
        worker_id: Unique identifier for this worker (0, 1, 2, ...)
        worker_prompt: Custom instructions for this worker's role

    Returns:
        Node function for the worker

    Example:
        >>> worker = create_worker_node(llm, worker_id=0,
        ...     worker_prompt="Focus on technical aspects")
        >>> state = {"subtasks": ["Analyze intro", ...], ...}
        >>> result = worker(state)
    """
    system_prompt = WORKER_SYSTEM_PROMPT.format(
        custom_instructions=worker_prompt or "Do your best work."
    )

    def worker_node(state: MapReduceState) -> dict:
        """Process assigned subtask."""
        subtasks = state.get("subtasks", [])

        # Get this worker's subtask
        if worker_id < len(subtasks):
            subtask = subtasks[worker_id]
        else:
            # Fallback if not enough subtasks
            subtask = f"Process part {worker_id + 1} of: {state['task']}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Main task: {state['task']}

Your subtask: {subtask}

Complete your subtask thoroughly and provide your results."""),
        ]

        response = llm.invoke(messages)

        return {
            "worker_results": [{
                "worker_id": worker_id,
                "subtask": subtask,
                "output": response.content,
            }],
        }

    return worker_node


def create_reducer_node(llm: "BaseChatModel") -> callable:
    """
    Create a reducer node that aggregates worker results.

    The reducer synthesizes outputs from all workers into a coherent
    final result, identifying patterns, resolving conflicts, and
    organizing information.

    Args:
        llm: Language model for the reducer

    Returns:
        Node function for the reducer

    Example:
        >>> reducer = create_reducer_node(llm)
        >>> state = {
        ...     "task": "Analyze paper",
        ...     "worker_results": [{"output": "..."}, ...],
        ...     ...
        ... }
        >>> result = reducer(state)
        >>> print(result["final_result"])
    """
    # Try to use structured output
    try:
        structured_llm = llm.with_structured_output(ReducerOutput)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def reducer_node(state: MapReduceState) -> dict:
        """Aggregate all worker results into final output."""
        worker_results = state.get("worker_results", [])

        if not worker_results:
            return {
                "final_result": "No results to aggregate - workers did not complete."
            }

        # Build context from all worker outputs
        worker_sections = []
        for result in worker_results:
            worker_id = result.get("worker_id", "unknown")
            subtask = result.get("subtask", "")
            output = result.get("output", "")

            worker_sections.append(f"""### Worker {worker_id}
**Subtask**: {subtask}
**Output**:
{output}""")

        workers_context = "\n\n".join(worker_sections)

        messages = [
            SystemMessage(content=REDUCER_SYSTEM_PROMPT),
            HumanMessage(content=f"""Original task: {state['task']}

Worker results:
{workers_context}

Synthesize these worker outputs into a comprehensive final result."""),
        ]

        if use_structured:
            output = structured_llm.invoke(messages)
            final_result = output.final_result
            summary = output.summary
        else:
            # Fallback: use response content directly
            response = structured_llm.invoke(messages)
            final_result = response.content
            summary = "Results aggregated from all workers"

        return {
            "final_result": final_result,
        }

    return reducer_node


# === Helper Functions ===

def fanout_to_workers(state: MapReduceState) -> list[str]:
    """
    Fanout function that routes to all workers in parallel.

    This is used with conditional_edges to send execution to
    multiple worker nodes simultaneously.

    Args:
        state: Current map-reduce state

    Returns:
        List of worker node names to execute in parallel

    Note:
        In LangGraph, returning a list from a conditional edge
        creates parallel execution paths.
    """
    # Get number of workers from subtasks
    num_workers = len(state.get("subtasks", []))

    if num_workers == 0:
        # No subtasks, skip to reducer
        return ["reducer"]

    # Return all worker node names
    return [f"worker_{i}" for i in range(num_workers)]


def collect_results(state: MapReduceState) -> str:
    """
    Collection function that routes to reducer after all workers complete.

    Args:
        state: Current map-reduce state

    Returns:
        Name of the next node (always "reducer")
    """
    return "reducer"


# === Graph Builder ===

def create_map_reduce_graph(
    llm: "BaseChatModel",
    num_workers: int = 3,
    worker_prompt: str = "",
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a map-reduce graph for parallel agent execution.

    This creates a graph that:
    1. Mapper splits task into subtasks
    2. Workers process subtasks in parallel
    3. Reducer aggregates results into final output

    Args:
        llm: Language model for all agents (mapper, workers, reducer)
        num_workers: Number of parallel workers (default: 3)
        worker_prompt: Custom instructions for worker agents
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> from langgraph_ollama_local import LocalAgentConfig
        >>> config = LocalAgentConfig()
        >>> llm = config.create_chat_client()
        >>>
        >>> graph = create_map_reduce_graph(llm, num_workers=3)
        >>> result = run_map_reduce_task(graph, "Analyze this document")
        >>> print(result["final_result"])
    """
    workflow = StateGraph(MapReduceState)

    # Create and add mapper node
    mapper = create_mapper_node(llm, num_workers)
    workflow.add_node("mapper", mapper)

    # Create and add worker nodes
    for i in range(num_workers):
        worker = create_worker_node(llm, i, worker_prompt)
        workflow.add_node(f"worker_{i}", worker)

    # Create and add reducer node
    reducer = create_reducer_node(llm)
    workflow.add_node("reducer", reducer)

    # Entry point: start at mapper
    workflow.add_edge(START, "mapper")

    # Mapper fans out to all workers in parallel
    # Note: Using add_edge to each worker for parallel execution
    for i in range(num_workers):
        workflow.add_edge("mapper", f"worker_{i}")

    # All workers converge to reducer
    for i in range(num_workers):
        workflow.add_edge(f"worker_{i}", "reducer")

    # Reducer leads to end
    workflow.add_edge("reducer", END)

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_map_reduce_task(
    graph: "CompiledStateGraph",
    task: str,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through the map-reduce system.

    This is a convenience function that sets up the initial state
    and invokes the graph with proper configuration.

    Args:
        graph: Compiled map-reduce graph
        task: Task description to be processed
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - final_result: Aggregated output from reducer
        - worker_results: List of individual worker outputs
        - subtasks: List of subtasks created by mapper

    Example:
        >>> result = run_map_reduce_task(
        ...     graph,
        ...     "Analyze the main themes in this research paper"
        ... )
        >>> print(result["final_result"])
        >>> print(f"Processed {len(result['worker_results'])} subtasks")
    """
    initial_state: MapReduceState = {
        "task": task,
        "subtasks": [],
        "worker_results": [],
        "final_result": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


def create_custom_map_reduce_graph(
    mapper_llm: "BaseChatModel",
    worker_llm: "BaseChatModel",
    reducer_llm: "BaseChatModel",
    num_workers: int = 3,
    worker_prompts: list[str] | None = None,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a map-reduce graph with different LLMs for each role.

    This advanced builder allows using different models for mapper,
    workers, and reducer - useful when different roles have different
    computational requirements.

    Args:
        mapper_llm: Language model for the mapper
        worker_llm: Language model for workers
        reducer_llm: Language model for the reducer
        num_workers: Number of parallel workers
        worker_prompts: Optional list of custom prompts per worker
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> # Use larger model for mapper and reducer, smaller for workers
        >>> mapper_llm = ChatOllama(model="llama3.1:70b")
        >>> worker_llm = ChatOllama(model="llama3.1:8b")
        >>> reducer_llm = ChatOllama(model="llama3.1:70b")
        >>>
        >>> graph = create_custom_map_reduce_graph(
        ...     mapper_llm, worker_llm, reducer_llm, num_workers=5
        ... )
    """
    workflow = StateGraph(MapReduceState)

    # Create mapper with dedicated LLM
    mapper = create_mapper_node(mapper_llm, num_workers)
    workflow.add_node("mapper", mapper)

    # Create workers with dedicated LLM and optional custom prompts
    for i in range(num_workers):
        prompt = worker_prompts[i] if worker_prompts and i < len(worker_prompts) else ""
        worker = create_worker_node(worker_llm, i, prompt)
        workflow.add_node(f"worker_{i}", worker)

    # Create reducer with dedicated LLM
    reducer = create_reducer_node(reducer_llm)
    workflow.add_node("reducer", reducer)

    # Build graph structure
    workflow.add_edge(START, "mapper")

    for i in range(num_workers):
        workflow.add_edge("mapper", f"worker_{i}")
        workflow.add_edge(f"worker_{i}", "reducer")

    workflow.add_edge("reducer", END)

    return workflow.compile(checkpointer=checkpointer)


# === Module Exports ===

__all__ = [
    # State
    "MapReduceState",
    "MapperOutput",
    "ReducerOutput",
    # Node creators
    "create_mapper_node",
    "create_worker_node",
    "create_reducer_node",
    # Helper functions
    "fanout_to_workers",
    "collect_results",
    # Graph builders
    "create_map_reduce_graph",
    "create_custom_map_reduce_graph",
    # Utilities
    "run_map_reduce_task",
]
