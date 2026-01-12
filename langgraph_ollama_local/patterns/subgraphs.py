"""
Subgraph Patterns Module.

This module provides utilities for composing LangGraph graphs using
subgraphs. Subgraphs enable modular, reusable graph components that
can be embedded within larger parent graphs.

Key concepts:
- **State Transformation**: Convert between parent and subgraph states
- **Encapsulation**: Subgraphs manage their own internal state
- **Reusability**: Same subgraph can be used in multiple contexts
- **Composition**: Build complex systems from simpler components

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      Parent Graph                          │
    │                                                             │
    │  ┌─────────┐    ┌─────────────────────────┐    ┌─────────┐ │
    │  │  Entry  │───▶│      Subgraph A         │───▶│  Exit   │ │
    │  │  Node   │    │  (Encapsulated Logic)   │    │  Node   │ │
    │  └─────────┘    └─────────────────────────┘    └─────────┘ │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    ```

Example:
    >>> from langgraph_ollama_local.patterns.subgraphs import create_subgraph_node
    >>>
    >>> # Define state transformations
    >>> def state_in(parent_state):
    ...     return {"query": parent_state["question"], "docs": [], "answer": ""}
    >>>
    >>> def state_out(subgraph_state, parent_state):
    ...     return {"rag_answer": subgraph_state["answer"]}
    >>>
    >>> # Wrap subgraph as node
    >>> rag_node = create_subgraph_node(rag_graph, state_in, state_out)
    >>> parent_graph.add_node("rag", rag_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Type variables for generic state handling
ParentState = TypeVar("ParentState")
SubgraphState = TypeVar("SubgraphState")


def create_subgraph_node(
    subgraph: "CompiledStateGraph",
    state_in: Callable[[ParentState], SubgraphState],
    state_out: Callable[[SubgraphState, ParentState], dict],
    config_transform: Callable[[dict], dict] | None = None,
) -> Callable[[ParentState], dict]:
    """
    Wrap a compiled subgraph as a node in a parent graph.

    This function creates a node that:
    1. Transforms parent state to subgraph input state
    2. Runs the subgraph
    3. Transforms subgraph output back to parent state updates

    Args:
        subgraph: Compiled subgraph to wrap
        state_in: Function to transform parent state to subgraph input
        state_out: Function to transform subgraph output to parent state updates
        config_transform: Optional function to transform config for subgraph

    Returns:
        Node function compatible with parent graph's add_node()

    Example:
        >>> def state_in(parent):
        ...     return {"query": parent["question"], "docs": [], "answer": ""}
        >>>
        >>> def state_out(sub, parent):
        ...     return {"rag_answer": sub["answer"]}
        >>>
        >>> rag_node = create_subgraph_node(rag_graph, state_in, state_out)
        >>> parent.add_node("rag", rag_node)
    """
    def node(state: ParentState, config: dict | None = None) -> dict:
        """Execute subgraph with state transformation."""
        # Transform parent state to subgraph input
        subgraph_input = state_in(state)

        # Optionally transform config
        subgraph_config = config
        if config_transform and config:
            subgraph_config = config_transform(config)

        # Run subgraph
        if subgraph_config:
            subgraph_output = subgraph.invoke(subgraph_input, config=subgraph_config)
        else:
            subgraph_output = subgraph.invoke(subgraph_input)

        # Transform output back to parent state updates
        return state_out(subgraph_output, state)

    return node


async def create_subgraph_node_async(
    subgraph: "CompiledStateGraph",
    state_in: Callable[[ParentState], SubgraphState],
    state_out: Callable[[SubgraphState, ParentState], dict],
    config_transform: Callable[[dict], dict] | None = None,
) -> Callable[[ParentState], dict]:
    """
    Async version of create_subgraph_node.

    Args:
        subgraph: Compiled subgraph to wrap
        state_in: Function to transform parent state to subgraph input
        state_out: Function to transform subgraph output to parent state updates
        config_transform: Optional function to transform config for subgraph

    Returns:
        Async node function compatible with parent graph
    """
    async def node(state: ParentState, config: dict | None = None) -> dict:
        """Execute subgraph asynchronously with state transformation."""
        subgraph_input = state_in(state)

        subgraph_config = config
        if config_transform and config:
            subgraph_config = config_transform(config)

        if subgraph_config:
            subgraph_output = await subgraph.ainvoke(subgraph_input, config=subgraph_config)
        else:
            subgraph_output = await subgraph.ainvoke(subgraph_input)

        return state_out(subgraph_output, state)

    return node


# === Common State Transformers ===

def passthrough_state_in(parent_state: dict) -> dict:
    """
    Pass parent state directly to subgraph (identity transform).

    Use when parent and subgraph share the same state schema.
    """
    return parent_state.copy()


def passthrough_state_out(subgraph_state: dict, parent_state: dict) -> dict:
    """
    Pass subgraph state directly back to parent (identity transform).

    Use when parent and subgraph share the same state schema.
    """
    return subgraph_state


def field_mapper_in(*field_mappings: tuple[str, str]):
    """
    Create a state_in function that maps specific fields.

    Args:
        field_mappings: Tuples of (parent_field, subgraph_field)

    Returns:
        state_in function

    Example:
        >>> state_in = field_mapper_in(
        ...     ("user_question", "query"),
        ...     ("context_docs", "documents"),
        ... )
    """
    def state_in(parent_state: dict) -> dict:
        result = {}
        for parent_field, subgraph_field in field_mappings:
            if parent_field in parent_state:
                result[subgraph_field] = parent_state[parent_field]
        return result

    return state_in


def field_mapper_out(*field_mappings: tuple[str, str]):
    """
    Create a state_out function that maps specific fields.

    Args:
        field_mappings: Tuples of (subgraph_field, parent_field)

    Returns:
        state_out function

    Example:
        >>> state_out = field_mapper_out(
        ...     ("answer", "rag_response"),
        ...     ("sources", "cited_sources"),
        ... )
    """
    def state_out(subgraph_state: dict, parent_state: dict) -> dict:
        result = {}
        for subgraph_field, parent_field in field_mappings:
            if subgraph_field in subgraph_state:
                result[parent_field] = subgraph_state[subgraph_field]
        return result

    return state_out


# === Subgraph Composition Utilities ===

def chain_subgraphs(
    subgraphs: list[tuple["CompiledStateGraph", Callable, Callable]],
) -> Callable[[Any], dict]:
    """
    Chain multiple subgraphs sequentially.

    Each subgraph's output becomes part of the state passed to the next.

    Args:
        subgraphs: List of (subgraph, state_in, state_out) tuples

    Returns:
        Node function that runs all subgraphs in sequence

    Example:
        >>> chain = chain_subgraphs([
        ...     (retrieval_graph, retrieve_in, retrieve_out),
        ...     (grading_graph, grade_in, grade_out),
        ...     (generation_graph, generate_in, generate_out),
        ... ])
    """
    def chained_node(state: dict) -> dict:
        current_state = state.copy()

        for subgraph, state_in, state_out in subgraphs:
            # Transform and run
            subgraph_input = state_in(current_state)
            subgraph_output = subgraph.invoke(subgraph_input)

            # Merge output into current state
            updates = state_out(subgraph_output, current_state)
            current_state.update(updates)

        # Return only the updates (difference from original)
        return {k: v for k, v in current_state.items() if k not in state or state[k] != v}

    return chained_node


def conditional_subgraph(
    condition: Callable[[Any], bool],
    true_subgraph: tuple["CompiledStateGraph", Callable, Callable],
    false_subgraph: tuple["CompiledStateGraph", Callable, Callable] | None = None,
) -> Callable[[Any], dict]:
    """
    Conditionally run one of two subgraphs based on state.

    Args:
        condition: Function that returns True/False based on state
        true_subgraph: (subgraph, state_in, state_out) to run if condition is True
        false_subgraph: Optional (subgraph, state_in, state_out) to run if False

    Returns:
        Node function that conditionally executes subgraphs

    Example:
        >>> def needs_web_search(state):
        ...     return len(state.get("documents", [])) < 2
        >>>
        >>> node = conditional_subgraph(
        ...     needs_web_search,
        ...     (web_search_graph, ws_in, ws_out),
        ...     (direct_answer_graph, da_in, da_out),
        ... )
    """
    def conditional_node(state: dict) -> dict:
        if condition(state):
            subgraph, state_in, state_out = true_subgraph
        elif false_subgraph:
            subgraph, state_in, state_out = false_subgraph
        else:
            return {}  # No-op if condition is False and no false_subgraph

        subgraph_input = state_in(state)
        subgraph_output = subgraph.invoke(subgraph_input)
        return state_out(subgraph_output, state)

    return conditional_node


def parallel_subgraphs(
    subgraphs: list[tuple["CompiledStateGraph", Callable, Callable]],
    merge_strategy: Callable[[list[dict]], dict] | None = None,
) -> Callable[[Any], dict]:
    """
    Run multiple subgraphs and merge their outputs.

    Note: This runs sequentially despite the name. For true parallelism,
    use asyncio with create_subgraph_node_async.

    Args:
        subgraphs: List of (subgraph, state_in, state_out) tuples
        merge_strategy: Function to merge outputs (default: dict.update order)

    Returns:
        Node function that runs all subgraphs and merges results

    Example:
        >>> def merge_by_priority(outputs):
        ...     # Later outputs override earlier ones
        ...     merged = {}
        ...     for out in outputs:
        ...         merged.update(out)
        ...     return merged
        >>>
        >>> node = parallel_subgraphs(
        ...     [(graph_a, in_a, out_a), (graph_b, in_b, out_b)],
        ...     merge_strategy=merge_by_priority,
        ... )
    """
    def default_merge(outputs: list[dict]) -> dict:
        merged = {}
        for output in outputs:
            merged.update(output)
        return merged

    merge = merge_strategy or default_merge

    def parallel_node(state: dict) -> dict:
        outputs = []

        for subgraph, state_in, state_out in subgraphs:
            subgraph_input = state_in(state)
            subgraph_output = subgraph.invoke(subgraph_input)
            output = state_out(subgraph_output, state)
            outputs.append(output)

        return merge(outputs)

    return parallel_node


# === Retry and Error Handling ===

def retry_subgraph(
    subgraph: "CompiledStateGraph",
    state_in: Callable[[Any], dict],
    state_out: Callable[[dict, Any], dict],
    should_retry: Callable[[dict, Any], bool],
    max_retries: int = 3,
    on_retry: Callable[[dict, Any, int], dict] | None = None,
) -> Callable[[Any], dict]:
    """
    Wrap a subgraph with retry logic.

    Args:
        subgraph: Subgraph to wrap
        state_in: Input state transformer
        state_out: Output state transformer
        should_retry: Function that returns True if retry is needed
        max_retries: Maximum retry attempts
        on_retry: Optional function to modify state before retry

    Returns:
        Node function with retry logic

    Example:
        >>> def should_retry(output, state):
        ...     return output.get("quality_score", 0) < 0.7
        >>>
        >>> def on_retry(output, state, attempt):
        ...     return {"retry_hint": f"Attempt {attempt}, improve quality"}
        >>>
        >>> node = retry_subgraph(
        ...     generation_graph, gen_in, gen_out,
        ...     should_retry, max_retries=3, on_retry=on_retry,
        ... )
    """
    def retry_node(state: dict) -> dict:
        current_state = state.copy()

        for attempt in range(max_retries + 1):
            subgraph_input = state_in(current_state)
            subgraph_output = subgraph.invoke(subgraph_input)

            if attempt >= max_retries or not should_retry(subgraph_output, current_state):
                return state_out(subgraph_output, current_state)

            # Prepare for retry
            if on_retry:
                retry_updates = on_retry(subgraph_output, current_state, attempt + 1)
                current_state.update(retry_updates)

        # Should not reach here, but return last output
        return state_out(subgraph_output, current_state)

    return retry_node


# === Module Exports ===

__all__ = [
    # Core functions
    "create_subgraph_node",
    "create_subgraph_node_async",
    # State transformers
    "passthrough_state_in",
    "passthrough_state_out",
    "field_mapper_in",
    "field_mapper_out",
    # Composition utilities
    "chain_subgraphs",
    "conditional_subgraph",
    "parallel_subgraphs",
    "retry_subgraph",
]
