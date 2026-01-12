"""
Language Agent Tree Search (LATS) Pattern Module.

This module implements LATS, which applies Monte Carlo Tree Search (MCTS)
algorithms to language agents for complex reasoning tasks. Instead of exploring
a single solution path, LATS explores multiple candidate solutions in parallel
using tree search with backpropagation.

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   LATS Tree Search                          │
    │                                                             │
    │                      [Root Node]                            │
    │                     value=0.5, visits=10                   │
    │                           │                                 │
    │            ┌──────────────┼──────────────┐                 │
    │            │              │              │                  │
    │       [Child 1]      [Child 2]      [Child 3]             │
    │      value=0.7       value=0.3       value=0.6            │
    │      visits=5        visits=3        visits=2             │
    │            │                              │                │
    │       ┌────┼────┐                    ┌───┴───┐           │
    │    [C1.1] [C1.2]                  [C3.1] [C3.2]          │
    │                                                             │
    │  • Select: Choose node via UCB (exploration vs exploitation)│
    │  • Expand: Generate N candidate actions from LLM           │
    │  • Simulate: Execute tools and evaluate with reflection    │
    │  • Backpropagate: Update values up to root                 │
    └─────────────────────────────────────────────────────────────┘
    ```

MCTS Algorithm:
    1. **Selection**: Start at root, recursively select child with highest UCB
    2. **Expansion**: Generate N candidates at selected node
    3. **Simulation**: Execute each candidate, get reflection/score
    4. **Backpropagation**: Update node values from child to root

    Repeat until solution found or max depth/iterations reached.

UCB Formula:
    UCB(node) = avg_reward + exploration_weight * sqrt(ln(parent_visits) / node_visits)

    - Higher avg_reward: Exploitation (use what works)
    - Higher exploration_weight: Exploration (try new paths)

Key Concepts:
- **Tree Search**: Explore multiple solution paths simultaneously
- **UCB Selection**: Balance exploring new paths vs exploiting good paths
- **Backpropagation**: Child success/failure propagates to ancestors
- **Reflection-Based Scoring**: LLM evaluates quality of each candidate
- **Complexity Limits**: Local models need depth/width constraints

Complexity Limits for Local Models:
    | Model Size | Max Depth | Max Width | Max Nodes | Timeout |
    |------------|-----------|-----------|-----------|---------|
    | 3B-8B      | 3-4       | 2-3       | 10-15     | 30s     |
    | 13B-34B    | 4-5       | 3-4       | 20-30     | 45s     |
    | 70B+       | 5-6       | 4-5       | 30-50     | 60s     |

Example:
    >>> from langgraph_ollama_local.patterns.lats import (
    ...     create_lats_graph,
    ...     run_lats_task,
    ... )
    >>> from langchain_ollama import ChatOllama
    >>>
    >>> llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
    >>> tools = [search_tool, calculator_tool]
    >>>
    >>> # Create LATS graph with complexity limits for local model
    >>> graph = create_lats_graph(
    ...     llm=llm,
    ...     tools=tools,
    ...     max_depth=3,  # Limit for 3B model
    ...     max_width=2,  # 2 candidates per expansion
    ...     exploration_weight=1.0,
    ... )
    >>>
    >>> # Run tree search on complex task
    >>> result = run_lats_task(
    ...     graph,
    ...     "What is the GDP of the country with the highest population in 2023?"
    ... )
    >>> print(result["best_solution"])
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


# === Reflection Model ===

class Reflection(BaseModel):
    """
    LLM-based value estimate for a tree node.

    The reflection model provides structured self-critique and scoring
    for each candidate action in the tree search.

    Attributes:
        reflections: Textual critique of the candidate's quality
        score: Numeric quality score from 0 (worst) to 10 (best)
        found_solution: Whether this candidate represents a complete solution
    """

    reflections: str = Field(
        description="Detailed critique of the candidate solution's quality, "
                    "completeness, and correctness"
    )
    score: int = Field(
        ge=0,
        le=10,
        description="Quality score from 0 (worst) to 10 (best)"
    )
    found_solution: bool = Field(
        description="Whether this candidate represents a complete and correct solution"
    )

    @property
    def normalized_score(self) -> float:
        """
        Get normalized score in range [0.0, 1.0].

        Returns:
            Score normalized to 0.0-1.0 range
        """
        return self.score / 10.0


# === Node Class ===

class Node:
    """
    Tree node with MCTS statistics.

    Each node represents a state in the solution tree, tracking:
    - Messages (conversation history at this node)
    - Parent/children relationships
    - MCTS statistics (value, visits)
    - Reflection-based evaluation

    The node automatically backpropagates scores when created and
    marks the tree as solved if it finds a solution.

    Attributes:
        messages: Conversation history at this node
        reflection: LLM-based evaluation of this node's quality
        parent: Parent node (None for root)
        children: List of child nodes
        value: Average reward (updated via backpropagation)
        visits: Number of times this node was visited
        depth: Depth in tree (root is depth 1)
    """

    def __init__(
        self,
        messages: list,
        reflection: Reflection | None = None,
        parent: Node | None = None,
    ):
        """
        Initialize a tree node.

        Args:
            messages: Conversation history at this node
            reflection: Optional reflection for scoring (required for non-root)
            parent: Parent node (None for root)
        """
        self.messages = messages
        self.parent = parent
        self.children: list[Node] = []
        self.value = 0.0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent else 1
        self._is_solved = reflection.found_solution if reflection else False

        # If this node found a solution, mark entire path as solved
        if self._is_solved:
            self._mark_tree_as_solved()

        # Backpropagate initial score
        if reflection:
            self.backpropagate(reflection.normalized_score)

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        """
        Calculate UCB1 score for this node.

        UCB (Upper Confidence Bound) balances exploitation (using nodes with
        high average reward) vs exploration (trying less-visited nodes).

        Formula: UCB = avg_reward + exploration_weight * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_weight: Controls exploration vs exploitation trade-off.
                              Higher values encourage more exploration.
                              Typical range: 0.5-2.0

        Returns:
            UCB score (infinity if never visited, prioritizing exploration)
        """
        if self.visits == 0:
            return float("inf")

        # Exploitation: average reward from this node
        average_reward = self.value / self.visits

        # Exploration: bonus for less-visited nodes
        if self.parent is None or self.parent.visits == 0:
            exploration_term = 0.0
        else:
            exploration_term = math.sqrt(
                math.log(self.parent.visits) / self.visits
            )

        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float) -> None:
        """
        Update node values from this node to root.

        Backpropagation propagates reward information up the tree,
        updating visit counts and average values for all ancestors.

        Args:
            reward: Reward value to propagate (typically normalized score 0-1)
        """
        node = self
        while node:
            node.visits += 1
            # Update running average: new_avg = (old_avg * (n-1) + new_value) / n
            node.value += (reward - node.value) / node.visits
            node = node.parent

    def get_trajectory(self) -> list:
        """
        Get full message path from root to this node.

        Returns:
            List of all messages from root to current node
        """
        messages = []
        node = self
        while node:
            messages = node.messages + messages
            node = node.parent
        return messages

    def _mark_tree_as_solved(self) -> None:
        """Mark all ancestors as containing a solved path."""
        node = self.parent
        while node:
            node._is_solved = True
            node = node.parent

    @property
    def is_solved(self) -> bool:
        """Whether this node or any descendant found a solution."""
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        """Whether this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def height(self) -> int:
        """
        Get maximum depth of tree from this node.

        Returns:
            Height of subtree (0 for leaf, max child height + 1 otherwise)
        """
        if not self.children:
            return 0
        return 1 + max(child.height for child in self.children)

    def _get_all_children(self) -> list[Node]:
        """
        Get all descendant nodes (DFS).

        Returns:
            List of all descendant nodes
        """
        all_nodes = []
        for child in self.children:
            all_nodes.append(child)
            all_nodes.extend(child._get_all_children())
        return all_nodes


# === State Definition ===

class TreeState(TypedDict):
    """
    State for LATS tree search.

    The state maintains the search tree and input task throughout
    the selection-expansion-backpropagation cycle.

    Attributes:
        root: Root node of the search tree
        input: The original task/question to solve
    """

    root: Node
    input: str


# === Selection ===

def select(root: Node) -> Node:
    """
    Select best leaf node to expand via UCB.

    Starting from root, recursively select the child with highest UCB
    until reaching a leaf node (one with no children).

    Args:
        root: Root node of search tree

    Returns:
        Best leaf node to expand
    """
    node = root

    # Traverse down tree following highest UCB children
    while node.children:
        # If any child hasn't been visited, prioritize it (UCB = inf)
        max_child = max(
            node.children,
            key=lambda c: c.upper_confidence_bound()
        )
        node = max_child

    return node


# === Prompts ===

EXPANSION_PROMPT = """You are an expert problem solver working on the following task:

{input}

Current conversation history:
{trajectory}

Generate your next action to make progress on this task. You can:
1. Use available tools to gather information
2. Reason about the information you have
3. Provide a final answer if you have enough information

Think carefully and take one clear action."""

REFLECTION_PROMPT = """You are an expert evaluator assessing the quality of a problem-solving attempt.

Original task:
{input}

Attempted solution path:
{trajectory}

Evaluate this attempt and provide:
1. Detailed critique of quality, completeness, and correctness
2. A score from 0 (completely wrong/incomplete) to 10 (perfect solution)
3. Whether this represents a complete and correct solution

Be critical but fair. A solution is only complete if it fully answers the original task."""


# === Expansion ===

def create_expansion_node(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_width: int = 3,
    exploration_weight: float = 1.0,
) -> callable:
    """
    Create expansion node that generates N candidate actions.

    The expansion node:
    1. Selects best leaf via UCB
    2. Generates N candidate actions using LLM with temperature
    3. Executes any tool calls in each candidate
    4. Gets reflection/score for each candidate
    5. Creates child nodes (which auto-backpropagate)

    Args:
        llm: Language model for generation (should have temperature > 0)
        tools: Available tools for the agent
        max_width: Number of candidates to generate per expansion
        exploration_weight: UCB exploration parameter

    Returns:
        Expansion node function
    """
    # Create structured LLM for reflections
    reflection_llm = llm.with_structured_output(Reflection)

    # Create tool node for execution
    tool_node = ToolNode(tools) if tools else None

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def expand(state: TreeState) -> dict:
        """
        Expand best leaf node with N candidates.

        Args:
            state: Current tree state

        Returns:
            Updated state (tree is modified in-place)
        """
        root = state["root"]
        task_input = state["input"]

        # Select best leaf node to expand
        best_node = select(root)
        trajectory = best_node.get_trajectory()

        # Build expansion prompt
        trajectory_str = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in trajectory
        )
        expansion_msg = HumanMessage(
            content=EXPANSION_PROMPT.format(
                input=task_input,
                trajectory=trajectory_str
            )
        )

        # Generate N candidates with temperature sampling
        candidates = []
        for _ in range(max_width):
            response = llm_with_tools.invoke(trajectory + [expansion_msg])

            # Execute any tool calls
            candidate_msgs = [response]
            if response.tool_calls and tool_node:
                # Execute tools
                tool_result = tool_node.invoke({"messages": [response]})
                candidate_msgs.extend(tool_result["messages"])

            candidates.append(candidate_msgs)

        # Reflect on each candidate
        for candidate_msgs in candidates:
            # Build trajectory for reflection
            full_trajectory = trajectory + candidate_msgs
            trajectory_str = "\n".join(
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in full_trajectory
            )

            # Get reflection/score
            reflection_msg = HumanMessage(
                content=REFLECTION_PROMPT.format(
                    input=task_input,
                    trajectory=trajectory_str
                )
            )
            reflection = reflection_llm.invoke([reflection_msg])

            # Create child node (auto-backpropagates)
            child = Node(
                messages=candidate_msgs,
                reflection=reflection,
                parent=best_node
            )
            best_node.children.append(child)

        return state

    return expand


# === Termination ===

def should_loop(
    state: TreeState,
    max_depth: int = 5,
    max_iterations: int = 20,
) -> Literal["expand", "__end__"]:
    """
    Determine whether to continue tree search.

    Search terminates when:
    1. A solution is found (any node has found_solution=True)
    2. Max depth is reached
    3. Max iterations reached (total number of nodes)

    Args:
        state: Current tree state
        max_depth: Maximum tree depth to explore
        max_iterations: Maximum number of nodes to create

    Returns:
        "expand" to continue, END to terminate
    """
    root = state["root"]

    # Check if solution found
    if root.is_solved:
        return END

    # Check max depth
    if root.height >= max_depth:
        return END

    # Check max iterations (total nodes created)
    total_nodes = 1 + len(root._get_all_children())
    if total_nodes >= max_iterations:
        return END

    return "expand"


# === Best Solution ===

def get_best_solution(root: Node) -> Node:
    """
    Return best solution from tree.

    Prioritizes:
    1. Solved leaf nodes (found_solution=True)
    2. Highest value among leaf nodes

    Args:
        root: Root of search tree

    Returns:
        Best solution node
    """
    # Get all nodes in tree
    all_nodes = [root] + root._get_all_children()

    # Get all terminal (leaf) nodes
    terminal_nodes = [n for n in all_nodes if n.is_terminal]

    if not terminal_nodes:
        # No terminal nodes yet, return root
        return root

    # Prioritize solved nodes, then by value
    return max(
        terminal_nodes,
        key=lambda n: (int(n.reflection.found_solution if n.reflection else False), n.value)
    )


# === Graph Construction ===

def create_lats_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_depth: int = 5,
    max_width: int = 3,
    max_iterations: int = 20,
    exploration_weight: float = 1.0,
) -> CompiledStateGraph:
    """
    Create LATS graph with tree search.

    The graph consists of:
    1. Expand node: Generates candidates, executes tools, reflects, backpropagates
    2. Conditional edge: Continues until solution found or limits reached

    Args:
        llm: Language model (should have temperature > 0 for diversity)
        tools: Available tools for the agent
        max_depth: Maximum tree depth (complexity limit)
        max_width: Candidates per expansion (complexity limit)
        max_iterations: Maximum total nodes (complexity limit)
        exploration_weight: UCB exploration parameter (0.5-2.0)

    Returns:
        Compiled LATS graph

    Example:
        >>> llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
        >>> graph = create_lats_graph(llm, tools, max_depth=3, max_width=2)
        >>> result = graph.invoke({
        ...     "root": Node(messages=[]),
        ...     "input": "Complex reasoning task"
        ... })
    """
    # Create expansion node
    expand_node = create_expansion_node(
        llm=llm,
        tools=tools,
        max_width=max_width,
        exploration_weight=exploration_weight,
    )

    # Build graph
    graph = StateGraph(TreeState)

    # Add nodes
    graph.add_node("expand", expand_node)

    # Add edges
    graph.add_edge(START, "expand")
    graph.add_conditional_edges(
        "expand",
        lambda state: should_loop(
            state,
            max_depth=max_depth,
            max_iterations=max_iterations
        ),
    )

    return graph.compile()


# === Convenience Runner ===

def run_lats_task(
    graph: CompiledStateGraph,
    task: str,
    initial_messages: list | None = None,
) -> dict[str, Any]:
    """
    Run LATS tree search on a task.

    This is a convenience function that:
    1. Initializes the tree with root node
    2. Runs the graph
    3. Extracts the best solution

    Args:
        graph: Compiled LATS graph
        task: Task description / question to solve
        initial_messages: Optional initial messages for context

    Returns:
        Dictionary with:
            - best_solution: Best solution node
            - best_trajectory: Full message path to best solution
            - root: Root node (for tree inspection)
            - total_nodes: Total nodes explored

    Example:
        >>> result = run_lats_task(
        ...     graph,
        ...     "What is the population of the largest city in France?"
        ... )
        >>> print(result["best_trajectory"][-1].content)
    """
    # Initialize root node
    initial_msgs = initial_messages or []
    root = Node(messages=initial_msgs, reflection=None, parent=None)

    # Run tree search
    final_state = graph.invoke({
        "root": root,
        "input": task,
    })

    # Get best solution
    best_node = get_best_solution(final_state["root"])

    return {
        "best_solution": best_node,
        "best_trajectory": best_node.get_trajectory(),
        "root": final_state["root"],
        "total_nodes": 1 + len(final_state["root"]._get_all_children()),
    }


__all__ = [
    # Models
    "Reflection",
    "Node",
    "TreeState",
    # Core functions
    "select",
    "create_expansion_node",
    "should_loop",
    "get_best_solution",
    # Graph
    "create_lats_graph",
    "run_lats_task",
]
