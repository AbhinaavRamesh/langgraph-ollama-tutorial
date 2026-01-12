"""
Tests for LATS (Language Agent Tree Search) Pattern.

This module tests the LATS implementation including Node class,
reflection model, tree search operations, and graph construction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    pass


# === Reflection Model Tests ===


class TestReflection:
    """Test Reflection model."""

    def test_reflection_creation(self) -> None:
        """Test creating a Reflection."""
        from langgraph_ollama_local.patterns.lats import Reflection

        reflection = Reflection(
            reflections="Good attempt but incomplete",
            score=6,
            found_solution=False,
        )
        assert reflection.score == 6
        assert reflection.found_solution is False
        assert "incomplete" in reflection.reflections

    def test_reflection_normalized_score(self) -> None:
        """Test normalized score property."""
        from langgraph_ollama_local.patterns.lats import Reflection

        reflection = Reflection(
            reflections="Perfect solution",
            score=10,
            found_solution=True,
        )
        assert reflection.normalized_score == 1.0

        reflection2 = Reflection(
            reflections="Halfway there",
            score=5,
            found_solution=False,
        )
        assert reflection2.normalized_score == 0.5

    def test_reflection_score_bounds(self) -> None:
        """Test score must be in valid range."""
        from langgraph_ollama_local.patterns.lats import Reflection

        # Valid scores
        Reflection(reflections="Bad", score=0, found_solution=False)
        Reflection(reflections="Good", score=10, found_solution=True)

        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            Reflection(reflections="Invalid", score=-1, found_solution=False)

        with pytest.raises(Exception):
            Reflection(reflections="Invalid", score=11, found_solution=False)


# === Node Class Tests ===


class TestNode:
    """Test Node class."""

    def test_node_creation_root(self) -> None:
        """Test creating a root node."""
        from langgraph_ollama_local.patterns.lats import Node

        root = Node(messages=[], reflection=None, parent=None)

        assert root.parent is None
        assert root.depth == 1
        assert len(root.children) == 0
        assert root.visits == 0
        assert root.value == 0.0
        assert root.is_terminal

    def test_node_creation_with_reflection(self) -> None:
        """Test creating node with reflection auto-backpropagates."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)

        reflection = Reflection(
            reflections="Good progress",
            score=7,
            found_solution=False,
        )
        child = Node(
            messages=[HumanMessage(content="test")],
            reflection=reflection,
            parent=root,
        )

        # Child should have score backpropagated
        assert child.visits == 1
        assert child.value == 0.7  # normalized score

        # Parent should be updated
        assert root.visits == 1
        assert root.value == 0.7

    def test_node_depth(self) -> None:
        """Test node depth calculation."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        assert root.depth == 1

        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child1 = Node(messages=[], reflection=reflection, parent=root)
        assert child1.depth == 2

        child2 = Node(messages=[], reflection=reflection, parent=child1)
        assert child2.depth == 3

    def test_ucb_unvisited_node(self) -> None:
        """Test UCB returns infinity for unvisited nodes."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child = Node(messages=[], reflection=reflection, parent=root)

        # Create another child that hasn't been visited yet
        child2 = Node.__new__(Node)
        child2.visits = 0
        child2.value = 0.0
        child2.parent = root

        ucb = child2.upper_confidence_bound()
        assert ucb == float("inf")

    def test_ucb_calculation(self) -> None:
        """Test UCB calculation with actual values."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)

        # Create child with known values
        reflection = Reflection(reflections="test", score=6, found_solution=False)
        child = Node(messages=[], reflection=reflection, parent=root)

        # Manually set visits for testing
        root.visits = 10
        child.visits = 3
        child.value = 0.6

        ucb = child.upper_confidence_bound(exploration_weight=1.0)

        # UCB = 0.6/3 + 1.0 * sqrt(ln(10)/3)
        expected_avg = 0.6 / 3
        expected_exploration = math.sqrt(math.log(10) / 3)
        expected_ucb = expected_avg + expected_exploration

        assert abs(ucb - expected_ucb) < 0.01

    def test_ucb_exploration_weight(self) -> None:
        """Test UCB with different exploration weights."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child = Node(messages=[], reflection=reflection, parent=root)

        root.visits = 10
        child.visits = 5

        ucb_low = child.upper_confidence_bound(exploration_weight=0.5)
        ucb_high = child.upper_confidence_bound(exploration_weight=2.0)

        # Higher exploration weight should give higher UCB
        assert ucb_high > ucb_low

    def test_backpropagate(self) -> None:
        """Test backpropagation updates ancestors."""
        from langgraph_ollama_local.patterns.lats import Node

        # Create 3-level tree
        root = Node(messages=[], reflection=None, parent=None)
        root.visits = 0  # Reset from auto-backprop

        child = Node.__new__(Node)
        child.messages = []
        child.parent = root
        child.children = []
        child.value = 0.0
        child.visits = 0

        grandchild = Node.__new__(Node)
        grandchild.messages = []
        grandchild.parent = child
        grandchild.children = []
        grandchild.value = 0.0
        grandchild.visits = 0

        # Backpropagate from grandchild
        grandchild.backpropagate(0.8)

        # All nodes should be updated
        assert grandchild.visits == 1
        assert grandchild.value == 0.8
        assert child.visits == 1
        assert child.value == 0.8
        assert root.visits == 1
        assert root.value == 0.8

        # Backpropagate again with different reward
        grandchild.backpropagate(0.6)

        assert grandchild.visits == 2
        assert abs(grandchild.value - 0.7) < 0.01  # (0.8 + 0.6) / 2

    def test_get_trajectory(self) -> None:
        """Test getting full message path from root."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        msg1 = HumanMessage(content="first")
        msg2 = AIMessage(content="second")
        msg3 = HumanMessage(content="third")

        root = Node(messages=[msg1], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child = Node(messages=[msg2], reflection=reflection, parent=root)
        grandchild = Node(messages=[msg3], reflection=reflection, parent=child)

        trajectory = grandchild.get_trajectory()

        assert len(trajectory) == 3
        assert trajectory[0].content == "first"
        assert trajectory[1].content == "second"
        assert trajectory[2].content == "third"

    def test_is_solved_propagation(self) -> None:
        """Test that finding solution marks ancestors as solved."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        assert not root.is_solved

        reflection_unsolved = Reflection(
            reflections="Progress",
            score=5,
            found_solution=False,
        )
        child1 = Node(messages=[], reflection=reflection_unsolved, parent=root)
        assert not root.is_solved
        assert not child1.is_solved

        # Create child with solution
        reflection_solved = Reflection(
            reflections="Complete!",
            score=10,
            found_solution=True,
        )
        child2 = Node(messages=[], reflection=reflection_solved, parent=root)

        # Both child and root should be marked as solved
        assert child2.is_solved
        assert root.is_solved

    def test_is_terminal(self) -> None:
        """Test terminal node detection."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        assert root.is_terminal

        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child = Node(messages=[], reflection=reflection, parent=root)
        root.children.append(child)

        assert not root.is_terminal
        assert child.is_terminal

    def test_height_calculation(self) -> None:
        """Test tree height calculation."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        assert root.height == 0

        reflection = Reflection(reflections="test", score=5, found_solution=False)
        child1 = Node(messages=[], reflection=reflection, parent=root)
        root.children.append(child1)
        assert root.height == 1

        child2 = Node(messages=[], reflection=reflection, parent=child1)
        child1.children.append(child2)
        assert root.height == 2
        assert child1.height == 1

    def test_get_all_children(self) -> None:
        """Test getting all descendants."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)

        # Create tree structure
        child1 = Node(messages=[], reflection=reflection, parent=root)
        child2 = Node(messages=[], reflection=reflection, parent=root)
        grandchild1 = Node(messages=[], reflection=reflection, parent=child1)
        grandchild2 = Node(messages=[], reflection=reflection, parent=child1)

        root.children = [child1, child2]
        child1.children = [grandchild1, grandchild2]

        all_children = root._get_all_children()

        assert len(all_children) == 4
        assert child1 in all_children
        assert child2 in all_children
        assert grandchild1 in all_children
        assert grandchild2 in all_children


# === State Tests ===


class TestTreeState:
    """Test TreeState definition."""

    def test_tree_state_structure(self) -> None:
        """Verify TreeState has required fields."""
        from langgraph_ollama_local.patterns.lats import Node, TreeState

        root = Node(messages=[], reflection=None, parent=None)
        state: TreeState = {
            "root": root,
            "input": "test task",
        }

        assert state["input"] == "test task"
        assert state["root"] is root


# === Selection Tests ===


class TestSelection:
    """Test selection function."""

    def test_select_returns_root_when_no_children(self) -> None:
        """Test selection returns root if it has no children."""
        from langgraph_ollama_local.patterns.lats import Node, select

        root = Node(messages=[], reflection=None, parent=None)
        selected = select(root)

        assert selected is root

    def test_select_chooses_highest_ucb(self) -> None:
        """Test selection chooses child with highest UCB."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, select

        root = Node(messages=[], reflection=None, parent=None)

        # Create children with different values
        reflection1 = Reflection(reflections="ok", score=5, found_solution=False)
        reflection2 = Reflection(reflections="better", score=8, found_solution=False)

        child1 = Node(messages=[], reflection=reflection1, parent=root)
        child2 = Node(messages=[], reflection=reflection2, parent=root)

        root.children = [child1, child2]

        # child2 should have higher UCB due to higher score
        selected = select(root)

        assert selected is child2

    def test_select_traverses_to_leaf(self) -> None:
        """Test selection traverses tree to leaf node."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, select

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=7, found_solution=False)

        child = Node(messages=[], reflection=reflection, parent=root)
        grandchild = Node(messages=[], reflection=reflection, parent=child)

        root.children = [child]
        child.children = [grandchild]

        selected = select(root)

        # Should traverse to leaf (grandchild)
        assert selected is grandchild


# === Expansion Tests ===


class TestExpansion:
    """Test expansion node creation."""

    def test_create_expansion_node(self) -> None:
        """Test creating expansion node."""
        from langgraph_ollama_local.patterns.lats import create_expansion_node

        mock_llm = MagicMock()
        mock_tools = []

        expand_node = create_expansion_node(
            llm=mock_llm,
            tools=mock_tools,
            max_width=2,
        )

        assert callable(expand_node)


# === Termination Tests ===


class TestTermination:
    """Test termination conditions."""

    def test_should_loop_terminates_on_solution(self) -> None:
        """Test termination when solution is found."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, should_loop
        from langgraph.graph import END

        root = Node(messages=[], reflection=None, parent=None)

        # Mark as solved
        reflection = Reflection(
            reflections="Complete!",
            score=10,
            found_solution=True,
        )
        child = Node(messages=[], reflection=reflection, parent=root)
        root.children.append(child)

        state = {"root": root, "input": "test"}

        result = should_loop(state, max_depth=5, max_iterations=20)
        assert result == END

    def test_should_loop_terminates_on_max_depth(self) -> None:
        """Test termination when max depth reached."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, should_loop
        from langgraph.graph import END

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)

        # Create deep tree
        child1 = Node(messages=[], reflection=reflection, parent=root)
        child2 = Node(messages=[], reflection=reflection, parent=child1)
        child3 = Node(messages=[], reflection=reflection, parent=child2)

        root.children = [child1]
        child1.children = [child2]
        child2.children = [child3]

        state = {"root": root, "input": "test"}

        # Max depth is 3, tree has height 3
        result = should_loop(state, max_depth=3, max_iterations=100)
        assert result == END

    def test_should_loop_terminates_on_max_iterations(self) -> None:
        """Test termination when max iterations reached."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, should_loop
        from langgraph.graph import END

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)

        # Create 5 children (6 total nodes including root)
        for _ in range(5):
            child = Node(messages=[], reflection=reflection, parent=root)
            root.children.append(child)

        state = {"root": root, "input": "test"}

        # Max iterations is 5, but we have 6 nodes
        result = should_loop(state, max_depth=10, max_iterations=5)
        assert result == END

    def test_should_loop_continues(self) -> None:
        """Test loop continues when conditions not met."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, should_loop

        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(reflections="test", score=5, found_solution=False)

        child = Node(messages=[], reflection=reflection, parent=root)
        root.children.append(child)

        state = {"root": root, "input": "test"}

        result = should_loop(state, max_depth=10, max_iterations=100)
        assert result == "expand"


# === Best Solution Tests ===


class TestBestSolution:
    """Test best solution extraction."""

    def test_get_best_solution_returns_root_if_no_terminal(self) -> None:
        """Test returns root if no terminal nodes."""
        from langgraph_ollama_local.patterns.lats import Node, get_best_solution

        root = Node(messages=[], reflection=None, parent=None)
        best = get_best_solution(root)

        assert best is root

    def test_get_best_solution_prioritizes_solved(self) -> None:
        """Test prioritizes solved nodes over higher scores."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, get_best_solution

        root = Node(messages=[], reflection=None, parent=None)

        # High score but not solved
        reflection_high = Reflection(
            reflections="Good but incomplete",
            score=9,
            found_solution=False,
        )
        child1 = Node(messages=[], reflection=reflection_high, parent=root)

        # Lower score but solved
        reflection_solved = Reflection(
            reflections="Complete!",
            score=7,
            found_solution=True,
        )
        child2 = Node(messages=[], reflection=reflection_solved, parent=root)

        root.children = [child1, child2]

        best = get_best_solution(root)

        # Should choose solved node even with lower score
        assert best is child2

    def test_get_best_solution_highest_value(self) -> None:
        """Test selects highest value among unsolved nodes."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, get_best_solution

        root = Node(messages=[], reflection=None, parent=None)

        reflection_low = Reflection(reflections="ok", score=4, found_solution=False)
        reflection_high = Reflection(reflections="better", score=8, found_solution=False)

        child1 = Node(messages=[], reflection=reflection_low, parent=root)
        child2 = Node(messages=[], reflection=reflection_high, parent=root)

        root.children = [child1, child2]

        best = get_best_solution(root)

        # Should choose higher value
        assert best is child2


# === Graph Construction Tests ===


class TestGraphConstruction:
    """Test LATS graph creation."""

    def test_create_lats_graph(self) -> None:
        """Test creating LATS graph."""
        from langgraph_ollama_local.patterns.lats import create_lats_graph

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_tools = []

        graph = create_lats_graph(
            llm=mock_llm,
            tools=mock_tools,
            max_depth=3,
            max_width=2,
            max_iterations=10,
        )

        assert graph is not None

    def test_create_lats_graph_with_tools(self) -> None:
        """Test creating LATS graph with tools."""
        from langgraph_ollama_local.patterns.lats import create_lats_graph
        from langchain_core.tools import tool

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return "result"

        graph = create_lats_graph(
            llm=mock_llm,
            tools=[test_tool],
            max_depth=4,
            max_width=3,
        )

        assert graph is not None


# === Runner Tests ===


class TestRunner:
    """Test convenience runner."""

    def test_run_lats_task(self) -> None:
        """Test running LATS task."""
        from langgraph_ollama_local.patterns.lats import Node, Reflection, run_lats_task

        # Create mock graph
        mock_graph = MagicMock()

        # Create a root with solved child
        root = Node(messages=[], reflection=None, parent=None)
        reflection = Reflection(
            reflections="Complete",
            score=10,
            found_solution=True,
        )
        child = Node(
            messages=[AIMessage(content="solution")],
            reflection=reflection,
            parent=root,
        )
        root.children.append(child)

        mock_graph.invoke = MagicMock(return_value={"root": root, "input": "test"})

        result = run_lats_task(mock_graph, "test task")

        assert "best_solution" in result
        assert "best_trajectory" in result
        assert "root" in result
        assert "total_nodes" in result
        assert result["total_nodes"] == 2  # root + 1 child


# === Integration Tests ===


@pytest.mark.integration
class TestLATSIntegration:
    """Integration tests with real LLM (requires Ollama)."""

    def test_lats_basic_task(self) -> None:
        """Test LATS with simple reasoning task."""
        pytest.skip("Integration test - requires Ollama")
        # from langchain_ollama import ChatOllama
        # from langgraph_ollama_local.patterns.lats import create_lats_graph, run_lats_task
        #
        # llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
        # graph = create_lats_graph(llm, tools=[], max_depth=2, max_width=2)
        #
        # result = run_lats_task(graph, "What is 2+2?")
        # assert result["total_nodes"] >= 1
