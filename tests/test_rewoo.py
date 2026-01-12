"""
Tests for ReWOO Pattern.

This module tests the ReWOO pattern implementation including
plan parsing, variable substitution, tool execution, and graph construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


# === State Tests ===


class TestReWOOState:
    """Test ReWOO state definition."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.patterns.rewoo import ReWOOState

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [],
            "results": {},
            "result": "",
        }
        assert state["task"] == "Test task"
        assert state["results"] == {}


# === Plan Parsing Tests ===


class TestPlanParsing:
    """Test plan parsing functionality."""

    def test_parse_simple_plan(self) -> None:
        """Test parsing a simple plan with one step."""
        from langgraph_ollama_local.patterns.rewoo import parse_plan

        plan = """
        Plan: Search for information
        #E1 = Google[search query]
        """

        steps = parse_plan(plan)

        assert len(steps) == 1
        reasoning, var, tool, args = steps[0]
        assert var == "#E1"
        assert tool == "Google"
        assert args == "search query"
        assert "Search for information" in reasoning

    def test_parse_multi_step_plan(self) -> None:
        """Test parsing a plan with multiple steps."""
        from langgraph_ollama_local.patterns.rewoo import parse_plan

        plan = """
        Plan: Search for NBA championship information
        #E1 = Google[2024 NBA championship winner]

        Plan: Analyze the search results
        #E2 = LLM[Who won according to #E1?]

        Plan: Get more details
        #E3 = Google[#E2 Finals MVP]
        """

        steps = parse_plan(plan)

        assert len(steps) == 3

        # Check first step
        reasoning1, var1, tool1, args1 = steps[0]
        assert var1 == "#E1"
        assert tool1 == "Google"
        assert "2024 NBA championship winner" in args1

        # Check second step
        reasoning2, var2, tool2, args2 = steps[1]
        assert var2 == "#E2"
        assert tool2 == "LLM"
        assert "#E1" in args2

        # Check third step
        reasoning3, var3, tool3, args3 = steps[2]
        assert var3 == "#E3"
        assert tool3 == "Google"
        assert "#E2" in args3

    def test_parse_empty_plan(self) -> None:
        """Test parsing an empty plan."""
        from langgraph_ollama_local.patterns.rewoo import parse_plan

        plan = ""
        steps = parse_plan(plan)

        assert len(steps) == 0

    def test_parse_plan_with_complex_arguments(self) -> None:
        """Test parsing plan with complex arguments."""
        from langgraph_ollama_local.patterns.rewoo import parse_plan

        plan = """
        Plan: Calculate a complex expression
        #E1 = Calculator[(25 + 15) * 2]

        Plan: Use the result in another calculation
        #E2 = Calculator[#E1 + 10]
        """

        steps = parse_plan(plan)

        assert len(steps) == 2
        _, _, _, args1 = steps[0]
        assert "(25 + 15) * 2" in args1

        _, _, _, args2 = steps[1]
        assert "#E1 + 10" in args2

    def test_parse_plan_with_whitespace(self) -> None:
        """Test parsing plan handles whitespace correctly."""
        from langgraph_ollama_local.patterns.rewoo import parse_plan

        plan = """
        Plan:   Search with extra spaces
        #E1 = Google[  query with spaces  ]
        """

        steps = parse_plan(plan)

        assert len(steps) == 1
        reasoning, var, tool, args = steps[0]
        assert var == "#E1"
        # Note: reasoning and args should be stripped
        assert reasoning.strip() == "Search with extra spaces"
        assert args.strip() == "query with spaces"


# === Planner Node Tests ===


class TestPlannerNode:
    """Test planner node creation and execution."""

    def test_create_planner_node(self) -> None:
        """Test planner node creation."""
        from langgraph_ollama_local.patterns.rewoo import create_planner_node

        mock_llm = MagicMock()
        planner = create_planner_node(mock_llm)

        assert callable(planner)

    def test_planner_generates_plan(self) -> None:
        """Test that planner generates a valid plan."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_planner_node,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """
        Plan: Search for information
        #E1 = Google[test query]

        Plan: Analyze results
        #E2 = LLM[Analyze #E1]
        """
        mock_llm.invoke.return_value = mock_response

        planner = create_planner_node(mock_llm)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [],
            "results": {},
            "result": "",
        }

        result = planner(state)

        # Check that LLM was called
        assert mock_llm.invoke.called

        # Check that plan_string was returned
        assert "plan_string" in result
        assert "#E1" in result["plan_string"]

        # Check that steps were parsed
        assert "steps" in result
        assert len(result["steps"]) == 2

    def test_planner_with_custom_tool_descriptions(self) -> None:
        """Test planner with custom tool descriptions."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_planner_node,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Plan: Test\n#E1 = Tool[input]"
        mock_llm.invoke.return_value = mock_response

        planner = create_planner_node(mock_llm)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [],
            "results": {},
            "result": "",
            "tool_descriptions": "Custom tool descriptions",  # type: ignore
        }

        result = planner(state)

        # Verify LLM was called with custom descriptions
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        prompt = messages[0].content

        assert "Custom tool descriptions" in prompt


# === Tool Executor Tests ===


class TestToolExecutor:
    """Test tool executor node."""

    def test_create_tool_executor(self) -> None:
        """Test tool executor creation."""
        from langgraph_ollama_local.patterns.rewoo import create_tool_executor

        tools = {"Google": MagicMock()}
        executor = create_tool_executor(tools)

        assert callable(executor)

    def test_executor_executes_first_step(self) -> None:
        """Test executor executes the first step."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Search results"

        tools = {"Google": mock_tool}
        executor = create_tool_executor(tools)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [("Reasoning", "#E1", "Google", "test query")],
            "results": {},
            "result": "",
        }

        result = executor(state)

        # Check tool was invoked
        assert mock_tool.invoke.called
        mock_tool.invoke.assert_called_with("test query")

        # Check result was stored
        assert "results" in result
        assert "#E1" in result["results"]
        assert result["results"]["#E1"] == "Search results"

    def test_executor_variable_substitution(self) -> None:
        """Test that executor substitutes variables correctly."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Second result"

        tools = {"Google": mock_tool}
        executor = create_tool_executor(tools)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [
                ("First", "#E1", "Google", "first query"),
                ("Second", "#E2", "Google", "search about #E1"),
            ],
            "results": {"#E1": "First result"},
            "result": "",
        }

        result = executor(state)

        # Check that variable was substituted
        mock_tool.invoke.assert_called_with("search about First result")

        # Check second result was stored
        assert "#E2" in result["results"]
        assert result["results"]["#E2"] == "Second result"

    def test_executor_with_llm_tool(self) -> None:
        """Test executor with LLM tool for reasoning."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM analysis"
        mock_llm.invoke.return_value = mock_response

        executor = create_tool_executor({}, mock_llm)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [("Analyze", "#E1", "LLM", "Analyze this: data")],
            "results": {},
            "result": "",
        }

        result = executor(state)

        # Check LLM was invoked
        assert mock_llm.invoke.called

        # Check result was stored
        assert "#E1" in result["results"]
        assert result["results"]["#E1"] == "LLM analysis"

    def test_executor_handles_missing_tool(self) -> None:
        """Test executor handles missing tool gracefully."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        executor = create_tool_executor({})

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [("Search", "#E1", "UnknownTool", "query")],
            "results": {},
            "result": "",
        }

        result = executor(state)

        # Check error message was stored
        assert "#E1" in result["results"]
        assert "not found" in result["results"]["#E1"]

    def test_executor_all_steps_complete(self) -> None:
        """Test executor returns empty when all steps complete."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        executor = create_tool_executor({})

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [("Step1", "#E1", "Tool", "input")],
            "results": {"#E1": "result"},  # Already complete
            "result": "",
        }

        result = executor(state)

        # Should return empty dict
        assert result == {}

    def test_executor_multiple_variable_substitution(self) -> None:
        """Test substituting multiple variables in one input."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_tool_executor,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Combined analysis"
        mock_llm.invoke.return_value = mock_response

        executor = create_tool_executor({}, mock_llm)

        # State with 3 steps, first 2 complete
        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [
                ("First", "#E1", "LLM", "First step"),
                ("Second", "#E2", "LLM", "Second step"),
                ("Combine", "#E3", "LLM", "Compare #E1 and #E2"),
            ],
            "results": {"#E1": "Result 1", "#E2": "Result 2"},
            "result": "",
        }

        result = executor(state)

        # Check LLM was invoked
        assert mock_llm.invoke.called

        # Check both variables were substituted in the input
        call_args = mock_llm.invoke.call_args
        if call_args:
            messages = call_args[0][0]
            prompt = messages[0].content
            assert "Result 1" in prompt
            assert "Result 2" in prompt


# === Solver Node Tests ===


class TestSolverNode:
    """Test solver node."""

    def test_create_solver_node(self) -> None:
        """Test solver node creation."""
        from langgraph_ollama_local.patterns.rewoo import create_solver_node

        mock_llm = MagicMock()
        solver = create_solver_node(mock_llm)

        assert callable(solver)

    def test_solver_synthesizes_answer(self) -> None:
        """Test that solver synthesizes final answer."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_solver_node,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Final synthesized answer"
        mock_llm.invoke.return_value = mock_response

        solver = create_solver_node(mock_llm)

        state: ReWOOState = {
            "task": "Who won?",
            "plan_string": "",
            "steps": [
                ("Search", "#E1", "Google", "winner"),
                ("Analyze", "#E2", "LLM", "Who won in #E1?"),
            ],
            "results": {"#E1": "Team A won", "#E2": "Team A"},
            "result": "",
        }

        result = solver(state)

        # Check LLM was called
        assert mock_llm.invoke.called

        # Check result contains final answer
        assert "result" in result
        assert result["result"] == "Final synthesized answer"

    def test_solver_includes_all_evidence(self) -> None:
        """Test that solver prompt includes all evidence."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            create_solver_node,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Answer"
        mock_llm.invoke.return_value = mock_response

        solver = create_solver_node(mock_llm)

        state: ReWOOState = {
            "task": "Test task",
            "plan_string": "",
            "steps": [
                ("Step1", "#E1", "Tool1", "input1"),
                ("Step2", "#E2", "Tool2", "input2"),
            ],
            "results": {"#E1": "Evidence 1", "#E2": "Evidence 2"},
            "result": "",
        }

        result = solver(state)

        # Check that prompt includes all evidence
        call_args = mock_llm.invoke.call_args[0][0]
        prompt = call_args[0].content

        assert "Evidence 1" in prompt
        assert "Evidence 2" in prompt


# === Routing Tests ===


class TestRouting:
    """Test routing logic."""

    def test_route_to_executor(self) -> None:
        """Test routing to executor when steps remain."""
        from langgraph_ollama_local.patterns.rewoo import ReWOOState, route_rewoo

        state: ReWOOState = {
            "task": "Test",
            "plan_string": "",
            "steps": [("S1", "#E1", "T1", "i1"), ("S2", "#E2", "T2", "i2")],
            "results": {"#E1": "result1"},  # Only first step done
            "result": "",
        }

        route = route_rewoo(state)

        assert route == "executor"

    def test_route_to_solver(self) -> None:
        """Test routing to solver when all steps complete."""
        from langgraph_ollama_local.patterns.rewoo import ReWOOState, route_rewoo

        state: ReWOOState = {
            "task": "Test",
            "plan_string": "",
            "steps": [("S1", "#E1", "T1", "i1"), ("S2", "#E2", "T2", "i2")],
            "results": {"#E1": "result1", "#E2": "result2"},  # All done
            "result": "",
        }

        route = route_rewoo(state)

        assert route == "solver"

    def test_route_no_steps(self) -> None:
        """Test routing when there are no steps."""
        from langgraph_ollama_local.patterns.rewoo import ReWOOState, route_rewoo

        state: ReWOOState = {
            "task": "Test",
            "plan_string": "",
            "steps": [],
            "results": {},
            "result": "",
        }

        route = route_rewoo(state)

        # No steps means go to solver
        assert route == "solver"


# === Helper Function Tests ===


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_current_step(self) -> None:
        """Test _get_current_step helper."""
        from langgraph_ollama_local.patterns.rewoo import (
            ReWOOState,
            _get_current_step,
        )

        # No results yet
        state: ReWOOState = {
            "task": "Test",
            "plan_string": "",
            "steps": [("S1", "#E1", "T", "i"), ("S2", "#E2", "T", "i")],
            "results": {},
            "result": "",
        }
        assert _get_current_step(state) == 0

        # First step done
        state["results"] = {"#E1": "result1"}
        assert _get_current_step(state) == 1

        # All steps done
        state["results"] = {"#E1": "result1", "#E2": "result2"}
        assert _get_current_step(state) is None

    def test_format_tool_descriptions(self) -> None:
        """Test format_tool_descriptions utility."""
        from langgraph_ollama_local.patterns.rewoo import format_tool_descriptions

        mock_tool1 = MagicMock()
        mock_tool1.description = "Tool 1 description"

        mock_tool2 = MagicMock()
        mock_tool2.description = "Tool 2 description"

        tools = {"Google": mock_tool1, "Calculator": mock_tool2}

        descriptions = format_tool_descriptions(tools)

        # Check format
        assert "Google" in descriptions
        assert "Calculator" in descriptions
        assert "Tool 1 description" in descriptions
        assert "Tool 2 description" in descriptions
        assert "LLM" in descriptions  # Always includes LLM


# === Graph Creation Tests ===


class TestGraphCreation:
    """Test graph creation."""

    def test_create_rewoo_graph(self) -> None:
        """Test creating ReWOO graph."""
        from langgraph_ollama_local.patterns.rewoo import create_rewoo_graph

        mock_llm = MagicMock()
        tools = {"Google": MagicMock()}

        graph = create_rewoo_graph(mock_llm, tools)

        assert graph is not None
        # Graph should be compiled
        assert hasattr(graph, "invoke")


# === Integration Tests ===


class TestReWOOIntegration:
    """Integration tests for full ReWOO workflow."""

    def test_run_rewoo_task(self) -> None:
        """Test running a complete ReWOO task."""
        from langgraph_ollama_local.patterns.rewoo import (
            create_rewoo_graph,
            run_rewoo_task,
        )

        # Mock LLM
        mock_llm = MagicMock()

        # Mock planner response
        plan_response = MagicMock()
        plan_response.content = """
        Plan: Search for information
        #E1 = Google[test query]

        Plan: Analyze the results
        #E2 = LLM[Analyze #E1]
        """

        # Mock solver response
        solver_response = MagicMock()
        solver_response.content = "Final answer based on evidence"

        # Mock LLM to return different responses
        mock_llm.invoke.side_effect = [
            plan_response,  # Planner call
            MagicMock(content="Analysis result"),  # LLM tool call
            solver_response,  # Solver call
        ]

        # Mock tool
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Search results"

        tools = {"Google": mock_tool}

        graph = create_rewoo_graph(mock_llm, tools)

        result = run_rewoo_task(graph, "Test task")

        # Verify structure
        assert "task" in result
        assert "plan_string" in result
        assert "steps" in result
        assert "results" in result
        assert "result" in result

        # Verify execution
        assert len(result["steps"]) == 2
        assert "#E1" in result["results"]
        assert "#E2" in result["results"]
        assert result["result"] == "Final answer based on evidence"


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected symbols are exported."""

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        from langgraph_ollama_local.patterns import rewoo

        expected_exports = [
            "ReWOOState",
            "PLAN_REGEX",
            "parse_plan",
            "create_planner_node",
            "create_tool_executor",
            "create_solver_node",
            "route_rewoo",
            "create_rewoo_graph",
            "run_rewoo_task",
            "format_tool_descriptions",
        ]

        for export in expected_exports:
            assert hasattr(rewoo, export), f"Missing export: {export}"
