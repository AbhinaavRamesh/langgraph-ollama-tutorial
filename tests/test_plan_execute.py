"""
Tests for Plan-and-Execute Pattern.

This module tests the plan-and-execute pattern implementation including
planner, executor, replanner nodes, graph construction, and execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    pass


# === Plan-Execute State Tests ===


class TestPlanExecuteState:
    """Test plan-execute state definition."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.patterns.plan_execute import PlanExecuteState

        state: PlanExecuteState = {
            "task": "test task",
            "plan": [],
            "past_steps": [],
            "current_step": 0,
            "response": "",
        }
        assert state["task"] == "test task"
        assert state["plan"] == []
        assert state["current_step"] == 0

    def test_plan_schema(self) -> None:
        """Test Plan Pydantic model."""
        from langgraph_ollama_local.patterns.plan_execute import Plan

        plan = Plan(
            steps=[
                "Step 1: Research topic",
                "Step 2: Analyze data",
                "Step 3: Write summary",
            ]
        )
        assert len(plan.steps) == 3
        assert "Research" in plan.steps[0]

    def test_response_schema(self) -> None:
        """Test Response Pydantic model."""
        from langgraph_ollama_local.patterns.plan_execute import Response

        response = Response(response="Final answer to the task")
        assert "Final answer" in response.response

    def test_act_schema_with_response(self) -> None:
        """Test Act schema with Response."""
        from langgraph_ollama_local.patterns.plan_execute import Act, Response

        act = Act(action=Response(response="Task complete"))
        assert isinstance(act.action, Response)
        assert "complete" in act.action.response

    def test_act_schema_with_plan(self) -> None:
        """Test Act schema with Plan."""
        from langgraph_ollama_local.patterns.plan_execute import Act, Plan

        act = Act(action=Plan(steps=["New step 1", "New step 2"]))
        assert isinstance(act.action, Plan)
        assert len(act.action.steps) == 2


# === Planner Node Tests ===


class TestPlannerNode:
    """Test planner node functionality."""

    def test_create_planner_node(self) -> None:
        """Test planner node creation."""
        from langgraph_ollama_local.patterns.plan_execute import create_planner_node

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            steps=[
                "Research the topic",
                "Analyze findings",
                "Write summary",
            ]
        )
        mock_llm.with_structured_output.return_value = mock_structured

        planner = create_planner_node(mock_llm)
        assert callable(planner)

    def test_planner_creates_plan(self) -> None:
        """Test planner creates valid plan."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_planner_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            steps=[
                "Search for GDP of France",
                "Search for GDP of Germany",
                "Compare the values",
                "Provide analysis",
            ]
        )
        mock_llm.with_structured_output.return_value = mock_structured

        planner = create_planner_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Compare GDP of France and Germany",
            "plan": [],
            "past_steps": [],
            "current_step": 0,
            "response": "",
        }

        result = planner(state)

        assert "plan" in result
        assert len(result["plan"]) == 4
        assert all(isinstance(s, str) for s in result["plan"])
        assert result["current_step"] == 0

    def test_planner_with_fallback(self) -> None:
        """Test planner fallback when structured output not supported."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_planner_node,
        )

        mock_llm = MagicMock()
        # Simulate no structured output support
        mock_llm.with_structured_output.side_effect = AttributeError(
            "Not supported"
        )
        mock_llm.invoke.return_value = AIMessage(
            content="""Here's the plan:
1. First step
2. Second step
3. Third step"""
        )

        planner = create_planner_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Test task",
            "plan": [],
            "past_steps": [],
            "current_step": 0,
            "response": "",
        }

        result = planner(state)

        assert "plan" in result
        assert len(result["plan"]) >= 1


# === Executor Node Tests ===


class TestExecutorNode:
    """Test executor node functionality."""

    def test_create_executor_node(self) -> None:
        """Test executor node creation."""
        from langgraph_ollama_local.patterns.plan_execute import create_executor_node

        mock_llm = MagicMock()
        executor = create_executor_node(mock_llm)
        assert callable(executor)

    def test_executor_processes_step(self) -> None:
        """Test executor processes current step."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_executor_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="France's GDP is $2.96 trillion"
        )

        executor = create_executor_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Compare GDPs",
            "plan": [
                "Search for GDP of France",
                "Search for GDP of Germany",
                "Compare",
            ],
            "past_steps": [],
            "current_step": 0,
            "response": "",
        }

        result = executor(state)

        assert "past_steps" in result
        assert len(result["past_steps"]) == 1
        assert result["past_steps"][0][0] == "Search for GDP of France"
        assert "GDP" in result["past_steps"][0][1]
        assert result["current_step"] == 1

    def test_executor_builds_context(self) -> None:
        """Test executor builds context from past steps."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_executor_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Germany's GDP is $4.31 trillion")

        executor = create_executor_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Compare GDPs",
            "plan": ["Search France", "Search Germany", "Compare"],
            "past_steps": [("Search France", "France GDP: $2.96T")],
            "current_step": 1,
            "response": "",
        }

        result = executor(state)

        # Verify context was built (check that invoke was called)
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        # Should include context with past steps
        assert any("France" in str(msg.content) for msg in call_args)

    def test_executor_returns_empty_when_done(self) -> None:
        """Test executor returns empty dict when all steps complete."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_executor_node,
        )

        mock_llm = MagicMock()
        executor = create_executor_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Task",
            "plan": ["Step 1", "Step 2"],
            "past_steps": [],
            "current_step": 2,  # Beyond plan length
            "response": "",
        }

        result = executor(state)
        assert result == {}

    def test_executor_with_tools(self) -> None:
        """Test executor with tools uses ReAct agent."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_executor_node,
        )

        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "search"

        # Mock ReAct agent
        with patch("langgraph.prebuilt.create_react_agent") as mock_react:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="Search result")]
            }
            mock_react.return_value = mock_agent

            executor = create_executor_node(mock_llm, tools=[mock_tool])

            state: PlanExecuteState = {
                "task": "Task",
                "plan": ["Search for info"],
                "past_steps": [],
                "current_step": 0,
                "response": "",
            }

            result = executor(state)

            assert "past_steps" in result
            assert mock_react.called


# === Replanner Node Tests ===


class TestReplannerNode:
    """Test replanner node functionality."""

    def test_create_replanner_node(self) -> None:
        """Test replanner node creation."""
        from langgraph_ollama_local.patterns.plan_execute import create_replanner_node

        mock_llm = MagicMock()
        replanner = create_replanner_node(mock_llm)
        assert callable(replanner)

    def test_replanner_finalizes_when_complete(self) -> None:
        """Test replanner returns response when task complete."""
        from langgraph_ollama_local.patterns.plan_execute import (
            Act,
            PlanExecuteState,
            Response,
            create_replanner_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = Act(
            action=Response(response="Task is complete. France has lower GDP than Germany.")
        )
        mock_llm.with_structured_output.return_value = mock_structured

        replanner = create_replanner_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Compare GDPs",
            "plan": ["Search France", "Search Germany", "Compare"],
            "past_steps": [
                ("Search France", "France GDP: $2.96T"),
                ("Search Germany", "Germany GDP: $4.31T"),
                ("Compare", "Germany has higher GDP"),
            ],
            "current_step": 3,
            "response": "",
        }

        result = replanner(state)

        assert "response" in result
        assert "complete" in result["response"].lower()

    def test_replanner_creates_new_plan(self) -> None:
        """Test replanner creates new plan when more work needed."""
        from langgraph_ollama_local.patterns.plan_execute import (
            Act,
            Plan,
            PlanExecuteState,
            create_replanner_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = Act(
            action=Plan(
                steps=[
                    "Search for population data",
                    "Calculate GDP per capita",
                    "Provide final comparison",
                ]
            )
        )
        mock_llm.with_structured_output.return_value = mock_structured

        replanner = create_replanner_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Detailed GDP comparison",
            "plan": ["Search France", "Search Germany"],
            "past_steps": [
                ("Search France", "France GDP: $2.96T"),
                ("Search Germany", "Germany GDP: $4.31T"),
            ],
            "current_step": 2,
            "response": "",
        }

        result = replanner(state)

        assert "plan" in result
        assert len(result["plan"]) == 3
        assert result["current_step"] == 0

    def test_replanner_with_fallback(self) -> None:
        """Test replanner fallback when structured output not supported."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            create_replanner_node,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = AttributeError("Not supported")
        mock_llm.invoke.return_value = AIMessage(
            content="In conclusion, the task is complete with this final answer."
        )

        replanner = create_replanner_node(mock_llm)

        state: PlanExecuteState = {
            "task": "Task",
            "plan": ["Step 1"],
            "past_steps": [("Step 1", "Done")],
            "current_step": 1,
            "response": "",
        }

        result = replanner(state)

        # Should detect conclusion and finalize
        assert "response" in result or "plan" in result


# === Routing Tests ===


class TestRoutingFunctions:
    """Test routing logic."""

    def test_route_after_executor_continues(self) -> None:
        """Test routing continues to executor when plan not complete."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            route_after_executor,
        )

        state: PlanExecuteState = {
            "task": "Task",
            "plan": ["Step 1", "Step 2", "Step 3"],
            "past_steps": [],
            "current_step": 1,  # Step 2 next
            "response": "",
        }

        result = route_after_executor(state)
        assert result == "executor"

    def test_route_after_executor_to_replanner(self) -> None:
        """Test routing goes to replanner when plan complete."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            route_after_executor,
        )

        state: PlanExecuteState = {
            "task": "Task",
            "plan": ["Step 1", "Step 2"],
            "past_steps": [],
            "current_step": 2,  # Plan complete
            "response": "",
        }

        result = route_after_executor(state)
        assert result == "replanner"

    def test_route_after_replanner_ends(self) -> None:
        """Test routing ends when response provided."""
        from langgraph_ollama_local.patterns.plan_execute import (
            END,
            PlanExecuteState,
            route_after_replanner,
        )

        state: PlanExecuteState = {
            "task": "Task",
            "plan": [],
            "past_steps": [],
            "current_step": 0,
            "response": "Final answer",
        }

        result = route_after_replanner(state)
        assert result == END

    def test_route_after_replanner_continues(self) -> None:
        """Test routing continues to executor with new plan."""
        from langgraph_ollama_local.patterns.plan_execute import (
            PlanExecuteState,
            route_after_replanner,
        )

        state: PlanExecuteState = {
            "task": "Task",
            "plan": ["New step 1", "New step 2"],
            "past_steps": [],
            "current_step": 0,
            "response": "",  # No response
        }

        result = route_after_replanner(state)
        assert result == "executor"


# === Graph Building Tests ===


class TestGraphBuilding:
    """Test graph construction."""

    def test_create_plan_execute_graph(self) -> None:
        """Test creating plan-execute graph."""
        from langgraph_ollama_local.patterns.plan_execute import (
            create_plan_execute_graph,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        graph = create_plan_execute_graph(mock_llm)

        assert graph is not None
        # Graph should compile successfully

    def test_graph_with_tools(self) -> None:
        """Test creating graph with tools."""
        from langgraph_ollama_local.patterns.plan_execute import (
            create_plan_execute_graph,
        )

        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        with patch("langgraph.prebuilt.create_react_agent"):
            graph = create_plan_execute_graph(mock_llm, tools=[mock_tool])
            assert graph is not None


# === Integration Tests ===


class TestPlanExecuteIntegration:
    """Integration tests for full graph execution."""

    def test_run_plan_execute_task(self) -> None:
        """Test convenience runner function."""
        from langgraph_ollama_local.patterns.plan_execute import (
            Plan,
            PlanExecuteState,
            Response,
            create_plan_execute_graph,
        )

        mock_llm = MagicMock()

        # Mock planner
        mock_planner_structured = MagicMock()
        mock_planner_structured.invoke.return_value = Plan(
            steps=["Step 1", "Step 2"]
        )

        # Mock executor
        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = [
            AIMessage(content="Result 1"),
            AIMessage(content="Result 2"),
        ]

        # Mock replanner
        from langgraph_ollama_local.patterns.plan_execute import Act

        mock_replanner_structured = MagicMock()
        mock_replanner_structured.invoke.return_value = Act(
            action=Response(response="Task complete")
        )

        # Setup LLM mocks
        def mock_with_structured_output(schema):
            if schema.__name__ == "Plan":
                return mock_planner_structured
            elif schema.__name__ == "Act":
                return mock_replanner_structured
            return MagicMock()

        mock_llm.with_structured_output.side_effect = mock_with_structured_output
        mock_llm.invoke = mock_executor.invoke

        # Create and run graph
        from langgraph_ollama_local.patterns.plan_execute import run_plan_execute_task

        graph = create_plan_execute_graph(mock_llm)
        result = run_plan_execute_task(graph, "Test task")

        assert "response" in result
        assert result["response"] != ""


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_state_exports(self) -> None:
        """Test state-related exports."""
        from langgraph_ollama_local.patterns.plan_execute import (
            Act,
            Plan,
            PlanExecuteState,
            Response,
        )

        assert PlanExecuteState is not None
        assert Plan is not None
        assert Response is not None
        assert Act is not None

    def test_node_creator_exports(self) -> None:
        """Test node creator exports."""
        from langgraph_ollama_local.patterns.plan_execute import (
            create_executor_node,
            create_planner_node,
            create_replanner_node,
        )

        assert callable(create_planner_node)
        assert callable(create_executor_node)
        assert callable(create_replanner_node)

    def test_routing_exports(self) -> None:
        """Test routing function exports."""
        from langgraph_ollama_local.patterns.plan_execute import (
            route_after_executor,
            route_after_replanner,
        )

        assert callable(route_after_executor)
        assert callable(route_after_replanner)

    def test_graph_builder_exports(self) -> None:
        """Test graph builder exports."""
        from langgraph_ollama_local.patterns.plan_execute import (
            create_plan_execute_graph,
        )

        assert callable(create_plan_execute_graph)

    def test_utility_exports(self) -> None:
        """Test utility function exports."""
        from langgraph_ollama_local.patterns.plan_execute import run_plan_execute_task

        assert callable(run_plan_execute_task)
