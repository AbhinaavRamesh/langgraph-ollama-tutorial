"""
Tests for Multi-Agent Evaluation Pattern.

This module tests the evaluation pattern implementation including
simulated users, evaluators, and metrics aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    pass


# === State Tests ===


class TestEvaluationState:
    """Test evaluation state definition and reducers."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.patterns.evaluation import EvaluationState

        state: EvaluationState = {
            "messages": [],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 0,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }
        assert state["max_turns"] == 10
        assert state["session_complete"] is False

    def test_evaluation_criteria_schema(self) -> None:
        """Test EvaluationCriteria Pydantic model."""
        from langgraph_ollama_local.patterns.evaluation import EvaluationCriteria

        criteria = EvaluationCriteria(
            helpfulness=4,
            accuracy=5,
            empathy=3,
            efficiency=4,
            goal_completion=1,
            reasoning="Agent was helpful and accurate",
        )
        assert criteria.helpfulness == 4
        assert criteria.accuracy == 5
        assert criteria.goal_completion == 1

    def test_evaluation_criteria_validation(self) -> None:
        """Test score validation in EvaluationCriteria."""
        from langgraph_ollama_local.patterns.evaluation import EvaluationCriteria

        # Valid scores
        criteria = EvaluationCriteria(
            helpfulness=1,
            accuracy=5,
            empathy=3,
            efficiency=2,
            goal_completion=0,
            reasoning="Test",
        )
        assert criteria.helpfulness == 1

        # Invalid scores should raise
        with pytest.raises(Exception):  # Pydantic validation error
            EvaluationCriteria(
                helpfulness=6,  # Out of range
                accuracy=5,
                empathy=3,
                efficiency=4,
                goal_completion=1,
                reasoning="Test",
            )


# === Configuration Tests ===


class TestSimulatedUser:
    """Test simulated user configuration."""

    def test_simulated_user_creation(self) -> None:
        """Test creating SimulatedUser config."""
        from langgraph_ollama_local.patterns.evaluation import SimulatedUser

        user = SimulatedUser(
            persona="Frustrated customer",
            goals=["Get refund", "Express dissatisfaction"],
            behavior="impatient",
        )
        assert user.persona == "Frustrated customer"
        assert len(user.goals) == 2
        assert user.behavior == "impatient"

    def test_simulated_user_with_initial_message(self) -> None:
        """Test SimulatedUser with initial message."""
        from langgraph_ollama_local.patterns.evaluation import SimulatedUser

        user = SimulatedUser(
            persona="Customer",
            goals=["Get help"],
            behavior="friendly",
            initial_message="Hello, I need help!",
        )
        assert user.initial_message == "Hello, I need help!"

    def test_simulated_user_default_behavior(self) -> None:
        """Test SimulatedUser default values."""
        from langgraph_ollama_local.patterns.evaluation import SimulatedUser

        user = SimulatedUser(
            persona="Customer",
            goals=["Get help"],
        )
        assert user.behavior == "friendly"
        assert user.initial_message is None


# === Node Creation Tests ===


class TestSimulatedUserNode:
    """Test simulated user node creation and execution."""

    def test_create_simulated_user_node(self) -> None:
        """Test simulated user node creation."""
        from langgraph_ollama_local.patterns.evaluation import (
            SimulatedUser,
            create_simulated_user_node,
        )

        mock_llm = MagicMock()
        user_config = SimulatedUser(
            persona="Test user",
            goals=["Test goal"],
            behavior="friendly",
        )

        node = create_simulated_user_node(mock_llm, user_config)
        assert callable(node)

    def test_simulated_user_uses_initial_message(self) -> None:
        """Test that simulated user uses initial message on first turn."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            SimulatedUser,
            create_simulated_user_node,
        )

        mock_llm = MagicMock()
        user_config = SimulatedUser(
            persona="Test user",
            goals=["Test"],
            behavior="friendly",
            initial_message="Hello there!",
        )

        node = create_simulated_user_node(mock_llm, user_config)

        state: EvaluationState = {
            "messages": [],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 0,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)

        # Should use initial message, not call LLM
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Hello there!"
        assert result["turn_count"] == 1

    def test_simulated_user_generates_response(self) -> None:
        """Test that simulated user generates response after first turn."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            SimulatedUser,
            create_simulated_user_node,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I need help with my order"
        mock_llm.invoke.return_value = mock_response

        user_config = SimulatedUser(
            persona="Test user",
            goals=["Get help"],
            behavior="friendly",
        )

        node = create_simulated_user_node(mock_llm, user_config)

        state: EvaluationState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="How can I help?"),
            ],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 1,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)

        # Should call LLM to generate response
        assert mock_llm.invoke.called
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "I need help with my order"
        assert result["turn_count"] == 2


class TestEvaluatorNode:
    """Test evaluator node creation and execution."""

    def test_create_evaluator_node(self) -> None:
        """Test evaluator node creation."""
        from langgraph_ollama_local.patterns.evaluation import create_evaluator_node

        mock_llm = MagicMock()
        node = create_evaluator_node(mock_llm)
        assert callable(node)

    def test_evaluator_scores_conversation(self) -> None:
        """Test that evaluator provides scores."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            create_evaluator_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_criteria = MagicMock()
        mock_criteria.helpfulness = 4
        mock_criteria.accuracy = 5
        mock_criteria.empathy = 3
        mock_criteria.efficiency = 4
        mock_criteria.goal_completion = 1
        mock_criteria.reasoning = "Good responses"
        mock_structured.invoke.return_value = mock_criteria
        mock_llm.with_structured_output.return_value = mock_structured

        node = create_evaluator_node(mock_llm)

        state: EvaluationState = {
            "messages": [
                HumanMessage(content="I need help"),
                AIMessage(content="I'm here to help!"),
            ],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 2,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)

        # Should produce scores
        assert len(result["evaluator_scores"]) == 1
        scores = result["evaluator_scores"][0]
        assert scores["helpfulness"] == 4
        assert scores["accuracy"] == 5
        assert scores["empathy"] == 3
        assert scores["efficiency"] == 4
        assert scores["goal_completion"] == 1
        assert "conversation" in result


class TestHelperNodes:
    """Test helper nodes."""

    def test_check_completion_max_turns(self) -> None:
        """Test completion check based on max turns."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            create_check_completion_node,
        )

        node = create_check_completion_node()

        state: EvaluationState = {
            "messages": [],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 10,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)
        assert result["session_complete"] is True

    def test_check_completion_satisfied_user(self) -> None:
        """Test completion check based on user satisfaction."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            create_check_completion_node,
        )

        node = create_check_completion_node()

        state: EvaluationState = {
            "messages": [
                HumanMessage(content="Thanks for your help!"),
            ],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 3,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)
        assert result["session_complete"] is True

    def test_check_completion_not_done(self) -> None:
        """Test completion check when not done."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            create_check_completion_node,
        )

        node = create_check_completion_node()

        state: EvaluationState = {
            "messages": [
                HumanMessage(content="I still need help"),
            ],
            "conversation": "",
            "evaluator_scores": [],
            "turn_count": 3,
            "max_turns": 10,
            "session_complete": False,
            "final_metrics": {},
        }

        result = node(state)
        assert result["session_complete"] is False

    def test_finalize_node(self) -> None:
        """Test finalize node aggregates metrics."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            create_finalize_node,
        )

        node = create_finalize_node()

        state: EvaluationState = {
            "messages": [],
            "conversation": "",
            "evaluator_scores": [
                {
                    "helpfulness": 4,
                    "accuracy": 5,
                    "empathy": 3,
                    "efficiency": 4,
                    "goal_completion": 1,
                },
                {
                    "helpfulness": 5,
                    "accuracy": 4,
                    "empathy": 4,
                    "efficiency": 5,
                    "goal_completion": 1,
                },
            ],
            "turn_count": 4,
            "max_turns": 10,
            "session_complete": True,
            "final_metrics": {},
        }

        result = node(state)

        metrics = result["final_metrics"]
        assert metrics["helpfulness_avg"] == 4.5
        assert metrics["accuracy_avg"] == 4.5
        assert metrics["empathy_avg"] == 3.5
        assert metrics["efficiency_avg"] == 4.5
        assert metrics["goal_completion_rate"] == 1.0
        assert metrics["num_scores"] == 2


# === Metrics Aggregation Tests ===


class TestAggregateScores:
    """Test metrics aggregation function."""

    def test_aggregate_scores_basic(self) -> None:
        """Test basic score aggregation."""
        from langgraph_ollama_local.patterns.evaluation import aggregate_scores

        scores = [
            {
                "helpfulness": 4,
                "accuracy": 5,
                "empathy": 3,
                "efficiency": 4,
                "goal_completion": 1,
            },
            {
                "helpfulness": 5,
                "accuracy": 4,
                "empathy": 4,
                "efficiency": 5,
                "goal_completion": 1,
            },
        ]

        metrics = aggregate_scores(scores)

        assert metrics["helpfulness_avg"] == 4.5
        assert metrics["accuracy_avg"] == 4.5
        assert metrics["empathy_avg"] == 3.5
        assert metrics["efficiency_avg"] == 4.5
        assert metrics["goal_completion_rate"] == 1.0
        assert metrics["num_scores"] == 2

    def test_aggregate_scores_empty(self) -> None:
        """Test aggregation with no scores."""
        from langgraph_ollama_local.patterns.evaluation import aggregate_scores

        metrics = aggregate_scores([])

        assert metrics["helpfulness_avg"] == 0.0
        assert metrics["accuracy_avg"] == 0.0
        assert metrics["empathy_avg"] == 0.0
        assert metrics["efficiency_avg"] == 0.0
        assert metrics["goal_completion_rate"] == 0.0
        assert metrics["num_scores"] == 0

    def test_aggregate_scores_single(self) -> None:
        """Test aggregation with single score."""
        from langgraph_ollama_local.patterns.evaluation import aggregate_scores

        scores = [
            {
                "helpfulness": 3,
                "accuracy": 4,
                "empathy": 2,
                "efficiency": 5,
                "goal_completion": 0,
            },
        ]

        metrics = aggregate_scores(scores)

        assert metrics["helpfulness_avg"] == 3.0
        assert metrics["accuracy_avg"] == 4.0
        assert metrics["empathy_avg"] == 2.0
        assert metrics["efficiency_avg"] == 5.0
        assert metrics["goal_completion_rate"] == 0.0
        assert metrics["num_scores"] == 1

    def test_aggregate_scores_partial_completion(self) -> None:
        """Test aggregation with partial goal completion."""
        from langgraph_ollama_local.patterns.evaluation import aggregate_scores

        scores = [
            {"helpfulness": 4, "accuracy": 4, "empathy": 3, "efficiency": 4, "goal_completion": 1},
            {"helpfulness": 5, "accuracy": 5, "empathy": 4, "efficiency": 5, "goal_completion": 0},
            {"helpfulness": 3, "accuracy": 3, "empathy": 3, "efficiency": 3, "goal_completion": 1},
        ]

        metrics = aggregate_scores(scores)

        assert metrics["goal_completion_rate"] == pytest.approx(2 / 3, rel=0.01)
        assert metrics["num_scores"] == 3


# === Graph Creation Tests ===


class TestGraphCreation:
    """Test evaluation graph creation."""

    def test_create_evaluation_graph(self) -> None:
        """Test creating evaluation graph."""
        from langgraph_ollama_local.patterns.evaluation import (
            SimulatedUser,
            create_evaluation_graph,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_llm

        def mock_agent(state):
            return {"messages": [AIMessage(content="Test response")]}

        user_config = SimulatedUser(
            persona="Test user",
            goals=["Test"],
            behavior="friendly",
        )

        graph = create_evaluation_graph(
            mock_llm,
            mock_agent,
            user_config,
        )

        assert graph is not None
        # Graph should be compiled
        assert hasattr(graph, "invoke")


# === Integration Tests ===


class TestEvaluationIntegration:
    """Integration tests for full evaluation flow."""

    def test_run_evaluation_session(self) -> None:
        """Test running a complete evaluation session."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            SimulatedUser,
            create_evaluation_graph,
            run_evaluation_session,
        )

        # Mock LLM
        mock_llm = MagicMock()

        # Mock structured output for evaluator
        mock_criteria = MagicMock()
        mock_criteria.helpfulness = 4
        mock_criteria.accuracy = 5
        mock_criteria.empathy = 3
        mock_criteria.efficiency = 4
        mock_criteria.goal_completion = 1
        mock_criteria.reasoning = "Good"

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_criteria
        mock_llm.with_structured_output.return_value = mock_structured

        # Mock user responses
        mock_response = MagicMock()
        mock_response.content = "Thank you!"
        mock_llm.invoke.return_value = mock_response

        # Simple agent
        def agent(state: EvaluationState):
            return {"messages": [AIMessage(content="How can I help?")]}

        user_config = SimulatedUser(
            persona="Test user",
            goals=["Get help"],
            behavior="friendly",
            initial_message="Hello!",
        )

        graph = create_evaluation_graph(mock_llm, agent, user_config, evaluate_every_n_turns=1)

        result = run_evaluation_session(graph, max_turns=3)

        # Verify structure
        assert "final_metrics" in result
        assert "evaluator_scores" in result
        assert "messages" in result
        assert result["turn_count"] > 0

    def test_run_multiple_evaluations(self) -> None:
        """Test running multiple evaluation sessions."""
        from langgraph_ollama_local.patterns.evaluation import (
            EvaluationState,
            SimulatedUser,
            create_evaluation_graph,
            run_multiple_evaluations,
        )

        # Mock LLM
        mock_llm = MagicMock()

        mock_criteria = MagicMock()
        mock_criteria.helpfulness = 4
        mock_criteria.accuracy = 4
        mock_criteria.empathy = 4
        mock_criteria.efficiency = 4
        mock_criteria.goal_completion = 1
        mock_criteria.reasoning = "Good"

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_criteria
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = "Thanks!"
        mock_llm.invoke.return_value = mock_response

        def agent(state: EvaluationState):
            return {"messages": [AIMessage(content="Help!")]}

        user_config = SimulatedUser(
            persona="User",
            goals=["Help"],
            behavior="friendly",
            initial_message="Hi",
        )

        graph = create_evaluation_graph(mock_llm, agent, user_config)

        results = run_multiple_evaluations(graph, num_sessions=3, max_turns=2)

        assert "sessions" in results
        assert "aggregate_metrics" in results
        assert len(results["sessions"]) == 3
        assert "helpfulness_avg" in results["aggregate_metrics"]


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected symbols are exported."""

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        from langgraph_ollama_local.patterns import evaluation

        expected_exports = [
            "EvaluationState",
            "SimulatedUser",
            "EvaluationCriteria",
            "create_simulated_user_node",
            "create_evaluator_node",
            "create_check_completion_node",
            "create_finalize_node",
            "create_evaluation_graph",
            "run_evaluation_session",
            "run_multiple_evaluations",
            "aggregate_scores",
        ]

        for export in expected_exports:
            assert hasattr(evaluation, export), f"Missing export: {export}"
