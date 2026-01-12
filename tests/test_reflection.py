"""
Tests for Reflection Pattern.

This module tests the reflection pattern implementation including
generator nodes, critic nodes, routing logic, and multi-criteria reflection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    pass


# === State Tests ===


class TestReflectionState:
    """Test reflection state definition and structure."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.patterns.reflection import ReflectionState

        state: ReflectionState = {
            "messages": [],
            "task": "Test task",
            "draft": "",
            "critique": "",
            "iteration": 0,
            "max_iterations": 3,
        }
        assert state["task"] == "Test task"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3

    def test_state_structure(self) -> None:
        """Test state has all required fields."""
        from langgraph_ollama_local.patterns.reflection import ReflectionState

        state: ReflectionState = {
            "messages": [HumanMessage(content="Hello")],
            "task": "Write an essay",
            "draft": "First draft",
            "critique": "Needs improvement",
            "iteration": 1,
            "max_iterations": 3,
        }

        assert len(state["messages"]) == 1
        assert state["draft"] == "First draft"
        assert state["critique"] == "Needs improvement"
        assert state["iteration"] == 1


# === Node Creation Tests ===


class TestGeneratorNode:
    """Test generator node creation and behavior."""

    def test_create_generator_node(self) -> None:
        """Test generator node can be created."""
        from langgraph_ollama_local.patterns.reflection import create_generator_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Generated draft")

        generator = create_generator_node(mock_llm)
        assert generator is not None
        assert callable(generator)

    def test_generator_first_iteration(self) -> None:
        """Test generator creates initial draft on first iteration."""
        from langgraph_ollama_local.patterns.reflection import create_generator_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Initial draft")

        generator = create_generator_node(mock_llm)

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "",
            "critique": "",
            "iteration": 0,
            "max_iterations": 3,
        }

        result = generator(state)

        assert result["draft"] == "Initial draft"
        assert result["iteration"] == 1
        assert len(result["messages"]) == 1
        assert mock_llm.invoke.called

    def test_generator_revision_iteration(self) -> None:
        """Test generator revises based on critique."""
        from langgraph_ollama_local.patterns.reflection import create_generator_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Revised draft")

        generator = create_generator_node(mock_llm)

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Initial draft",
            "critique": "Add more examples",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = generator(state)

        assert result["draft"] == "Revised draft"
        assert result["iteration"] == 2
        # Verify critique was included in prompt
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = str(call_args)
        assert "critique" in prompt_text.lower() or "Add more examples" in prompt_text


class TestCriticNode:
    """Test critic node creation and behavior."""

    def test_create_critic_node(self) -> None:
        """Test critic node can be created."""
        from langgraph_ollama_local.patterns.reflection import create_critic_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Needs improvement")

        critic = create_critic_node(mock_llm)
        assert critic is not None
        assert callable(critic)

    def test_critic_provides_feedback(self) -> None:
        """Test critic provides feedback on draft."""
        from langgraph_ollama_local.patterns.reflection import create_critic_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Add more specific examples and improve clarity"
        )

        critic = create_critic_node(mock_llm)

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "AI is important for the future",
            "critique": "",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = critic(state)

        assert "critique" in result
        assert len(result["critique"]) > 0
        assert len(result["messages"]) == 1
        assert mock_llm.invoke.called

    def test_critic_can_approve(self) -> None:
        """Test critic can approve with APPROVED signal."""
        from langgraph_ollama_local.patterns.reflection import create_critic_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="APPROVED")

        critic = create_critic_node(mock_llm)

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Excellent draft about AI",
            "critique": "",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = critic(state)

        assert result["critique"] == "APPROVED"


class TestMultiCriteriaCritic:
    """Test multi-criteria critic node."""

    def test_create_multi_criteria_critic(self) -> None:
        """Test multi-criteria critic can be created."""
        from langgraph_ollama_local.patterns.reflection import (
            create_multi_criteria_critic_node,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(side_effect=AttributeError)
        mock_llm.invoke.return_value = MagicMock(content="Critique text")

        critic = create_multi_criteria_critic_node(mock_llm, approval_threshold=7)
        assert critic is not None
        assert callable(critic)

    def test_multi_criteria_fallback(self) -> None:
        """Test multi-criteria critic falls back to text if structured output fails."""
        from langgraph_ollama_local.patterns.reflection import (
            create_multi_criteria_critic_node,
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(side_effect=AttributeError)
        mock_llm.invoke.return_value = MagicMock(
            content="Clarity: 8/10\nAccuracy: 7/10\nNeeds revision"
        )

        critic = create_multi_criteria_critic_node(mock_llm)

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft text",
            "critique": "",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = critic(state)

        assert "critique" in result
        assert len(result["critique"]) > 0


# === Routing Tests ===


class TestShouldContinue:
    """Test routing logic for reflection loop."""

    def test_should_continue_if_not_approved(self) -> None:
        """Test routing continues if not approved."""
        from langgraph.graph import END
        from langgraph_ollama_local.patterns.reflection import should_continue

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft",
            "critique": "Needs improvement",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = should_continue(state)
        assert result == "generator"

    def test_should_end_if_approved(self) -> None:
        """Test routing ends if APPROVED."""
        from langgraph.graph import END
        from langgraph_ollama_local.patterns.reflection import should_continue

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft",
            "critique": "APPROVED",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = should_continue(state)
        assert result == END

    def test_should_end_if_approved_lowercase(self) -> None:
        """Test routing ends if approved (case insensitive)."""
        from langgraph.graph import END
        from langgraph_ollama_local.patterns.reflection import should_continue

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft",
            "critique": "This is approved and ready",
            "iteration": 1,
            "max_iterations": 3,
        }

        result = should_continue(state)
        assert result == END

    def test_should_end_if_max_iterations(self) -> None:
        """Test routing ends if max iterations reached."""
        from langgraph.graph import END
        from langgraph_ollama_local.patterns.reflection import should_continue

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft",
            "critique": "Still needs work",
            "iteration": 3,
            "max_iterations": 3,
        }

        result = should_continue(state)
        assert result == END

    def test_should_continue_under_max_iterations(self) -> None:
        """Test routing continues if under max iterations."""
        from langgraph_ollama_local.patterns.reflection import should_continue

        state = {
            "messages": [],
            "task": "Write about AI",
            "draft": "Draft",
            "critique": "Needs work",
            "iteration": 2,
            "max_iterations": 5,
        }

        result = should_continue(state)
        assert result == "generator"


# === Graph Building Tests ===


class TestReflectionGraph:
    """Test reflection graph creation and compilation."""

    def test_create_reflection_graph(self) -> None:
        """Test basic reflection graph can be created."""
        from langgraph_ollama_local.patterns.reflection import create_reflection_graph

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test")

        graph = create_reflection_graph(llm=mock_llm, max_iterations=3)

        assert graph is not None
        # Verify graph has expected structure
        assert hasattr(graph, "invoke")

    def test_create_multi_criteria_graph(self) -> None:
        """Test multi-criteria reflection graph can be created."""
        from langgraph_ollama_local.patterns.reflection import create_reflection_graph

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test")
        mock_llm.with_structured_output = MagicMock(side_effect=AttributeError)

        graph = create_reflection_graph(
            llm=mock_llm,
            max_iterations=4,
            use_multi_criteria=True,
            approval_threshold=8,
        )

        assert graph is not None

    def test_create_multi_model_graph(self) -> None:
        """Test multi-model reflection graph can be created."""
        from langgraph_ollama_local.patterns.reflection import (
            create_multi_model_reflection_graph,
        )

        mock_generator = MagicMock()
        mock_generator.invoke.return_value = MagicMock(content="Generated")

        mock_critic = MagicMock()
        mock_critic.invoke.return_value = MagicMock(content="Critique")

        graph = create_multi_model_reflection_graph(
            generator_llm=mock_generator,
            critic_llm=mock_critic,
            max_iterations=3,
        )

        assert graph is not None


# === Integration Tests ===


class TestReflectionIntegration:
    """Test end-to-end reflection workflows."""

    def test_reflection_workflow_with_approval(self) -> None:
        """Test reflection workflow that gets approved."""
        from langgraph_ollama_local.patterns.reflection import create_reflection_graph

        # Mock LLM that approves on second iteration
        mock_llm = MagicMock()
        responses = [
            MagicMock(content="First draft"),  # Generator iteration 1
            MagicMock(content="Needs improvement"),  # Critic iteration 1
            MagicMock(content="Improved draft"),  # Generator iteration 2
            MagicMock(content="APPROVED"),  # Critic iteration 2
        ]
        mock_llm.invoke.side_effect = responses

        graph = create_reflection_graph(llm=mock_llm, max_iterations=5)

        result = graph.invoke(
            {
                "messages": [],
                "task": "Write about AI",
                "draft": "",
                "critique": "",
                "iteration": 0,
                "max_iterations": 5,
            }
        )

        assert result["iteration"] == 2
        assert "APPROVED" in result["critique"]
        assert result["draft"] == "Improved draft"

    def test_reflection_workflow_max_iterations(self) -> None:
        """Test reflection workflow that reaches max iterations."""
        from langgraph_ollama_local.patterns.reflection import create_reflection_graph

        # Mock LLM that never approves
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Needs more work")

        graph = create_reflection_graph(llm=mock_llm, max_iterations=2)

        result = graph.invoke(
            {
                "messages": [],
                "task": "Write about AI",
                "draft": "",
                "critique": "",
                "iteration": 0,
                "max_iterations": 2,
            }
        )

        assert result["iteration"] == 2
        assert "APPROVED" not in result["critique"]

    def test_run_reflection_task(self) -> None:
        """Test convenience function for running reflection."""
        from langgraph_ollama_local.patterns.reflection import (
            create_reflection_graph,
            run_reflection_task,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="Draft"),
            MagicMock(content="APPROVED"),
        ]

        graph = create_reflection_graph(llm=mock_llm, max_iterations=3)

        result = run_reflection_task(
            graph=graph,
            task="Write a test essay",
            max_iterations=3,
        )

        assert "draft" in result
        assert "critique" in result
        assert result["iteration"] >= 1


# === Structured Output Tests ===


class TestMultiCriteriaFeedback:
    """Test multi-criteria feedback model."""

    def test_multi_criteria_feedback_schema(self) -> None:
        """Test MultiCriteriaFeedback schema validation."""
        from langgraph_ollama_local.patterns.reflection import MultiCriteriaFeedback

        feedback = MultiCriteriaFeedback(
            clarity_score=8,
            accuracy_score=9,
            completeness_score=7,
            overall_feedback="Good work, needs minor improvements",
            approved=False,
        )

        assert feedback.clarity_score == 8
        assert feedback.accuracy_score == 9
        assert feedback.completeness_score == 7
        assert feedback.approved is False

    def test_multi_criteria_feedback_validation(self) -> None:
        """Test MultiCriteriaFeedback score validation."""
        from langgraph_ollama_local.patterns.reflection import MultiCriteriaFeedback

        # Valid scores (1-10)
        feedback = MultiCriteriaFeedback(
            clarity_score=1,
            accuracy_score=10,
            completeness_score=5,
            overall_feedback="Test",
            approved=True,
        )
        assert feedback.clarity_score == 1

        # Invalid scores should raise
        with pytest.raises(Exception):  # Pydantic validation error
            MultiCriteriaFeedback(
                clarity_score=11,  # Out of range
                accuracy_score=5,
                completeness_score=5,
                overall_feedback="Test",
                approved=False,
            )

        with pytest.raises(Exception):  # Pydantic validation error
            MultiCriteriaFeedback(
                clarity_score=5,
                accuracy_score=0,  # Out of range
                completeness_score=5,
                overall_feedback="Test",
                approved=False,
            )


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_state_export(self) -> None:
        """Test ReflectionState is exported."""
        from langgraph_ollama_local.patterns.reflection import ReflectionState

        assert ReflectionState is not None

    def test_node_creator_exports(self) -> None:
        """Test node creator functions are exported."""
        from langgraph_ollama_local.patterns.reflection import (
            create_critic_node,
            create_generator_node,
            create_multi_criteria_critic_node,
        )

        assert create_generator_node is not None
        assert create_critic_node is not None
        assert create_multi_criteria_critic_node is not None

    def test_graph_builder_exports(self) -> None:
        """Test graph builder functions are exported."""
        from langgraph_ollama_local.patterns.reflection import (
            create_multi_model_reflection_graph,
            create_reflection_graph,
        )

        assert create_reflection_graph is not None
        assert create_multi_model_reflection_graph is not None

    def test_convenience_exports(self) -> None:
        """Test convenience functions are exported."""
        from langgraph_ollama_local.patterns.reflection import run_reflection_task

        assert run_reflection_task is not None

    def test_routing_exports(self) -> None:
        """Test routing functions are exported."""
        from langgraph_ollama_local.patterns.reflection import should_continue

        assert should_continue is not None

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        from langgraph_ollama_local.patterns import reflection

        expected_exports = [
            "ReflectionState",
            "MultiCriteriaFeedback",
            "create_generator_node",
            "create_critic_node",
            "create_multi_criteria_critic_node",
            "should_continue",
            "create_reflection_graph",
            "create_multi_model_reflection_graph",
            "run_reflection_task",
        ]

        for export in expected_exports:
            assert export in reflection.__all__, f"{export} not in __all__"
