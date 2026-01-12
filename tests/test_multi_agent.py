"""
Tests for Multi-Agent Patterns (Phase 4).

This module tests the multi-agent collaboration, hierarchical teams,
and subgraph patterns implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    pass


# === Multi-Agent Collaboration Tests ===


class TestMultiAgentState:
    """Test multi-agent state definition and reducers."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.agents.multi_agent import MultiAgentState

        # Should not raise
        state: MultiAgentState = {
            "messages": [],
            "task": "test task",
            "next_agent": "",
            "agent_outputs": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }
        assert state["task"] == "test task"

    def test_supervisor_decision_schema(self) -> None:
        """Test SupervisorDecision Pydantic model."""
        from langgraph_ollama_local.agents.multi_agent import SupervisorDecision

        decision = SupervisorDecision(
            next_agent="researcher",
            reasoning="Need to gather information first",
        )
        assert decision.next_agent == "researcher"
        assert "information" in decision.reasoning


class TestSupervisorNode:
    """Test supervisor routing logic."""

    def test_create_supervisor_node(self) -> None:
        """Test supervisor node creation."""
        from langgraph_ollama_local.agents.multi_agent import create_supervisor_node

        mock_llm = MagicMock()
        # Mock structured output
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="researcher",
            reasoning="Need research",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        supervisor = create_supervisor_node(mock_llm)
        assert callable(supervisor)

    def test_supervisor_routes_to_agent(self) -> None:
        """Test supervisor routing to an agent."""
        from langgraph_ollama_local.agents.multi_agent import (
            MultiAgentState,
            create_supervisor_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="coder",
            reasoning="Ready to implement",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        supervisor = create_supervisor_node(mock_llm)

        state: MultiAgentState = {
            "messages": [],
            "task": "Build calculator",
            "next_agent": "",
            "agent_outputs": [{"agent": "researcher", "output": "Analysis done"}],
            "iteration": 1,
            "max_iterations": 10,
            "final_result": "",
        }

        result = supervisor(state)

        assert result["next_agent"] == "coder"
        assert result["iteration"] == 2

    def test_supervisor_routes_to_finish(self) -> None:
        """Test supervisor routing to FINISH."""
        from langgraph_ollama_local.agents.multi_agent import (
            MultiAgentState,
            create_supervisor_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="FINISH",
            reasoning="Task complete",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        supervisor = create_supervisor_node(mock_llm)

        state: MultiAgentState = {
            "messages": [],
            "task": "Build calculator",
            "next_agent": "",
            "agent_outputs": [
                {"agent": "researcher", "output": "Analysis"},
                {"agent": "coder", "output": "Code"},
                {"agent": "reviewer", "output": "Approved"},
            ],
            "iteration": 3,
            "max_iterations": 10,
            "final_result": "",
        }

        result = supervisor(state)

        assert result["next_agent"] == "FINISH"


class TestAgentNodes:
    """Test individual agent nodes."""

    def test_create_agent_node(self) -> None:
        """Test agent node creation."""
        from langgraph_ollama_local.agents.multi_agent import create_agent_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Research complete")

        agent = create_agent_node(mock_llm, "researcher")
        assert callable(agent)

    def test_agent_produces_output(self) -> None:
        """Test agent produces structured output."""
        from langgraph_ollama_local.agents.multi_agent import (
            MultiAgentState,
            create_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="I analyzed the requirements.")

        agent = create_agent_node(mock_llm, "researcher")

        state: MultiAgentState = {
            "messages": [],
            "task": "Build calculator",
            "next_agent": "researcher",
            "agent_outputs": [],
            "iteration": 1,
            "max_iterations": 10,
            "final_result": "",
        }

        result = agent(state)

        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0]["agent"] == "researcher"
        assert "analyzed" in result["agent_outputs"][0]["output"]


class TestSynthesizeNode:
    """Test the synthesize node."""

    def test_synthesize_combines_outputs(self) -> None:
        """Test synthesize combines all agent outputs."""
        from langgraph_ollama_local.agents.multi_agent import (
            MultiAgentState,
            synthesize_node,
        )

        state: MultiAgentState = {
            "messages": [],
            "task": "Build calculator",
            "next_agent": "FINISH",
            "agent_outputs": [
                {"agent": "researcher", "output": "Requirements analyzed"},
                {"agent": "coder", "output": "def add(a, b): return a + b"},
                {"agent": "reviewer", "output": "Code looks good"},
            ],
            "iteration": 4,
            "max_iterations": 10,
            "final_result": "",
        }

        result = synthesize_node(state)

        assert "final_result" in result
        assert "Researcher" in result["final_result"]
        assert "Coder" in result["final_result"]
        assert "Reviewer" in result["final_result"]

    def test_synthesize_empty_outputs(self) -> None:
        """Test synthesize handles empty outputs."""
        from langgraph_ollama_local.agents.multi_agent import (
            MultiAgentState,
            synthesize_node,
        )

        state: MultiAgentState = {
            "messages": [],
            "task": "Build calculator",
            "next_agent": "FINISH",
            "agent_outputs": [],
            "iteration": 1,
            "max_iterations": 10,
            "final_result": "",
        }

        result = synthesize_node(state)

        assert "No work" in result["final_result"]


class TestRouteSupervisor:
    """Test routing function."""

    def test_route_to_agent(self) -> None:
        """Test routing to an agent."""
        from langgraph_ollama_local.agents.multi_agent import route_supervisor

        state = {
            "next_agent": "researcher",
            "iteration": 1,
            "max_iterations": 10,
        }

        assert route_supervisor(state) == "researcher"

    def test_route_to_finish(self) -> None:
        """Test routing to synthesize on FINISH."""
        from langgraph_ollama_local.agents.multi_agent import route_supervisor

        state = {
            "next_agent": "FINISH",
            "iteration": 3,
            "max_iterations": 10,
        }

        assert route_supervisor(state) == "synthesize"

    def test_route_max_iterations(self) -> None:
        """Test routing to synthesize at max iterations."""
        from langgraph_ollama_local.agents.multi_agent import route_supervisor

        state = {
            "next_agent": "researcher",  # Would normally go to researcher
            "iteration": 10,
            "max_iterations": 10,
        }

        assert route_supervisor(state) == "synthesize"


class TestMultiAgentGraph:
    """Test complete multi-agent graph."""

    def test_create_graph(self) -> None:
        """Test graph creation."""
        from langgraph_ollama_local.agents.multi_agent import create_multi_agent_graph

        mock_llm = MagicMock()
        # Mock for supervisor
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="FINISH",
            reasoning="Done",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        # Mock for agents
        mock_llm.invoke.return_value = AIMessage(content="Output")

        graph = create_multi_agent_graph(mock_llm)

        assert graph is not None
        # Should be a compiled graph
        assert hasattr(graph, "invoke")


# === Hierarchical Teams Tests ===


class TestTeamState:
    """Test team state definitions."""

    def test_team_state_imports(self) -> None:
        """Verify team state can be imported."""
        from langgraph_ollama_local.agents.hierarchical import TeamState

        state: TeamState = {
            "messages": [],
            "task": "Research task",
            "team_name": "research",
            "next_member": "",
            "member_outputs": [],
            "iteration": 0,
            "max_iterations": 5,
            "team_result": "",
        }
        assert state["team_name"] == "research"

    def test_hierarchical_state_imports(self) -> None:
        """Verify hierarchical state can be imported."""
        from langgraph_ollama_local.agents.hierarchical import HierarchicalState

        state: HierarchicalState = {
            "messages": [],
            "task": "Complex task",
            "active_team": "",
            "team_results": {},
            "iteration": 0,
            "max_iterations": 10,
            "final_result": "",
        }
        assert state["task"] == "Complex task"


class TestTeamGraph:
    """Test team graph creation."""

    def test_create_team_graph(self) -> None:
        """Test team graph creation."""
        from langgraph_ollama_local.agents.hierarchical import create_team_graph

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_member="DONE",
            reasoning="Complete",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Member output")

        graph = create_team_graph(
            mock_llm,
            "research",
            members=[
                ("searcher", "Search for information", None),
                ("analyst", "Analyze findings", None),
            ],
        )

        assert graph is not None
        assert hasattr(graph, "invoke")


class TestHierarchicalGraph:
    """Test hierarchical graph creation."""

    def test_create_hierarchical_graph(self) -> None:
        """Test hierarchical graph creation."""
        from langgraph_ollama_local.agents.hierarchical import (
            create_hierarchical_graph,
            create_team_graph,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_member="DONE",
            reasoning="Complete",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Output")

        # Create teams
        team1 = create_team_graph(mock_llm, "team1", [("member1", "Do work", None)])
        team2 = create_team_graph(mock_llm, "team2", [("member2", "Do work", None)])

        # Create hierarchical graph
        graph = create_hierarchical_graph(mock_llm, {"team1": team1, "team2": team2})

        assert graph is not None
        assert hasattr(graph, "invoke")


# === Subgraph Patterns Tests ===


class TestSubgraphNode:
    """Test subgraph node wrapper."""

    def test_create_subgraph_node(self) -> None:
        """Test subgraph node creation."""
        from langgraph_ollama_local.patterns.subgraphs import create_subgraph_node

        mock_subgraph = MagicMock()
        mock_subgraph.invoke.return_value = {"output": "result"}

        def state_in(parent):
            return {"input": parent["parent_input"]}

        def state_out(sub, parent):
            return {"parent_output": sub["output"]}

        node = create_subgraph_node(mock_subgraph, state_in, state_out)

        assert callable(node)

    def test_subgraph_node_transforms_state(self) -> None:
        """Test state transformation in subgraph node."""
        from langgraph_ollama_local.patterns.subgraphs import create_subgraph_node

        mock_subgraph = MagicMock()
        mock_subgraph.invoke.return_value = {"answer": "42"}

        def state_in(parent):
            return {"query": parent["question"]}

        def state_out(sub, parent):
            return {"response": sub["answer"]}

        node = create_subgraph_node(mock_subgraph, state_in, state_out)

        parent_state = {"question": "What is the answer?"}
        result = node(parent_state)

        # Verify subgraph was called with transformed input
        mock_subgraph.invoke.assert_called_once()
        call_args = mock_subgraph.invoke.call_args[0][0]
        assert call_args["query"] == "What is the answer?"

        # Verify output was transformed
        assert result["response"] == "42"


class TestFieldMappers:
    """Test field mapper utilities."""

    def test_field_mapper_in(self) -> None:
        """Test field_mapper_in creates correct transformer."""
        from langgraph_ollama_local.patterns.subgraphs import field_mapper_in

        mapper = field_mapper_in(
            ("parent_field", "sub_field"),
            ("another", "mapped"),
        )

        parent = {"parent_field": "value1", "another": "value2", "extra": "ignored"}
        result = mapper(parent)

        assert result["sub_field"] == "value1"
        assert result["mapped"] == "value2"
        assert "extra" not in result

    def test_field_mapper_out(self) -> None:
        """Test field_mapper_out creates correct transformer."""
        from langgraph_ollama_local.patterns.subgraphs import field_mapper_out

        mapper = field_mapper_out(
            ("sub_field", "parent_field"),
            ("mapped", "another"),
        )

        sub = {"sub_field": "result1", "mapped": "result2", "extra": "ignored"}
        parent = {"existing": "value"}
        result = mapper(sub, parent)

        assert result["parent_field"] == "result1"
        assert result["another"] == "result2"
        assert "extra" not in result


class TestChainSubgraphs:
    """Test subgraph chaining."""

    def test_chain_subgraphs(self) -> None:
        """Test chaining multiple subgraphs."""
        from langgraph_ollama_local.patterns.subgraphs import chain_subgraphs

        # Create mock subgraphs
        sub1 = MagicMock()
        sub1.invoke.return_value = {"step1_out": "result1"}

        sub2 = MagicMock()
        sub2.invoke.return_value = {"step2_out": "result2"}

        chained = chain_subgraphs([
            (sub1, lambda s: {"in1": s["input"]}, lambda out, s: {"mid": out["step1_out"]}),
            (sub2, lambda s: {"in2": s["mid"]}, lambda out, s: {"output": out["step2_out"]}),
        ])

        state = {"input": "start"}
        result = chained(state)

        assert "output" in result
        assert result["output"] == "result2"


class TestConditionalSubgraph:
    """Test conditional subgraph execution."""

    def test_conditional_true(self) -> None:
        """Test conditional subgraph when condition is True."""
        from langgraph_ollama_local.patterns.subgraphs import conditional_subgraph

        true_sub = MagicMock()
        true_sub.invoke.return_value = {"result": "true_path"}

        false_sub = MagicMock()
        false_sub.invoke.return_value = {"result": "false_path"}

        node = conditional_subgraph(
            lambda s: s.get("flag", False),
            (true_sub, lambda s: s, lambda out, s: out),
            (false_sub, lambda s: s, lambda out, s: out),
        )

        result = node({"flag": True})

        assert result["result"] == "true_path"
        true_sub.invoke.assert_called_once()
        false_sub.invoke.assert_not_called()

    def test_conditional_false(self) -> None:
        """Test conditional subgraph when condition is False."""
        from langgraph_ollama_local.patterns.subgraphs import conditional_subgraph

        true_sub = MagicMock()
        true_sub.invoke.return_value = {"result": "true_path"}

        false_sub = MagicMock()
        false_sub.invoke.return_value = {"result": "false_path"}

        node = conditional_subgraph(
            lambda s: s.get("flag", False),
            (true_sub, lambda s: s, lambda out, s: out),
            (false_sub, lambda s: s, lambda out, s: out),
        )

        result = node({"flag": False})

        assert result["result"] == "false_path"
        false_sub.invoke.assert_called_once()
        true_sub.invoke.assert_not_called()


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_agents_module_exports(self) -> None:
        """Test agents module exports."""
        from langgraph_ollama_local.agents import (
            MultiAgentState,
            SupervisorDecision,
            create_agent_node,
            create_multi_agent_graph,
            create_supervisor_node,
            route_supervisor,
            run_multi_agent_task,
            synthesize_node,
        )

        # Should not raise
        assert MultiAgentState is not None
        assert create_multi_agent_graph is not None

    def test_patterns_module_exports(self) -> None:
        """Test patterns module exports."""
        from langgraph_ollama_local.patterns import (
            create_subgraph_node,
            create_subgraph_node_async,
        )

        assert create_subgraph_node is not None
        assert create_subgraph_node_async is not None

    def test_hierarchical_exports(self) -> None:
        """Test hierarchical module exports."""
        from langgraph_ollama_local.agents.hierarchical import (
            HierarchicalState,
            TeamState,
            create_hierarchical_graph,
            create_team_graph,
            run_hierarchical_task,
        )

        assert TeamState is not None
        assert create_hierarchical_graph is not None
