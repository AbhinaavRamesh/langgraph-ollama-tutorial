"""
Tests for Agent Swarm/Network Pattern.

This module tests the agent swarm pattern implementation including
swarm state, agent configuration, routing, and graph construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    pass


# === State Tests ===


class TestSwarmState:
    """Test swarm state definition."""

    def test_state_imports(self) -> None:
        """Verify swarm state can be imported."""
        from langgraph_ollama_local.patterns.swarm import SwarmState

        state: SwarmState = {
            "messages": [],
            "task": "test task",
            "agents_state": {},
            "shared_context": [],
            "current_agent": "",
            "iteration": 0,
            "max_iterations": 10,
            "final_result": "",
        }
        assert state["task"] == "test task"

    def test_state_has_agents_state(self) -> None:
        """Verify agents_state tracks per-agent data."""
        from langgraph_ollama_local.patterns.swarm import SwarmState

        state: SwarmState = {
            "messages": [],
            "task": "test",
            "agents_state": {
                "agent_a": {"last_output": "work", "work_count": 1},
                "agent_b": {"last_output": "more work", "work_count": 2},
            },
            "shared_context": [],
            "current_agent": "agent_b",
            "iteration": 2,
            "max_iterations": 10,
            "final_result": "",
        }
        assert "agent_a" in state["agents_state"]
        assert state["agents_state"]["agent_b"]["work_count"] == 2


# === Agent Configuration Tests ===


class TestSwarmAgent:
    """Test SwarmAgent configuration model."""

    def test_swarm_agent_creation(self) -> None:
        """Test creating a SwarmAgent."""
        from langgraph_ollama_local.patterns.swarm import SwarmAgent

        agent = SwarmAgent(
            name="researcher",
            system_prompt="Research and gather information",
            connections=["analyst", "writer"],
        )
        assert agent.name == "researcher"
        assert len(agent.connections) == 2
        assert "analyst" in agent.connections

    def test_swarm_agent_no_connections(self) -> None:
        """Test agent with no connections (terminal node)."""
        from langgraph_ollama_local.patterns.swarm import SwarmAgent

        agent = SwarmAgent(
            name="finalizer",
            system_prompt="Finalize output",
            connections=[],
        )
        assert agent.name == "finalizer"
        assert len(agent.connections) == 0

    def test_swarm_agent_with_tools(self) -> None:
        """Test agent with tools."""
        from langgraph_ollama_local.patterns.swarm import SwarmAgent

        mock_tool = MagicMock()
        agent = SwarmAgent(
            name="researcher",
            system_prompt="Research",
            connections=["writer"],
            tools=[mock_tool],
        )
        assert agent.tools is not None
        assert len(agent.tools) == 1


# === Routing Decision Tests ===


class TestSwarmRouting:
    """Test SwarmRouting structured output."""

    def test_routing_decision_creation(self) -> None:
        """Test creating a routing decision."""
        from langgraph_ollama_local.patterns.swarm import SwarmRouting

        decision = SwarmRouting(
            next_agent="analyst",
            reasoning="Need analysis of findings",
            share_context=True,
        )
        assert decision.next_agent == "analyst"
        assert decision.share_context is True

    def test_routing_decision_done(self) -> None:
        """Test routing to DONE."""
        from langgraph_ollama_local.patterns.swarm import SwarmRouting

        decision = SwarmRouting(
            next_agent="DONE",
            reasoning="Task complete",
            share_context=True,
        )
        assert decision.next_agent == "DONE"

    def test_routing_decision_no_share(self) -> None:
        """Test routing without sharing context."""
        from langgraph_ollama_local.patterns.swarm import SwarmRouting

        decision = SwarmRouting(
            next_agent="writer",
            reasoning="Internal work only",
            share_context=False,
        )
        assert decision.share_context is False


# === Node Creation Tests ===


class TestCreateSwarmNode:
    """Test swarm agent node creation."""

    def test_create_swarm_node(self) -> None:
        """Test creating a swarm agent node."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="writer",
            reasoning="Ready to write",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Research complete")

        agent = SwarmAgent(
            name="researcher",
            system_prompt="Research",
            connections=["writer"],
        )

        node = create_swarm_node(mock_llm, agent)
        assert callable(node)

    def test_swarm_node_execution(self) -> None:
        """Test executing a swarm agent node."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            SwarmState,
            create_swarm_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="analyst",
            reasoning="Need analysis",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Gathered information")

        agent = SwarmAgent(
            name="researcher",
            system_prompt="Research",
            connections=["analyst"],
        )

        node = create_swarm_node(mock_llm, agent)

        state: SwarmState = {
            "messages": [],
            "task": "Research AI",
            "agents_state": {},
            "shared_context": [],
            "current_agent": "",
            "iteration": 0,
            "max_iterations": 10,
            "final_result": "",
        }

        result = node(state)

        assert result["current_agent"] == "analyst"
        assert result["iteration"] == 1
        assert len(result["shared_context"]) == 1
        assert result["shared_context"][0]["agent"] == "researcher"

    def test_swarm_node_sees_shared_context(self) -> None:
        """Test that agent node sees shared context."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            SwarmState,
            create_swarm_node,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="DONE",
            reasoning="Complete",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Analysis done")

        agent = SwarmAgent(
            name="analyst",
            system_prompt="Analyze",
            connections=[],
        )

        node = create_swarm_node(mock_llm, agent)

        state: SwarmState = {
            "messages": [],
            "task": "Analyze data",
            "agents_state": {},
            "shared_context": [
                {"agent": "researcher", "content": "Found key insights"},
            ],
            "current_agent": "analyst",
            "iteration": 1,
            "max_iterations": 10,
            "final_result": "",
        }

        result = node(state)

        # Verify LLM was invoked with context
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        # Check that shared context was included in prompt
        human_msg = call_args[1]
        assert "researcher" in human_msg.content


# === Routing Tests ===


class TestRouteSwarm:
    """Test swarm routing function."""

    def test_route_to_agent(self) -> None:
        """Test routing to a named agent."""
        from langgraph_ollama_local.patterns.swarm import route_swarm

        state = {
            "current_agent": "analyst",
            "iteration": 2,
            "max_iterations": 10,
        }

        assert route_swarm(state) == "analyst"

    def test_route_to_done(self) -> None:
        """Test routing when agent says DONE."""
        from langgraph_ollama_local.patterns.swarm import route_swarm

        state = {
            "current_agent": "DONE",
            "iteration": 3,
            "max_iterations": 10,
        }

        assert route_swarm(state) == "aggregate"

    def test_route_max_iterations(self) -> None:
        """Test routing at max iterations."""
        from langgraph_ollama_local.patterns.swarm import route_swarm

        state = {
            "current_agent": "researcher",
            "iteration": 10,
            "max_iterations": 10,
        }

        assert route_swarm(state) == "aggregate"

    def test_route_empty_agent(self) -> None:
        """Test routing with empty current_agent."""
        from langgraph_ollama_local.patterns.swarm import route_swarm

        state = {
            "current_agent": "",
            "iteration": 1,
            "max_iterations": 10,
        }

        assert route_swarm(state) == "aggregate"


# === Aggregate Node Tests ===


class TestAggregateNode:
    """Test aggregate node."""

    def test_aggregate_combines_outputs(self) -> None:
        """Test aggregate combines all agent outputs."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmState,
            create_aggregate_node,
        )

        aggregate = create_aggregate_node()

        state: SwarmState = {
            "messages": [],
            "task": "Test task",
            "agents_state": {},
            "shared_context": [
                {"agent": "researcher", "content": "Research findings", "iteration": 1},
                {"agent": "analyst", "content": "Analysis results", "iteration": 2},
                {"agent": "writer", "content": "Final report", "iteration": 3},
            ],
            "current_agent": "DONE",
            "iteration": 3,
            "max_iterations": 10,
            "final_result": "",
        }

        result = aggregate(state)

        assert "final_result" in result
        assert "Researcher" in result["final_result"]
        assert "Analyst" in result["final_result"]
        assert "Writer" in result["final_result"]

    def test_aggregate_empty_context(self) -> None:
        """Test aggregate with no shared context."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmState,
            create_aggregate_node,
        )

        aggregate = create_aggregate_node()

        state: SwarmState = {
            "messages": [],
            "task": "Test task",
            "agents_state": {},
            "shared_context": [],
            "current_agent": "DONE",
            "iteration": 0,
            "max_iterations": 10,
            "final_result": "",
        }

        result = aggregate(state)

        assert "No work" in result["final_result"]


# === Graph Building Tests ===


class TestCreateSwarmGraph:
    """Test swarm graph creation."""

    def test_create_simple_graph(self) -> None:
        """Test creating a simple swarm graph."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="DONE",
            reasoning="Complete",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Work done")

        agents = [
            SwarmAgent(name="agent_a", system_prompt="Do work", connections=[]),
        ]

        graph = create_swarm_graph(mock_llm, agents)

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_create_graph_with_connections(self) -> None:
        """Test creating graph with agent connections."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="DONE",
            reasoning="Complete",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Work done")

        agents = [
            SwarmAgent(name="agent_a", system_prompt="Work", connections=["agent_b"]),
            SwarmAgent(name="agent_b", system_prompt="More work", connections=[]),
        ]

        graph = create_swarm_graph(mock_llm, agents)

        assert graph is not None

    def test_create_graph_validates_connections(self) -> None:
        """Test that invalid connections raise error."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        mock_llm = MagicMock()

        agents = [
            SwarmAgent(
                name="agent_a",
                system_prompt="Work",
                connections=["nonexistent_agent"],  # Invalid
            ),
        ]

        with pytest.raises(ValueError, match="unknown agent"):
            create_swarm_graph(mock_llm, agents)

    def test_create_graph_empty_agents(self) -> None:
        """Test that empty agents list raises error."""
        from langgraph_ollama_local.patterns.swarm import create_swarm_graph

        mock_llm = MagicMock()

        with pytest.raises(ValueError, match="at least one agent"):
            create_swarm_graph(mock_llm, [])

    def test_create_graph_custom_entry(self) -> None:
        """Test creating graph with custom entry agent."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            next_agent="DONE",
            reasoning="Complete",
            share_context=True,
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = AIMessage(content="Work done")

        agents = [
            SwarmAgent(name="agent_a", system_prompt="Work", connections=["agent_b"]),
            SwarmAgent(name="agent_b", system_prompt="More work", connections=[]),
        ]

        graph = create_swarm_graph(mock_llm, agents, entry_agent="agent_b")

        assert graph is not None

    def test_create_graph_invalid_entry(self) -> None:
        """Test that invalid entry agent raises error."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        mock_llm = MagicMock()

        agents = [
            SwarmAgent(name="agent_a", system_prompt="Work", connections=[]),
        ]

        with pytest.raises(ValueError, match="not found"):
            create_swarm_graph(mock_llm, agents, entry_agent="invalid")


# === Utility Tests ===


class TestBroadcastMessage:
    """Test broadcast_message utility."""

    def test_broadcast_message(self) -> None:
        """Test broadcasting a message to the swarm."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmState,
            broadcast_message,
        )

        state: SwarmState = {
            "messages": [],
            "task": "Test",
            "agents_state": {},
            "shared_context": [],
            "current_agent": "",
            "iteration": 2,
            "max_iterations": 10,
            "final_result": "",
        }

        result = broadcast_message(state, "Important update", "coordinator")

        assert len(result["shared_context"]) == 1
        assert result["shared_context"][0]["agent"] == "coordinator"
        assert result["shared_context"][0]["content"] == "Important update"
        assert result["shared_context"][0]["broadcast"] is True


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_patterns_module_exports(self) -> None:
        """Test patterns module exports swarm functions."""
        from langgraph_ollama_local.patterns import (
            SwarmAgent,
            SwarmState,
            create_swarm_graph,
            run_swarm_task,
        )

        assert SwarmAgent is not None
        assert SwarmState is not None
        assert create_swarm_graph is not None
        assert run_swarm_task is not None

    def test_swarm_module_exports(self) -> None:
        """Test swarm module exports."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            SwarmRouting,
            SwarmState,
            broadcast_message,
            create_aggregate_node,
            create_swarm_graph,
            create_swarm_node,
            route_swarm,
            run_swarm_task,
        )

        # Should not raise
        assert SwarmState is not None
        assert SwarmAgent is not None
        assert SwarmRouting is not None
        assert create_swarm_node is not None
        assert create_aggregate_node is not None
        assert route_swarm is not None
        assert create_swarm_graph is not None
        assert run_swarm_task is not None
        assert broadcast_message is not None


# === Integration Tests ===


class TestSwarmIntegration:
    """Integration tests for complete swarm workflows."""

    def test_simple_swarm_flow(self) -> None:
        """Test a simple two-agent swarm flow."""
        from langgraph_ollama_local.patterns.swarm import (
            SwarmAgent,
            create_swarm_graph,
        )

        # Create mock LLM
        mock_llm = MagicMock()

        # Mock structured output for routing
        mock_structured = MagicMock()

        # First call: agent_a routes to agent_b
        # Second call: agent_b routes to DONE
        routing_responses = [
            MagicMock(next_agent="agent_b", reasoning="Need B", share_context=True),
            MagicMock(next_agent="DONE", reasoning="Complete", share_context=True),
        ]
        mock_structured.invoke.side_effect = routing_responses
        mock_llm.with_structured_output.return_value = mock_structured

        # Mock work output
        mock_llm.invoke.return_value = AIMessage(content="Work output")

        agents = [
            SwarmAgent(name="agent_a", system_prompt="Do A", connections=["agent_b"]),
            SwarmAgent(name="agent_b", system_prompt="Do B", connections=[]),
        ]

        graph = create_swarm_graph(mock_llm, agents)

        # Run the swarm
        initial_state = {
            "messages": [],
            "task": "Test task",
            "agents_state": {},
            "shared_context": [],
            "current_agent": "",
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }

        result = graph.invoke(initial_state)

        # Verify completion
        assert result["iteration"] > 0
        assert result["final_result"] != ""
        assert len(result["shared_context"]) >= 2  # At least two agents contributed
