"""
Tests for Agent Handoffs Pattern.

This module tests the agent handoff pattern implementation including
handoff tools, agent nodes, routing logic, and graph compilation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    pass


# === State Definition Tests ===


class TestHandoffState:
    """Test handoff state definition and reducers."""

    def test_state_imports(self) -> None:
        """Verify state can be imported."""
        from langgraph_ollama_local.patterns.handoffs import HandoffState

        # Should not raise
        state: HandoffState = {
            "messages": [],
            "task": "test task",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }
        assert state["task"] == "test task"
        assert state["current_agent"] == "sales"


# === Handoff Tool Tests ===


class TestHandoffTool:
    """Test handoff tool creation and usage."""

    def test_create_handoff_tool(self) -> None:
        """Test handoff tool creation."""
        from langgraph_ollama_local.patterns.handoffs import create_handoff_tool

        tool = create_handoff_tool(
            target_agent="support",
            description="Transfer for technical issues",
        )

        assert tool is not None
        assert tool.name == "handoff_to_support"
        assert "technical" in tool.description.lower()

    def test_handoff_tool_invocation(self) -> None:
        """Test handoff tool can be invoked."""
        from langgraph_ollama_local.patterns.handoffs import create_handoff_tool

        tool = create_handoff_tool("billing", "Transfer for payments")

        result = tool.invoke({"reason": "Customer needs invoice help"})

        assert "billing" in result.lower()
        assert "invoice" in result.lower() or "reason" in result.lower()

    def test_multiple_handoff_tools(self) -> None:
        """Test creating multiple handoff tools."""
        from langgraph_ollama_local.patterns.handoffs import create_handoff_tool

        tool1 = create_handoff_tool("support", "Tech issues")
        tool2 = create_handoff_tool("billing", "Payment issues")
        tool3 = create_handoff_tool("sales", "Product questions")

        assert tool1.name == "handoff_to_support"
        assert tool2.name == "handoff_to_billing"
        assert tool3.name == "handoff_to_sales"

        # All should be different tools
        assert tool1.name != tool2.name
        assert tool2.name != tool3.name


# === Agent Node Tests ===


class TestHandoffAgentNode:
    """Test handoff agent node creation and behavior."""

    def test_create_agent_node(self) -> None:
        """Test agent node creation."""
        from langgraph_ollama_local.patterns.handoffs import (
            create_handoff_agent_node,
            create_handoff_tool,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content="I can help with that!",
            tool_calls=[],
        )

        handoff_tool = create_handoff_tool("support", "Tech help")
        agent = create_handoff_agent_node(
            mock_llm,
            "sales",
            "Handle sales inquiries",
            handoff_tools=[handoff_tool],
        )

        assert callable(agent)

    def test_agent_completes_task(self) -> None:
        """Test agent completing task without handoff."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_handoff_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        # No tool calls = agent completed task
        mock_llm.invoke.return_value = AIMessage(
            content="Here's the pricing information",
            tool_calls=[],
        )

        agent = create_handoff_agent_node(
            mock_llm,
            "sales",
            "Handle sales",
            handoff_tools=[],
        )

        state: HandoffState = {
            "messages": [],
            "task": "What's your pricing?",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }

        result = agent(state)

        assert "context" in result
        assert len(result["context"]) == 1
        assert result["context"][0]["agent"] == "sales"
        assert result.get("handoff_target", "") == ""  # No handoff

    def test_agent_initiates_handoff(self) -> None:
        """Test agent initiating handoff via tool call."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_handoff_agent_node,
            create_handoff_tool,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Tool call = agent wants to hand off
        mock_llm.invoke.return_value = AIMessage(
            content="Let me transfer you to support",
            tool_calls=[
                {
                    "name": "handoff_to_support",
                    "args": {"reason": "Technical issue"},
                    "id": "call_123",
                }
            ],
        )

        handoff_tool = create_handoff_tool("support", "Tech help")
        agent = create_handoff_agent_node(
            mock_llm,
            "sales",
            "Handle sales",
            handoff_tools=[handoff_tool],
        )

        state: HandoffState = {
            "messages": [],
            "task": "App is crashing",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }

        result = agent(state)

        assert result["handoff_target"] == "support"
        assert len(result["handoff_history"]) == 1
        assert "sales -> support" in result["handoff_history"][0]

    def test_agent_uses_context(self) -> None:
        """Test agent receives context from previous agents."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_handoff_agent_node,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Based on previous info...",
            tool_calls=[],
        )

        agent = create_handoff_agent_node(
            mock_llm,
            "support",
            "Handle support",
            handoff_tools=[],
        )

        state: HandoffState = {
            "messages": [],
            "task": "Help with account",
            "current_agent": "support",
            "handoff_target": "",
            "context": [
                {"agent": "sales", "work": "Customer needs account help"},
            ],
            "handoff_history": ["sales -> support"],
            "iteration": 1,
            "max_iterations": 5,
            "final_result": "",
        }

        result = agent(state)

        # Verify LLM was called
        mock_llm.invoke.assert_called_once()
        # The call should include the task
        call_messages = mock_llm.invoke.call_args[0][0]
        assert any("Help with account" in str(msg.content) for msg in call_messages)


# === Routing Tests ===


class TestRouteHandoffs:
    """Test handoff routing logic."""

    def test_route_to_target_agent(self) -> None:
        """Test routing to handoff target."""
        from langgraph_ollama_local.patterns.handoffs import route_handoffs

        state = {
            "handoff_target": "support",
            "iteration": 1,
            "max_iterations": 5,
        }

        result = route_handoffs(state)
        assert result == "support"

    def test_route_to_complete_no_handoff(self) -> None:
        """Test routing to complete when no handoff."""
        from langgraph_ollama_local.patterns.handoffs import route_handoffs

        state = {
            "handoff_target": "",
            "iteration": 1,
            "max_iterations": 5,
        }

        result = route_handoffs(state)
        assert result == "complete"

    def test_route_to_complete_max_iterations(self) -> None:
        """Test routing to complete at max iterations."""
        from langgraph_ollama_local.patterns.handoffs import route_handoffs

        state = {
            "handoff_target": "support",  # Would normally route to support
            "iteration": 5,
            "max_iterations": 5,
        }

        result = route_handoffs(state)
        assert result == "complete"


# === Completion Node Tests ===


class TestCompletionNode:
    """Test the completion node."""

    def test_completion_combines_context(self) -> None:
        """Test completion node combines all agent work."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_completion_node,
        )

        complete = create_completion_node()

        state: HandoffState = {
            "messages": [],
            "task": "Help needed",
            "current_agent": "support",
            "handoff_target": "",
            "context": [
                {"agent": "sales", "work": "Transferred to support"},
                {"agent": "support", "work": "Here's how to fix it"},
            ],
            "handoff_history": ["sales -> support"],
            "iteration": 2,
            "max_iterations": 5,
            "final_result": "",
        }

        result = complete(state)

        assert "final_result" in result
        assert "sales" in result["final_result"].lower()
        assert "support" in result["final_result"].lower()

    def test_completion_empty_context(self) -> None:
        """Test completion with no agent work."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_completion_node,
        )

        complete = create_completion_node()

        state: HandoffState = {
            "messages": [],
            "task": "Help needed",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }

        result = complete(state)

        assert "final_result" in result
        assert "no" in result["final_result"].lower()


# === Graph Building Tests ===


class TestHandoffGraph:
    """Test complete handoff graph creation."""

    def test_create_handoff_graph(self) -> None:
        """Test graph creation."""
        from langgraph_ollama_local.patterns.handoffs import (
            create_handoff_graph,
            create_handoff_tool,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content="Response",
            tool_calls=[],
        )

        # Create tools
        h_support = create_handoff_tool("support", "Tech help")
        h_billing = create_handoff_tool("billing", "Payment help")
        h_sales = create_handoff_tool("sales", "Product help")

        # Create graph
        graph = create_handoff_graph(
            mock_llm,
            agents={
                "sales": ("Handle sales", [h_support, h_billing]),
                "support": ("Handle tech", [h_billing, h_sales]),
                "billing": ("Handle billing", [h_sales, h_support]),
            },
            entry_agent="sales",
        )

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_invocation(self) -> None:
        """Test invoking the handoff graph."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_handoff_graph,
            create_handoff_tool,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        # Agent completes task without handoff
        mock_llm.invoke.return_value = AIMessage(
            content="Here's the answer",
            tool_calls=[],
        )

        h_support = create_handoff_tool("support", "Tech")
        h_billing = create_handoff_tool("billing", "Pay")

        graph = create_handoff_graph(
            mock_llm,
            agents={
                "sales": ("Sales", [h_support, h_billing]),
            },
            entry_agent="sales",
        )

        initial_state: HandoffState = {
            "messages": [HumanMessage(content="Question")],
            "task": "Question",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        }

        result = graph.invoke(initial_state)

        assert "final_result" in result
        assert result.get("iteration", 0) >= 0


# === Convenience Function Tests ===


class TestRunHandoffConversation:
    """Test the convenience function for running conversations."""

    def test_run_conversation(self) -> None:
        """Test running a conversation."""
        from langgraph_ollama_local.patterns.handoffs import (
            create_handoff_graph,
            create_handoff_tool,
            run_handoff_conversation,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content="Response",
            tool_calls=[],
        )

        h_support = create_handoff_tool("support", "Tech")

        graph = create_handoff_graph(
            mock_llm,
            agents={"sales": ("Sales", [h_support])},
        )

        result = run_handoff_conversation(
            graph,
            "Help me",
            entry_agent="sales",
            max_iterations=3,
        )

        assert "final_result" in result
        assert "task" in result
        assert result["task"] == "Help me"


# === Integration Tests ===


class TestHandoffIntegration:
    """Integration tests for complete handoff workflows."""

    def test_sales_to_support_handoff(self) -> None:
        """Test a sales -> support handoff scenario."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_handoff_graph,
            create_handoff_tool,
        )

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Sales agent hands off to support
        def sales_response(*args, **kwargs):
            return AIMessage(
                content="Transferring to support",
                tool_calls=[
                    {
                        "name": "handoff_to_support",
                        "args": {"reason": "Technical issue"},
                        "id": "call_1",
                    }
                ],
            )

        # Support agent completes task
        def support_response(*args, **kwargs):
            return AIMessage(content="Here's the fix", tool_calls=[])

        # Mock LLM returns different responses based on call count
        mock_llm.invoke.side_effect = [sales_response(), support_response()]

        h_support = create_handoff_tool("support", "Tech")
        h_sales = create_handoff_tool("sales", "Sales")

        graph = create_handoff_graph(
            mock_llm,
            agents={
                "sales": ("Sales", [h_support]),
                "support": ("Support", [h_sales]),
            },
            entry_agent="sales",
        )

        result = graph.invoke({
            "messages": [],
            "task": "App crashed",
            "current_agent": "sales",
            "handoff_target": "",
            "context": [],
            "handoff_history": [],
            "iteration": 0,
            "max_iterations": 5,
            "final_result": "",
        })

        # Should have work from both agents
        assert len(result["context"]) == 2
        assert result["context"][0]["agent"] == "sales"
        assert result["context"][1]["agent"] == "support"

        # Should have handoff history
        assert len(result["handoff_history"]) == 1
        assert "sales -> support" in result["handoff_history"][0]


# === Module Exports Tests ===


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_patterns_module_exports(self) -> None:
        """Test patterns module exports handoff utilities."""
        from langgraph_ollama_local.patterns.handoffs import (
            HandoffState,
            create_completion_node,
            create_handoff_agent_node,
            create_handoff_graph,
            create_handoff_tool,
            route_handoffs,
            run_handoff_conversation,
            update_current_agent,
        )

        # Should not raise
        assert HandoffState is not None
        assert create_handoff_tool is not None
        assert create_handoff_agent_node is not None
        assert create_completion_node is not None
        assert route_handoffs is not None
        assert update_current_agent is not None
        assert create_handoff_graph is not None
        assert run_handoff_conversation is not None
