"""
Extended integration tests for LangGraph patterns.

These tests require a running Ollama server and test real LLM interactions.
Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

import json
import time
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for agent tests."""

    messages: Annotated[list, add_messages]


@pytest.fixture
def ollama_available():
    """Check if Ollama is available."""
    import httpx

    try:
        response = httpx.get("http://192.168.1.114:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    pytest.skip("Ollama server not available")


@pytest.fixture
def llm(ollama_available):
    """Get configured LLM."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model="llama3.1:8b",
        base_url="http://192.168.1.114:11434",
        temperature=0,
    )


# === Streaming Tests ===


@pytest.mark.integration
class TestStreaming:
    """Test streaming functionality."""

    def test_stream_produces_chunks(self, llm) -> None:
        """Test that streaming produces multiple chunks."""
        chunks = []
        for chunk in llm.stream("Say hello in one word"):
            chunks.append(chunk)

        assert len(chunks) >= 1, "Should produce at least one chunk"

    def test_graph_stream(self, llm) -> None:
        """Test streaming through a graph."""

        def chatbot(state: AgentState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile()

        chunks = list(graph.stream({"messages": [HumanMessage(content="Hi")]}))
        assert len(chunks) >= 1


# === Persistence Tests ===


@pytest.mark.integration
class TestPersistence:
    """Test persistence functionality."""

    def test_sqlite_saver(self, llm) -> None:
        """Test SqliteSaver persistence."""
        from langgraph.checkpoint.sqlite import SqliteSaver

        def chatbot(state: AgentState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        # Use in-memory SQLite
        with SqliteSaver.from_conn_string(":memory:") as memory:
            graph = graph_builder.compile(checkpointer=memory)

            config = {"configurable": {"thread_id": "test-1"}}
            result = graph.invoke(
                {"messages": [HumanMessage(content="Hello")]},
                config=config,
            )

            assert len(result["messages"]) >= 2

    def test_state_history(self, llm) -> None:
        """Test that state history is preserved."""
        memory = MemorySaver()

        def chatbot(state: AgentState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "history-test"}}

        # Multiple invocations
        graph.invoke({"messages": [HumanMessage(content="First")]}, config=config)
        graph.invoke({"messages": [HumanMessage(content="Second")]}, config=config)

        # Get history
        history = list(graph.get_state_history(config))
        assert len(history) >= 2, "Should have state history"


# === Error Handling Tests ===


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling."""

    def test_tool_error_handling(self, llm) -> None:
        """Test that tool errors are handled gracefully."""

        @tool
        def failing_tool(x: str) -> str:
            """A tool that always fails."""
            raise ValueError("Tool failed intentionally")

        tools = [failing_tool]
        tools_by_name = {t.name: t for t in tools}

        def tool_node(state: AgentState) -> dict:
            """Execute tools with error handling."""
            outputs = []
            for tc in state["messages"][-1].tool_calls:
                try:
                    result = tools_by_name[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    result = f"Error: {e}"
                outputs.append(
                    ToolMessage(
                        content=json.dumps(result),
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )
            return {"messages": outputs}

        # Simulate a tool call
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "failing_tool", "args": {"x": "test"}, "id": "1"}],
        )

        result = tool_node({"messages": [ai_message]})
        assert "Error" in result["messages"][0].content


# === Performance Tests ===


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance and latency tests."""

    def test_response_latency(self, llm) -> None:
        """Test that response latency is reasonable."""
        start = time.time()
        llm.invoke("Say 'hello' and nothing else")
        elapsed = time.time() - start

        assert elapsed < 30, f"Response took too long: {elapsed:.2f}s"

    def test_multiple_invocations(self, llm) -> None:
        """Test multiple sequential invocations."""
        def chatbot(state: AgentState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile()

        for i in range(3):
            result = graph.invoke({"messages": [HumanMessage(content=f"Count: {i}")]})
            assert len(result["messages"]) >= 2


# === Tool Calling Validation ===


@pytest.mark.integration
class TestToolCallingValidation:
    """Validate tool calling behavior."""

    def test_tool_args_parsing(self, llm) -> None:
        """Test that tool arguments are parsed correctly."""

        @tool
        def calculate(a: int, b: int, operation: str) -> int:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        tools = [calculate]
        llm_with_tools = llm.bind_tools(tools)

        response = llm_with_tools.invoke("Add 5 and 3")

        if hasattr(response, "tool_calls") and response.tool_calls:
            tc = response.tool_calls[0]
            assert tc["name"] == "calculate"
            assert "a" in tc["args"]
            assert "b" in tc["args"]


# === Graph Validation Tests ===


class TestGraphValidation:
    """Test graph structure validation."""

    def test_graph_has_required_nodes(self) -> None:
        """Test that graphs have required structure."""

        def node(state: AgentState) -> dict:
            return {"messages": []}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("test_node", node)
        graph_builder.add_edge(START, "test_node")
        graph_builder.add_edge("test_node", END)
        graph = graph_builder.compile()

        graph_repr = graph.get_graph()
        node_names = [n.name for n in graph_repr.nodes.values()]

        assert "test_node" in node_names
        assert "__start__" in node_names
        assert "__end__" in node_names

    def test_conditional_edges_coverage(self) -> None:
        """Test that conditional edges cover all cases."""

        def node(state: AgentState) -> dict:
            return {"messages": []}

        def router(state: AgentState) -> str:
            return "end"

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("router_node", node)
        graph_builder.add_edge(START, "router_node")
        graph_builder.add_conditional_edges(
            "router_node",
            router,
            {"continue": "router_node", "end": END},
        )
        graph = graph_builder.compile()

        # Should compile without error
        assert graph is not None
