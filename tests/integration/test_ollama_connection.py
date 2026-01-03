"""
Integration tests for Ollama connection.

These tests require a running Ollama instance and are skipped if unavailable.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph_ollama_local import list_models
from langgraph_ollama_local.config import LocalAgentConfig


# State for ReAct agent tests (must be at module level for type hints)
class AgentState(TypedDict):
    """State for ReAct agent integration tests."""

    messages: Annotated[list, add_messages]


@pytest.mark.integration
class TestOllamaConnection:
    """Integration tests for Ollama connectivity."""

    def test_connection_check(self, ollama_available: bool) -> None:
        """Test basic connection to Ollama."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        import httpx

        config = LocalAgentConfig()
        response = httpx.get(f"{config.ollama.base_url}/api/tags", timeout=5)
        assert response.status_code == 200

    def test_create_chat_client(self, ollama_available: bool) -> None:
        """Test creating a chat client that can connect."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        client = config.create_chat_client()
        assert client is not None

    @pytest.mark.slow
    def test_simple_inference(self, ollama_available: bool) -> None:
        """Test a simple inference call."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        # Use the configured model (from .env or defaults)
        client = config.create_chat_client()

        response = client.invoke("Say 'hello' and nothing else.")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    def test_list_models(self, ollama_available: bool) -> None:
        """Test listing available models."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        models = list_models(host=config.ollama.host, port=config.ollama.port)

        assert isinstance(models, list)
        assert len(models) > 0
        print(f"Available models: {models}")


@pytest.mark.integration
class TestMultiModelInference:
    """Integration tests for multiple models."""

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["gpt-oss:20b", "llama3.1:8b"])
    def test_basic_inference(self, ollama_available: bool, model: str) -> None:
        """Test basic inference with different models."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        models = list_models(host=config.ollama.host, port=config.ollama.port)

        if model not in models:
            pytest.skip(f"Model {model} not available")

        client = config.create_chat_client(model=model)
        response = client.invoke("What is 2+2? Answer with just the number.")

        assert response is not None
        assert hasattr(response, "content")
        assert "4" in response.content
        print(f"Model {model} response: {response.content}")


@pytest.mark.integration
class TestToolCalling:
    """Integration tests for tool calling with different models."""

    @pytest.fixture
    def math_tools(self):
        """Create math tools for testing."""

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together."""
            return a * b

        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

        return [multiply, add]

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["llama3.1:8b"])
    def test_tool_binding(self, ollama_available: bool, model: str, math_tools) -> None:
        """Test that tools can be bound to the model."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        models = list_models(host=config.ollama.host, port=config.ollama.port)

        if model not in models:
            pytest.skip(f"Model {model} not available")

        client = config.create_chat_client(model=model, temperature=0)
        client_with_tools = client.bind_tools(math_tools)

        assert client_with_tools is not None
        print(f"Tools bound successfully to {model}")

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["llama3.1:8b"])
    def test_tool_calling_multiply(
        self, ollama_available: bool, model: str, math_tools
    ) -> None:
        """Test that the model can call the multiply tool."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        models = list_models(host=config.ollama.host, port=config.ollama.port)

        if model not in models:
            pytest.skip(f"Model {model} not available")

        client = config.create_chat_client(model=model, temperature=0)
        client_with_tools = client.bind_tools(math_tools)

        response = client_with_tools.invoke("What is 7 times 8?")

        # Check if tool was called
        assert hasattr(response, "tool_calls")
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call["name"] == "multiply"
            print(f"Tool call: {tool_call}")
        else:
            # Some models may answer directly
            print(f"Model answered directly: {response.content}")

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["llama3.1:8b"])
    def test_full_react_loop(
        self, ollama_available: bool, model: str, math_tools
    ) -> None:
        """Test a full ReAct loop with tool execution."""
        if not ollama_available:
            pytest.skip("Ollama not available")

        config = LocalAgentConfig()
        models = list_models(host=config.ollama.host, port=config.ollama.port)

        if model not in models:
            pytest.skip(f"Model {model} not available")

        # Setup
        client = config.create_chat_client(model=model, temperature=0)
        llm_with_tools = client.bind_tools(math_tools)
        tools_by_name = {t.name: t for t in math_tools}

        def agent_node(state: AgentState) -> dict:
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        def tool_node(state: AgentState) -> dict:
            outputs = []
            for tc in state["messages"][-1].tool_calls:
                result = tools_by_name[tc["name"]].invoke(tc["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(result),
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )
            return {"messages": outputs}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        graph = workflow.compile()

        # Run
        result = graph.invoke({"messages": [("user", "What is 6 times 9?")]})

        assert "messages" in result
        assert len(result["messages"]) >= 2  # At least user + response

        final_message = result["messages"][-1]
        print(f"Final response: {final_message.content}")

        # Check that 54 appears in the response
        assert "54" in final_message.content or "54" in str(result["messages"])
