"""
Tests for tutorial notebook patterns.

These tests verify that the code patterns used in tutorials work correctly.
They use mock LLMs for fast, deterministic testing.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# === Tutorial 01: Basic Chatbot ===


class ChatbotState(TypedDict):
    """State for basic chatbot (Tutorial 01)."""

    messages: Annotated[list, add_messages]


class TestTutorial01ChatbotBasics:
    """Tests for Tutorial 01: Basic Chatbot patterns."""

    def test_state_with_add_messages_reducer(self) -> None:
        """Test that add_messages reducer appends messages."""
        # Create a simple graph to test the reducer
        def echo_node(state: ChatbotState) -> dict:
            last_msg = state["messages"][-1]
            return {"messages": [AIMessage(content=f"Echo: {last_msg.content}")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("echo", echo_node)
        graph_builder.add_edge(START, "echo")
        graph_builder.add_edge("echo", END)
        graph = graph_builder.compile()

        # Invoke and check messages are accumulated
        result = graph.invoke({"messages": [HumanMessage(content="Hello")]})

        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "Hello"
        assert result["messages"][1].content == "Echo: Hello"

    def test_basic_graph_structure(self) -> None:
        """Test basic graph with START -> node -> END."""

        def simple_node(state: ChatbotState) -> dict:
            return {"messages": [AIMessage(content="Response")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("chatbot", simple_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile()

        # Should be able to invoke
        result = graph.invoke({"messages": []})
        assert "messages" in result

    def test_graph_visualization(self) -> None:
        """Test that graph can generate visualization."""

        def node(state: ChatbotState) -> dict:
            return {"messages": []}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("chatbot", node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        graph = graph_builder.compile()

        # Should be able to get graph representation
        graph_repr = graph.get_graph()
        assert graph_repr is not None

        # Should have nodes including chatbot
        node_names = [node.name for node in graph_repr.nodes.values()]
        assert "chatbot" in node_names


# === Tutorial 02: Tool Calling ===


class AgentState(TypedDict):
    """State for ReAct agent (Tutorial 02)."""

    messages: Annotated[list, add_messages]


@tool
def mock_multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def mock_add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


class TestTutorial02ToolCalling:
    """Tests for Tutorial 02: Tool Calling patterns."""

    def test_tool_definition(self) -> None:
        """Test that tools are properly defined."""
        assert mock_multiply.name == "mock_multiply"
        assert mock_add.name == "mock_add"
        assert "Multiply" in mock_multiply.description
        assert "Add" in mock_add.description

    def test_tool_execution(self) -> None:
        """Test that tools can be executed."""
        result = mock_multiply.invoke({"a": 5, "b": 3})
        assert result == 15

        result = mock_add.invoke({"a": 10, "b": 20})
        assert result == 30

    def test_tool_node_pattern(self) -> None:
        """Test the tool node execution pattern."""
        tools = [mock_multiply, mock_add]
        tools_by_name = {t.name: t for t in tools}

        def tool_node(state: AgentState) -> dict:
            """Execute tools from last message."""
            outputs = []
            last_message = state["messages"][-1]

            for tool_call in last_message.tool_calls:
                tool = tools_by_name[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

        # Create a mock AI message with tool calls
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "mock_multiply", "args": {"a": 7, "b": 8}, "id": "call_123"}
            ],
        )

        result = tool_node({"messages": [ai_message]})

        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "mock_multiply"
        assert json.loads(result["messages"][0].content) == 56

    def test_conditional_routing_pattern(self) -> None:
        """Test the conditional routing logic."""

        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Message with tool calls
        ai_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "test", "args": {}, "id": "1"}],
        )
        assert should_continue({"messages": [ai_with_tools]}) == "tools"

        # Message without tool calls
        ai_without_tools = AIMessage(content="Final answer")
        assert should_continue({"messages": [ai_without_tools]}) == "end"

    def test_react_graph_structure(self) -> None:
        """Test ReAct graph with conditional edges."""

        def agent_node(state: AgentState) -> dict:
            # Return message without tool calls to end
            return {"messages": [AIMessage(content="Done")]}

        def tool_node(state: AgentState) -> dict:
            return {"messages": [ToolMessage(content="result", tool_call_id="1")]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        graph = workflow.compile()

        # Should complete without error
        result = graph.invoke({"messages": [HumanMessage(content="Test")]})
        assert "messages" in result

    def test_multiple_tool_calls(self) -> None:
        """Test handling multiple tool calls in one message."""
        tools = [mock_multiply, mock_add]
        tools_by_name = {t.name: t for t in tools}

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

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "mock_multiply", "args": {"a": 2, "b": 3}, "id": "call_1"},
                {"name": "mock_add", "args": {"a": 10, "b": 5}, "id": "call_2"},
            ],
        )

        result = tool_node({"messages": [ai_message]})

        assert len(result["messages"]) == 2
        assert json.loads(result["messages"][0].content) == 6
        assert json.loads(result["messages"][1].content) == 15


class TestGraphVisualization:
    """Tests for graph visualization functionality."""

    def test_mermaid_generation(self) -> None:
        """Test that graphs can generate mermaid diagrams."""

        def node(state: ChatbotState) -> dict:
            return {"messages": []}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("test", node)
        graph_builder.add_edge(START, "test")
        graph_builder.add_edge("test", END)
        graph = graph_builder.compile()

        # Should be able to get mermaid string
        mermaid = graph.get_graph().draw_mermaid()
        assert "test" in mermaid
        assert "START" in mermaid or "__start__" in mermaid


# === Tutorial 03: Memory & Persistence ===


class TestTutorial03MemoryPersistence:
    """Tests for Tutorial 03: Memory & Persistence patterns."""

    def test_memory_saver_basic(self) -> None:
        """Test MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        def chatbot(state: ChatbotState) -> dict:
            return {"messages": [AIMessage(content="Hello!")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        # First invocation
        config = {"configurable": {"thread_id": "test-thread-1"}}
        result = graph.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

        assert len(result["messages"]) == 2

    def test_thread_isolation(self) -> None:
        """Test that different threads have separate state."""
        from langgraph.checkpoint.memory import MemorySaver

        def counter_node(state: ChatbotState) -> dict:
            count = len(state["messages"])
            return {"messages": [AIMessage(content=f"Count: {count}")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("counter", counter_node)
        graph_builder.add_edge(START, "counter")
        graph_builder.add_edge("counter", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        # Thread 1
        config1 = {"configurable": {"thread_id": "thread-1"}}
        result1 = graph.invoke({"messages": [HumanMessage(content="A")]}, config=config1)

        # Thread 2
        config2 = {"configurable": {"thread_id": "thread-2"}}
        result2 = graph.invoke({"messages": [HumanMessage(content="B")]}, config=config2)

        # Both should have independent counts
        assert "Count: 1" in result1["messages"][-1].content
        assert "Count: 1" in result2["messages"][-1].content

    def test_state_persistence_across_invocations(self) -> None:
        """Test that state persists across multiple invocations."""
        from langgraph.checkpoint.memory import MemorySaver

        def echo(state: ChatbotState) -> dict:
            last_msg = state["messages"][-1]
            return {"messages": [AIMessage(content=f"Echo: {last_msg.content}")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("echo", echo)
        graph_builder.add_edge(START, "echo")
        graph_builder.add_edge("echo", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "persist-test"}}

        # First call
        graph.invoke({"messages": [HumanMessage(content="First")]}, config=config)

        # Second call on same thread
        result = graph.invoke({"messages": [HumanMessage(content="Second")]}, config=config)

        # Should have all 4 messages (2 human + 2 AI)
        assert len(result["messages"]) == 4

    def test_get_state(self) -> None:
        """Test retrieving graph state."""
        from langgraph.checkpoint.memory import MemorySaver

        def node(state: ChatbotState) -> dict:
            return {"messages": [AIMessage(content="Response")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("node", node)
        graph_builder.add_edge(START, "node")
        graph_builder.add_edge("node", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "state-test"}}
        graph.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

        # Get state
        state = graph.get_state(config)
        assert state.values["messages"] is not None
        assert len(state.values["messages"]) == 2


# === Tutorial 04: Human-in-the-Loop ===


class TestTutorial04HumanInTheLoop:
    """Tests for Tutorial 04: Human-in-the-Loop patterns."""

    def test_interrupt_before(self) -> None:
        """Test interrupt_before pauses execution."""
        from langgraph.checkpoint.memory import MemorySaver

        def agent_node(state: AgentState) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "dangerous_action", "args": {}, "id": "1"}],
                    )
                ]
            }

        def tool_node(state: AgentState) -> dict:
            return {"messages": [ToolMessage(content="executed", tool_call_id="1")]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "interrupt-test"}}
        graph.invoke({"messages": [HumanMessage(content="Do something")]}, config=config)

        # Should be paused before tools
        state = graph.get_state(config)
        assert state.next == ("tools",)

    def test_resume_after_interrupt(self) -> None:
        """Test resuming execution after interrupt."""
        from langgraph.checkpoint.memory import MemorySaver

        execution_log = []

        def agent_node(state: AgentState) -> dict:
            execution_log.append("agent")
            if len(state["messages"]) == 1:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "action", "args": {}, "id": "1"}],
                        )
                    ]
                }
            return {"messages": [AIMessage(content="Done")]}

        def tool_node(state: AgentState) -> dict:
            execution_log.append("tools")
            return {"messages": [ToolMessage(content="result", tool_call_id="1")]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

        config = {"configurable": {"thread_id": "resume-test"}}

        # First invocation - should pause
        graph.invoke({"messages": [HumanMessage(content="Start")]}, config=config)
        assert "agent" in execution_log
        assert "tools" not in execution_log

        # Resume
        graph.invoke(None, config=config)
        assert "tools" in execution_log

    def test_state_next_empty_when_complete(self) -> None:
        """Test that state.next is empty when graph completes."""
        from langgraph.checkpoint.memory import MemorySaver

        def node(state: ChatbotState) -> dict:
            return {"messages": [AIMessage(content="Done")]}

        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("node", node)
        graph_builder.add_edge(START, "node")
        graph_builder.add_edge("node", END)

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "complete-test"}}
        graph.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

        state = graph.get_state(config)
        assert state.next == ()  # Empty tuple means complete


# === Tutorial 05: Reflection ===


class ReflectionState(TypedDict):
    """State for reflection pattern."""

    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int


class TestTutorial05Reflection:
    """Tests for Tutorial 05: Reflection patterns."""

    def test_reflection_state(self) -> None:
        """Test reflection state structure."""
        state: ReflectionState = {
            "messages": [],
            "task": "Write a poem",
            "draft": "First draft",
            "critique": "",
            "iteration": 0,
        }
        assert state["task"] == "Write a poem"
        assert state["iteration"] == 0

    def test_generate_critique_loop(self) -> None:
        """Test generate -> critique loop pattern."""
        iterations = []

        def generate(state: ReflectionState) -> dict:
            iteration = state.get("iteration", 0)
            iterations.append(("generate", iteration))
            return {
                "draft": f"Draft v{iteration + 1}",
                "iteration": iteration + 1,
            }

        def critique(state: ReflectionState) -> dict:
            iterations.append(("critique", state["iteration"]))
            if state["iteration"] >= 2:
                return {"critique": "APPROVED"}
            return {"critique": "Needs improvement"}

        def should_continue(state: ReflectionState) -> str:
            if "APPROVED" in state.get("critique", "").upper():
                return "end"
            if state["iteration"] >= 3:
                return "end"
            return "generate"

        workflow = StateGraph(ReflectionState)
        workflow.add_node("generate", generate)
        workflow.add_node("critique", critique)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique", should_continue, {"generate": "generate", "end": END}
        )
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "task": "Test",
                "draft": "",
                "critique": "",
                "iteration": 0,
            }
        )

        # Should have gone through 2 iterations
        assert result["iteration"] == 2
        assert "APPROVED" in result["critique"]

    def test_max_iterations_limit(self) -> None:
        """Test that max iterations prevents infinite loops."""
        MAX_ITERATIONS = 3

        def generate(state: ReflectionState) -> dict:
            return {
                "draft": "Draft",
                "iteration": state.get("iteration", 0) + 1,
            }

        def critique(state: ReflectionState) -> dict:
            return {"critique": "Needs more work"}  # Never approves

        def should_continue(state: ReflectionState) -> str:
            if "APPROVED" in state.get("critique", "").upper():
                return "end"
            if state["iteration"] >= MAX_ITERATIONS:
                return "end"
            return "generate"

        workflow = StateGraph(ReflectionState)
        workflow.add_node("generate", generate)
        workflow.add_node("critique", critique)
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique", should_continue, {"generate": "generate", "end": END}
        )
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "task": "Test",
                "draft": "",
                "critique": "",
                "iteration": 0,
            }
        )

        # Should stop at max iterations
        assert result["iteration"] == MAX_ITERATIONS


# === Tutorial 06: Plan and Execute ===


class PlanExecuteState(TypedDict):
    """State for plan-and-execute pattern."""

    messages: Annotated[list, add_messages]
    task: str
    plan: List[str]
    current_step: int
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str


class TestTutorial06PlanAndExecute:
    """Tests for Tutorial 06: Plan and Execute patterns."""

    def test_plan_execute_state(self) -> None:
        """Test plan-execute state structure."""
        state: PlanExecuteState = {
            "messages": [],
            "task": "Do something",
            "plan": ["Step 1", "Step 2"],
            "current_step": 0,
            "past_steps": [],
            "response": "",
        }
        assert len(state["plan"]) == 2
        assert state["current_step"] == 0

    def test_past_steps_reducer(self) -> None:
        """Test that past_steps accumulates via operator.add."""

        def executor(state: PlanExecuteState) -> dict:
            step = state["plan"][state["current_step"]]
            return {
                "past_steps": [(step, f"Result of {step}")],
                "current_step": state["current_step"] + 1,
            }

        def should_continue(state: PlanExecuteState) -> str:
            if state["current_step"] < len(state["plan"]):
                return "executor"
            return "end"

        workflow = StateGraph(PlanExecuteState)
        workflow.add_node("executor", executor)
        workflow.add_edge(START, "executor")
        workflow.add_conditional_edges(
            "executor", should_continue, {"executor": "executor", "end": END}
        )
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "task": "Test",
                "plan": ["Step 1", "Step 2", "Step 3"],
                "current_step": 0,
                "past_steps": [],
                "response": "",
            }
        )

        # Should have accumulated all steps
        assert len(result["past_steps"]) == 3
        assert result["past_steps"][0] == ("Step 1", "Result of Step 1")
        assert result["past_steps"][2] == ("Step 3", "Result of Step 3")

    def test_planner_executor_finalizer_pattern(self) -> None:
        """Test the full planner -> executor -> finalizer pattern."""

        def planner(state: PlanExecuteState) -> dict:
            return {
                "plan": ["Research", "Write", "Review"],
                "current_step": 0,
            }

        def executor(state: PlanExecuteState) -> dict:
            if state["current_step"] >= len(state["plan"]):
                return {}
            step = state["plan"][state["current_step"]]
            return {
                "past_steps": [(step, f"Done: {step}")],
                "current_step": state["current_step"] + 1,
            }

        def finalizer(state: PlanExecuteState) -> dict:
            summary = ", ".join([s for s, r in state["past_steps"]])
            return {"response": f"Completed: {summary}"}

        def should_continue(state: PlanExecuteState) -> str:
            if state["current_step"] < len(state["plan"]):
                return "executor"
            return "finalizer"

        workflow = StateGraph(PlanExecuteState)
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("finalizer", finalizer)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor", should_continue, {"executor": "executor", "finalizer": "finalizer"}
        )
        workflow.add_edge("finalizer", END)
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "task": "Test task",
                "plan": [],
                "current_step": 0,
                "past_steps": [],
                "response": "",
            }
        )

        assert "Research" in result["response"]
        assert "Write" in result["response"]
        assert "Review" in result["response"]


# === Tutorial 07: Research Assistant ===


class ResearchState(TypedDict):
    """State for research assistant."""

    messages: Annotated[list, add_messages]
    query: str
    research_plan: List[str]
    current_step: int
    sources: Annotated[List[dict], operator.add]
    findings: Annotated[List[str], operator.add]
    critique: str
    gaps: List[str]
    iteration: int
    report: str


class TestTutorial07ResearchAssistant:
    """Tests for Tutorial 07: Research Assistant patterns."""

    def test_research_state(self) -> None:
        """Test research state structure."""
        state: ResearchState = {
            "messages": [],
            "query": "What is LangGraph?",
            "research_plan": [],
            "current_step": 0,
            "sources": [],
            "findings": [],
            "critique": "",
            "gaps": [],
            "iteration": 0,
            "report": "",
        }
        assert state["query"] == "What is LangGraph?"

    def test_sources_accumulator(self) -> None:
        """Test that sources accumulate via operator.add."""

        def researcher(state: ResearchState) -> dict:
            return {
                "sources": [{"title": f"Source {state['current_step']}", "url": "http://example.com"}],
                "current_step": state["current_step"] + 1,
            }

        def should_continue(state: ResearchState) -> str:
            if state["current_step"] < 3:
                return "researcher"
            return "end"

        workflow = StateGraph(ResearchState)
        workflow.add_node("researcher", researcher)
        workflow.add_edge(START, "researcher")
        workflow.add_conditional_edges(
            "researcher", should_continue, {"researcher": "researcher", "end": END}
        )
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "query": "Test",
                "research_plan": [],
                "current_step": 0,
                "sources": [],
                "findings": [],
                "critique": "",
                "gaps": [],
                "iteration": 0,
                "report": "",
            }
        )

        assert len(result["sources"]) == 3

    def test_reflection_with_gaps(self) -> None:
        """Test reflection identifying gaps and looping back."""
        loop_count = [0]

        def researcher(state: ResearchState) -> dict:
            loop_count[0] += 1
            return {
                "findings": [f"Finding {loop_count[0]}"],
                "gaps": [],  # Clear gaps after addressing
            }

        def reflector(state: ResearchState) -> dict:
            if state["iteration"] == 0:
                return {
                    "critique": "Missing details",
                    "gaps": ["Need more info"],
                    "iteration": state["iteration"] + 1,
                }
            return {
                "critique": "COMPLETE",
                "gaps": [],
                "iteration": state["iteration"] + 1,
            }

        def route_reflection(state: ResearchState) -> str:
            if "COMPLETE" in state.get("critique", "").upper():
                return "end"
            if state.get("gaps"):
                return "researcher"
            return "end"

        workflow = StateGraph(ResearchState)
        workflow.add_node("researcher", researcher)
        workflow.add_node("reflector", reflector)
        workflow.add_edge(START, "researcher")
        workflow.add_edge("researcher", "reflector")
        workflow.add_conditional_edges(
            "reflector", route_reflection, {"researcher": "researcher", "end": END}
        )
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "query": "Test",
                "research_plan": [],
                "current_step": 0,
                "sources": [],
                "findings": [],
                "critique": "",
                "gaps": [],
                "iteration": 0,
                "report": "",
            }
        )

        # Should have looped back once
        assert loop_count[0] == 2
        assert len(result["findings"]) == 2

    def test_full_research_pipeline(self) -> None:
        """Test the complete research pipeline."""

        def planner(state: ResearchState) -> dict:
            return {
                "research_plan": ["Search", "Analyze"],
                "current_step": 0,
                "iteration": 0,
            }

        def researcher(state: ResearchState) -> dict:
            if state["current_step"] >= len(state["research_plan"]):
                return {}
            step = state["research_plan"][state["current_step"]]
            return {
                "findings": [f"Found: {step}"],
                "current_step": state["current_step"] + 1,
                "gaps": [],
            }

        def analyzer(state: ResearchState) -> dict:
            return {"findings": ["Analysis complete"]}

        def reflector(state: ResearchState) -> dict:
            return {
                "critique": "COMPLETE",
                "gaps": [],
                "iteration": state["iteration"] + 1,
            }

        def synthesizer(state: ResearchState) -> dict:
            return {"report": "Final report based on findings"}

        def route_research(state: ResearchState) -> str:
            if state["current_step"] < len(state["research_plan"]):
                return "researcher"
            return "analyzer"

        def route_reflection(state: ResearchState) -> str:
            if "COMPLETE" in state.get("critique", "").upper():
                return "synthesizer"
            return "researcher"

        workflow = StateGraph(ResearchState)
        workflow.add_node("planner", planner)
        workflow.add_node("researcher", researcher)
        workflow.add_node("analyzer", analyzer)
        workflow.add_node("reflector", reflector)
        workflow.add_node("synthesizer", synthesizer)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_conditional_edges(
            "researcher", route_research, {"researcher": "researcher", "analyzer": "analyzer"}
        )
        workflow.add_edge("analyzer", "reflector")
        workflow.add_conditional_edges(
            "reflector", route_reflection, {"researcher": "researcher", "synthesizer": "synthesizer"}
        )
        workflow.add_edge("synthesizer", END)
        graph = workflow.compile()

        result = graph.invoke(
            {
                "messages": [],
                "query": "What is LangGraph?",
                "research_plan": [],
                "current_step": 0,
                "sources": [],
                "findings": [],
                "critique": "",
                "gaps": [],
                "iteration": 0,
                "report": "",
            }
        )

        assert result["report"] == "Final report based on findings"
        assert len(result["findings"]) == 3  # 2 research + 1 analysis
