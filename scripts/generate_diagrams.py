#!/usr/bin/env python3
"""
Generate Mermaid diagrams as PNG images for tutorial documentation.

This script creates graph visualizations for each tutorial and saves them
to docs/images/. Run this after making changes to graph structures.

Usage:
    python scripts/generate_diagrams.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Output directory
DOCS_IMAGES = Path(__file__).parent.parent / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)


class ChatbotState(TypedDict):
    """State for basic chatbot."""

    messages: Annotated[list, add_messages]


class AgentState(TypedDict):
    """State for ReAct agent."""

    messages: Annotated[list, add_messages]


class MemoryState(TypedDict):
    """State for memory/persistence tutorial."""

    messages: Annotated[list, add_messages]


class HumanInLoopState(TypedDict):
    """State for human-in-the-loop tutorial."""

    messages: Annotated[list, add_messages]


class ReflectionState(TypedDict):
    """State for reflection tutorial."""

    messages: Annotated[list, add_messages]
    draft: str
    critique: str
    iteration: int


class PlanExecuteState(TypedDict):
    """State for plan-and-execute tutorial."""

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: int
    results: list[str]


class ResearchState(TypedDict):
    """State for research assistant tutorial."""

    messages: Annotated[list, add_messages]
    query: str
    research_plan: list[str]
    current_step: int
    findings: list[str]
    critique: str
    gaps: list[str]
    iteration: int
    report: str


def generate_01_chatbot():
    """Generate Tutorial 01: Basic Chatbot graph."""

    def chatbot(state: ChatbotState) -> dict:
        return {"messages": []}

    graph_builder = StateGraph(ChatbotState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    # Save PNG
    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "01-chatbot-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_02_react():
    """Generate Tutorial 02: ReAct Agent graph."""

    def agent_node(state: AgentState) -> dict:
        return {"messages": []}

    def tool_node(state: AgentState) -> dict:
        return {"messages": []}

    def should_continue(state: AgentState) -> str:
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

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "02-react-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_03_memory():
    """Generate Tutorial 03: Memory & Persistence graph."""

    def chatbot(state: MemoryState) -> dict:
        return {"messages": []}

    graph_builder = StateGraph(MemoryState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # With checkpointer (shown in docs as conceptual)
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "03-memory-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_04_human_in_loop():
    """Generate Tutorial 04: Human-in-the-Loop graph."""

    def agent_node(state: HumanInLoopState) -> dict:
        return {"messages": []}

    def tool_node(state: HumanInLoopState) -> dict:
        return {"messages": []}

    def should_continue(state: HumanInLoopState) -> str:
        return "end"

    workflow = StateGraph(HumanInLoopState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "04-human-in-loop-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_05_reflection():
    """Generate Tutorial 05: Reflection graph."""

    def generate(state: ReflectionState) -> dict:
        return {"draft": ""}

    def critique(state: ReflectionState) -> dict:
        return {"critique": ""}

    def should_continue(state: ReflectionState) -> str:
        return "end"

    workflow = StateGraph(ReflectionState)
    workflow.add_node("generate", generate)
    workflow.add_node("critique", critique)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_conditional_edges(
        "critique", should_continue, {"generate": "generate", "end": END}
    )
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "05-reflection-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_06_plan_execute():
    """Generate Tutorial 06: Plan-and-Execute graph."""

    def planner(state: PlanExecuteState) -> dict:
        return {"plan": []}

    def executor(state: PlanExecuteState) -> dict:
        return {"results": []}

    def finalizer(state: PlanExecuteState) -> dict:
        return {"results": []}

    def should_continue(state: PlanExecuteState) -> str:
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

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "06-plan-execute-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_07_research():
    """Generate Tutorial 07: Research Assistant graph."""

    def planner(state: ResearchState) -> dict:
        return {"research_plan": []}

    def researcher(state: ResearchState) -> dict:
        return {"findings": []}

    def analyzer(state: ResearchState) -> dict:
        return {"findings": []}

    def reflector(state: ResearchState) -> dict:
        return {"critique": ""}

    def synthesizer(state: ResearchState) -> dict:
        return {"report": ""}

    def route_research(state: ResearchState) -> str:
        return "analyzer"

    def route_reflection(state: ResearchState) -> str:
        return "synthesizer"

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

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = DOCS_IMAGES / "07-research-assistant-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def main():
    """Generate all tutorial diagrams."""
    print("Generating tutorial diagrams...")
    print(f"Output directory: {DOCS_IMAGES}")
    print()

    generate_01_chatbot()
    generate_02_react()
    generate_03_memory()
    generate_04_human_in_loop()
    generate_05_reflection()
    generate_06_plan_execute()
    generate_07_research()

    print()
    print("Done! All diagrams generated.")


if __name__ == "__main__":
    main()
