#!/usr/bin/env python3
"""
Generate Mermaid diagrams as PNG images for tutorial documentation.

This script creates graph visualizations for each tutorial and saves them
to docs/images/. Run this after making changes to graph structures.

Usage:
    python scripts/generate_diagrams.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.documents import Document

# Output directories
DOCS_ROOT = Path(__file__).parent.parent / "docs"
CORE_IMAGES = DOCS_ROOT / "core_patterns" / "images"
RAG_IMAGES = DOCS_ROOT / "rag_patterns" / "images"

# Legacy location for backwards compatibility
DOCS_IMAGES = DOCS_ROOT / "images"

# Create directories
CORE_IMAGES.mkdir(parents=True, exist_ok=True)
RAG_IMAGES.mkdir(parents=True, exist_ok=True)
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


# === RAG Pattern States ===


class BasicRAGState(TypedDict):
    """State for basic RAG tutorial."""

    question: str
    documents: List[Document]
    generation: str


class SelfRAGState(TypedDict):
    """State for Self-RAG tutorial."""

    question: str
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    hallucination_check: bool
    answer_useful: bool
    retry_count: int


class CRAGState(TypedDict):
    """State for Corrective RAG tutorial."""

    question: str
    documents: List[Document]
    relevant_documents: List[Document]
    web_results: List[Document]
    all_documents: List[Document]
    generation: str


class AdaptiveRAGState(TypedDict):
    """State for Adaptive RAG tutorial."""

    question: str
    route: str
    documents: List[Document]
    generation: str


class AgenticRAGState(TypedDict):
    """State for Agentic RAG tutorial."""

    messages: Annotated[list, add_messages]
    documents: List[Document]


class PerplexityState(TypedDict):
    """State for Perplexity-style research assistant."""

    question: str
    local_sources: List[Document]
    web_sources: List[Document]
    all_sources: List[Document]
    generation: str
    follow_up_questions: List[str]


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


# === RAG Pattern Generators ===


def generate_08_basic_rag():
    """Generate Tutorial 08: Basic RAG graph."""

    def retrieve(state: BasicRAGState) -> dict:
        return {"documents": []}

    def generate(state: BasicRAGState) -> dict:
        return {"generation": ""}

    workflow = StateGraph(BasicRAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "08-basic-rag-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_09_self_rag():
    """Generate Tutorial 09: Self-RAG graph."""

    def retrieve(state: SelfRAGState) -> dict:
        return {"documents": []}

    def grade_documents(state: SelfRAGState) -> dict:
        return {"relevant_documents": []}

    def generate(state: SelfRAGState) -> dict:
        return {"generation": ""}

    def check_hallucination(state: SelfRAGState) -> dict:
        return {"hallucination_check": True}

    def check_answer(state: SelfRAGState) -> dict:
        return {"answer_useful": True}

    def route_after_grade(state: SelfRAGState) -> str:
        return "generate"

    def route_after_hallucination(state: SelfRAGState) -> str:
        return "check_answer"

    def route_after_answer(state: SelfRAGState) -> str:
        return "end"

    workflow = StateGraph(SelfRAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("check_answer", check_answer)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "retrieve": "retrieve"},
    )
    workflow.add_edge("generate", "check_hallucination")
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"check_answer": "check_answer", "generate": "generate"},
    )
    workflow.add_conditional_edges(
        "check_answer",
        route_after_answer,
        {"end": END, "retrieve": "retrieve"},
    )
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "09-self-rag-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_10_crag():
    """Generate Tutorial 10: Corrective RAG graph."""

    def retrieve(state: CRAGState) -> dict:
        return {"documents": []}

    def grade_documents(state: CRAGState) -> dict:
        return {"relevant_documents": []}

    def web_search(state: CRAGState) -> dict:
        return {"web_results": []}

    def combine_documents(state: CRAGState) -> dict:
        return {"all_documents": []}

    def generate(state: CRAGState) -> dict:
        return {"generation": ""}

    def route_after_grade(state: CRAGState) -> str:
        return "generate"

    workflow = StateGraph(CRAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("combine_documents", combine_documents)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "combine_documents")
    workflow.add_edge("combine_documents", "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "10-crag-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_11_adaptive_rag():
    """Generate Tutorial 11: Adaptive RAG graph."""

    def route_query(state: AdaptiveRAGState) -> dict:
        return {"route": "vectorstore"}

    def retrieve(state: AdaptiveRAGState) -> dict:
        return {"documents": []}

    def web_search(state: AdaptiveRAGState) -> dict:
        return {"documents": []}

    def generate(state: AdaptiveRAGState) -> dict:
        return {"generation": ""}

    def direct_answer(state: AdaptiveRAGState) -> dict:
        return {"generation": ""}

    def get_route(state: AdaptiveRAGState) -> str:
        return "vectorstore"

    workflow = StateGraph(AdaptiveRAGState)
    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_answer", direct_answer)

    workflow.add_edge(START, "route_query")
    workflow.add_conditional_edges(
        "route_query",
        get_route,
        {
            "vectorstore": "retrieve",
            "websearch": "web_search",
            "direct": "direct_answer",
        },
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_answer", END)
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "11-adaptive-rag-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_12_agentic_rag():
    """Generate Tutorial 12: Agentic RAG graph."""

    def agent(state: AgenticRAGState) -> dict:
        return {"messages": []}

    def tools(state: AgenticRAGState) -> dict:
        return {"messages": []}

    def should_continue(state: AgenticRAGState) -> str:
        return "end"

    workflow = StateGraph(AgenticRAGState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "12-agentic-rag-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def generate_13_perplexity():
    """Generate Tutorial 13: Perplexity-style Research Assistant graph."""

    def gather_sources(state: PerplexityState) -> dict:
        return {"all_sources": []}

    def generate_answer(state: PerplexityState) -> dict:
        return {"generation": ""}

    def generate_followups(state: PerplexityState) -> dict:
        return {"follow_up_questions": []}

    workflow = StateGraph(PerplexityState)
    workflow.add_node("gather_sources", gather_sources)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("generate_followups", generate_followups)

    workflow.add_edge(START, "gather_sources")
    workflow.add_edge("gather_sources", "generate_answer")
    workflow.add_edge("generate_answer", "generate_followups")
    workflow.add_edge("generate_followups", END)
    graph = workflow.compile()

    png_data = graph.get_graph().draw_mermaid_png()
    output_path = RAG_IMAGES / "13-perplexity-graph.png"
    output_path.write_bytes(png_data)
    print(f"Generated: {output_path}")


def main():
    """Generate all tutorial diagrams."""
    print("Generating tutorial diagrams...")
    print(f"Core patterns output: {CORE_IMAGES}")
    print(f"RAG patterns output: {RAG_IMAGES}")
    print(f"Legacy output: {DOCS_IMAGES}")
    print()

    print("=== Core Patterns (01-07) ===")
    generate_01_chatbot()
    generate_02_react()
    generate_03_memory()
    generate_04_human_in_loop()
    generate_05_reflection()
    generate_06_plan_execute()
    generate_07_research()

    print()
    print("=== RAG Patterns (08-13) ===")
    generate_08_basic_rag()
    generate_09_self_rag()
    generate_10_crag()
    generate_11_adaptive_rag()
    generate_12_agentic_rag()
    generate_13_perplexity()

    print()
    print("Done! All diagrams generated.")


if __name__ == "__main__":
    main()
