---
title: Research Assistant
description: Combine all patterns from previous tutorials into a comprehensive research assistant agent that plans, searches, reflects, and synthesizes findings into reports.
prev:
  text: 'Tutorial 06: Plan and Execute'
  link: '/tutorials/core/06-plan-and-execute'
next:
  text: 'Basic RAG'
  link: '/tutorials/rag/08-basic-rag'
---

# Tutorial 07: Research Assistant

This capstone tutorial combines all patterns from previous tutorials into a comprehensive research assistant agent.

## What You'll Learn

- **Multi-step planning**: Breaking research tasks into phases
- **Tool integration**: Search, retrieve, and analyze
- **Reflection**: Self-critique and improve findings
- **Synthesis**: Combine results into coherent reports

## Prerequisites

- Completed [Tutorial 06: Plan and Execute](06-plan-and-execute.md)
- Understanding of all previous patterns

---

## Architecture Overview

The research assistant combines:
- **Tutorial 02**: Tool calling for search and retrieval
- **Tutorial 03**: Memory for tracking research context
- **Tutorial 05**: Reflection for quality improvement
- **Tutorial 06**: Plan-and-execute for structured workflow

### The Research Pipeline

The research assistant orchestrates planning, researching, analyzing, reflecting, and synthesizing into a complete pipeline.

---

## Core Concepts

### 1. Research State

Tracks the full research lifecycle:

```python
from typing import Annotated, List
import operator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]

    # Task
    query: str                               # Research query

    # Planning
    research_plan: List[str]                 # Research steps
    current_step: int                        # Current step index

    # Research
    sources: Annotated[List[dict], operator.add]  # Gathered sources
    findings: Annotated[List[str], operator.add]  # Key findings

    # Reflection
    critique: str                            # Current critique
    gaps: List[str]                          # Identified gaps
    iteration: int                           # Reflection iteration

    # Output
    report: str                              # Final report
```

### 2. Research Tools

```python
from langchain_core.tools import tool
import json

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # In production: use Tavily, SerpAPI, etc.
    return json.dumps([{"title": f"Results for {query}", "snippet": "..."}])

@tool
def fetch_document(url: str) -> str:
    """Fetch content from a URL."""
    # In production: use httpx, requests, etc.
    return f"Document content from {url}..."

@tool
def take_notes(topic: str, notes: str) -> str:
    """Record research notes."""
    return f"Notes recorded: {notes}"
```

### 3. Multi-Phase Workflow

The research assistant uses five distinct phases:

| Phase | Purpose |
|-------|---------|
| **Planner** | Creates structured research steps |
| **Researcher** | Executes steps with tools |
| **Analyzer** | Organizes and interprets findings |
| **Reflector** | Critiques quality, identifies gaps |
| **Synthesizer** | Creates final report |

---

## Building the Research Assistant

### Step 1: Define State

```python
from typing import Annotated, List
import operator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
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
```

### Step 2: Create Nodes

```python
import json
import re
from langchain_core.messages import HumanMessage

MAX_ITERATIONS = 2

def planner_node(state: ResearchState) -> dict:
    response = llm.invoke([
        HumanMessage(content=f"Create 3-5 research steps (JSON array): {state['query']}")
    ])

    try:
        plan = json.loads(re.search(r'\[.*?\]', response.content, re.DOTALL).group())
    except:
        plan = [f"Research: {state['query']}"]

    return {"research_plan": plan, "current_step": 0, "iteration": 0}

def researcher_node(state: ResearchState) -> dict:
    if state["current_step"] >= len(state["research_plan"]):
        return {}

    step = state["research_plan"][state["current_step"]]
    gaps = state.get("gaps", [])

    gap_context = f"\nAlso address: {gaps}" if gaps else ""
    response = llm_with_tools.invoke([
        HumanMessage(content=f"Research: {step}{gap_context}")
    ])

    # Execute tool calls and gather findings
    findings = [f"{step}: {response.content[:200]}"]

    return {
        "findings": findings,
        "current_step": state["current_step"] + 1,
        "gaps": []  # Clear gaps after addressing
    }

def analyzer_node(state: ResearchState) -> dict:
    findings = state.get("findings", [])

    response = llm.invoke([
        HumanMessage(content=f"Analyze these findings:\n{findings[-5:]}")
    ])

    return {"findings": [f"Analysis: {response.content}"]}

def reflector_node(state: ResearchState) -> dict:
    findings = state.get("findings", [])

    response = llm.invoke([
        HumanMessage(content=f"Critique (say COMPLETE if done):\n{findings[-1]}")
    ])

    # Parse gaps if not complete
    gaps = []
    if "COMPLETE" not in response.content.upper():
        try:
            match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if match:
                gaps = json.loads(match.group())
        except:
            pass

    return {
        "critique": response.content,
        "gaps": gaps,
        "iteration": state["iteration"] + 1
    }

def synthesizer_node(state: ResearchState) -> dict:
    findings = state.get("findings", [])
    query = state["query"]

    response = llm.invoke([
        HumanMessage(content=f"Write a report on '{query}':\n{findings}")
    ])

    return {"report": response.content}
```

### Step 3: Define Routing

```python
def route_after_research(state: ResearchState) -> str:
    if state["current_step"] < len(state["research_plan"]):
        return "researcher"
    return "analyzer"

def route_after_reflection(state: ResearchState) -> str:
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    gaps = state.get("gaps", [])

    if "COMPLETE" in critique.upper() or iteration >= MAX_ITERATIONS:
        return "synthesizer"
    if gaps:
        return "researcher"  # Fill gaps
    return "synthesizer"
```

### Step 4: Build Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("synthesizer", synthesizer_node)

# Add edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges(
    "researcher", route_after_research,
    {"researcher": "researcher", "analyzer": "analyzer"}
)
workflow.add_edge("analyzer", "reflector")
workflow.add_conditional_edges(
    "reflector", route_after_reflection,
    {"researcher": "researcher", "synthesizer": "synthesizer"}
)
workflow.add_edge("synthesizer", END)

graph = workflow.compile()
```

### Step 5: Use It

```python
result = graph.invoke({
    "query": "What is LangGraph?",
    "messages": [],
    "research_plan": [],
    "current_step": 0,
    "sources": [],
    "findings": [],
    "critique": "",
    "gaps": [],
    "iteration": 0,
    "report": ""
})

print(result["report"])
```

---

## Complete Code

```python
import json
import re
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_ollama_local import LocalAgentConfig

# === State ===
class ResearchState(TypedDict):
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

# === LLM & Tools ===
config = LocalAgentConfig()
llm = ChatOllama(model=config.ollama.model, base_url=config.ollama.base_url, temperature=0)

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return json.dumps([{"title": f"Results for {query}", "snippet": f"Info about {query}"}])

tools = [web_search]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# === Nodes ===
def planner(state):
    response = llm.invoke([HumanMessage(content=f"Create 3-5 research steps (JSON array): {state['query']}")])
    try:
        plan = json.loads(re.search(r'\[.*?\]', response.content, re.DOTALL).group())
    except:
        plan = [f"Research: {state['query']}"]
    return {"research_plan": plan, "current_step": 0, "iteration": 0}

def researcher(state):
    if state["current_step"] >= len(state["research_plan"]):
        return {}
    step = state["research_plan"][state["current_step"]]
    response = llm_with_tools.invoke([HumanMessage(content=f"Research: {step}")])
    return {"findings": [f"{step}: {response.content[:200]}"], "current_step": state["current_step"] + 1, "gaps": []}

def analyzer(state):
    response = llm.invoke([HumanMessage(content=f"Analyze: {state['findings'][-3:]}")])
    return {"findings": [f"Analysis: {response.content}"]}

def reflector(state):
    response = llm.invoke([HumanMessage(content=f"Critique (say COMPLETE if done): {state['findings'][-1]}")])
    gaps = []
    if "COMPLETE" not in response.content.upper():
        try:
            match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if match:
                gaps = json.loads(match.group())
        except:
            pass
    return {"critique": response.content, "gaps": gaps, "iteration": state["iteration"] + 1}

def synthesizer(state):
    response = llm.invoke([HumanMessage(content=f"Write report on {state['query']}:\n{state['findings']}")])
    return {"report": response.content}

def route_research(state):
    return "researcher" if state["current_step"] < len(state["research_plan"]) else "analyzer"

def route_reflection(state):
    if "COMPLETE" in state.get("critique", "").upper() or state["iteration"] >= 2:
        return "synthesizer"
    return "researcher" if state.get("gaps") else "synthesizer"

# === Graph ===
workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner)
workflow.add_node("researcher", researcher)
workflow.add_node("analyzer", analyzer)
workflow.add_node("reflector", reflector)
workflow.add_node("synthesizer", synthesizer)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges("researcher", route_research, {"researcher": "researcher", "analyzer": "analyzer"})
workflow.add_edge("analyzer", "reflector")
workflow.add_conditional_edges("reflector", route_reflection, {"researcher": "researcher", "synthesizer": "synthesizer"})
workflow.add_edge("synthesizer", END)
graph = workflow.compile()

# === Use ===
result = graph.invoke({
    "query": "What is LangGraph?",
    "messages": [], "research_plan": [], "current_step": 0,
    "sources": [], "findings": [], "critique": "", "gaps": [],
    "iteration": 0, "report": ""
})
print(result["report"])
```

---

## Common Pitfalls

### 1. Empty Findings List

```python
# WRONG - accessing empty list
def analyzer(state):
    response = llm.invoke([HumanMessage(content=f"Analyze: {state['findings'][-1]}")])
    # Error if findings is empty!

# CORRECT - check before accessing
def analyzer(state):
    findings = state.get("findings", [])
    if not findings:
        return {"findings": ["Analysis: No findings to analyze yet"]}

    response = llm.invoke([HumanMessage(content=f"Analyze: {findings[-3:]}")])
    return {"findings": [f"Analysis: {response.content}"]}
```

### 2. Infinite Reflection Loop

```python
# WRONG - gap detection always finds something
def reflector(state):
    response = llm.invoke([...])
    # LLM always suggests improvements = infinite loop

# CORRECT - hard limits and explicit completion
MAX_ITERATIONS = 2

def reflector(state):
    response = llm.invoke([...])

    # Parse gaps
    gaps = []
    if "COMPLETE" not in response.content.upper():
        try:
            match = re.search(r'\[.*?\]', response.content, re.DOTALL)
            if match:
                gaps = json.loads(match.group())
        except:
            pass

    return {
        "critique": response.content,
        "gaps": gaps[:3],  # Limit gaps to prevent explosion
        "iteration": state["iteration"] + 1
    }
```

### 3. Not Clearing Gaps After Addressing

```python
# WRONG - gaps accumulate forever
def researcher(state):
    # Process step...
    return {"findings": [...], "current_step": state["current_step"] + 1}
    # gaps remain!

# CORRECT - clear gaps after addressing
def researcher(state):
    gaps = state.get("gaps", [])
    gap_context = f"\nAlso address: {gaps}" if gaps else ""

    # Process step with gap context...
    return {
        "findings": [...],
        "current_step": state["current_step"] + 1,
        "gaps": []  # Clear after addressing
    }
```

---

## Quiz

Test your understanding of the research assistant:

<Quiz
  question="Which patterns from previous tutorials are combined in the Research Assistant?"
  tutorial-id="07-research-assistant"
  :options="[
    { text: 'Tool calling and memory only', correct: false },
    { text: 'Planning and reflection only', correct: false },
    { text: 'Tool calling, memory, planning, reflection, and synthesis', correct: true },
    { text: 'Memory and human-in-the-loop only', correct: false }
  ]"
  explanation="The Research Assistant is a capstone that combines: tool calling (Tutorial 02) for search/retrieval, memory (Tutorial 03) for tracking context, reflection (Tutorial 05) to identify gaps, plan-and-execute (Tutorial 06) for structured workflow, plus synthesis for the final report."
  :hints="[
    { text: 'This tutorial is meant to bring together concepts from most previous tutorials', penalty: 10 },
    { text: 'Check the Architecture Overview section listing which tutorials are combined', penalty: 15 }
  ]"
/>

<Quiz
  question="What is the purpose of the 'gaps' field in ResearchState?"
  tutorial-id="07-research-assistant"
  :options="[
    { text: 'To track missing data in the database', correct: false },
    { text: 'To identify areas needing more research after reflection', correct: true },
    { text: 'To log errors during execution', correct: false },
    { text: 'To store incomplete tool results', correct: false }
  ]"
  explanation="The 'gaps' field stores areas identified by the reflector node as needing more research. When gaps exist, the routing logic sends execution back to the researcher node to fill them before synthesizing the final report."
  :hints="[
    { text: 'The reflector node evaluates what is missing from the current findings', penalty: 10 },
    { text: 'Look at route_after_reflection - what does it check?', penalty: 15 }
  ]"
/>

<Quiz
  question="Why should gaps be cleared after the researcher processes them?"
  tutorial-id="07-research-assistant"
  :options="[
    { text: 'To save memory and reduce state size', correct: false },
    { text: 'To prevent infinite loops from accumulating gaps', correct: true },
    { text: 'To improve query performance', correct: false },
    { text: 'To satisfy Python type constraints', correct: false }
  ]"
  explanation="Gaps must be cleared after addressing them. If you don't clear gaps, they accumulate across iterations. The reflector might keep adding new gaps, and old ones persist, potentially creating an infinite loop."
  :hints="[
    { text: 'Think about what happens if the same gaps are never marked as resolved', penalty: 10 },
    { text: 'Check the Common Pitfall about Not Clearing Gaps After Addressing', penalty: 15 }
  ]"
/>

<Quiz
  question="The research assistant uses the same LLM for all phases (planner, researcher, analyzer, etc.)."
  tutorial-id="07-research-assistant"
  type="true-false"
  :options="[
    { text: 'True', correct: false },
    { text: 'False', correct: true }
  ]"
  explanation="While the tutorial example uses a single LLM for simplicity, in practice you can use different models for different phases. For example, use a faster model for research and a more capable model for analysis and synthesis."
  :hints="[
    { text: 'The researcher node uses llm_with_tools which has tool binding', penalty: 10 },
    { text: 'Think about optimizing for cost vs quality across different phases', penalty: 15 }
  ]"
/>

<Quiz
  question="What are the five phases/nodes in the Research Assistant workflow?"
  tutorial-id="07-research-assistant"
  :options="[
    { text: 'Input, Process, Validate, Output, Cleanup', correct: false },
    { text: 'Planner, Researcher, Analyzer, Reflector, Synthesizer', correct: true },
    { text: 'Query, Search, Filter, Rank, Display', correct: false },
    { text: 'Start, Execute, Review, Revise, End', correct: false }
  ]"
  explanation="The Research Assistant has five phases: Planner (creates research steps), Researcher (executes steps with tools), Analyzer (interprets findings), Reflector (critiques and identifies gaps), and Synthesizer (creates the final report)."
  :hints="[
    { text: 'Check the Multi-Phase Workflow table that lists each phase and its purpose', penalty: 10 },
    { text: 'Each phase is a distinct node in the StateGraph', penalty: 15 }
  ]"
/>

---

## Congratulations!

You've completed the LangGraph tutorial series! You now know how to build:

1. **Basic chatbots** (Tutorial 01)
2. **Tool-calling ReAct agents** (Tutorial 02)
3. **Persistent memory systems** (Tutorial 03)
4. **Human-in-the-loop workflows** (Tutorial 04)
5. **Self-improving reflection loops** (Tutorial 05)
6. **Plan-and-execute agents** (Tutorial 06)
7. **Full research assistants** (Tutorial 07)

Happy building!
