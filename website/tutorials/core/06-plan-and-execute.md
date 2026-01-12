---
title: Plan and Execute
description: Learn how to build agents that plan before they act, breaking complex tasks into manageable steps with planner, executor, and finalizer nodes.
prev:
  text: 'Tutorial 05: Reflection'
  link: '/tutorials/core/05-reflection'
next:
  text: 'Tutorial 07: Research Assistant'
  link: '/tutorials/core/07-research-assistant'
---

# Tutorial 06: Plan and Execute

This tutorial teaches how to build agents that plan before they act, breaking complex tasks into manageable steps.

## What You'll Learn

- **Planning**: Breaking tasks into actionable steps
- **Execution**: Processing steps sequentially
- **Re-planning**: Adjusting plans based on results
- **Structured outputs**: Using JSON for plans

## Prerequisites

- Completed [Tutorial 05: Reflection](05-reflection.md)
- Understanding of conditional edges

---

## Why Plan and Execute?

ReAct agents (Tutorial 02) decide step-by-step, which works well for simple tasks. But for complex tasks:

- **Multi-step problems** need thinking ahead
- **Resource efficiency** - use a stronger model for planning, faster for execution
- **Visibility** - explicit plans are easier to review and debug

---

## Core Concepts

### 1. State for Plan-and-Execute

```python
from typing import Annotated, List, Tuple
import operator
from typing_extensions import TypedDict

class PlanExecuteState(TypedDict):
    task: str                    # Original task
    plan: List[str]              # List of steps
    current_step: int            # Current step index
    past_steps: Annotated[List[Tuple[str, str]], operator.add]  # (step, result)
    response: str                # Final response
```

The `operator.add` reducer accumulates step results as tuples.

### 2. The Planner Node

Creates a plan from the task:

```python
def planner_node(state: PlanExecuteState) -> dict:
    prompt = f"Break this into 3-5 steps: {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON array from response
    plan = json.loads(response.content)

    return {"plan": plan, "current_step": 0}
```

### 3. The Executor Node

Processes one step at a time:

```python
def executor_node(state: PlanExecuteState) -> dict:
    step = state["plan"][state["current_step"]]

    # Execute this step
    response = llm.invoke([HumanMessage(content=f"Execute: {step}")])

    return {
        "past_steps": [(step, response.content)],  # Accumulates via reducer
        "current_step": state["current_step"] + 1
    }
```

### 4. Routing Logic

```python
def should_continue(state: PlanExecuteState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "executor"  # More steps
    return "finalizer"     # All done
```

---

## Building the Agent

### Step 1: Define State

```python
from typing import Annotated, List, Tuple
import operator
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: List[str]
    current_step: int
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str
```

### Step 2: Create Nodes

```python
import json
import re
from langchain_core.messages import HumanMessage

def planner(state: PlanExecuteState) -> dict:
    prompt = f"Break into 3-5 steps (JSON array): {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON
    try:
        plan = json.loads(re.search(r'\[.*?\]', response.content, re.DOTALL).group())
    except:
        plan = [response.content]

    return {"plan": plan, "current_step": 0}

def executor(state: PlanExecuteState) -> dict:
    if state["current_step"] >= len(state["plan"]):
        return {}

    step = state["plan"][state["current_step"]]
    response = llm.invoke([HumanMessage(content=f"Execute: {step}")])

    return {
        "past_steps": [(step, response.content)],
        "current_step": state["current_step"] + 1
    }

def finalizer(state: PlanExecuteState) -> dict:
    summary = "\n".join([f"{s}: {r}" for s, r in state["past_steps"]])
    prompt = f"Summarize results for '{state['task']}':\n{summary}"
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"response": response.content}
```

### Step 3: Build Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("finalizer", finalizer)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {"executor": "executor", "finalizer": "finalizer"}
)
workflow.add_edge("finalizer", END)

graph = workflow.compile()
```

### Step 4: Use It

```python
result = graph.invoke({
    "task": "Explain 3 benefits of Python",
    "messages": [],
    "plan": [],
    "current_step": 0,
    "past_steps": [],
    "response": ""
})

print(result["response"])
```

---

## Complete Code

```python
import json
import re
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_ollama_local import LocalAgentConfig

# === State ===
class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: List[str]
    current_step: int
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(model=config.ollama.model, base_url=config.ollama.base_url, temperature=0)

# === Nodes ===
def planner(state):
    response = llm.invoke([HumanMessage(content=f"Break into 3-5 steps (JSON array): {state['task']}")])
    try:
        plan = json.loads(re.search(r'\[.*?\]', response.content, re.DOTALL).group())
    except:
        plan = [response.content]
    return {"plan": plan, "current_step": 0}

def executor(state):
    if state["current_step"] >= len(state["plan"]):
        return {}
    step = state["plan"][state["current_step"]]
    response = llm.invoke([HumanMessage(content=f"Execute: {step}")])
    return {"past_steps": [(step, response.content)], "current_step": state["current_step"] + 1}

def finalizer(state):
    summary = "\n".join([f"{s}: {r}" for s, r in state["past_steps"]])
    response = llm.invoke([HumanMessage(content=f"Summarize: {summary}")])
    return {"response": response.content}

def should_continue(state):
    return "executor" if state["current_step"] < len(state["plan"]) else "finalizer"

# === Graph ===
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("finalizer", finalizer)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {"executor": "executor", "finalizer": "finalizer"})
workflow.add_edge("finalizer", END)
graph = workflow.compile()

# === Use ===
result = graph.invoke({"task": "List 3 benefits of Python", "messages": [], "plan": [], "current_step": 0, "past_steps": [], "response": ""})
print(result["response"])
```

---

## Advantages Over ReAct

| Aspect | ReAct | Plan-and-Execute |
|--------|-------|------------------|
| **Planning** | Implicit (step-by-step) | Explicit (upfront plan) |
| **Visibility** | Limited | Full plan visible |
| **Debugging** | Harder | Easier |
| **Model usage** | Same model for all | Can use different models |

---

## Common Pitfalls

### 1. Unparseable Plans

```python
# WRONG - hoping the LLM returns valid JSON
response = llm.invoke([HumanMessage(content="Create a plan")])
plan = json.loads(response.content)  # May fail!

# CORRECT - robust parsing with fallback
def parse_plan(response_content: str) -> list[str]:
    try:
        # Try to find JSON array
        match = re.search(r'\[.*?\]', response_content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines or numbers
    lines = response_content.split('\n')
    steps = [re.sub(r'^\d+[\.\)]\s*', '', line.strip())
             for line in lines if line.strip()]
    return steps if steps else [response_content]
```

### 2. Missing Step Index Bounds

```python
# WRONG - no bounds check
def executor(state):
    step = state["plan"][state["current_step"]]  # May be out of range!

# CORRECT - always check bounds
def executor(state):
    if state["current_step"] >= len(state["plan"]):
        return {}  # No more steps

    step = state["plan"][state["current_step"]]
    ...
```

### 3. Not Accumulating Results

```python
# WRONG - overwriting past_steps
class State(TypedDict):
    past_steps: List[Tuple[str, str]]  # No reducer!

def executor(state):
    return {"past_steps": [(step, result)]}  # Replaces previous!

# CORRECT - use operator.add reducer
class State(TypedDict):
    past_steps: Annotated[List[Tuple[str, str]], operator.add]

def executor(state):
    return {"past_steps": [(step, result)]}  # Appends!
```

---

## Quiz

Test your understanding of plan-and-execute patterns:

<Quiz
  question="What is the main advantage of Plan-and-Execute over ReAct?"
  tutorial-id="06-plan-and-execute"
  :options="[
    { text: 'Faster execution speed', correct: false },
    { text: 'Explicit upfront planning with better visibility', correct: true },
    { text: 'Uses less memory', correct: false },
    { text: 'Requires fewer LLM calls', correct: false }
  ]"
  explanation="Plan-and-Execute creates an explicit plan upfront, making it easier to review, debug, and understand what the agent will do. You can inspect the plan before execution begins. ReAct plans implicitly step-by-step."
  :hints="[
    { text: 'Consider what information is available before vs after execution starts', penalty: 10 },
    { text: 'Check the Advantages Over ReAct table in the tutorial', penalty: 15 }
  ]"
/>

<Quiz
  question="Why is the operator.add reducer used for past_steps in PlanExecuteState?"
  tutorial-id="06-plan-and-execute"
  :options="[
    { text: 'To calculate the sum of step numbers', correct: false },
    { text: 'To append new step results without replacing old ones', correct: true },
    { text: 'To merge duplicate steps together', correct: false },
    { text: 'To sort steps chronologically', correct: false }
  ]"
  explanation="operator.add is a list concatenation reducer that appends new items to the list. Each executor call returns a single (step, result) tuple, and the reducer accumulates all of them without overwriting previous results."
  :hints="[
    { text: 'Think about what happens when you use + with two lists in Python', penalty: 10 },
    { text: 'Check the Common Pitfall about Not Accumulating Results', penalty: 15 }
  ]"
/>

<Quiz
  question="What should the planner return if JSON parsing fails?"
  tutorial-id="06-plan-and-execute"
  :options="[
    { text: 'An empty list', correct: false },
    { text: 'Raise an exception to halt execution', correct: false },
    { text: 'A fallback plan with at least one step', correct: true },
    { text: 'Return None to skip planning entirely', correct: false }
  ]"
  explanation="Robust parsers should have fallback logic. If JSON parsing fails, return a minimal valid plan (like treating the whole response as a single step) rather than failing completely or returning an empty list."
  :hints="[
    { text: 'The agent should continue working even if the LLM output is not perfectly formatted', penalty: 10 },
    { text: 'Look at the try/except block in the planner node', penalty: 15 }
  ]"
/>

<Quiz
  question="The executor node should check if current_step is within bounds before accessing the plan."
  tutorial-id="06-plan-and-execute"
  type="true-false"
  :options="[
    { text: 'True', correct: true },
    { text: 'False', correct: false }
  ]"
  explanation="Always check bounds before accessing list elements. Without a bounds check, accessing plan[current_step] could raise an IndexError if current_step equals or exceeds the plan length. The tutorial shows returning an empty dict when out of bounds."
  :hints="[
    { text: 'What happens in Python when you access a list index that does not exist?', penalty: 10 },
    { text: 'Check the Common Pitfall about Missing Step Index Bounds', penalty: 15 }
  ]"
/>

<Quiz
  question="What are the three main node types in a Plan-and-Execute agent?"
  tutorial-id="06-plan-and-execute"
  :options="[
    { text: 'Planner, Worker, Reporter', correct: false },
    { text: 'Planner, Executor, Finalizer', correct: true },
    { text: 'Scheduler, Runner, Summarizer', correct: false },
    { text: 'Designer, Builder, Reviewer', correct: false }
  ]"
  explanation="Plan-and-Execute uses three node types: Planner (creates the step list), Executor (processes steps one at a time), and Finalizer (synthesizes results into a final response)."
  :hints="[
    { text: 'Look at the workflow.add_node() calls in the Build Graph section', penalty: 10 },
    { text: 'The names describe what each node does: plan, execute, finalize', penalty: 15 }
  ]"
/>

---

## What's Next?

[Tutorial 07: Research Assistant](07-research-assistant.md) - Combine all patterns into a comprehensive research agent that plans, searches, reflects, and synthesizes.
