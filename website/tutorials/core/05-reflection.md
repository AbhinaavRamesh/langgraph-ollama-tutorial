---
title: Reflection
description: Learn how to build self-critiquing agents that iteratively improve their outputs through reflection loops with generate-critique-revise patterns.
prev:
  text: 'Tutorial 04: Human-in-the-Loop'
  link: '/tutorials/core/04-human-in-the-loop'
next:
  text: 'Tutorial 06: Plan and Execute'
  link: '/tutorials/core/06-plan-and-execute'
---

# Tutorial 05: Reflection

This tutorial teaches how to build self-critiquing agents that iteratively improve their outputs through reflection loops.

## What You'll Learn

- **Reflection loops**: Generate → Critique → Revise patterns
- **Self-improvement**: Using LLMs to evaluate their own outputs
- **Iteration control**: When to stop refining
- **Quality enhancement**: Producing better outputs through feedback

## Prerequisites

- Completed [Tutorial 04: Human-in-the-Loop](04-human-in-the-loop.md)
- Understanding of conditional edges

---

## What is Reflection?

Reflection is a pattern where an LLM:
1. **Generates** an initial output
2. **Critiques** its own work
3. **Revises** based on the critique
4. **Repeats** until satisfied

This mirrors how humans improve their work through drafts and revisions.

### Why Use Reflection?

Single-shot LLM outputs are often good but not great. Through reflection:
- Errors get caught and corrected
- Missing information gets added
- Clarity improves with each revision
- Quality approaches human-level editing

### Use Cases

- **Writing**: Essays, reports, documentation
- **Code generation**: Write, review, refactor
- **Analysis**: Initial assessment → deeper analysis → conclusions
- **Problem-solving**: Solution → evaluation → refinement

---

## Core Concepts

### 1. The Reflection Loop

The graph cycles between generation and critique until approved or max iterations reached.

### 2. State for Reflection

We track more than just messages:

```python
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]  # History
    task: str           # Original task
    draft: str          # Current draft
    critique: str       # Latest critique
    iteration: int      # Current iteration
```

### 3. Stopping Conditions

Two common ways to exit the loop:
1. **Approval signal**: Critique says "APPROVED" or "No changes needed"
2. **Max iterations**: Prevent infinite loops (e.g., max 3 iterations)

```python
def should_continue(state):
    if "APPROVED" in state["critique"].upper():
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "generate"
```

---

## Building a Reflection Agent

### Step 1: Define State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int
```

### Step 2: Create Generator Node

```python
from langchain_core.messages import HumanMessage, SystemMessage

GENERATOR_PROMPT = """You are a skilled writer.
If this is the first draft, write a complete response.
If you have critique, revise your draft to address the feedback."""

def generate_node(state: ReflectionState) -> dict:
    iteration = state.get("iteration", 0)

    if iteration == 0:
        prompt = f"Write a response: {state['task']}"
    else:
        prompt = f"Revise based on critique:\nDraft: {state['draft']}\nCritique: {state['critique']}"

    response = llm.invoke([
        SystemMessage(content=GENERATOR_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {
        "draft": response.content,
        "iteration": iteration + 1
    }
```

### Step 3: Create Critique Node

```python
CRITIQUE_PROMPT = """You are a thoughtful editor.
If the draft is excellent, respond with exactly: "APPROVED"
Otherwise, provide specific feedback for improvement."""

def critique_node(state: ReflectionState) -> dict:
    prompt = f"Critique this draft:\n{state['draft']}"

    response = llm.invoke([
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {"critique": response.content}
```

### Step 4: Define Routing

```python
MAX_ITERATIONS = 3

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "generate"
```

### Step 5: Build Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ReflectionState)

workflow.add_node("generate", generate_node)
workflow.add_node("critique", critique_node)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "critique")
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {"generate": "generate", "end": END}
)

graph = workflow.compile()
```

### Step 6: Use It

```python
result = graph.invoke({
    "task": "Explain LangGraph in 2 sentences.",
    "messages": [],
    "draft": "",
    "critique": "",
    "iteration": 0
})

print(result["draft"])  # Final, refined output
```

---

## Complete Code

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_ollama_local import LocalAgentConfig

# === State ===
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0.7,
)

# === Nodes ===
def generate(state: ReflectionState) -> dict:
    iteration = state.get("iteration", 0)
    if iteration == 0:
        prompt = f"Write a response: {state['task']}"
    else:
        prompt = f"Revise based on critique:\nDraft: {state['draft']}\nCritique: {state['critique']}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "iteration": iteration + 1}

def critique(state: ReflectionState) -> dict:
    prompt = f"Critique this (say APPROVED if perfect):\n{state['draft']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": response.content}

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    if state["iteration"] >= 3:
        return "end"
    return "generate"

# === Graph ===
workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate)
workflow.add_node("critique", critique)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "critique")
workflow.add_conditional_edges("critique", should_continue, {"generate": "generate", "end": END})
graph = workflow.compile()

# === Use ===
result = graph.invoke({
    "task": "Explain recursion in 2 sentences.",
    "messages": [],
    "draft": "",
    "critique": "",
    "iteration": 0
})
print(result["draft"])
```

---

## Variations

### Two-Model Reflection

Use a stronger model for critique:

```python
generator = ChatOllama(model="llama3.2:3b")  # Fast
critic = ChatOllama(model="llama3.1:70b")    # Thorough
```

### Structured Feedback

Use JSON for specific improvement areas:

```python
CRITIQUE_PROMPT = """Return JSON with:
{
    "approved": true/false,
    "clarity": "feedback on clarity",
    "accuracy": "feedback on accuracy",
    "completeness": "feedback on completeness"
}"""
```

---

## Common Pitfalls

### 1. Infinite Loops

```python
# WRONG - no termination condition
def should_continue(state):
    return "generate"  # Always continues!

# CORRECT - multiple exit conditions
MAX_ITERATIONS = 3

def should_continue(state):
    # Exit on approval
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    # Exit on max iterations
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    # Continue refining
    return "generate"
```

### 2. Critique Ignoring Instructions

```python
# WRONG - vague critique prompt
"Give feedback on this draft"

# CORRECT - explicit approval signal
CRITIQUE_PROMPT = """You are a thoughtful editor.

Evaluate this draft against these criteria:
1. Clarity - Is the message clear?
2. Accuracy - Are all claims correct?
3. Completeness - Is anything missing?

If the draft meets all criteria, respond with exactly: "APPROVED"
Otherwise, provide specific, actionable feedback for each issue."""
```

### 3. Generator Not Using Critique

```python
# WRONG - ignoring previous critique
def generate(state):
    prompt = f"Write about: {state['task']}"  # No reference to critique
    ...

# CORRECT - incorporate critique
def generate(state):
    if state["iteration"] == 0:
        prompt = f"Write about: {state['task']}"
    else:
        prompt = f"""Revise this draft to address the critique:

Original task: {state['task']}
Current draft: {state['draft']}
Critique to address: {state['critique']}

Produce an improved version that specifically addresses each point in the critique."""
    ...
```

---

## Quiz

Test your understanding of reflection patterns:

<Quiz
  question="What is the primary purpose of a reflection loop in LangGraph?"
  tutorial-id="05-reflection"
  :options="[
    { text: 'To save computation by caching results', correct: false },
    { text: 'To iteratively improve outputs through self-critique', correct: true },
    { text: 'To handle errors and retry failed operations', correct: false },
    { text: 'To parallelize LLM calls for faster execution', correct: false }
  ]"
  explanation="Reflection loops allow an LLM to generate output, critique its own work, and revise based on feedback - iteratively improving quality through multiple drafts, similar to how humans improve their work."
  :hints="[
    { text: 'Think about how writers improve their drafts over multiple revisions', penalty: 10 },
    { text: 'The pattern is Generate -> Critique -> Revise -> Repeat', penalty: 15 }
  ]"
/>

<Quiz
  question="Why is MAX_ITERATIONS critical in reflection loops?"
  tutorial-id="05-reflection"
  :options="[
    { text: 'To limit token usage and reduce costs', correct: false },
    { text: 'To prevent infinite loops', correct: true },
    { text: 'To ensure consistent output quality', correct: false },
    { text: 'To satisfy API rate limits', correct: false }
  ]"
  explanation="MAX_ITERATIONS prevents infinite loops. Without it, if the critique never approves or the LLM keeps finding issues, the loop would run forever consuming resources indefinitely."
  :hints="[
    { text: 'What happens if the critic always finds something to improve?', penalty: 10 },
    { text: 'Check the Common Pitfalls section about infinite loops', penalty: 15 }
  ]"
/>

<Quiz
  question="What are the two common ways to exit a reflection loop?"
  tutorial-id="05-reflection"
  :options="[
    { text: 'Timeout or user cancellation', correct: false },
    { text: 'Error thrown or completion flag', correct: false },
    { text: 'Approval signal or max iterations reached', correct: true },
    { text: 'Quality threshold or resource limit', correct: false }
  ]"
  explanation="Reflection loops typically exit when either: 1) The critique explicitly approves (e.g., contains 'APPROVED'), or 2) The maximum number of iterations is reached to prevent infinite loops."
  :hints="[
    { text: 'One exit condition is based on the critique content, the other is a counter', penalty: 10 },
    { text: 'Look at the should_continue function in the tutorial', penalty: 15 }
  ]"
/>

<Quiz
  question="The generator node should use the critique feedback when producing revisions."
  tutorial-id="05-reflection"
  type="true-false"
  :options="[
    { text: 'True', correct: true },
    { text: 'False', correct: false }
  ]"
  explanation="The generator must incorporate the critique when revising. The tutorial shows checking iteration count - on first iteration, just generate; on subsequent iterations, include both the current draft AND the critique in the prompt so the LLM knows what to fix."
  :hints="[
    { text: 'Review the Common Pitfall about Generator Not Using Critique', penalty: 10 },
    { text: 'If the generator ignores critique, how would it know what to improve?', penalty: 15 }
  ]"
/>

<Quiz
  question="What field in ReflectionState tracks the current version of the output being refined?"
  tutorial-id="05-reflection"
  type="fill-blank"
  :accepted-answers="['draft', 'the draft', 'draft field']"
  explanation="The 'draft' field in ReflectionState holds the current version of the output. It gets updated after each generate node execution and is evaluated by the critique node."
  :hints="[
    { text: 'Look at what the generate node returns and what the critique node reads', penalty: 10 },
    { text: 'This field represents the work-in-progress that gets refined', penalty: 15 }
  ]"
/>

---

## What's Next?

[Tutorial 06: Plan and Execute](06-plan-and-execute.md) - Learn how to break complex tasks into steps, plan before executing, and re-plan based on results.
