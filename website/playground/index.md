---
layout: doc
title: Python Playground
description: Interactive browser-based Python sandbox for testing LangGraph concepts
---

<script setup>
const example1 = `from typing import TypedDict, Annotated
from operator import add

# Define a simple state schema
class GraphState(TypedDict):
    messages: Annotated[list[str], add]
    counter: int
    user_input: str

# Create initial state
state = GraphState(
    messages=['System initialized'],
    counter=0,
    user_input='Hello'
)

print('Initial State:')
print(f'Messages: {state["messages"]}')
print(f'Counter: {state["counter"]}')
print(f'User Input: {state["user_input"]}')

# Simulate state update
state['messages'].append('User said: ' + state['user_input'])
state['counter'] += 1

print('\\nUpdated State:')
print(f'Messages: {state["messages"]}')
print(f'Counter: {state["counter"]}')`

const example2 = `from typing import TypedDict

# Define state
class State(TypedDict):
    input: str
    output: str
    step: int

# Define node functions
def node_a(state: State) -> dict:
    print(f'Executing Node A (Step {state["step"]})')
    return {
        'output': state['input'].upper(),
        'step': state['step'] + 1
    }

def node_b(state: State) -> dict:
    print(f'Executing Node B (Step {state["step"]})')
    return {
        'output': state['output'] + ' - processed',
        'step': state['step'] + 1
    }

# Simulate graph execution
state = {'input': 'hello langgraph', 'output': '', 'step': 1}

print('Starting graph execution...\\n')
state = {**state, **node_a(state)}
state = {**state, **node_b(state)}

print(f'\\nFinal Output: {state["output"]}')
print(f'Total Steps: {state["step"] - 1}')`

const example3 = `from typing import TypedDict, Literal

class RouterState(TypedDict):
    message: str
    sentiment: str
    route: str

def analyze_sentiment(state: RouterState) -> dict:
    message = state['message'].lower()
    positive = ['good', 'great', 'excellent', 'happy', 'love']
    negative = ['bad', 'terrible', 'sad', 'hate', 'angry']

    pos = sum(1 for w in positive if w in message)
    neg = sum(1 for w in negative if w in message)

    sentiment = 'positive' if pos > neg else 'negative' if neg > pos else 'neutral'
    print(f'Analyzed: "{message}" -> {sentiment}')
    return {'sentiment': sentiment}

def route_message(state: RouterState) -> str:
    return state['sentiment']

# Test messages
messages = ['I love this!', 'This is terrible', 'It is okay']

for msg in messages:
    print(f'\\n{"=" * 40}')
    state = {'message': msg, 'sentiment': '', 'route': ''}
    state = {**state, **analyze_sentiment(state)}
    route = route_message(state)
    print(f'Routed to: {route.upper()} handler')`
</script>

# Python Playground

An interactive Python sandbox powered by [Pyodide](https://pyodide.org/) — run Python entirely in your browser via WebAssembly.

::: tip Features
- Full Python 3.11 runtime
- Edit and run code instantly
- No setup required
:::

::: warning Limitations
- Cannot call actual LLMs (no Ollama/OpenAI APIs)
- Cannot install system-dependent packages
:::

---

## Example 1: Basic State

Explore how LangGraph's `TypedDict` state works.

<CodePlayground title="Basic State Management" :code="example1" />

**Key Concepts:**
- `TypedDict` defines state structure
- `Annotated[list, add]` merges list updates
- State flows between nodes

---

## Example 2: Graph Nodes

Build a basic graph structure with nodes.

<CodePlayground title="Graph Structure" :code="example2" />

**Key Concepts:**
- Nodes are functions that take and return state
- State updates are merged automatically
- LangGraph manages flow between nodes

---

## Example 3: Conditional Routing

Learn conditional routing based on state.

<CodePlayground title="Conditional Routing" :code="example3" />

**Key Concepts:**
- Routing functions return next node name
- Enables dynamic flow control
- Use `add_conditional_edges()` in LangGraph

---

## Next Steps

Ready for real LLMs? Start with:

- [01. Chatbot Basics](/tutorials/core/01-chatbot-basics)
- [02. Tool Calling](/tutorials/core/02-tool-calling)
- [03. Memory & Persistence](/tutorials/core/03-memory-persistence)

::: info Having Issues?
Pyodide takes a few seconds to initialize. Refresh if needed.
:::
