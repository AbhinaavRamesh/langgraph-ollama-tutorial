---
title: Build a Basic Chatbot
description: Learn the fundamental concepts of LangGraph by building a simple chatbot. Understand StateGraph, State, Nodes, Edges, and Reducers.
---

# Tutorial 01: Build a Basic Chatbot

This tutorial teaches the fundamental concepts of LangGraph by building a simple chatbot. By the end, you'll understand the core abstractions that power all LangGraph applications.

## What You'll Learn

- **StateGraph**: The core abstraction for building LLM applications
- **State**: How data flows through your application
- **Nodes**: Functions that process and update state
- **Edges**: Connections that define execution flow
- **Reducers**: Functions that control how state is updated

## Prerequisites

1. Ollama running locally or on your LAN
2. A model pulled (see below)
3. This package installed (`pip install -e .`)

**Pull a model before running:**
```bash
ollama pull llama3.2:3b
ollama list
```

If using [ollama-local-serve](https://github.com/abhinaavramesh/ollama-local-serve) on a LAN server:
```bash
curl http://your-server:11434/api/pull -d '{"name": "llama3.2:3b"}'

# Or programmatically
from langgraph_ollama_local import ensure_model
ensure_model("llama3.2:3b", host="192.168.1.100")
```

---

## Why LangGraph?

### The Problem with Simple LLM Calls

A basic LLM call is stateless:
```python
response = llm.invoke("What is Python?")
# Next call has no memory of this conversation
```

Real applications need:
- **Conversation memory** across multiple turns
- **Decision making** based on LLM outputs
- **Tool execution** with loops until task completion
- **Error recovery** and retry logic

### LangGraph vs Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Raw LLM calls** | Simple, direct | No state, no control flow |
| **LangChain chains** | Easy composition | Limited branching, no cycles |
| **Custom code** | Full control | Reinvent the wheel, error-prone |
| **LangGraph** | State + cycles + control | Learning curve |

LangGraph is the right choice when you need:
- Loops (agent retries, reflection)
- Conditional branching (tool calling decisions)
- State that persists across steps
- Human-in-the-loop workflows

### When NOT to Use LangGraph

Don't overcomplicate simple use cases:
- Single LLM call with formatting → Use raw LangChain
- Linear pipeline (A → B → C) → Use simple chains
- No state needed → Direct API calls are fine

---

## Core Concepts Deep Dive

### 1. StateGraph Architecture

A `StateGraph` represents your application as a directed graph where:
- **Nodes** are processing functions
- **Edges** define transitions
- **State** flows through, accumulating changes

```
                    ┌─────────────┐
     Input ──────► │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Node A    │ ◄── Processes state
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     END     │
                    └─────────────┘
```

**Key insight**: The graph is compiled once, then invoked many times. Compilation validates structure and optimizes execution.

### 2. State Design Principles

State is a `TypedDict` defining your application's data schema:

```python
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]  # With reducer
    context: str                              # Without reducer (overwrites)
    counter: int                              # Without reducer (overwrites)
```

**Design principles:**

1. **Keep state minimal** - Only include data that needs to flow between nodes
2. **Use reducers for accumulation** - Messages, logs, results that append
3. **Use plain fields for latest value** - Current step, status flags

**Common state patterns:**

```python
# Chat application
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Multi-step task
class TaskState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    steps_completed: Annotated[list, operator.add]
    final_result: str

# Agent with memory
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    scratchpad: str  # Working memory
    tools_used: Annotated[list, operator.add]
```

### 3. Reducers Explained

Reducers control how node outputs merge with existing state.

**Without reducer** (default): New value replaces old
```python
# State: {"count": 5}
# Node returns: {"count": 10}
# Result: {"count": 10}  ← Replaced
```

**With reducer**: Custom merge logic
```python
# State: {"items": [1, 2]}
# Node returns: {"items": [3]}
# With operator.add reducer
# Result: {"items": [1, 2, 3]}  ← Appended
```

**Built-in reducers:**

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `add_messages` | Appends, dedupes by ID | Conversation history |
| `operator.add` | List concatenation | Accumulating results |
| Custom function | Any logic you define | Complex merging |

**Custom reducer example:**
```python
def max_reducer(current: int, update: int) -> int:
    """Keep the maximum value."""
    return max(current, update)

class State(TypedDict):
    high_score: Annotated[int, max_reducer]
```

### 4. Message Types

LangChain messages carry conversation context:

| Type | Purpose | Example |
|------|---------|---------|
| `HumanMessage` | User input | "What is Python?" |
| `AIMessage` | LLM response | "Python is a programming language..." |
| `SystemMessage` | Instructions | "You are a helpful assistant" |
| `ToolMessage` | Tool results | `{"result": 42}` |

**Message creation shortcuts:**
```python
# Tuple shorthand
messages = [
    ("system", "You are helpful."),
    ("user", "Hello!"),
]

# Explicit objects
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello!"),
]
```

**Accessing message properties:**
```python
msg = result["messages"][-1]
print(msg.content)        # The text
print(msg.type)           # "human", "ai", "system", "tool"
print(msg.id)             # Unique identifier
print(msg.additional_kwargs)  # Extra metadata
```

---

## Building the Chatbot

### Step 1: Define State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """Chatbot state with message history."""
    messages: Annotated[list, add_messages]
```

### Step 2: Create the LLM

```python
from langchain_ollama import ChatOllama
from langgraph_ollama_local import LocalAgentConfig

config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0.7,  # Adjust for creativity vs consistency
)
```

**Temperature guide:**
- `0.0` - Deterministic, consistent outputs
- `0.3-0.5` - Balanced, slight variation
- `0.7-1.0` - Creative, diverse outputs
- `>1.0` - Unpredictable, experimental

### Step 3: Define the Node

```python
def chatbot(state: State) -> dict:
    """Generate a response from conversation history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### Step 4: Build and Compile

```python
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

### Step 5: Invoke

```python
result = graph.invoke({
    "messages": [("user", "What is LangGraph?")]
})
print(result["messages"][-1].content)
```

---

## Graph Visualization


---

## Common Pitfalls

### 1. Forgetting the Reducer

```python
# WRONG - messages will be overwritten each invocation
class State(TypedDict):
    messages: list

# CORRECT - messages accumulate
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### 2. Returning Wrong Format

```python
# WRONG - returning message directly
def node(state):
    return llm.invoke(state["messages"])

# CORRECT - return dict with state updates
def node(state):
    return {"messages": [llm.invoke(state["messages"])]}
```

### 3. Modifying State Directly

```python
# WRONG - mutating state
def node(state):
    state["messages"].append(new_msg)
    return state

# CORRECT - return updates only
def node(state):
    return {"messages": [new_msg]}
```

### 4. Missing Edges

```python
# WRONG - no path from START
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)
# Error: No edge from START

# CORRECT - complete path
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

---

## Streaming

LangGraph supports multiple streaming modes:

### Stream Full State Updates
```python
for event in graph.stream({"messages": [("user", "Hi")]}):
    print(event)
# {"chatbot": {"messages": [AIMessage(...)]}}
```

### Stream Values Only
```python
for event in graph.stream(
    {"messages": [("user", "Hi")]},
    stream_mode="values"
):
    print(event["messages"][-1].content)
```

### Stream LLM Tokens (Requires async)
```python
async for event in graph.astream_events(
    {"messages": [("user", "Hi")]},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## Testing Your Graph

### Unit Testing Nodes

```python
def test_chatbot_returns_message():
    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Hello!")

    # Test node in isolation
    result = chatbot({"messages": [HumanMessage(content="Hi")]})

    assert "messages" in result
    assert len(result["messages"]) == 1
```

### Integration Testing

```python
def test_graph_execution():
    result = graph.invoke({
        "messages": [("user", "Say hello")]
    })

    assert len(result["messages"]) == 2
    assert result["messages"][-1].type == "ai"
```

### Snapshot Testing

```python
def test_graph_structure():
    graph_repr = graph.get_graph()
    assert "chatbot" in [n.name for n in graph_repr.nodes.values()]
```

---

## Performance Considerations

### 1. Model Selection

| Model | Speed | Quality | Memory |
|-------|-------|---------|--------|
| `llama3.2:1b` | Fast | Basic | ~2GB |
| `llama3.2:3b` | Medium | Good | ~4GB |
| `llama3.1:8b` | Slower | Better | ~8GB |
| `llama3.1:70b` | Slow | Best | ~40GB |

### 2. Reduce Context Length

Each message adds to the context. For long conversations:
```python
def chatbot(state: State) -> dict:
    # Only use last N messages
    recent = state["messages"][-10:]
    response = llm.invoke(recent)
    return {"messages": [response]}
```

### 3. Connection Pooling

For high-throughput applications, reuse connections:
```python
import httpx

# Create a persistent client
client = httpx.Client(timeout=60.0)
llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    client=client,  # Reuse connection
)
```

---

## Production Checklist

- [ ] **Error handling**: Wrap LLM calls in try/except
- [ ] **Timeouts**: Set reasonable timeouts for LLM calls
- [ ] **Logging**: Log inputs/outputs for debugging
- [ ] **Rate limiting**: Implement if using shared server
- [ ] **Monitoring**: Track latency, errors, token usage
- [ ] **Graceful degradation**: Handle model unavailability

---

## Complete Code

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph_ollama_local import LocalAgentConfig

class State(TypedDict):
    messages: Annotated[list, add_messages]

config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

result = graph.invoke({"messages": [("user", "Hello!")]})
print(result["messages"][-1].content)
```

---

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Messages](https://python.langchain.com/docs/concepts/messages/)
- [Ollama Model Library](https://ollama.com/library)

---

## Quiz

Test your understanding of the concepts covered in this tutorial:

<Quiz
  question="What is the purpose of a reducer in LangGraph state?"
  tutorial-id="01-chatbot-basics"
  :options="[
    { text: 'To delete old state values', correct: false },
    { text: 'To control how node outputs merge with existing state', correct: true },
    { text: 'To compress state for storage', correct: false },
    { text: 'To validate state types', correct: false }
  ]"
  explanation="Reducers define how new values from nodes combine with existing state values. For example, add_messages appends to message history instead of replacing it."
  :hints="[
    { text: 'Think about what happens when multiple nodes update the same state field', penalty: 10 },
    { text: 'Consider the add_messages reducer - what does it do?', penalty: 15 }
  ]"
/>

<Quiz
  question="What should a node function return in LangGraph?"
  tutorial-id="01-chatbot-basics"
  :options="[
    { text: 'The complete state object', correct: false },
    { text: 'A single message', correct: false },
    { text: 'A dictionary with state updates', correct: true },
    { text: 'A boolean indicating success', correct: false }
  ]"
  explanation="Nodes should return a dictionary containing only the state fields they want to update. LangGraph merges these updates with the existing state using the defined reducers."
  :hints="[
    { text: 'Look at the chatbot node function - what does it return?', penalty: 10 },
    { text: 'The return value should only contain fields that changed', penalty: 15 }
  ]"
/>

<Quiz
  question="Which statement is true about StateGraph compilation?"
  tutorial-id="01-chatbot-basics"
  :options="[
    { text: 'You must recompile after each invocation', correct: false },
    { text: 'Compilation validates structure and optimizes execution', correct: true },
    { text: 'Compilation is optional for simple graphs', correct: false },
    { text: 'Compiled graphs cannot be invoked more than once', correct: false }
  ]"
  explanation="Compilation validates the graph structure (checking for unreachable nodes, missing edges, etc.) and optimizes execution. You compile once and then invoke the graph many times."
  :hints="[
    { text: 'Think about the compile() method - when is it called?', penalty: 10 }
  ]"
/>

<Quiz
  question="What happens if you define a state field without a reducer?"
  tutorial-id="01-chatbot-basics"
  type="multiple-choice"
  :options="[
    { text: 'An error is raised', correct: false },
    { text: 'The new value replaces the old value', correct: true },
    { text: 'The values are automatically appended', correct: false },
    { text: 'The field becomes read-only', correct: false }
  ]"
  explanation="Without a reducer, the default behavior is to overwrite - the new value completely replaces the old value. This is fine for fields like 'current_step' but problematic for message history."
  :hints="[
    { text: 'This is the default Python dictionary behavior', penalty: 10 }
  ]"
/>

<Quiz
  question="The add_messages reducer in LangGraph is used for _______"
  tutorial-id="01-chatbot-basics"
  type="fill-blank"
  :accepted-answers="['appending messages', 'accumulating messages', 'message history', 'conversation history', 'adding messages']"
  explanation="add_messages is a built-in reducer that appends new messages to the existing list while also handling message deduplication by ID."
  :hints="[
    { text: 'Think about what you want to happen with chat messages over time', penalty: 10 }
  ]"
/>

---

## What's Next?

This basic chatbot lacks:
1. **Tools** - Can't take actions or access information
2. **Memory** - Each `invoke()` starts fresh
3. **Branching** - Always follows the same path

Continue to [Tutorial 02: Tool Calling](02-tool-calling.md) to learn how to give your chatbot the ability to use tools.
