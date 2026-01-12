---
title: Memory & Persistence
description: Learn how to add memory to your LangGraph agents so they can remember previous conversations across multiple interactions using checkpointers and thread IDs.
prev:
  text: 'Tutorial 02: Tool Calling'
  link: '/tutorials/core/02-tool-calling'
next:
  text: 'Tutorial 04: Human-in-the-Loop'
  link: '/tutorials/core/04-human-in-the-loop'
---

# Tutorial 03: Memory & Persistence

This tutorial teaches how to add memory to your LangGraph agents so they can remember previous conversations across multiple interactions.

## What You'll Learn

- **Checkpointers**: How LangGraph persists state
- **Thread IDs**: Maintaining separate conversations
- **MemorySaver**: In-memory persistence for development
- **SqliteSaver**: File-based persistence for production
- **State inspection**: Viewing and debugging conversation history

## Prerequisites

- Completed [Tutorial 02: Tool Calling](02-tool-calling.md)
- For SqliteSaver: `pip install langgraph-checkpoint-sqlite`

---

## Understanding the Memory Problem

### Why Agents Need Memory

In Tutorial 01, we built a basic chatbot. But each call to `invoke()` was independent:

```python
result1 = graph.invoke({"messages": [("user", "My name is Alice")]})
result2 = graph.invoke({"messages": [("user", "What's my name?")]})
# The chatbot has no idea who Alice is!
```

This is because LangGraph graphs are stateless by default. Each invocation:
1. Receives fresh input
2. Processes through nodes
3. Returns output
4. **Forgets everything**

Real conversational agents need to remember:
- Previous messages in the conversation
- User preferences and context
- Decisions made earlier in the session

### The Solution: Checkpointers

LangGraph solves this with **checkpointers** - components that save the graph state after each step.

> When you compile a graph with a checkpointer, the checkpointer saves a checkpoint of the graph state at every super-step. Those checkpoints are saved to a thread, which can be accessed after graph execution.

This enables:
- **Conversation memory**: Messages accumulate across calls
- **Human-in-the-loop**: Pause and resume execution
- **Time travel**: Go back to previous states
- **Fault tolerance**: Resume from failures

---

## Core Concepts

### 1. Checkpointers

A checkpointer is an object that implements the `BaseCheckpointSaver` interface. It's responsible for:
- Saving graph state after each node execution
- Loading state when a thread is resumed
- Managing state versions for time travel

You add a checkpointer when compiling your graph:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### 2. Thread IDs

Threads are how LangGraph organizes saved states. Think of a thread as a conversation ID:

- Each unique thread ID has its own conversation history
- Same thread ID = continue the same conversation
- Different thread ID = start fresh

```python
config = {"configurable": {"thread_id": "user-123-conversation-1"}}
result = graph.invoke(input, config=config)
```

**Important**: When using a checkpointer, you **must** provide a thread ID in the config. Without it, LangGraph raises an error.

### 3. State Snapshots

At any point, you can inspect the current state of a thread:

```python
state = graph.get_state(config)
print(state.values["messages"])  # All messages in the conversation
```

You can also view the history of all states:

```python
for state in graph.get_state_history(config):
    print(f"Step: {state.next}, Messages: {len(state.values['messages'])}")
```

---

## Available Checkpointers

LangGraph provides several checkpointer implementations:

### MemorySaver (Development)

Stores state in Python dictionaries. Fast and simple, but:
- **Lost when the process ends**
- Single-process only
- Perfect for development and testing

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### SqliteSaver (Local Persistence)

Stores state in a SQLite database file. Good for:
- Local applications
- Demos and prototypes
- Single-server deployments

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory SQLite (for testing)
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)

# File-based SQLite (persistent)
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
```

**Note**: Requires `pip install langgraph-checkpoint-sqlite`

### PostgresSaver (Production)

Stores state in PostgreSQL. Ideal for:
- Production deployments
- Multi-server architectures
- High availability requirements

```python
from langgraph.checkpoint.postgres import PostgresSaver

memory = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
```

**Note**: Requires `pip install langgraph-checkpoint-postgres`

---

## Building a Chatbot with Memory

### Step 1: Define State

Same as before - we track messages:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### Step 2: Create the Graph

```python
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph_ollama_local import LocalAgentConfig

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
```

### Step 3: Compile with Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### Step 4: Use Thread IDs

```python
# Start a conversation
config = {"configurable": {"thread_id": "conversation-1"}}

result = graph.invoke(
    {"messages": [("user", "My name is Alice.")]},
    config=config
)
print(result["messages"][-1].content)

# Continue the same conversation
result = graph.invoke(
    {"messages": [("user", "What's my name?")]},
    config=config  # Same thread ID
)
print(result["messages"][-1].content)  # "Your name is Alice!"
```

---

## Complete Code

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langgraph_ollama_local import LocalAgentConfig

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
)

# === Node ===
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# === Graph ===
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# === Compile with Memory ===
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# === Helper Function ===
def chat(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", user_input)]}, config=config)
    return result["messages"][-1].content

# === Use It ===
print(chat("Hi! My name is Bob."))
print(chat("What's my name?"))  # Remembers!
```

---

## Advanced Patterns

### Multiple Conversations

Handle multiple users with different thread IDs:

```python
def chat_with_user(user_id: str, message: str):
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    result = graph.invoke({"messages": [("user", message)]}, config=config)
    return result["messages"][-1].content

# Each user has their own conversation history
chat_with_user("alice", "My favorite color is blue")
chat_with_user("bob", "My favorite color is red")

# Each remembers their own preferences
chat_with_user("alice", "What's my favorite color?")  # "Blue"
chat_with_user("bob", "What's my favorite color?")    # "Red"
```

### Inspecting Conversation History

```python
config = {"configurable": {"thread_id": "my-thread"}}
state = graph.get_state(config)

print(f"Total messages: {len(state.values['messages'])}")
for msg in state.values["messages"]:
    print(f"  [{msg.type}]: {msg.content[:50]}...")
```

### Persistent Storage for Production

```python
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

# Ensure directory exists
db_path = Path("data/conversations.db")
db_path.parent.mkdir(exist_ok=True)

# Create persistent checkpointer
conn = sqlite3.connect(str(db_path), check_same_thread=False)
memory = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=memory)

# Conversations now survive restarts!
```

---

## Common Pitfalls

### 1. Forgetting the Thread ID

```python
# WRONG - no config with thread_id
result = graph.invoke({"messages": [("user", "Hi")]})
# Error: RunnableConfig must contain 'thread_id' key

# CORRECT - always include thread_id in config
config = {"configurable": {"thread_id": "conversation-1"}}
result = graph.invoke({"messages": [("user", "Hi")]}, config=config)
```

### 2. Expecting Memory Without Checkpointer

```python
# WRONG - no checkpointer = no memory
graph = graph_builder.compile()

# CORRECT - add checkpointer for persistence
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### 3. Reusing Thread IDs Unintentionally

```python
# WRONG - all users share the same thread
for user_id in users:
    config = {"configurable": {"thread_id": "default"}}  # Same ID!
    graph.invoke(...)

# CORRECT - unique thread per user
for user_id in users:
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    graph.invoke(...)
```

---

## Checkpointer Selection Guide

| Scenario | Recommended Checkpointer |
|----------|-------------------------|
| Development/Testing | `MemorySaver` |
| Local demos | `SqliteSaver` with `:memory:` |
| Desktop applications | `SqliteSaver` with file |
| Web applications | `PostgresSaver` |
| Multi-server deployment | `PostgresSaver` |

---

## Quiz

Test your understanding of memory and persistence:

<Quiz
  question="What is required to enable memory in a LangGraph application?"
  tutorial-id="03-memory-persistence"
  :options="[
    { text: 'Setting temperature to 0', correct: false },
    { text: 'Compiling the graph with a checkpointer', correct: true },
    { text: 'Using a special Memory State type', correct: false },
    { text: 'Installing the langgraph-memory package', correct: false }
  ]"
  explanation="Memory requires compiling the graph with a checkpointer (like MemorySaver or SqliteSaver). The checkpointer saves state after each step, allowing conversations to persist across invocations."
  :hints="[
    { text: 'Look at how the graph is compiled differently when memory is needed', penalty: 10 },
    { text: 'The compile() method accepts an optional parameter for persistence', penalty: 15 }
  ]"
/>

<Quiz
  question="What happens if you don't provide a thread_id when using a checkpointer?"
  tutorial-id="03-memory-persistence"
  :options="[
    { text: 'The graph uses a default thread_id automatically', correct: false },
    { text: 'An error is raised', correct: true },
    { text: 'Memory is stored in a temporary location', correct: false },
    { text: 'The checkpointer is silently ignored', correct: false }
  ]"
  explanation="When a checkpointer is configured, you MUST provide a thread_id in the config parameter. Without it, LangGraph raises an error because it doesn't know which conversation thread to save state to."
  :hints="[
    { text: 'Thread IDs identify which conversation the state belongs to', penalty: 10 },
    { text: 'Check the Common Pitfalls section for this specific error', penalty: 15 }
  ]"
/>

<Quiz
  question="Which checkpointer is recommended for production web applications with multiple servers?"
  tutorial-id="03-memory-persistence"
  :options="[
    { text: 'MemorySaver', correct: false },
    { text: 'SqliteSaver', correct: false },
    { text: 'PostgresSaver', correct: true },
    { text: 'FileSaver', correct: false }
  ]"
  explanation="PostgresSaver is recommended for production web applications because it supports multi-server deployments, high availability, and can be shared across multiple processes. MemorySaver is in-memory only, and SqliteSaver is single-server file-based."
  :hints="[
    { text: 'Consider what happens when you have multiple application servers', penalty: 10 },
    { text: 'Which database type is designed for concurrent multi-process access?', penalty: 15 }
  ]"
/>

<Quiz
  question="MemorySaver persists data across application restarts."
  tutorial-id="03-memory-persistence"
  type="true-false"
  :options="[
    { text: 'True', correct: false },
    { text: 'False', correct: true }
  ]"
  explanation="MemorySaver stores state in Python dictionaries and is lost when the process ends. It's perfect for development and testing but not for production persistence. Use SqliteSaver or PostgresSaver for data that survives restarts."
  :hints="[
    { text: 'Consider where in-memory data is stored and what happens when a process terminates', penalty: 10 },
    { text: 'The tutorial mentions MemorySaver is single-process only', penalty: 15 }
  ]"
/>

<Quiz
  question="What method is used to inspect the current state of a conversation thread?"
  tutorial-id="03-memory-persistence"
  type="fill-blank"
  :accepted-answers="['get_state', 'graph.get_state', 'get_state()', 'graph.get_state()']"
  explanation="Use graph.get_state(config) to inspect the current state of a thread. This returns a StateSnapshot object containing the values (like messages) and metadata about the conversation."
  :hints="[
    { text: 'The method name describes what it does - retrieving the current state', penalty: 10 },
    { text: 'Look at the State Snapshots section in the tutorial', penalty: 15 }
  ]"
/>

---

## What's Next?

[Tutorial 04: Human-in-the-Loop](04-human-in-the-loop.md) - Learn how to pause agent execution for human approval before taking sensitive actions.
