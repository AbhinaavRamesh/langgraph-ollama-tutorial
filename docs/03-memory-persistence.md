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

LangGraph solves this with **checkpointers** - components that save the graph state after each step. According to the [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/persistence):

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

## Graph Visualization

![Memory Graph](images/03-memory-graph.png)

The graph structure is the same as Tutorial 01. The difference is in how it's compiled - with a checkpointer that saves state between calls.

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

### 4. SQLite Threading Issues

```python
# WRONG - SQLite connection without thread safety
conn = sqlite3.connect("checkpoints.db")
memory = SqliteSaver(conn)
# Error when used from multiple threads

# CORRECT - enable thread safety
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
```

### 5. Not Closing Database Connections

```python
# WRONG - connection left open
conn = sqlite3.connect("checkpoints.db")
memory = SqliteSaver(conn)
# ... use graph ...
# Connection never closed

# CORRECT - use context manager or explicit close
with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    graph = graph_builder.compile(checkpointer=memory)
    # ... use graph ...
# Connection automatically closed
```

---

## Testing Persistence

### Unit Testing State Updates

```python
def test_messages_accumulate():
    """Test that messages are properly accumulated."""
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "test-1"}}

    # First message
    graph.invoke({"messages": [("user", "Hello")]}, config=config)

    # Second message
    graph.invoke({"messages": [("user", "How are you?")]}, config=config)

    # Check state
    state = graph.get_state(config)
    assert len(state.values["messages"]) >= 4  # 2 user + 2 AI
```

### Testing Thread Isolation

```python
def test_threads_are_isolated():
    """Test that different threads don't share state."""
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Thread 1
    config1 = {"configurable": {"thread_id": "user-alice"}}
    graph.invoke({"messages": [("user", "I'm Alice")]}, config=config1)

    # Thread 2
    config2 = {"configurable": {"thread_id": "user-bob"}}
    graph.invoke({"messages": [("user", "I'm Bob")]}, config=config2)

    # Check isolation
    state1 = graph.get_state(config1)
    state2 = graph.get_state(config2)

    assert "Alice" in str(state1.values["messages"])
    assert "Alice" not in str(state2.values["messages"])
```

### Testing State History

```python
def test_state_history_preserved():
    """Test that state history is maintained."""
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "history-test"}}

    # Multiple invocations
    graph.invoke({"messages": [("user", "First")]}, config=config)
    graph.invoke({"messages": [("user", "Second")]}, config=config)
    graph.invoke({"messages": [("user", "Third")]}, config=config)

    # Get history
    history = list(graph.get_state_history(config))
    assert len(history) >= 3
```

### Integration Testing with SqliteSaver

```python
import tempfile
import os

@pytest.mark.integration
def test_sqlite_persistence_across_restarts():
    """Test that SQLite persists across process restarts."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # First "session"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "persist-test"}}
        graph.invoke({"messages": [("user", "Remember: the code is 42")]}, config=config)
        conn.close()

        # Second "session" (simulating restart)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
        graph = graph_builder.compile(checkpointer=memory)

        state = graph.get_state(config)
        assert "42" in str(state.values["messages"])
        conn.close()
    finally:
        os.unlink(db_path)
```

---

## Performance Considerations

### 1. Message History Growth

Large conversation histories slow down:
- LLM inference (more tokens)
- Checkpointer serialization
- Memory usage

**Mitigation strategies:**

```python
def chatbot(state: State):
    # Only use recent messages for LLM context
    recent_messages = state["messages"][-20:]  # Last 20 messages

    # Optionally, summarize older messages
    if len(state["messages"]) > 20:
        summary = summarize(state["messages"][:-20])
        context = [SystemMessage(content=f"Previous context: {summary}")] + recent_messages
    else:
        context = recent_messages

    return {"messages": [llm.invoke(context)]}
```

### 2. Checkpointer Selection Impact

| Checkpointer | Write Speed | Read Speed | Persistence | Scalability |
|--------------|-------------|------------|-------------|-------------|
| MemorySaver | Very Fast | Very Fast | None | Single process |
| SqliteSaver | Fast | Fast | Local file | Single server |
| PostgresSaver | Medium | Fast | Database | Multi-server |

### 3. Reducing Checkpoint Size

Store only what you need:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # These get persisted

    # Don't include large, non-essential data
    # AVOID: full_document_text: str  (large)
    # AVOID: raw_api_responses: list  (can grow)
```

### 4. Connection Pooling

For production SqliteSaver with multiple threads:

```python
from contextlib import contextmanager
from queue import Queue

class ConnectionPool:
    def __init__(self, db_path: str, size: int = 5):
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self.pool.put(conn)

    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)
```

---

## State Management Best Practices

### 1. Namespace Thread IDs

Organize threads by context:

```python
def get_thread_id(user_id: str, conversation_type: str = "general") -> str:
    """Create namespaced thread ID."""
    return f"{conversation_type}:{user_id}:{uuid.uuid4().hex[:8]}"

# Examples:
# "support:user-123:a1b2c3d4"
# "sales:user-456:e5f6g7h8"
# "general:user-789:i9j0k1l2"
```

### 2. Implement Conversation Expiry

Clean up old conversations:

```python
import time
from datetime import datetime, timedelta

def cleanup_old_threads(memory: SqliteSaver, max_age_days: int = 30):
    """Remove threads older than max_age_days."""
    cutoff = datetime.now() - timedelta(days=max_age_days)

    # Implementation depends on checkpointer
    # For SqliteSaver, you can query the underlying database
    conn = memory.conn
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM checkpoints
        WHERE created_at < ?
    """, (cutoff.isoformat(),))
    conn.commit()
```

### 3. State Validation

Validate state before processing:

```python
def chatbot(state: State):
    # Validate state
    if not state.get("messages"):
        raise ValueError("State must contain messages")

    messages = state["messages"]
    if len(messages) > 1000:
        raise ValueError("Conversation too long, please start a new thread")

    return {"messages": [llm.invoke(messages)]}
```

---

## Migration Between Checkpointers

### Development to Production

```python
def migrate_memory_to_sqlite(
    memory_saver: MemorySaver,
    sqlite_path: str,
    thread_ids: list[str]
):
    """Migrate threads from MemorySaver to SqliteSaver."""
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    sqlite_saver = SqliteSaver(conn)

    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}

        # Get all checkpoints from memory
        for checkpoint in memory_saver.list(config):
            # Put into SQLite
            sqlite_saver.put(
                config,
                checkpoint.checkpoint,
                checkpoint.metadata,
                {}  # new_versions
            )

    conn.close()
```

### SQLite to PostgreSQL

```python
def migrate_sqlite_to_postgres(
    sqlite_path: str,
    postgres_url: str,
    thread_ids: list[str]
):
    """Migrate from SQLite to PostgreSQL."""
    from langgraph.checkpoint.postgres import PostgresSaver

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_saver = SqliteSaver(sqlite_conn)

    postgres_saver = PostgresSaver.from_conn_string(postgres_url)

    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}

        for checkpoint in sqlite_saver.list(config):
            postgres_saver.put(
                config,
                checkpoint.checkpoint,
                checkpoint.metadata,
                {}
            )

    sqlite_conn.close()
```

---

## Production Checklist

- [ ] **Checkpointer selection**: Appropriate for deployment environment
- [ ] **Thread ID strategy**: Consistent naming convention
- [ ] **Connection management**: Proper connection pooling and cleanup
- [ ] **State size limits**: Prevent unbounded growth
- [ ] **Cleanup policy**: Old threads are archived or deleted
- [ ] **Backup strategy**: Database is backed up regularly
- [ ] **Monitoring**: Track checkpoint sizes and counts
- [ ] **Error handling**: Handle checkpointer failures gracefully
- [ ] **Testing**: Persistence tested across restarts
- [ ] **Security**: Sensitive data encrypted at rest

---

## Running the Notebook

```bash
cd examples
jupyter lab 03_memory_persistence.ipynb
```

---

## Key Takeaways

| Concept | What It Does |
|---------|--------------|
| **Checkpointer** | Saves graph state after each step |
| **Thread ID** | Identifies a conversation (in config) |
| **MemorySaver** | In-memory storage (development) |
| **SqliteSaver** | File-based storage (local persistence) |
| **PostgresSaver** | Database storage (production) |
| **get_state()** | Inspect current thread state |

## Checkpointer Selection Guide

| Scenario | Recommended Checkpointer |
|----------|-------------------------|
| Development/Testing | `MemorySaver` |
| Local demos | `SqliteSaver` with `:memory:` |
| Desktop applications | `SqliteSaver` with file |
| Web applications | `PostgresSaver` |
| Multi-server deployment | `PostgresSaver` |

---

## What's Next?

[Tutorial 04: Human-in-the-Loop](04-human-in-the-loop.md) - Learn how to pause agent execution for human approval before taking sensitive actions.

---

## Sources

- [LangGraph Persistence Documentation](https://docs.langchain.com/oss/python/langgraph/persistence)
- [langgraph-checkpoint-sqlite on PyPI](https://pypi.org/project/langgraph-checkpoint-sqlite/)
- [LangGraph v0.2 Release Blog](https://blog.langchain.com/langgraph-v0-2/)
