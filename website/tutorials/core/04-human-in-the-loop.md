---
title: Human-in-the-Loop
description: Learn how to pause agent execution for human review and approval before taking sensitive actions using interrupts, checkpointers, and the Command API.
prev:
  text: 'Tutorial 03: Memory & Persistence'
  link: '/tutorials/core/03-memory-persistence'
next:
  text: 'Tutorial 05: Reflection'
  link: '/tutorials/core/05-reflection'
---

# Tutorial 04: Human-in-the-Loop

This tutorial teaches how to pause agent execution for human review and approval before taking sensitive actions.

## What You'll Learn

- **Interrupts**: Pausing graph execution at specific points
- **interrupt_before**: Static breakpoints defined at compile time
- **interrupt()**: Dynamic breakpoints at runtime
- **Command**: Resuming execution with human input
- **Approval workflows**: Review and approve agent actions

## Prerequisites

- Completed [Tutorial 03: Memory & Persistence](03-memory-persistence.md)
- Understanding of checkpointers (required for interrupts)

---

## Why Human-in-the-Loop?

Agents are powerful but need oversight. Before an agent:
- Sends an email or message
- Makes a purchase or payment
- Deletes or modifies data
- Calls external APIs with side effects

You want a human to review and approve the action.

### Common Patterns

According to the LangGraph documentation, there are four typical patterns:

1. **Approve/Reject**: Pause before a critical step, review, and approve or reject
2. **Edit State**: Pause to review and modify the graph state
3. **Review Tool Calls**: Inspect and edit tool calls before execution
4. **Provide Input**: Ask the human for additional information

---

## Core Concepts

### 1. Interrupts and Checkpointers

Interrupts use LangGraph's persistence layer. When you call an interrupt:
1. Graph execution pauses
2. Current state is saved to the checkpointer
3. The thread is marked as "interrupted"
4. You can inspect the state and decide what to do
5. Resume with `invoke(None, config)` or `invoke(Command(resume=value), config)`

**Important**: Interrupts require a checkpointer. Without one, there's no way to save and resume state.

### 2. interrupt_before and interrupt_after

These are the simplest ways to add interrupts - specify them at compile time:

```python
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"],  # Pause BEFORE tools node
    interrupt_after=["agent"],   # Pause AFTER agent node
)
```

### 3. Checking Interrupt Status

After invoking a graph, check if it's paused:

```python
state = graph.get_state(config)

if state.next:  # If there's a next node, we're paused
    print(f"Paused before: {state.next}")
else:
    print("Execution complete")
```

### 4. Resuming Execution

To continue after approval:

```python
# Simple resume (continue as-is)
result = graph.invoke(None, config=config)

# Resume with input (using Command)
from langgraph.types import Command
result = graph.invoke(Command(resume="approved"), config=config)
```

---

## Building an Approval Workflow

### Step 1: Define Sensitive Tools

```python
from langchain_core.tools import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (sensitive action!)."""
    return f"Email sent to {to}"

@tool
def get_weather(location: str) -> str:
    """Get weather (safe action)."""
    return f"Weather in {location}: Sunny, 72°F"

# Mark which tools are sensitive
SENSITIVE_TOOLS = {"send_email"}
```

### Step 2: Build the Graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# Compile with interrupt
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # Pause before ANY tool execution
)
```

### Step 3: Run with Approval

```python
config = {"configurable": {"thread_id": "approval-1"}}

# Start execution
result = graph.invoke(
    {"messages": [("user", "Send email to alice@example.com")]},
    config=config
)

# Check if paused
state = graph.get_state(config)
if state.next:
    # Show pending action
    last_msg = state.values["messages"][-1]
    for tc in last_msg.tool_calls:
        print(f"Pending: {tc['name']}({tc['args']})")

    # Get human approval
    if input("Approve? (y/n): ").lower() == 'y':
        result = graph.invoke(None, config=config)
    else:
        print("Rejected!")
```

---

## Selective Interrupts

Not all tools need approval. Route sensitive tools to a node with interrupts:

```python
def should_continue(state: State) -> str:
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        # Check if any tool is sensitive
        for tc in last_msg.tool_calls:
            if tc["name"] in SENSITIVE_TOOLS:
                return "sensitive_tools"
        return "safe_tools"
    return "end"

# Build graph with separate paths
workflow.add_node("sensitive_tools", tool_node)
workflow.add_node("safe_tools", tool_node)
workflow.add_conditional_edges(
    "agent", should_continue,
    {"sensitive_tools": "sensitive_tools", "safe_tools": "safe_tools", "end": END}
)

# Only interrupt for sensitive tools
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_tools"]  # NOT safe_tools
)
```

---

## Complete Code

```python
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph_ollama_local import LocalAgentConfig

# === Tools ===
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to} with subject: {subject}"

tools = [send_email]
tools_by_name = {t.name: t for t in tools}

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0,
).bind_tools(tools)

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

# === Nodes ===
def agent_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

def tool_node(state: State) -> dict:
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        ))
    return {"messages": outputs}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# === Graph ===
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# === Compile with Interrupt ===
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

# === Approval Helper ===
def run_with_approval(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", user_input)]}, config=config)

    while True:
        state = graph.get_state(config)
        if not state.next:
            break

        # Show pending action
        last_msg = state.values["messages"][-1]
        print(f"Pending: {last_msg.tool_calls}")

        # In production: get approval from UI
        result = graph.invoke(None, config=config)  # Auto-approve for demo

    return result["messages"][-1].content

# === Use ===
response = run_with_approval("Send email to test@example.com", "demo-1")
print(response)
```

---

## Advanced Patterns

### 1. Dynamic Interrupts with `interrupt()`

Instead of compile-time `interrupt_before`, use runtime `interrupt()`:

```python
from langgraph.types import interrupt

def tool_node(state: State) -> dict:
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        # Dynamic interrupt for sensitive tools only
        if tc["name"] in SENSITIVE_TOOLS:
            approval = interrupt({
                "action": tc["name"],
                "args": tc["args"],
                "message": f"Approve {tc['name']} with args {tc['args']}?"
            })

            if approval.get("approved") != True:
                outputs.append(ToolMessage(
                    content="Action rejected by user",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                ))
                continue

        result = tools_by_name[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        ))

    return {"messages": outputs}
```

### 2. Editing State Before Resuming

Modify tool calls before execution:

```python
from langgraph.types import Command

# Get paused state
state = graph.get_state(config)
last_msg = state.values["messages"][-1]

# Review tool calls
for tc in last_msg.tool_calls:
    print(f"Tool: {tc['name']}, Args: {tc['args']}")

# Modify if needed
edited_args = {"to": "safe@example.com", "subject": "Modified", "body": "Safe content"}
last_msg.tool_calls[0]["args"] = edited_args

# Update state with modifications
graph.update_state(config, {"messages": [last_msg]})

# Resume with edited state
result = graph.invoke(None, config=config)
```

---

## Common Pitfalls

### 1. No Checkpointer

```python
# WRONG - interrupts without checkpointer
graph = workflow.compile(interrupt_before=["tools"])
# RuntimeError: Checkpointer required for interrupts

# CORRECT - always include checkpointer
memory = MemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)
```

### 2. Resuming Without Interrupt

```python
# WRONG - trying to resume when not paused
result = graph.invoke({"messages": [("user", "Hi")]}, config=config)
result = graph.invoke(None, config=config)  # Nothing to resume!

# CORRECT - check if paused first
state = graph.get_state(config)
if state.next:
    result = graph.invoke(None, config=config)
```

### 3. Wrong Config After Resume

```python
# WRONG - different thread_id loses state
result = graph.invoke(input, config={"configurable": {"thread_id": "a"}})
result = graph.invoke(None, config={"configurable": {"thread_id": "b"}})
# Error: No state to resume

# CORRECT - same thread_id
config = {"configurable": {"thread_id": "consistent-id"}}
result = graph.invoke(input, config=config)
# ... later ...
result = graph.invoke(None, config=config)  # Same config
```

---

## Quiz

Test your understanding of human-in-the-loop patterns:

<Quiz
  question="What two things are required to use interrupts in LangGraph?"
  tutorial-id="04-human-in-the-loop"
  :options="[
    { text: 'A checkpointer and a special interrupt() function', correct: false },
    { text: 'A thread_id and error handling', correct: false },
    { text: 'A checkpointer and a thread_id in the config', correct: true },
    { text: 'A callback function and a checkpointer', correct: false }
  ]"
  explanation="Interrupts require both a checkpointer (to save/resume state) and a thread_id (to identify which conversation to pause). Without a checkpointer, there's no way to save and resume the paused state."
  :hints="[
    { text: 'Think about what needs to happen when execution pauses - state must be saved somewhere', penalty: 10 },
    { text: 'Review what you learned in Tutorial 03 about persistence requirements', penalty: 15 }
  ]"
/>

<Quiz
  question="How do you resume execution after an interrupt?"
  tutorial-id="04-human-in-the-loop"
  :options="[
    { text: 'Call invoke() with new user input', correct: false },
    { text: 'Call invoke(None, config) with the same thread_id', correct: true },
    { text: 'Call resume() on the graph object', correct: false },
    { text: 'The graph resumes automatically after a timeout', correct: false }
  ]"
  explanation="To resume after an interrupt, call invoke(None, config) where None indicates 'continue from where you left off' and config must have the same thread_id that was paused."
  :hints="[
    { text: 'The first argument to invoke() indicates whether to provide new input or continue', penalty: 10 },
    { text: 'Passing None means there is no new input - just continue execution', penalty: 15 }
  ]"
/>

<Quiz
  question="What does state.next indicate when inspecting a paused graph?"
  tutorial-id="04-human-in-the-loop"
  :options="[
    { text: 'The next user message to process', correct: false },
    { text: 'The next tool that will be executed', correct: false },
    { text: 'A tuple of next nodes to execute (empty if done)', correct: true },
    { text: 'The iteration number for the current step', correct: false }
  ]"
  explanation="state.next is a tuple containing the names of the next nodes to execute. If it's empty or falsy, execution is complete. If it has values, the graph is paused before those nodes."
  :hints="[
    { text: 'This field tells you where in the graph execution will continue', penalty: 10 },
    { text: 'An empty tuple means there are no more nodes to visit', penalty: 15 }
  ]"
/>

<Quiz
  question="Interrupts can work without a checkpointer if you handle state manually."
  tutorial-id="04-human-in-the-loop"
  type="true-false"
  :options="[
    { text: 'True', correct: false },
    { text: 'False', correct: true }
  ]"
  explanation="Interrupts absolutely require a checkpointer - there is no manual workaround. The checkpointer is the mechanism that saves state when pausing and loads it when resuming. Without it, LangGraph will raise a RuntimeError."
  :hints="[
    { text: 'Check the Common Pitfalls section for errors about missing checkpointers', penalty: 10 },
    { text: 'The tutorial states this is a hard requirement, not optional', penalty: 15 }
  ]"
/>

<Quiz
  question="What compile-time parameter pauses execution BEFORE a specific node runs?"
  tutorial-id="04-human-in-the-loop"
  type="fill-blank"
  :accepted-answers="['interrupt_before', 'interrupt-before']"
  explanation="Use interrupt_before=['node_name'] when compiling to pause execution before that node runs. There's also interrupt_after for pausing after a node completes."
  :hints="[
    { text: 'The parameter name describes when the pause happens relative to the node', penalty: 10 },
    { text: 'Look at the compile() call in the Building an Approval Workflow section', penalty: 15 }
  ]"
/>

---

## What's Next?

[Tutorial 05: Reflection](05-reflection.md) - Learn how to build self-critiquing agents that iteratively improve their outputs.
