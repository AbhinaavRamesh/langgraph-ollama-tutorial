# Tutorial 02: Tool Calling & ReAct Agent

This tutorial teaches how to build a ReAct (Reasoning + Acting) agent from scratch using LangGraph and local Ollama. You'll learn how to give your LLM the ability to take actions in the real world.

## What You'll Learn

- How to define **tools** for your agent
- **Binding tools** to an LLM
- **Conditional edges** for dynamic routing
- The **ReAct loop**: Agent → Tools → Agent
- Building agents that can take real actions

## Prerequisites

- Completed [Tutorial 01: Chatbot Basics](01-chatbot-basics.md)
- A model that supports tool calling (see [Ollama Tools Models](https://ollama.com/search?c=tools))

**Recommended models for tool calling:**
- `llama3.1` (8B+) - Best overall for function calling
- `llama3.2` (1B, 3B) - Good for resource-constrained environments
- `mistral` (7B) - Balance of performance and efficiency
- `qwen3` - Featured in official Ollama examples
- `granite4` - IBM's tool-optimized models

**Pull a model before running:**
```bash
# Pull a model that supports tool calling
ollama pull llama3.1:8b

# Or for a smaller model
ollama pull llama3.2:3b

# Verify it's available
ollama list
```

If using [ollama-local-serve](https://github.com/abhinaavramesh/ollama-local-serve) on a LAN server:
```bash
# On the server, or via API
curl http://your-server:11434/api/pull -d '{"name": "llama3.1:8b"}'

# Or use the helper function
from langgraph_ollama_local import ensure_model
ensure_model("llama3.1:8b", host="192.168.1.100")
```

---

## Understanding Agents

### What is an Agent?

In the context of LLMs, an **agent** is a system that can:
1. **Reason** about what actions to take
2. **Execute** those actions using tools
3. **Observe** the results
4. **Iterate** until the task is complete

The key distinction from a simple chatbot is that agents can **take actions** beyond just generating text. They can search the web, query databases, call APIs, perform calculations, and more.

### Why Do We Need Agents?

Consider these limitations of a basic LLM:
- **No real-time information**: Training data has a cutoff date
- **No computation**: Can make arithmetic errors on complex math
- **No external access**: Cannot check weather, stock prices, or your calendar
- **No actions**: Cannot send emails, create files, or update databases

Agents solve these problems by giving the LLM access to **tools** that extend its capabilities.

---

## The ReAct Pattern

### What is ReAct?

ReAct (**Re**asoning and **Act**ing) is a paradigm introduced in the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). It combines two capabilities:

1. **Reasoning**: The LLM thinks through problems step-by-step
2. **Acting**: The LLM uses tools to gather information or take actions

The key insight is that reasoning and acting should be **interleaved**. The LLM reasons about what tool to use, uses it, observes the result, then reasons about what to do next.

### The ReAct Loop

The pattern follows this cycle:

1. **User Query** → Agent receives a question or task
2. **Reasoning** → Agent thinks about what information it needs
3. **Action** → Agent decides to call a tool
4. **Observation** → Tool returns results
5. **Repeat** → Agent reasons with new information, may call more tools
6. **Response** → When satisfied, agent responds to user

This continues until the agent has enough information to provide a complete answer.

---

## Core Concepts

### 1. Tools

Tools are Python functions that the LLM can decide to call. They're the bridge between the LLM's reasoning and real-world actions.

#### Defining a Tool

Use the `@tool` decorator from LangChain:

```python
from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b
```

#### Tool Requirements

Every tool needs three things:

1. **The `@tool` decorator**: Converts your function into a LangChain Tool object
2. **A docstring**: **Critical!** The LLM reads this to understand what the tool does and when to use it
3. **Type hints**: Tell the LLM what types of arguments are expected

**Why docstrings matter so much:**

The LLM has no way to see your function's implementation. It decides whether to use a tool based entirely on:
- The tool's name
- The docstring description
- The argument names and types

A vague docstring like `"Does math"` will confuse the LLM. Be specific: `"Multiply two numbers together. Use this for multiplication operations."`

#### Tool Examples

```python
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Use this tool when you need to calculate a sum.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of a and b
    """
    return a + b

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Use this to check weather conditions in any city.

    Args:
        location: City name (e.g., "San Francisco", "New York")

    Returns:
        A string describing the current weather
    """
    # In production, this would call a weather API
    return f"Weather in {location}: Sunny, 72°F"
```

### 2. Tool Binding

Before the LLM can use tools, you must tell it what tools are available. This is called **binding**.

```python
from langchain_ollama import ChatOllama

# Create the base LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Bind tools to create a tool-aware LLM
llm_with_tools = llm.bind_tools([multiply, add, get_weather])
```

**What happens during binding:**
1. Each tool's schema (name, description, parameters) is extracted
2. This schema is formatted according to the model's expected format
3. The schema is included in every request to the LLM

**Important**: The original `llm` object is unchanged. `bind_tools()` returns a new object. Always use the `llm_with_tools` version when you want tool calling.

### 3. Tool Calls

When you invoke an LLM with bound tools, it may decide to call a tool instead of responding directly.

```python
response = llm_with_tools.invoke("What is 5 times 3?")

# The response is an AIMessage
print(type(response))  # AIMessage

# Check if tools were called
print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 5, 'b': 3}, 'id': 'call_abc123'}]

# The content might be empty when tools are called
print(response.content)  # "" or reasoning text
```

**Tool call structure:**
```python
{
    "name": "multiply",      # Which tool to call
    "args": {"a": 5, "b": 3}, # Arguments to pass
    "id": "call_abc123"      # Unique ID for tracking
}
```

The `id` is important - when you return tool results, you must include this ID so the LLM knows which call the result corresponds to.

### 4. The Tool Node

The tool node is responsible for actually executing tool calls. It:
1. Reads the tool calls from the last message
2. Executes each tool with its arguments
3. Returns `ToolMessage` objects with the results

```python
import json
from langchain_core.messages import ToolMessage

# Create a lookup dictionary for tools
tools = [multiply, add, get_weather]
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state):
    """Execute tool calls from the last AI message."""
    outputs = []

    # Get tool calls from the last message
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        # Look up the tool
        tool = tools_by_name[tool_call["name"]]

        # Execute it
        result = tool.invoke(tool_call["args"])

        # Create a ToolMessage with the result
        outputs.append(ToolMessage(
            content=json.dumps(result),  # Must be a string
            name=tool_call["name"],
            tool_call_id=tool_call["id"],  # Must match!
        ))

    return {"messages": outputs}
```

**Why `ToolMessage`?**

The LLM needs to see tool results in a specific format. `ToolMessage` is a message type that:
- Contains the tool's output
- Links back to the original tool call via `tool_call_id`
- Has a special role that LLMs understand

### 5. Conditional Edges

In Tutorial 01, all edges were unconditional - they always went from A to B. For ReAct, we need **conditional edges** that route based on state.

```python
from langgraph.graph import END

def should_continue(state):
    """Decide whether to continue to tools or end.

    Returns:
        "tools" - if the agent requested tool calls
        "end" - if the agent is done (no tool calls)
    """
    last_message = state["messages"][-1]

    # Check if there are any tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tool calls means the agent is done
    return "end"

# Add conditional edge
workflow.add_conditional_edges(
    "agent",                           # From this node
    should_continue,                   # Use this function to decide
    {"tools": "tools", "end": END}     # Map return values to destinations
)
```

**How it works:**
1. After `agent` node runs, LangGraph calls `should_continue(state)`
2. The function returns either `"tools"` or `"end"`
3. LangGraph looks up the return value in the mapping dictionary
4. Execution continues to the mapped node (`"tools"` node or `END`)

---

## Building the ReAct Agent

Now let's build a complete ReAct agent step by step.

### Step 1: Define Tools

```python
from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b

# Collect tools and create lookup
tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}
```

### Step 2: Create LLM with Tool Binding

```python
from langchain_ollama import ChatOllama
from langgraph_ollama_local import LocalAgentConfig

config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0,  # Deterministic for reliable tool calling
)

# Bind tools
llm_with_tools = llm.bind_tools(tools)
```

### Step 3: Define State

Same as Tutorial 01 - we track messages:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State for our ReAct agent."""
    messages: Annotated[list, add_messages]
```

### Step 4: Define the Agent Node

The agent node calls the LLM:

```python
def agent_node(state: AgentState) -> dict:
    """Call the LLM to decide what to do.

    The LLM will either:
    - Return a text response (done)
    - Request tool calls (continue)
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### Step 5: Define the Tool Node

The tool node executes requested tools:

```python
import json
from langchain_core.messages import ToolMessage

def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the last message."""
    outputs = []

    for tc in state["messages"][-1].tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        ))

    return {"messages": outputs}
```

### Step 6: Define Routing Logic

```python
from langgraph.graph import END

def should_continue(state: AgentState) -> str:
    """Route based on whether tools were called."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
```

### Step 7: Build the Graph

```python
from langgraph.graph import StateGraph, START, END

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "agent")  # Start with agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")  # After tools, back to agent

# Compile
graph = workflow.compile()
```

### Step 8: Use It!

```python
result = graph.invoke({
    "messages": [("user", "What is 7 times 8?")]
})
print(result["messages"][-1].content)
# "7 times 8 equals 56"
```

---

## Complete Code

Here's everything in one place:

```python
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_ollama_local import LocalAgentConfig

# === Tools ===
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0,
).bind_tools(tools)

# === State ===
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# === Nodes ===
def agent_node(state):
    return {"messages": [llm.invoke(state["messages"])]}

def tool_node(state):
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        ))
    return {"messages": outputs}

# === Routing ===
def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"

# === Graph ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# === Use ===
result = graph.invoke({"messages": [("user", "What is 7 times 8?")]})
print(result["messages"][-1].content)
```

## Graph Visualization

![ReAct Agent Graph](./images/02-react-graph.png)

The graph shows the ReAct loop:
- Execution starts at `__start__`
- Goes to `agent` node which calls the LLM
- **Conditional edge** checks if tools were called:
  - If yes → goes to `tools` node → back to `agent`
  - If no → goes to `__end__`

This loop continues until the agent responds without requesting tools.

---

## Understanding the Flow

Let's trace through what happens with the query "What is 7 times 8?":

1. **User message** added to state: `[HumanMessage("What is 7 times 8?")]`

2. **Agent node** runs:
   - LLM sees the message and available tools
   - Decides to call `multiply(7, 8)`
   - Returns AIMessage with `tool_calls=[{name: "multiply", args: {a: 7, b: 8}}]`

3. **should_continue** returns `"tools"` (tool calls exist)

4. **Tool node** runs:
   - Executes `multiply.invoke({a: 7, b: 8})` → `56`
   - Returns `ToolMessage(content="56", tool_call_id=...)`

5. **Agent node** runs again:
   - LLM sees: user question, its tool call, tool result (56)
   - Now has enough info to respond
   - Returns AIMessage with content "7 times 8 equals 56"

6. **should_continue** returns `"end"` (no tool calls)

7. **Execution ends**, final state returned

---

## Common Patterns

### Multiple Tool Calls

The LLM can request multiple tools in one response:

```python
# Query: "What is 5 + 3, and what is 10 * 2?"
# LLM returns:
tool_calls = [
    {"name": "add", "args": {"a": 5, "b": 3}, "id": "call_1"},
    {"name": "multiply", "args": {"a": 10, "b": 2}, "id": "call_2"},
]
```

The tool node executes all of them and returns multiple ToolMessages.

### Sequential Tool Calls

For multi-step problems, the agent may call tools across multiple iterations:

```python
# Query: "Add 5 and 3, then multiply the result by 2"

# Iteration 1:
# Agent calls add(5, 3) → 8

# Iteration 2:
# Agent sees result (8), calls multiply(8, 2) → 16

# Iteration 3:
# Agent responds: "5 + 3 = 8, then 8 × 2 = 16"
```

### System Prompts for Better Tool Use

```python
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When you need to perform calculations, use the available tools.
Always show your reasoning before and after using tools.

Available tools:
- add: Add two numbers
- multiply: Multiply two numbers
"""

def agent_node(state):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

---

## Troubleshooting

### Tool Calling Not Working?

Not all local models support tool calling. Check the [Ollama Tools Models](https://ollama.com/search?c=tools) page for the official list.

**Models that work well:**
- `llama3.1:8b` or larger - Best overall performance
- `llama3.2:3b` - Good for smaller deployments
- `mistral:7b` - Efficient and reliable
- `qwen3` - Featured in Ollama docs
- `granite4` - Tool-optimized by IBM

**Models that often struggle:**
- Very small models (< 1B parameters) with complex prompts
- Base (non-instruct/chat) models
- Models not listed in the Tools category

### Inconsistent Results?

Set `temperature=0` for deterministic behavior:

```python
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,  # Critical for reliable tool calling
)
```

### Model Ignoring Tools?

Make your tool descriptions very explicit about **when** to use them:

```python
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    USE THIS TOOL when you need to:
    - Add numbers
    - Calculate a sum
    - Find a total

    DO NOT use for multiplication or other operations.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of a and b
    """
    return a + b
```

### Tool Errors?

Add error handling in your tool node:

```python
def tool_node(state):
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        try:
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            content = json.dumps(result)
        except Exception as e:
            content = f"Error: {str(e)}"

        outputs.append(ToolMessage(
            content=content,
            name=tc["name"],
            tool_call_id=tc["id"],
        ))
    return {"messages": outputs}
```

---

## Common Pitfalls

### 1. Forgetting to Bind Tools

```python
# WRONG - using base LLM
response = llm.invoke("What is 5 + 3?")
# LLM will try to answer without tools

# CORRECT - using tool-bound LLM
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke("What is 5 + 3?")
```

### 2. Missing Tool Call ID

```python
# WRONG - no tool_call_id
ToolMessage(content="8", name="add")
# Error: ToolMessage requires tool_call_id

# CORRECT - include the ID from the original call
ToolMessage(
    content="8",
    name="add",
    tool_call_id=tool_call["id"]  # Must match!
)
```

### 3. Non-String Content in ToolMessage

```python
# WRONG - returning raw Python objects
ToolMessage(content={"result": 56}, ...)
# Error: content must be a string

# CORRECT - serialize to JSON
ToolMessage(content=json.dumps({"result": 56}), ...)
```

### 4. Vague Tool Descriptions

```python
# WRONG - LLM won't know when to use this
@tool
def calc(x, y):
    """Do math."""
    return x + y

# CORRECT - specific description with use cases
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Use this for:
    - Computing sums
    - Adding quantities
    - Calculating totals

    Args:
        a: First number to add
        b: Second number to add
    """
    return a + b
```

### 5. Infinite Loops

```python
# WRONG - no exit condition
def should_continue(state):
    return "tools"  # Always continues!

# CORRECT - check for completion
def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
```

### 6. Checking tool_calls on Wrong Message Type

```python
# WRONG - may fail on HumanMessage or ToolMessage
if state["messages"][-1].tool_calls:
    ...

# CORRECT - check message type first
last = state["messages"][-1]
if hasattr(last, "tool_calls") and last.tool_calls:
    ...
```

---

## Testing Your Agent

### Unit Testing Tools

Test tools in isolation before integrating:

```python
def test_multiply_tool():
    result = multiply.invoke({"a": 7, "b": 8})
    assert result == 56

def test_add_tool():
    result = add.invoke({"a": 5, "b": 3})
    assert result == 8
```

### Testing the Tool Node

```python
from langchain_core.messages import AIMessage

def test_tool_node_executes_calls():
    # Create a mock AI message with tool calls
    ai_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "multiply",
            "args": {"a": 4, "b": 5},
            "id": "call_123"
        }]
    )

    result = tool_node({"messages": [ai_message]})

    assert len(result["messages"]) == 1
    assert "20" in result["messages"][0].content
    assert result["messages"][0].tool_call_id == "call_123"
```

### Testing Routing Logic

```python
def test_should_continue_with_tools():
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "add", "args": {}, "id": "1"}]
    )
    assert should_continue({"messages": [ai_msg]}) == "tools"

def test_should_continue_without_tools():
    ai_msg = AIMessage(content="The answer is 42")
    assert should_continue({"messages": [ai_msg]}) == "end"
```

### Integration Testing

```python
@pytest.mark.integration
def test_agent_uses_tools():
    result = graph.invoke({
        "messages": [("user", "What is 7 times 8?")]
    })

    # Check that tools were used
    messages = result["messages"]
    has_tool_call = any(
        hasattr(m, "tool_calls") and m.tool_calls
        for m in messages
    )
    assert has_tool_call

    # Check final answer
    assert "56" in messages[-1].content
```

### Snapshot Testing

```python
def test_graph_structure():
    graph_repr = graph.get_graph()
    node_names = [n.name for n in graph_repr.nodes.values()]

    assert "agent" in node_names
    assert "tools" in node_names
    assert "__start__" in node_names
    assert "__end__" in node_names
```

---

## Performance Considerations

### 1. Tool Execution Time

Tools should be fast. Long-running operations should:
- Use async versions
- Implement timeouts
- Consider caching

```python
import asyncio

@tool
async def slow_api_call(query: str) -> str:
    """Call an external API (with timeout)."""
    async with asyncio.timeout(10):  # 10 second timeout
        return await external_api.fetch(query)
```

### 2. Minimize Tool Count

More tools = more tokens in every request:

```python
# Inefficient - 10 separate tools
@tool
def add(a, b): ...
@tool
def subtract(a, b): ...
@tool
def multiply(a, b): ...
# ... 7 more

# Better - one tool with operation parameter
@tool
def calculate(a: float, b: float, operation: str) -> float:
    """Perform arithmetic: add, subtract, multiply, divide."""
    ops = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b if b != 0 else "Error: division by zero",
    }
    return ops.get(operation, lambda: "Unknown operation")()
```

### 3. Batch Tool Calls

When the LLM requests multiple tools, execute them in parallel:

```python
import asyncio

async def tool_node_async(state: AgentState) -> dict:
    """Execute multiple tool calls concurrently."""
    tool_calls = state["messages"][-1].tool_calls

    async def execute_one(tc):
        result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
        return ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        )

    outputs = await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
    return {"messages": list(outputs)}
```

### 4. Model Selection for Tool Calling

| Model | Tool Calling Quality | Speed | Memory |
|-------|---------------------|-------|--------|
| `llama3.1:8b` | Excellent | Medium | ~8GB |
| `llama3.2:3b` | Good | Fast | ~4GB |
| `llama3.2:1b` | Basic | Very Fast | ~2GB |
| `mistral:7b` | Good | Medium | ~6GB |
| `qwen3:8b` | Excellent | Medium | ~8GB |

---

## Security Considerations

### 1. Input Validation

Tools receive user-influenced input. Always validate:

```python
@tool
def query_database(table: str, query: str) -> str:
    """Query a database table."""
    # Validate table name
    ALLOWED_TABLES = {"users", "products", "orders"}
    if table not in ALLOWED_TABLES:
        return f"Error: Table '{table}' not allowed"

    # Sanitize query (never use string formatting for SQL!)
    # Use parameterized queries
    return execute_safe_query(table, query)
```

### 2. Limit Tool Capabilities

Give tools minimal permissions:

```python
@tool
def read_file(path: str) -> str:
    """Read a file from the allowed directory."""
    import os

    # Restrict to safe directory
    safe_base = "/app/data"
    full_path = os.path.normpath(os.path.join(safe_base, path))

    # Prevent directory traversal
    if not full_path.startswith(safe_base):
        return "Error: Access denied"

    with open(full_path) as f:
        return f.read()
```

### 3. Audit Tool Usage

Log all tool calls for security review:

```python
import logging

logger = logging.getLogger("agent_tools")

def tool_node(state: AgentState) -> dict:
    outputs = []
    for tc in state["messages"][-1].tool_calls:
        # Log before execution
        logger.info(f"Tool call: {tc['name']}, args: {tc['args']}")

        try:
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            logger.info(f"Tool result: {tc['name']} -> success")
        except Exception as e:
            result = f"Error: {e}"
            logger.error(f"Tool error: {tc['name']} -> {e}")

        outputs.append(ToolMessage(
            content=json.dumps(result),
            name=tc["name"],
            tool_call_id=tc["id"],
        ))
    return {"messages": outputs}
```

---

## Production Checklist

- [ ] **Input validation**: All tools validate their inputs
- [ ] **Error handling**: Tool errors are caught and returned gracefully
- [ ] **Timeouts**: External API calls have timeouts
- [ ] **Rate limiting**: Limit tool calls per request/session
- [ ] **Logging**: All tool usage is logged for audit
- [ ] **Testing**: Unit and integration tests pass
- [ ] **Model selection**: Using a model that reliably calls tools
- [ ] **Temperature**: Set to 0 for deterministic tool calling
- [ ] **Monitoring**: Track tool success/failure rates
- [ ] **Fallbacks**: Handle cases when tools fail

---

## Running the Notebook

```bash
cd examples
jupyter lab 02_tool_calling.ipynb
```

---

## Key Takeaways

| Concept | What It Does |
|---------|--------------|
| **Tool** | Function decorated with `@tool` that LLM can call |
| **bind_tools()** | Tells the LLM what tools are available |
| **tool_calls** | List of tool invocations requested by the LLM |
| **ToolMessage** | Message containing a tool's output, linked by ID |
| **Conditional Edge** | Edge that routes based on a condition function |
| **ReAct Loop** | Agent → Tools → Agent cycle until done |

## What's Missing?

This agent has limitations:

1. **No memory** - Each invoke starts fresh
2. **Limited tools** - Only math operations
3. **No error recovery** - Fails on bad tool calls

These are addressed in later tutorials:
- [Tutorial 03: Memory & Persistence](03-memory-persistence.md) - Remember conversations
- Tutorial 04: Human-in-the-Loop - Add approval steps

---

## Next Steps

[Tutorial 03: Memory & Persistence](03-memory-persistence.md) - Learn how to add conversation memory so your agent remembers previous interactions.
