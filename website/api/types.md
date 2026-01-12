---
title: State Types API
description: TypedDict state schemas for all patterns
---

# State Types API

Complete reference for all TypedDict state schemas used across different agent patterns. These types define the structure of state that flows through LangGraph graphs.

## Multi-Agent States

### MultiAgentState

State schema for supervisor-based multi-agent collaboration.

```python
from typing_extensions import TypedDict
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    next_agent: str
    agent_outputs: Annotated[list[dict], operator.add]
    iteration: int
    max_iterations: int
    final_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Conversation history (accumulates) |
| `task` | `str` | Replace | The current task description |
| `next_agent` | `str` | Replace | Which agent should run next (set by supervisor) |
| `agent_outputs` | `list[dict]` | `operator.add` | Accumulated outputs from all agents |
| `iteration` | `int` | Replace | Current iteration count |
| `max_iterations` | `int` | Replace | Maximum allowed iterations |
| `final_result` | `str` | Replace | The synthesized final result |

#### Usage

```python
from langgraph_ollama_local.agents import MultiAgentState
from langchain_core.messages import HumanMessage

initial_state: MultiAgentState = {
    "messages": [HumanMessage(content="Task: Build calculator")],
    "task": "Build a calculator app",
    "next_agent": "",
    "agent_outputs": [],
    "iteration": 0,
    "max_iterations": 10,
    "final_result": "",
}
```

#### Related

- [create_multi_agent_graph()](/api/agents#create_multi_agent_graph)
- [Tutorial: Multi-Agent Collaboration](/tutorials/multi-agent/14-multi-agent-collaboration)

---

### TeamState

State schema for a single team within a hierarchical structure.

```python
class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    team_name: str
    next_member: str
    member_outputs: Annotated[list[dict], operator.add]
    iteration: int
    max_iterations: int
    team_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Team conversation history |
| `task` | `str` | Replace | Task assigned to the team |
| `team_name` | `str` | Replace | Name of this team |
| `next_member` | `str` | Replace | Which team member should work next |
| `member_outputs` | `list[dict]` | `operator.add` | Accumulated outputs from members |
| `iteration` | `int` | Replace | Current iteration within the team |
| `max_iterations` | `int` | Replace | Maximum iterations for this team |
| `team_result` | `str` | Replace | Final synthesized result from the team |

#### Usage

```python
from langgraph_ollama_local.agents.hierarchical import TeamState

team_state: TeamState = {
    "messages": [],
    "task": "Research quantum computing",
    "team_name": "research",
    "next_member": "",
    "member_outputs": [],
    "iteration": 0,
    "max_iterations": 5,
    "team_result": "",
}
```

#### Related

- [create_team_graph()](/api/agents#create_team_graph)
- [Tutorial: Hierarchical Teams](/tutorials/multi-agent/15-hierarchical-teams)

---

### HierarchicalState

State schema for hierarchical agent teams with multiple sub-teams.

```python
class HierarchicalState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    active_team: str
    team_results: dict[str, str]
    iteration: int
    max_iterations: int
    final_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Top-level conversation history |
| `task` | `str` | Replace | Overall task description |
| `active_team` | `str` | Replace | Which team is currently working |
| `team_results` | `dict[str, str]` | Replace | Results from each team (team_name -> result) |
| `iteration` | `int` | Replace | Top-level iteration count |
| `max_iterations` | `int` | Replace | Maximum top-level iterations |
| `final_result` | `str` | Replace | Final synthesized result from all teams |

#### Usage

```python
from langgraph_ollama_local.agents.hierarchical import HierarchicalState

hierarchical_state: HierarchicalState = {
    "messages": [],
    "task": "Build microservices architecture",
    "active_team": "",
    "team_results": {},
    "iteration": 0,
    "max_iterations": 10,
    "final_result": "",
}
```

#### Related

- [create_hierarchical_graph()](/api/agents#create_hierarchical_graph)
- [Tutorial: Hierarchical Teams](/tutorials/multi-agent/15-hierarchical-teams)

---

## Pattern States

### SwarmState

State schema for decentralized agent swarm/network.

```python
class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    agents_state: dict[str, dict[str, Any]]
    shared_context: Annotated[list[dict], operator.add]
    current_agent: str
    iteration: int
    max_iterations: int
    final_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Conversation history |
| `task` | `str` | Replace | The overall task for the swarm |
| `agents_state` | `dict[str, dict]` | Replace | Per-agent state (agent_name -> agent_data) |
| `shared_context` | `list[dict]` | `operator.add` | Accumulated shared findings from all agents |
| `current_agent` | `str` | Replace | Name of the currently active agent |
| `iteration` | `int` | Replace | Current iteration count |
| `max_iterations` | `int` | Replace | Maximum iterations before completion |
| `final_result` | `str` | Replace | The synthesized final result |

#### Usage

```python
from langgraph_ollama_local.patterns.swarm import SwarmState

swarm_state: SwarmState = {
    "messages": [],
    "task": "Research AI trends",
    "agents_state": {},
    "shared_context": [],
    "current_agent": "",
    "iteration": 0,
    "max_iterations": 10,
    "final_result": "",
}
```

#### Related

- [create_swarm_graph()](/api/patterns#create_swarm_graph)
- [Tutorial: Agent Swarm](/tutorials/multi-agent/18-agent-swarm)

---

### HandoffState

State schema for agent handoff pattern with explicit control transfer.

```python
class HandoffState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    current_agent: str
    handoff_target: str
    context: Annotated[list[dict], operator.add]
    handoff_history: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    final_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Conversation history |
| `task` | `str` | Replace | The original task or query |
| `current_agent` | `str` | Replace | Currently active agent name |
| `handoff_target` | `str` | Replace | Agent to hand off to (empty if no handoff) |
| `context` | `list[dict]` | `operator.add` | Shared context accumulating across handoffs |
| `handoff_history` | `list[str]` | `operator.add` | List of handoff events for tracking |
| `iteration` | `int` | Replace | Number of handoffs that have occurred |
| `max_iterations` | `int` | Replace | Maximum allowed handoffs |
| `final_result` | `str` | Replace | The final response to the user |

#### Usage

```python
from langgraph_ollama_local.patterns.handoffs import HandoffState

handoff_state: HandoffState = {
    "messages": [],
    "task": "I need help with my invoice",
    "current_agent": "sales",
    "handoff_target": "",
    "context": [],
    "handoff_history": [],
    "iteration": 0,
    "max_iterations": 10,
    "final_result": "",
}
```

#### Related

- [create_handoff_graph()](/api/patterns#create_handoff_graph)
- [Tutorial: Agent Handoffs](/tutorials/multi-agent/17-agent-handoffs)

---

### MapReduceState

State schema for map-reduce pattern with parallel execution.

```python
class MapReduceState(TypedDict):
    task: str
    subtasks: list[str]
    worker_results: Annotated[list[dict], operator.add]
    final_result: str
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `task` | `str` | Replace | The main task to be processed |
| `subtasks` | `list[str]` | Replace | List of subtasks created by mapper (one per worker) |
| `worker_results` | `list[dict]` | `operator.add` | Accumulated results from all workers |
| `final_result` | `str` | Replace | The aggregated final result from reducer |

#### Usage

```python
from langgraph_ollama_local.patterns.map_reduce import MapReduceState

map_reduce_state: MapReduceState = {
    "task": "Analyze research paper",
    "subtasks": [],
    "worker_results": [],
    "final_result": "",
}
```

#### Related

- [create_map_reduce_graph()](/api/patterns#create_map_reduce_graph)
- [Tutorial: Map-Reduce Agents](/tutorials/multi-agent/19-map-reduce-agents)

---

### EvaluationState

State schema for agent evaluation sessions with simulated users.

```python
class EvaluationState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation: str
    evaluator_scores: Annotated[list[dict], operator.add]
    turn_count: int
    max_turns: int
    session_complete: bool
    final_metrics: dict[str, float]
```

#### Fields

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `messages` | `list` | `add_messages` | Full conversation between agent and simulated user |
| `conversation` | `str` | Replace | Formatted conversation for evaluator review |
| `evaluator_scores` | `list[dict]` | `operator.add` | List of score dicts from evaluator agent |
| `turn_count` | `int` | Replace | Current conversation turn number |
| `max_turns` | `int` | Replace | Maximum number of conversation turns |
| `session_complete` | `bool` | Replace | Whether the evaluation session is done |
| `final_metrics` | `dict[str, float]` | Replace | Aggregated metrics summary |

#### Usage

```python
from langgraph_ollama_local.patterns.evaluation import EvaluationState

evaluation_state: EvaluationState = {
    "messages": [],
    "conversation": "",
    "evaluator_scores": [],
    "turn_count": 0,
    "max_turns": 10,
    "session_complete": False,
    "final_metrics": {},
}
```

#### Related

- [create_evaluation_graph()](/api/patterns#create_evaluation_graph)
- [Tutorial: Multi-Agent Evaluation](/tutorials/multi-agent/20-multi-agent-evaluation)

---

## Understanding Reducers

LangGraph uses reducers to control how state fields are updated when multiple nodes write to the same field.

### Common Reducers

#### Replace (Default)

New value replaces the old value.

```python
class MyState(TypedDict):
    task: str  # Uses replace reducer by default
```

#### add_messages

Intelligent message accumulation that handles duplicates and updates.

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
```

**Behavior:**
- Appends new messages to the list
- Handles message updates by ID
- Removes duplicates intelligently

#### operator.add

Concatenates lists or adds numbers.

```python
import operator
from typing import Annotated

class MyState(TypedDict):
    outputs: Annotated[list[dict], operator.add]
    count: Annotated[int, operator.add]
```

**Behavior:**
- For lists: Concatenates `[1, 2] + [3, 4] = [1, 2, 3, 4]`
- For numbers: Adds `5 + 3 = 8`

---

## Creating Custom States

### Basic Custom State

```python
from typing_extensions import TypedDict

class MyCustomState(TypedDict):
    input: str
    output: str
    metadata: dict
```

### Custom State with Reducers

```python
from typing_extensions import TypedDict
from typing import Annotated
import operator
from langgraph.graph.message import add_messages

class MyCustomState(TypedDict):
    # Messages with intelligent accumulation
    messages: Annotated[list, add_messages]

    # Simple replacement
    current_step: str

    # List accumulation
    results: Annotated[list[dict], operator.add]

    # Counter
    iteration: int
```

### Using Custom States

```python
from langgraph.graph import StateGraph

# Define custom state
class MyState(TypedDict):
    input: str
    output: str

# Create graph with custom state
workflow = StateGraph(MyState)

def my_node(state: MyState) -> dict:
    return {"output": f"Processed: {state['input']}"}

workflow.add_node("process", my_node)
workflow.set_entry_point("process")
workflow.set_finish_point("process")

graph = workflow.compile()

# Run with custom state
result = graph.invoke({"input": "Hello", "output": ""})
print(result["output"])  # "Processed: Hello"
```

---

## State Best Practices

### 1. Use Appropriate Reducers

```python
# Good: Use add_messages for message history
messages: Annotated[list, add_messages]

# Good: Use operator.add for accumulating results
agent_outputs: Annotated[list[dict], operator.add]

# Good: Use replace (default) for counters and flags
iteration: int
is_complete: bool
```

### 2. Initialize All Fields

```python
# Good: Provide initial values for all fields
initial_state: MultiAgentState = {
    "messages": [],
    "task": "My task",
    "next_agent": "",
    "agent_outputs": [],
    "iteration": 0,
    "max_iterations": 10,
    "final_result": "",
}
```

### 3. Document State Purpose

```python
class MyState(TypedDict):
    """
    State for my custom pattern.

    Attributes:
        messages: Conversation history between agents
        task: The current task being processed
        results: Accumulated results from all workers
    """
    messages: Annotated[list, add_messages]
    task: str
    results: Annotated[list, operator.add]
```

### 4. Keep State Flat When Possible

```python
# Good: Flat structure
class GoodState(TypedDict):
    user_id: str
    user_name: str
    user_email: str

# Avoid: Deep nesting makes updates complex
class AvoidState(TypedDict):
    user: dict[str, dict[str, str]]  # Harder to update
```

---

## Common State Patterns

### Pattern 1: Iterative Processing

```python
class IterativeState(TypedDict):
    task: str
    current_iteration: int
    max_iterations: int
    results: Annotated[list, operator.add]
    is_complete: bool
```

### Pattern 2: Agent Coordination

```python
class CoordinationState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: str
    agent_outputs: Annotated[list[dict], operator.add]
    routing_decision: str
```

### Pattern 3: Evaluation & Metrics

```python
class MetricsState(TypedDict):
    input_data: list[dict]
    scores: Annotated[list[float], operator.add]
    metrics: dict[str, float]
    best_score: float
```

---

## Related

- [Multi-Agent API](/api/agents) - Functions using these state types
- [Patterns API](/api/patterns) - Pattern-specific state types
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - LangGraph state management
