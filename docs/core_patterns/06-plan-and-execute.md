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

### The Pattern

![Plan and Execute Graph](./images/06-plan-execute-graph.png)

The planner creates steps, the executor processes each one sequentially, and the finalizer combines results.

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

## Graph Visualization

![Plan and Execute Graph](./images/06-plan-execute-graph.png)

Planner creates steps → Executor loops through each → Finalizer combines results.

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

## Variations

### Re-planning

Add a re-planner node that updates the plan based on execution results:

```python
def replanner(state):
    if needs_replanning(state):
        new_plan = create_new_plan(state)
        return {"plan": new_plan, "current_step": 0}
    return {}
```

### Parallel Execution

For independent steps, execute them in parallel:

```python
import asyncio

async def parallel_executor(state: PlanExecuteState) -> dict:
    """Execute independent steps in parallel."""
    plan = state["plan"]
    current = state["current_step"]

    # Identify independent steps (no dependencies)
    independent_steps = identify_independent_steps(plan[current:])

    async def execute_step(step: str) -> tuple[str, str]:
        response = await llm.ainvoke([HumanMessage(content=f"Execute: {step}")])
        return (step, response.content)

    # Execute in parallel
    results = await asyncio.gather(*[
        execute_step(step) for step in independent_steps
    ])

    return {
        "past_steps": list(results),
        "current_step": current + len(independent_steps)
    }
```

### Tool-Augmented Execution

Combine with tool calling for real-world tasks:

```python
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

executor_tools = [web_search, calculate]
executor_llm = llm.bind_tools(executor_tools)

def tool_executor(state: PlanExecuteState) -> dict:
    """Execute step with tool access."""
    step = state["plan"][state["current_step"]]

    # Execute with tools
    messages = [HumanMessage(content=f"Execute this step: {step}")]
    response = executor_llm.invoke(messages)

    # Handle tool calls
    while response.tool_calls:
        tool_results = execute_tools(response.tool_calls)
        messages.extend([response] + tool_results)
        response = executor_llm.invoke(messages)

    return {
        "past_steps": [(step, response.content)],
        "current_step": state["current_step"] + 1
    }
```

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

### 4. Plan Too Granular

```python
# WRONG - too many tiny steps
plan = [
    "Open browser",
    "Type URL",
    "Press Enter",
    "Wait for page",
    "Find search box",
    ...
]

# CORRECT - meaningful, actionable steps
plan = [
    "Search for Python documentation on the official website",
    "Extract the key features mentioned",
    "Summarize in 3 bullet points"
]
```

### 5. Finalizer Ignoring Failed Steps

```python
# WRONG - assuming all steps succeeded
def finalizer(state):
    summary = "\n".join([f"{s}: {r}" for s, r in state["past_steps"]])
    return {"response": summarize(summary)}

# CORRECT - handle failures
def finalizer(state):
    successful = [(s, r) for s, r in state["past_steps"]
                  if "Error:" not in r]
    failed = [(s, r) for s, r in state["past_steps"]
              if "Error:" in r]

    if failed:
        # Include failure information
        response = f"Completed {len(successful)}/{len(state['past_steps'])} steps.\n"
        response += f"Failed steps: {[s for s, r in failed]}"
    else:
        response = summarize(successful)

    return {"response": response}
```

---

## Testing Plan-and-Execute

### Unit Testing the Planner

```python
def test_planner_creates_valid_plan():
    """Test that planner creates a parseable plan."""
    state = {"task": "Explain Python in 3 points"}

    result = planner(state)

    assert "plan" in result
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) >= 1
    assert result["current_step"] == 0

def test_planner_handles_complex_task():
    """Test planning for multi-part task."""
    state = {"task": "Compare Python and JavaScript for web development"}

    result = planner(state)

    assert len(result["plan"]) >= 2
```

### Testing the Executor

```python
def test_executor_processes_step():
    """Test that executor processes one step."""
    state = {
        "plan": ["Step 1", "Step 2", "Step 3"],
        "current_step": 0,
        "past_steps": []
    }

    result = executor(state)

    assert "past_steps" in result
    assert len(result["past_steps"]) == 1
    assert result["past_steps"][0][0] == "Step 1"
    assert result["current_step"] == 1

def test_executor_handles_empty_plan():
    """Test executor with completed plan."""
    state = {
        "plan": ["Step 1"],
        "current_step": 1,  # Past end
        "past_steps": [("Step 1", "Done")]
    }

    result = executor(state)

    assert result == {}  # No action needed
```

### Testing the Routing

```python
def test_should_continue_more_steps():
    """Test routing when steps remain."""
    state = {"plan": ["A", "B", "C"], "current_step": 1}
    assert should_continue(state) == "executor"

def test_should_continue_all_done():
    """Test routing when all steps complete."""
    state = {"plan": ["A", "B"], "current_step": 2}
    assert should_continue(state) == "finalizer"
```

### Integration Testing

```python
@pytest.mark.integration
def test_full_plan_execute_flow():
    """Test complete plan-and-execute cycle."""
    result = graph.invoke({
        "task": "List 3 benefits of exercise",
        "messages": [],
        "plan": [],
        "current_step": 0,
        "past_steps": [],
        "response": ""
    })

    # Should have created a plan
    assert len(result["plan"]) >= 1

    # Should have executed all steps
    assert len(result["past_steps"]) == len(result["plan"])

    # Should have final response
    assert len(result["response"]) > 50
```

---

## Advanced Patterns

### Re-planning with Context

```python
def replanner(state: PlanExecuteState) -> dict:
    """Re-plan based on execution results."""
    if not needs_replanning(state):
        return {}

    past_results = "\n".join([f"{s}: {r}" for s, r in state["past_steps"]])

    prompt = f"""Original task: {state['task']}

Completed steps:
{past_results}

Remaining plan:
{state['plan'][state['current_step']:]}

Based on what we've learned, create an updated plan for remaining work.
Return as JSON array."""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_plan = parse_plan(response.content)

    return {
        "plan": state["plan"][:state["current_step"]] + new_plan,
        # Keep current_step - we continue from here
    }

def needs_replanning(state: PlanExecuteState) -> bool:
    """Determine if replanning is needed."""
    # Replan every N steps
    if state["current_step"] > 0 and state["current_step"] % 3 == 0:
        return True

    # Replan if last step failed
    if state["past_steps"]:
        last_result = state["past_steps"][-1][1]
        if "Error:" in last_result or "failed" in last_result.lower():
            return True

    return False
```

### Hierarchical Planning

Break complex tasks into sub-plans:

```python
class HierarchicalState(TypedDict):
    task: str
    high_level_plan: List[str]
    current_phase: int
    detailed_plan: List[str]
    current_step: int
    results: Annotated[List[dict], operator.add]
    response: str

def high_level_planner(state):
    """Create high-level phases."""
    prompt = f"Break this into 2-3 major phases: {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"high_level_plan": parse_plan(response.content), "current_phase": 0}

def detailed_planner(state):
    """Create detailed steps for current phase."""
    phase = state["high_level_plan"][state["current_phase"]]
    prompt = f"Break this phase into specific steps: {phase}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"detailed_plan": parse_plan(response.content), "current_step": 0}
```

### Two-Model Approach

Use different models for planning vs execution:

```python
# Strong model for planning (quality matters)
planner_llm = ChatOllama(
    model="llama3.1:70b",
    temperature=0,  # Deterministic
)

# Fast model for execution (throughput matters)
executor_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
)

def planner(state):
    response = planner_llm.invoke([...])
    return {"plan": parse_plan(response.content), "current_step": 0}

def executor(state):
    response = executor_llm.invoke([...])
    return {"past_steps": [(step, response.content)], ...}
```

---

## Performance Considerations

### 1. Plan Size vs Latency

| Steps | Typical Latency | Use Case |
|-------|-----------------|----------|
| 2-3 | Fast | Simple tasks |
| 4-5 | Medium | Moderate complexity |
| 6-10 | Slower | Complex research |
| 10+ | Very slow | Consider hierarchical |

### 2. Efficient JSON Prompting

```python
# WRONG - ambiguous, often fails
"Create a plan for the task"

# CORRECT - explicit format requirement
"""Create a plan for this task. Return ONLY a JSON array of strings:
["Step 1 description", "Step 2 description", ...]

Task: {task}"""
```

### 3. Caching Common Plans

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_plan(task_type: str) -> list[str]:
    """Return cached plan template for common task types."""
    templates = {
        "comparison": ["Research item A", "Research item B", "Compare and contrast"],
        "explanation": ["Define concept", "Provide examples", "Summarize key points"],
        "analysis": ["Gather data", "Identify patterns", "Draw conclusions"],
    }
    return templates.get(task_type, [])
```

---

## Production Checklist

- [ ] **Plan parsing**: Robust parsing with fallbacks
- [ ] **Bounds checking**: Executor checks step index
- [ ] **Error handling**: Handle step execution failures
- [ ] **Replanning**: Trigger replanning on failures
- [ ] **Logging**: Log plan, steps, and results
- [ ] **Timeouts**: Set per-step timeouts
- [ ] **Testing**: Plan, execute, finalize paths tested
- [ ] **Metrics**: Track steps per task, success rates
- [ ] **Fallback**: Return partial results on failure
- [ ] **State size**: Limit plan length to prevent bloat

---

## Running the Notebook

```bash
cd examples
jupyter lab 06_plan_and_execute.ipynb
```

---

## Key Takeaways

| Concept | What It Does |
|---------|--------------|
| **Planner** | Creates list of steps from task |
| **Executor** | Processes steps one at a time |
| **Finalizer** | Combines results into response |
| **past_steps** | Accumulates (step, result) pairs |
| **operator.add** | Reducer that appends to list |
| **Replanning** | Adjust plan based on results |
| **Hierarchical** | Phases → detailed steps |

---

## What's Next?

[Tutorial 07: Research Assistant](07-research-assistant.md) - Combine all patterns into a comprehensive research agent that plans, searches, reflects, and synthesizes.
