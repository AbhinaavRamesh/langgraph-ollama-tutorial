---
title: Plan-and-Execute Pattern
description: Learn how to build agents that separate strategic planning from tactical execution for efficient multi-step task completion with LangGraph and Ollama
---

# Tutorial 21: Plan-and-Execute Pattern

## Overview

The **plan-and-execute pattern** is a sophisticated approach for tackling complex multi-step tasks by separating strategic planning from tactical execution. Instead of making step-by-step decisions reactively (like ReAct agents), this pattern creates an upfront plan and then executes it systematically, with optional replanning based on results.

**Key Innovation**: Two-phase approach with explicit planning before execution, enabling better strategic decisions and more efficient resource usage.

## Architecture

```mermaid
flowchart LR
    Planner[Planner Node]:::planner
    Executor[Executor Node]:::executor
    Replanner[Replanner Node]:::replanner
    ExecuteNext[Execute Next]:::executor
    End[END]:::end

    Planner --> Executor
    Executor --> ExecuteNext
    Executor --> Replanner
    ExecuteNext -.Loop until<br/>plan complete.-> Executor
    Replanner --> End
    Replanner -.Finalize or<br/>new plan.-> ExecuteNext

    classDef planner fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef executor fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef replanner fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef end fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

### Graph Flow

```mermaid
flowchart TD
    Start([START]):::start
    Planner[Planner<br/>Creates 3-7 step plan]:::planner
    Executor[Executor<br/>Executes one step at a time]:::executor
    Decision{Plan<br/>done?}:::decision
    Replanner[Replanner<br/>Decides: finalize or create new plan]:::replanner
    ResponseCheck{Response<br/>ready?}:::decision
    End([END]):::end

    Start --> Planner
    Planner --> Executor
    Executor --> Decision
    Decision -->|More steps| Executor
    Decision -->|Plan complete| Replanner
    Replanner --> ResponseCheck
    ResponseCheck -->|Yes| End
    ResponseCheck -->|No, new plan| Executor

    classDef start fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef end fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef planner fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef executor fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef replanner fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef decision fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

## Core Components

### 1. State Schema

```python
class PlanExecuteState(TypedDict):
    """State for plan-and-execute pattern."""

    # Original task to accomplish
    task: str

    # Current plan (list of step descriptions)
    plan: list[str]

    # History of (step, result) pairs
    past_steps: Annotated[list[tuple[str, str]], operator.add]

    # Index of current step (0-based)
    current_step: int

    # Final response when complete
    response: str
```

**Key Fields:**
- `plan`: List of actionable steps created by planner
- `past_steps`: Accumulates execution history using `operator.add`
- `current_step`: Tracks progress through the plan
- `response`: Final answer, set by replanner when task is complete

### 2. Planner Node

Creates a structured step-by-step plan from the task.

```python
class Plan(BaseModel):
    """Structured plan output."""
    steps: list[str] = Field(
        description="List of 3-7 actionable steps"
    )

def create_planner_node(llm: BaseChatModel) -> callable:
    """Create a planner that generates step-by-step plans."""
    structured_llm = llm.with_structured_output(Plan)

    def planner(state: PlanExecuteState) -> dict:
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Create a plan for: {state['task']}")
        ]
        output = structured_llm.invoke(messages)
        return {"plan": output.steps, "current_step": 0}

    return planner
```

**Responsibilities:**
- Analyze task thoroughly
- Create 3-7 concrete, actionable steps
- Ensure logical ordering
- Consider dependencies between steps
- Optimize for efficiency

**Example Plan:**
```
Task: "Research and compare the GDP of France and Germany"

Plan:
1. Search for current GDP of France
2. Search for current GDP of Germany
3. Compare the two values
4. Analyze the difference and provide context
```

### 3. Executor Node

Executes steps sequentially, building on previous results.

```python
def create_executor_node(
    llm: BaseChatModel,
    tools: list[Any] | None = None,
) -> callable:
    """Create executor that processes one step at a time."""

    # Optionally use ReAct agent with tools
    if tools:
        from langgraph.prebuilt import create_react_agent
        react_agent = create_react_agent(llm, tools)

    def executor(state: PlanExecuteState) -> dict:
        # Get current step
        step = state["plan"][state["current_step"]]

        # Build context from past steps
        context = build_context(state["task"], state["past_steps"])

        # Execute step (with or without tools)
        result = execute_step(step, context, react_agent if tools else llm)

        return {
            "past_steps": [(step, result)],
            "current_step": state["current_step"] + 1,
        }

    return executor
```

**Features:**
- Executes one step at a time
- Builds context from previous step results
- Can use tools via ReAct agent
- Updates execution history
- Advances to next step

**Context Building:**
```python
# Context includes:
Original task: Research and compare GDP of France and Germany

Steps completed so far:
1. Search for current GDP of France
   Result: France's GDP in 2023 is approximately $2.96 trillion...

2. Search for current GDP of Germany
   Result: Germany's GDP in 2023 is approximately $4.31 trillion...

Now execute: Compare the two values
```

### 4. Replanner Node

Decides whether to finalize or create a new plan based on results.

```python
class Response(BaseModel):
    """Final response."""
    response: str = Field(description="Final answer")

class Act(BaseModel):
    """Action decision."""
    action: Union[Response, Plan] = Field(
        description="Either Response or new Plan"
    )

def create_replanner_node(llm: BaseChatModel) -> callable:
    """Create replanner for adaptive decision making."""
    structured_llm = llm.with_structured_output(Act)

    def replanner(state: PlanExecuteState) -> dict:
        # Review completed steps
        steps_summary = format_past_steps(state["past_steps"])

        messages = [
            SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"""
                Task: {state['task']}
                Completed: {steps_summary}

                Decide: Respond or create new plan?
            """)
        ]

        output = structured_llm.invoke(messages)

        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:  # New Plan
            return {"plan": output.action.steps, "current_step": 0}

    return replanner
```

**Decision Criteria:**
- **Finalize** if:
  - Original task is fully accomplished
  - All necessary information is gathered
  - No additional steps would add value
- **Replan** if:
  - Task needs more work
  - New information suggests different approach
  - Original plan was incomplete

### 5. Routing Functions

Control flow between nodes based on state.

```python
def route_after_executor(state: PlanExecuteState) -> str:
    """Route after executor."""
    if state["current_step"] < len(state["plan"]):
        return "executor"  # More steps to execute
    return "replanner"  # Plan complete, review results

def route_after_replanner(state: PlanExecuteState) -> str:
    """Route after replanner."""
    if state.get("response"):
        return END  # Task complete
    return "executor"  # New plan, continue executing
```

## API Reference

### Graph Builder

```python
def create_plan_execute_graph(
    llm: BaseChatModel,
    tools: list[Any] | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """
    Create a plan-and-execute graph.

    Args:
        llm: Language model for all nodes
        tools: Optional tools for executor (enables ReAct)
        checkpointer: Optional state persistence

    Returns:
        Compiled graph ready for execution
    """
```

### Convenience Runner

```python
def run_plan_execute_task(
    graph: CompiledStateGraph,
    task: str,
    thread_id: str = "default",
) -> dict:
    """
    Run a task through plan-and-execute.

    Args:
        graph: Compiled graph
        task: Task description
        thread_id: Thread ID for checkpointing

    Returns:
        Final state with response and execution history
    """
```

### Node Creators

```python
def create_planner_node(llm: BaseChatModel) -> callable:
    """Create planner node."""

def create_executor_node(
    llm: BaseChatModel,
    tools: list[Any] | None = None,
) -> callable:
    """Create executor node."""

def create_replanner_node(llm: BaseChatModel) -> callable:
    """Create replanner node."""
```

## Usage Examples

### Basic Usage

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.patterns import (
    create_plan_execute_graph,
    run_plan_execute_task,
)

# Setup
config = LocalAgentConfig()
llm = config.create_chat_client()

# Create graph
graph = create_plan_execute_graph(llm)

# Run task
result = run_plan_execute_task(
    graph,
    "Explain the benefits of microservices architecture with 3 advantages and 2 challenges"
)

print(result["response"])
print(f"Executed {len(result['past_steps'])} steps")
```

### With Tools

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    # Implementation
    return search_results

@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expression."""
    return eval(expression)

tools = [search, calculate]

# Create graph with tools
graph = create_plan_execute_graph(llm, tools=tools)

# Executor will use ReAct pattern with tools
result = run_plan_execute_task(
    graph,
    "Find the population of Tokyo and New York, then calculate their difference"
)
```

### Custom Implementation

```python
from langgraph.graph import StateGraph, START, END
from langgraph_ollama_local.patterns.plan_execute import (
    PlanExecuteState,
    create_planner_node,
    create_executor_node,
    create_replanner_node,
    route_after_executor,
    route_after_replanner,
)

# Build custom graph
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", create_planner_node(planner_llm))
workflow.add_node("executor", create_executor_node(executor_llm, tools))
workflow.add_node("replanner", create_replanner_node(replanner_llm))

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    route_after_executor,
    {"executor": "executor", "replanner": "replanner"}
)
workflow.add_conditional_edges(
    "replanner",
    route_after_replanner,
    {"executor": "executor", END: END}
)

graph = workflow.compile()
```

## Comparison with Other Patterns

### Plan-Execute vs ReAct

| Aspect | Plan-Execute | ReAct |
|--------|--------------|-------|
| **Planning** | Upfront, explicit | Step-by-step, implicit |
| **Visibility** | Full plan visible before execution | No visibility into future steps |
| **Token Efficiency** | Higher (plan once, execute many) | Lower (full context each step) |
| **Adaptability** | Replanning after plan completion | Continuous adaptation |
| **Best For** | Multi-step tasks with dependencies | Exploratory tasks, simple problems |
| **Model Flexibility** | Can use different models per node | Single model |
| **Debugging** | Easy (inspect plan and execution) | Harder (no explicit strategy) |

### Plan-Execute vs ReWOO

| Aspect | Plan-Execute | ReWOO |
|--------|--------------|-------|
| **Planning** | Adaptive (can replan) | Fixed (single plan) |
| **Execution** | Sequential | Parallel |
| **LLM Calls** | Multiple (planner + executor + replanner) | Two (planner + solver) |
| **Tool Use** | ReAct per step | Variable substitution |
| **Flexibility** | High (replanning) | Low (no replanning) |
| **Best For** | Complex tasks needing adaptation | Known workflows, token optimization |

### Decision Guide

```
Does your task require multiple coordinated steps?
├─ YES → Can you plan everything upfront?
│   ├─ YES → Use ReWOO (Tutorial 25)
│   │         • Fixed workflow
│   │         • Maximum token efficiency
│   │         • Parallel tool execution
│   │
│   └─ NO → Use Plan-Execute (Tutorial 21)
│             • Adaptive replanning
│             • Sequential execution
│             • Better for uncertain tasks
│
└─ NO → Does the task need exploration?
    ├─ YES → Use ReAct (Tutorial 02)
    │         • Step-by-step decisions
    │         • Reactive adaptation
    │
    └─ NO → Use direct LLM
              • Simple single-step tasks
```

## Advanced Patterns

### Multi-Model Architecture

Use different models for different nodes:

```python
# Large model for strategic planning
planner_llm = ChatOllama(model="llama3.1:70b")

# Small, fast model for execution
executor_llm = ChatOllama(model="llama3.2:3b")

# Large model for replanning decisions
replanner_llm = ChatOllama(model="llama3.1:70b")

workflow.add_node("planner", create_planner_node(planner_llm))
workflow.add_node("executor", create_executor_node(executor_llm, tools))
workflow.add_node("replanner", create_replanner_node(replanner_llm))
```

**Benefits:**
- Strategic decisions use powerful models
- Routine execution uses fast models
- Cost and latency optimization

### Hierarchical Planning

Combine with subgraphs for nested planning:

```python
# High-level planner creates strategic plan
high_level_graph = create_plan_execute_graph(llm)

# Each step uses detailed plan-execute subgraph
from langgraph_ollama_local.patterns import create_subgraph_node

detailed_executor = create_subgraph_node(
    create_plan_execute_graph(llm),
    input_fn=lambda state: {"task": state["plan"][state["current_step"]]},
    output_fn=lambda result: {"result": result["response"]}
)
```

### Plan Validation

Add validation node before execution:

```python
def validate_plan_node(state: PlanExecuteState) -> dict:
    """Validate plan before execution."""
    plan = state["plan"]

    # Check for issues
    if len(plan) < 2:
        return {"plan": [state["task"]]}  # Too simple, execute directly

    if len(plan) > 10:
        # Too complex, ask planner to simplify
        return {"needs_replanning": True}

    # Check for circular dependencies, etc.

    return {}

workflow.add_node("validate", validate_plan_node)
workflow.add_edge("planner", "validate")
workflow.add_conditional_edges(
    "validate",
    lambda s: "planner" if s.get("needs_replanning") else "executor"
)
```

## Best Practices

### 1. Plan Quality

**Good Plans:**
- 3-7 concrete, actionable steps
- Clear success criteria per step
- Logical ordering
- Independent where possible

**Poor Plans:**
```
❌ "Research the topic"  (too vague)
❌ "Do step 1, then step 2, then step 3"  (no content)
❌ 15 micro-steps  (too granular)
```

**Good Plans:**
```
✓ "Search for France's GDP data from reliable sources"
✓ "Search for Germany's GDP data from reliable sources"
✓ "Compare the GDP values and calculate the difference"
✓ "Provide context on economic factors explaining the difference"
```

### 2. Context Management

Keep context concise but informative:

```python
# Truncate long results
result_preview = (
    result[:200] + "..."
    if len(result) > 200
    else result
)

# Include only relevant past steps
recent_steps = state["past_steps"][-3:]  # Last 3 steps
```

### 3. Replanning Strategy

**When to replan:**
- Original plan was based on incorrect assumptions
- Execution revealed new information
- Task scope changed during execution

**When to finalize:**
- All objectives accomplished
- No additional steps would improve quality
- Diminishing returns on further work

### 4. Tool Integration

Structure tools for plan-execute compatibility:

```python
@tool
def focused_search(query: str, domain: str = "general") -> str:
    """
    Search for specific information.

    Args:
        query: Specific search query
        domain: Knowledge domain (tech, science, etc.)
    """
    # Implementation
```

Use focused, single-purpose tools rather than broad, multi-purpose ones.

### 5. Error Handling

Handle execution failures gracefully:

```python
def robust_executor(state: PlanExecuteState) -> dict:
    try:
        result = execute_step(...)
    except Exception as e:
        # Log error as step result
        result = f"Step failed: {str(e)}"

    return {
        "past_steps": [(step, result)],
        "current_step": state["current_step"] + 1,
    }
```

## Common Issues

### Issue 1: Plans Too Abstract

**Problem:** Planner creates vague steps like "Analyze the data"

**Solution:** Improve planner prompt with examples:

```python
PLANNER_PROMPT = """...

Good step: "Calculate the mean and standard deviation of the dataset"
Bad step: "Analyze the data"

Good step: "Search for recent studies on topic X published after 2020"
Bad step: "Research the topic"
"""
```

### Issue 2: Infinite Replanning Loop

**Problem:** Replanner keeps creating new plans without finalizing

**Solution:** Add iteration limit:

```python
class PlanExecuteState(TypedDict):
    # ...existing fields...
    iteration_count: int
    max_iterations: int

def route_after_replanner(state: PlanExecuteState) -> str:
    if state.get("response"):
        return END

    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        # Force finalization
        return "force_finalize"

    return "executor"
```

### Issue 3: Context Too Long

**Problem:** Accumulated context exceeds token limits

**Solution:** Implement context summarization:

```python
def summarize_past_steps(past_steps: list[tuple[str, str]]) -> str:
    """Summarize execution history."""
    if len(past_steps) <= 3:
        return format_steps(past_steps)

    # Summarize older steps, keep recent ones detailed
    summary = llm.invoke([
        HumanMessage(content=f"Summarize these steps: {past_steps[:-2]}")
    ])

    return summary.content + "\n\n" + format_steps(past_steps[-2:])
```

## Performance Considerations

### Token Usage

Plan-execute typically uses:
- **Planner**: 500-1000 tokens
- **Executor** (per step): 300-800 tokens
- **Replanner**: 600-1200 tokens

**Optimization:**
- Use smaller models for execution
- Truncate intermediate results
- Limit number of steps in plan

### Latency

Sequential execution introduces latency:
- 3-step plan: ~15-30 seconds (with typical local models)
- 7-step plan: ~35-70 seconds

**Mitigation:**
- Use faster models for execution
- Parallelize independent steps (hybrid with map-reduce)
- Consider ReWOO for latency-critical tasks

### Model Recommendations

| Model Size | Max Plan Steps | Use Case |
|------------|---------------|----------|
| 3B-8B | 3-4 | Simple multi-step tasks |
| 13B-34B | 4-6 | Moderate complexity |
| 70B+ | 5-7 | Complex reasoning, strategic planning |

## Testing

See `tests/test_plan_execute.py` for comprehensive test suite.

**Key test areas:**
- State schema validation
- Planner creates valid plans
- Executor advances through steps
- Replanner routing logic
- Graph compilation and invocation
- Fallback for models without structured output

## References

- **LangGraph Tutorial**: [Plan-and-Execute](https://github.com/langchain-ai/langgraph/tree/main/docs/docs/tutorials/plan-and-execute)
- **Implementation**: `langgraph_ollama_local/patterns/plan_execute.py`
- **Tutorial**: `examples/advanced_reasoning/21_plan_and_execute.ipynb`
- **Tests**: `tests/test_plan_execute.py`

## Related Tutorials

- **Tutorial 02**: Tool Calling and ReAct (foundation for executor)
- **Tutorial 22**: Reflection (iterative improvement)
- **Tutorial 25**: ReWOO (alternative planning pattern)
- **Tutorial 19**: Map-Reduce (parallel execution pattern)

## Quiz

Test your understanding of the Plan-and-Execute pattern:

<Quiz
  question="What is the key innovation of the Plan-and-Execute pattern compared to ReAct?"
  tutorial-id="21-plan-and-execute"
  :options="[
    { text: 'It requires fewer LLM calls for all tasks', correct: false },
    { text: 'It separates strategic planning from tactical execution with explicit upfront planning', correct: true },
    { text: 'It can only work with structured output models', correct: false },
    { text: 'It cannot use external tools', correct: false }
  ]"
  explanation="The Plan-and-Execute pattern's key innovation is the two-phase approach: creating an explicit upfront plan before execution. This provides better visibility into the strategy, enables different models for planning vs execution, and allows for more efficient resource usage."
  :hints="[
    { text: 'Think about what happens BEFORE any tool is called', penalty: 10 },
    { text: 'Consider the difference between reactive (step-by-step) and proactive (planned) approaches', penalty: 15 }
  ]"
/>

<Quiz
  question="When should the Replanner node create a new plan instead of finalizing with a response?"
  tutorial-id="21-plan-and-execute"
  :options="[
    { text: 'After every execution step to ensure accuracy', correct: false },
    { text: 'Only when all steps in the original plan fail', correct: false },
    { text: 'When the task needs more work, new information suggests a different approach, or the original plan was incomplete', correct: true },
    { text: 'Never - plans should always be executed exactly as originally created', correct: false }
  ]"
  explanation="The Replanner should create a new plan when: the original plan is complete but the task needs more work, execution revealed new information requiring a different approach, or the original plan was incomplete. This adaptive replanning capability is what makes Plan-and-Execute more flexible than ReWOO."
  :hints="[
    { text: 'Consider scenarios where the initial plan assumptions were incorrect', penalty: 10 },
    { text: 'The Replanner evaluates whether the task objective has been fully achieved', penalty: 15 }
  ]"
/>

<Quiz
  question="What is the recommended number of steps in a well-designed plan?"
  tutorial-id="21-plan-and-execute"
  :options="[
    { text: '1-2 steps', correct: false },
    { text: '3-7 concrete, actionable steps', correct: true },
    { text: '10-15 detailed micro-steps', correct: false },
    { text: 'As many steps as possible for thoroughness', correct: false }
  ]"
  explanation="A good plan should contain 3-7 concrete, actionable steps. Too few steps (1-2) indicates the task might be too simple for plan-execute. Too many steps (10+) leads to complexity issues, context length problems, and makes execution harder to track."
  :hints="[
    { text: 'Consider the balance between granularity and manageability', penalty: 10 },
    { text: 'The tutorial mentions specific numbers in the Best Practices section', penalty: 15 }
  ]"
/>

<Quiz
  question="What is the purpose of the past_steps field in the PlanExecuteState using operator.add?"
  tutorial-id="21-plan-and-execute"
  :options="[
    { text: 'To count how many steps have been executed', correct: false },
    { text: 'To accumulate execution history as (step, result) pairs for context building', correct: true },
    { text: 'To add new steps to the plan dynamically', correct: false },
    { text: 'To calculate the total execution time', correct: false }
  ]"
  explanation="The past_steps field uses Annotated[list[tuple[str, str]], operator.add] to automatically accumulate execution history. Each executor run appends a (step, result) tuple. This history provides context for subsequent steps and helps the replanner understand what has been accomplished."
  :hints="[
    { text: 'Look at how the executor node returns past_steps in its output', penalty: 10 },
    { text: 'operator.add in TypedDict annotations enables automatic list concatenation', penalty: 15 }
  ]"
/>

<Quiz
  question="How does Plan-and-Execute compare to ReWOO in terms of adaptability?"
  tutorial-id="21-plan-and-execute"
  type="true-false"
  :options="[
    { text: 'Plan-and-Execute is more adaptable because it can replan based on execution results, while ReWOO uses a fixed single plan', correct: true },
    { text: 'ReWOO is more adaptable because it executes tools in parallel without waiting for results', correct: false }
  ]"
  explanation="Plan-and-Execute is more adaptable than ReWOO. While ReWOO creates a single fixed plan and executes it without modification, Plan-and-Execute includes a Replanner node that can create entirely new plans based on execution results. This makes Plan-and-Execute better for uncertain tasks where the approach may need to change."
  :hints="[
    { text: 'Consider what happens when tool results reveal unexpected information', penalty: 10 },
    { text: 'The comparison table in the tutorial shows Flexibility: High vs Low', penalty: 15 }
  ]"
/>

---

<div class="tutorial-nav">
  <a href="/tutorials/multi-agent/20-multi-agent-evaluation" class="prev">← Tutorial 20: Multi-Agent Evaluation</a>
  <a href="/tutorials/advanced/22-reflection" class="next">Tutorial 22: Reflection →</a>
</div>
