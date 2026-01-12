# Map-Reduce Agents

## Overview

The map-reduce pattern enables parallel agent execution with result aggregation. This pattern is ideal for tasks that can be split into independent subtasks, processed in parallel, and then synthesized into a final result.

## Architecture

```mermaid
flowchart TD
    START([START]):::startNode --> Mapper[Mapper<br/>Split Task]:::mapperNode

    Mapper --> Worker1[Worker 1<br/>Parallel]:::workerNode
    Mapper --> Worker2[Worker 2<br/>Parallel]:::workerNode
    Mapper --> Worker3[Worker 3<br/>Parallel]:::workerNode

    Worker1 --> Reducer[Reducer<br/>Aggregate]:::reducerNode
    Worker2 --> Reducer
    Worker3 --> Reducer

    Reducer --> END([END]):::endNode

    classDef startNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef endNode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef mapperNode fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef workerNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef reducerNode fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

## When to Use

Use map-reduce when:

- **Tasks are parallelizable**: Subtasks can be completed independently
- **Speed matters**: Parallel execution is faster than sequential
- **Scale is needed**: Large workloads benefit from distributed processing
- **No interdependencies**: Workers don't need to coordinate with each other

## Common Use Cases

### 1. Document Analysis
Each worker analyzes a section of a document, and the reducer synthesizes findings.

```python
task = "Analyze this research paper in detail"
# Mapper splits into: intro, methods, results, conclusion
# Workers analyze their sections in parallel
# Reducer combines all analyses
```

### 2. Multi-Perspective Analysis
Each worker takes a different perspective on the same topic.

```python
task = "Analyze AI impact on society"
# Mapper creates: economic, ethical, technological perspectives
# Workers each analyze from their perspective
# Reducer synthesizes all perspectives
```

### 3. Large-Scale Data Processing
Each worker processes a chunk of data independently.

```python
task = "Process customer feedback from Q4"
# Mapper splits by category or time period
# Workers process their chunks
# Reducer aggregates insights
```

## Key Components

### 1. State Schema

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class MapReduceState(TypedDict):
    task: str                                        # Main task
    subtasks: list[str]                              # Created by mapper
    worker_results: Annotated[list[dict], operator.add]  # Accumulated outputs
    final_result: str                                # Aggregated result
```

### 2. Mapper Node

The mapper splits the task into independent subtasks:

```python
from pydantic import BaseModel, Field

class MapperOutput(BaseModel):
    subtasks: list[str] = Field(description="List of independent subtasks")
    reasoning: str = Field(description="Explanation of split")

def create_mapper_node(llm, num_workers=3):
    structured_llm = llm.with_structured_output(MapperOutput)

    def mapper(state):
        output = structured_llm.invoke([
            SystemMessage(content=f"Split into {num_workers} subtasks"),
            HumanMessage(content=f"Task: {state['task']}")
        ])
        return {"subtasks": output.subtasks[:num_workers]}

    return mapper
```

### 3. Worker Nodes

Workers process subtasks independently:

```python
def create_worker_node(llm, worker_id, worker_prompt=""):
    def worker(state):
        subtask = state["subtasks"][worker_id]

        response = llm.invoke([
            SystemMessage(content=f"Process your subtask. {worker_prompt}"),
            HumanMessage(content=f"Subtask: {subtask}")
        ])

        return {
            "worker_results": [{
                "worker_id": worker_id,
                "subtask": subtask,
                "output": response.content,
            }]
        }

    return worker
```

### 4. Reducer Node

The reducer aggregates all worker results:

```python
class ReducerOutput(BaseModel):
    final_result: str = Field(description="Synthesized result")
    summary: str = Field(description="Brief summary")

def create_reducer_node(llm):
    structured_llm = llm.with_structured_output(ReducerOutput)

    def reducer(state):
        # Combine all worker outputs
        results_text = "\n\n".join([
            f"Worker {r['worker_id']}: {r['output']}"
            for r in state["worker_results"]
        ])

        output = structured_llm.invoke([
            SystemMessage(content="Synthesize all results"),
            HumanMessage(content=f"Results:\n{results_text}")
        ])

        return {"final_result": output.final_result}

    return reducer
```

## Graph Construction

### Basic Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MapReduceState)

# Add nodes
workflow.add_node("mapper", create_mapper_node(llm, num_workers=3))
for i in range(3):
    workflow.add_node(f"worker_{i}", create_worker_node(llm, i))
workflow.add_node("reducer", create_reducer_node(llm))

# Build structure
workflow.add_edge(START, "mapper")

# Fan-out: mapper to all workers (parallel)
for i in range(3):
    workflow.add_edge("mapper", f"worker_{i}")

# Fan-in: all workers to reducer
for i in range(3):
    workflow.add_edge(f"worker_{i}", "reducer")

workflow.add_edge("reducer", END)

graph = workflow.compile()
```

### Execution

```python
result = graph.invoke({
    "task": "Analyze this document comprehensively",
    "subtasks": [],
    "worker_results": [],
    "final_result": "",
})

print(result["final_result"])
```

## Advanced Patterns

### Specialized Workers

Give each worker a specific role:

```python
worker_prompts = [
    "Focus on technical aspects",
    "Focus on business impact",
    "Focus on user experience",
]

for i in range(3):
    workflow.add_node(
        f"worker_{i}",
        create_worker_node(llm, i, worker_prompts[i])
    )
```

### Custom LLMs per Role

Use different models for different roles:

```python
from langgraph_ollama_local.patterns import create_custom_map_reduce_graph

graph = create_custom_map_reduce_graph(
    mapper_llm=ChatOllama(model="llama3.1:70b"),    # Larger for planning
    worker_llm=ChatOllama(model="llama3.1:8b"),     # Smaller for processing
    reducer_llm=ChatOllama(model="llama3.1:70b"),   # Larger for synthesis
    num_workers=5
)
```

### Dynamic Worker Count

Adjust workers based on task complexity:

```python
def determine_workers(task: str) -> int:
    """Determine optimal worker count based on task."""
    # Simple heuristic: more workers for longer tasks
    if len(task) > 1000:
        return 5
    elif len(task) > 500:
        return 3
    else:
        return 2

num_workers = determine_workers(task)
graph = create_map_reduce_graph(llm, num_workers=num_workers)
```

## API Reference

### State

```python
class MapReduceState(TypedDict):
    """State for map-reduce pattern."""
    task: str                                        # Main task description
    subtasks: list[str]                              # Subtasks from mapper
    worker_results: Annotated[list[dict], operator.add]  # Worker outputs
    final_result: str                                # Final aggregated result
```

### Node Creators

```python
def create_mapper_node(
    llm: BaseChatModel,
    num_workers: int = 3,
) -> callable:
    """Create mapper that splits tasks into subtasks."""

def create_worker_node(
    llm: BaseChatModel,
    worker_id: int,
    worker_prompt: str = "",
) -> callable:
    """Create worker that processes a subtask."""

def create_reducer_node(llm: BaseChatModel) -> callable:
    """Create reducer that aggregates worker results."""
```

### Graph Builders

```python
def create_map_reduce_graph(
    llm: BaseChatModel,
    num_workers: int = 3,
    worker_prompt: str = "",
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Create basic map-reduce graph."""

def create_custom_map_reduce_graph(
    mapper_llm: BaseChatModel,
    worker_llm: BaseChatModel,
    reducer_llm: BaseChatModel,
    num_workers: int = 3,
    worker_prompts: list[str] | None = None,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Create map-reduce graph with different LLMs per role."""
```

### Utilities

```python
def run_map_reduce_task(
    graph: CompiledStateGraph,
    task: str,
    thread_id: str = "default",
) -> dict:
    """Run a task through map-reduce system."""
```

## Pattern Comparison

### Map-Reduce vs Supervisor Pattern

| Aspect | Map-Reduce | Supervisor |
|--------|------------|------------|
| **Execution** | Parallel workers | Sequential agents |
| **Coordination** | None (independent) | High (supervisor decides) |
| **Use Case** | Independent subtasks | Interdependent tasks |
| **Speed** | Fast (parallel) | Slower (sequential) |
| **Complexity** | Simple (fan-out/fan-in) | Complex (routing logic) |

### Map-Reduce vs Hierarchical Teams

| Aspect | Map-Reduce | Hierarchical |
|--------|------------|--------------|
| **Structure** | Flat (mapper-workers-reducer) | Nested (teams and supervisors) |
| **Coordination** | Minimal | High (multi-level) |
| **Scalability** | Horizontal (add workers) | Vertical (add teams) |
| **Complexity** | Low | High |

## Best Practices

### 1. Task Decomposition

Ensure subtasks are truly independent:

```python
# Good: Independent sections
subtasks = [
    "Analyze introduction section",
    "Analyze methodology section",
    "Analyze results section",
]

# Bad: Sequential dependencies
subtasks = [
    "Read the document",
    "Based on reading, identify themes",  # Depends on first
    "Based on themes, write summary",     # Depends on second
]
```

### 2. Load Balancing

Create subtasks of similar complexity:

```python
# Good: Balanced subtasks
subtasks = [
    "Analyze chapters 1-3",
    "Analyze chapters 4-6",
    "Analyze chapters 7-9",
]

# Bad: Unbalanced
subtasks = [
    "Analyze chapter 1",
    "Analyze chapters 2-9",  # Much more work
]
```

### 3. Error Handling

Handle cases where workers fail:

```python
def reducer_with_error_handling(state):
    results = state.get("worker_results", [])

    if not results:
        return {"final_result": "No results available"}

    # Check for incomplete results
    if len(results) < expected_workers:
        partial_note = f"Note: Only {len(results)}/{expected_workers} workers completed."
    else:
        partial_note = ""

    # Synthesize available results
    synthesis = synthesize_results(results)

    return {"final_result": f"{synthesis}\n\n{partial_note}"}
```

### 4. Result Quality

Ensure reducer does true synthesis, not just concatenation:

```python
# Good: Synthesis with analysis
reducer_prompt = """
Synthesize the worker results by:
1. Identifying common themes across all outputs
2. Resolving any conflicts or inconsistencies
3. Organizing information logically
4. Providing integrated insights
"""

# Bad: Simple concatenation
# Don't just join worker outputs with "\n\n".join()
```

## Performance Considerations

### Worker Count

More workers ≠ always better:

- **Too few**: Underutilized parallelism
- **Too many**: Overhead, diminishing returns
- **Sweet spot**: Usually 3-5 workers for most tasks

### LLM Selection

Choose models based on role requirements:

```python
# Mapper: Needs good reasoning (larger model)
mapper_llm = ChatOllama(model="llama3.1:70b")

# Workers: Can use smaller models (parallel execution)
worker_llm = ChatOllama(model="llama3.1:8b")

# Reducer: Needs synthesis ability (larger model)
reducer_llm = ChatOllama(model="llama3.1:70b")
```

## Complete Example

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.patterns import (
    create_map_reduce_graph,
    run_map_reduce_task,
)

# Setup
config = LocalAgentConfig()
llm = config.create_chat_client()

# Create graph
graph = create_map_reduce_graph(
    llm,
    num_workers=3,
    worker_prompt="Provide detailed analysis with examples."
)

# Run task
result = run_map_reduce_task(
    graph,
    """Analyze the environmental, economic, and social impacts
    of renewable energy adoption in developing countries."""
)

print(f"Processed {len(result['worker_results'])} subtasks")
print(f"\nFinal Result:\n{result['final_result']}")
```

## Troubleshooting

### Issue: Workers receiving same subtask

**Problem**: All workers process the same subtask.

**Solution**: Ensure mapper creates unique subtasks:

```python
# Check mapper output
assert len(set(state["subtasks"])) == num_workers, "Subtasks must be unique"
```

### Issue: Reducer missing some results

**Problem**: Not all worker results in final output.

**Solution**: Verify all workers are connected to reducer:

```python
for i in range(num_workers):
    workflow.add_edge(f"worker_{i}", "reducer")  # Ensure all connected
```

### Issue: Poor synthesis quality

**Problem**: Reducer just concatenates instead of synthesizing.

**Solution**: Improve reducer prompt with specific instructions:

```python
reducer_prompt = """
You must synthesize, not just concatenate.
1. Identify themes across ALL worker outputs
2. Remove redundancy
3. Resolve conflicts
4. Organize logically
"""
```

## Further Reading

- [Tutorial 19: Map-Reduce Agents](../../examples/multi_agent_patterns/19_map_reduce_agents.ipynb)
- [LangGraph Documentation: Parallel Execution](https://langchain-ai.github.io/langgraph/)
- [MapReduce Pattern (Wikipedia)](https://en.wikipedia.org/wiki/MapReduce)
