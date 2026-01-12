---
title: Graph Visualizer
description: Visualize LangGraph state graphs interactively
---

# Graph Visualizer

Visualize and understand LangGraph state machine flows.

## How Graphs Work

LangGraph uses a **state machine** architecture where:

1. **Nodes** are functions that process and update state
2. **Edges** define the flow between nodes
3. **Conditional edges** enable dynamic routing based on state

## Example Architectures

### Simple Chatbot

```mermaid
flowchart LR
    START([START]) --> Agent[Agent Node]
    Agent --> END([END])

    style START fill:#e1f5fe,stroke:#01579b
    style END fill:#e8f5e9,stroke:#2e7d32
    style Agent fill:#e3f2fd,stroke:#1565c0
```

### ReAct Agent with Tools

```mermaid
flowchart TD
    START([START]) --> Agent[Agent]
    Agent --> ShouldContinue{Should<br/>Continue?}
    ShouldContinue -->|Yes, use tools| Tools[Tool Executor]
    ShouldContinue -->|No, done| END([END])
    Tools --> Agent

    style START fill:#e1f5fe,stroke:#01579b
    style END fill:#e8f5e9,stroke:#2e7d32
    style Agent fill:#fff3e0,stroke:#ef6c00
    style Tools fill:#e3f2fd,stroke:#1565c0
    style ShouldContinue fill:#f3e5f5,stroke:#7b1fa2
```

### Multi-Agent Supervisor

```mermaid
flowchart TD
    START([START]) --> Supervisor
    Supervisor[Supervisor] --> Route{Route to?}
    Route -->|Research| Researcher[Researcher]
    Route -->|Code| Coder[Coder]
    Route -->|Review| Reviewer[Reviewer]
    Route -->|Done| Synthesize[Synthesize]
    Researcher --> Supervisor
    Coder --> Supervisor
    Reviewer --> Supervisor
    Synthesize --> END([END])

    style START fill:#e1f5fe,stroke:#01579b
    style END fill:#e8f5e9,stroke:#2e7d32
    style Supervisor fill:#fff3e0,stroke:#ef6c00
    style Researcher fill:#e3f2fd,stroke:#1565c0
    style Coder fill:#e3f2fd,stroke:#1565c0
    style Reviewer fill:#e3f2fd,stroke:#1565c0
    style Route fill:#f3e5f5,stroke:#7b1fa2
    style Synthesize fill:#fce4ec,stroke:#c2185b
```

### RAG Pipeline

```mermaid
flowchart TD
    START([START]) --> Retrieve[Retrieve Documents]
    Retrieve --> Grade{Documents<br/>Relevant?}
    Grade -->|Yes| Generate[Generate Answer]
    Grade -->|No| WebSearch[Web Search]
    WebSearch --> Generate
    Generate --> CheckHallucination{Grounded<br/>in Facts?}
    CheckHallucination -->|Yes| END([END])
    CheckHallucination -->|No| Generate

    style START fill:#e1f5fe,stroke:#01579b
    style END fill:#e8f5e9,stroke:#2e7d32
    style Retrieve fill:#fff3e0,stroke:#ef6c00
    style Generate fill:#fce4ec,stroke:#c2185b
    style WebSearch fill:#e0f7fa,stroke:#00838f
    style Grade fill:#f3e5f5,stroke:#7b1fa2
    style CheckHallucination fill:#f3e5f5,stroke:#7b1fa2
```

### Plan-and-Execute

```mermaid
flowchart TD
    START([START]) --> Planner[Planner]
    Planner --> Executor[Executor]
    Executor --> Done{Plan<br/>Complete?}
    Done -->|No| Executor
    Done -->|Yes| Replanner[Replanner]
    Replanner --> HasResponse{Has<br/>Response?}
    HasResponse -->|Yes| END([END])
    HasResponse -->|No| Executor

    style START fill:#e1f5fe,stroke:#01579b
    style END fill:#e8f5e9,stroke:#2e7d32
    style Planner fill:#fff3e0,stroke:#ef6c00
    style Executor fill:#e3f2fd,stroke:#1565c0
    style Replanner fill:#f3e5f5,stroke:#7b1fa2
    style Done fill:#fce4ec,stroke:#c2185b
    style HasResponse fill:#fce4ec,stroke:#c2185b
```

## Color Legend

| Color | Meaning |
|-------|---------|
| <span style="display:inline-block;width:20px;height:20px;background:#e1f5fe;border:2px solid #01579b;border-radius:4px;vertical-align:middle"></span> Light Blue | START node |
| <span style="display:inline-block;width:20px;height:20px;background:#e8f5e9;border:2px solid #2e7d32;border-radius:4px;vertical-align:middle"></span> Green | END node |
| <span style="display:inline-block;width:20px;height:20px;background:#fff3e0;border:2px solid #ef6c00;border-radius:4px;vertical-align:middle"></span> Orange | Supervisor/Planner |
| <span style="display:inline-block;width:20px;height:20px;background:#e3f2fd;border:2px solid #1565c0;border-radius:4px;vertical-align:middle"></span> Blue | Worker/Agent nodes |
| <span style="display:inline-block;width:20px;height:20px;background:#f3e5f5;border:2px solid #7b1fa2;border-radius:4px;vertical-align:middle"></span> Purple | Decision/Routing |
| <span style="display:inline-block;width:20px;height:20px;background:#fce4ec;border:2px solid #c2185b;border-radius:4px;vertical-align:middle"></span> Pink | Output/Generate |

## Building Your Own

Learn to build these patterns step-by-step in our [tutorials](/tutorials/).
