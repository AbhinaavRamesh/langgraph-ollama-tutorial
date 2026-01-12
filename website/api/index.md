---
title: API Reference
description: Complete API reference for langgraph-ollama-local package
---

# API Reference

Complete reference documentation for the `langgraph-ollama-local` package. All functions, classes, and patterns for building local AI agents with LangGraph and Ollama.

## Quick Navigation

<div class="api-nav-grid">

### [Configuration](/api/configuration)
Setup and configuration for local LLMs and LangGraph
- `LocalAgentConfig` - Main configuration class
- `OllamaConfig` - Ollama server settings
- `LangGraphConfig` - LangGraph execution settings
- Environment variable reference

### [RAG](/api/rag)
Document loading, indexing, and retrieval
- `DocumentLoader` - Multi-format document loading
- `DocumentIndexer` - ChromaDB indexing pipeline
- `LocalEmbeddings` - Local embedding models
- Graders: Document, Hallucination, Answer, Query Router

### [Multi-Agent](/api/agents)
Multi-agent collaboration patterns
- `create_multi_agent_graph()` - Supervisor-based coordination
- `create_hierarchical_graph()` - Nested team structures
- `MultiAgentState`, `TeamState`, `HierarchicalState`
- Supervisor and team patterns

### [Patterns](/api/patterns)
Advanced multi-agent patterns
- Swarm - Decentralized agent networks
- Handoff - Peer-to-peer agent transfers
- Map-Reduce - Parallel agent execution
- Evaluation - Automated agent testing

### [State Types](/api/types)
TypedDict state schemas for all patterns
- Multi-Agent states
- Pattern-specific states
- Custom state creation
- Reducer reference

</div>

---

## Getting Started

### Installation & Setup

```bash
# Install the package
pip install langgraph-ollama-local

# Install with RAG support
pip install langgraph-ollama-local[rag]

# Install with all features
pip install langgraph-ollama-local[all]
```

### Basic Configuration

```python
from langgraph_ollama_local import LocalAgentConfig

# Create configuration
config = LocalAgentConfig()

# Create LLM client
llm = config.create_chat_client()

# Create checkpointer for persistence
checkpointer = config.create_checkpointer(backend="memory")
```

### Your First Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Create agent
agent = create_react_agent(llm, [multiply])

# Run agent
result = agent.invoke({
    "messages": [("user", "What is 25 times 17?")]
})

print(result["messages"][-1].content)
```

---

## Core Concepts

### Configuration

The `LocalAgentConfig` class provides a unified interface for configuring:
- **Ollama connection**: Model selection, server URL, temperature
- **LangGraph settings**: Recursion limits, checkpointing, streaming
- **Environment variables**: Load from `.env` files

```python
from langgraph_ollama_local import LocalAgentConfig

config = LocalAgentConfig()
llm = config.create_chat_client(model="llama3.2:7b")
```

[Learn more →](/api/configuration)

---

### RAG (Retrieval-Augmented Generation)

Build knowledge-grounded agents with document retrieval:

```python
from langgraph_ollama_local.rag import DocumentIndexer, DocumentGrader

# Index documents
indexer = DocumentIndexer()
indexer.index_directory("sources/")

# Grade relevance
grader = DocumentGrader(llm)
is_relevant = grader.grade(document, question)
```

[Learn more →](/api/rag)

---

### Multi-Agent Collaboration

Coordinate multiple specialized agents with supervisors:

```python
from langgraph_ollama_local.agents import (
    create_multi_agent_graph,
    run_multi_agent_task
)

graph = create_multi_agent_graph(llm)

result = run_multi_agent_task(
    graph,
    task="Create a Python function to validate email addresses"
)
```

[Learn more →](/api/agents)

---

### Advanced Patterns

Implement sophisticated multi-agent patterns:

#### Swarm Pattern
```python
from langgraph_ollama_local.patterns.swarm import SwarmAgent, create_swarm_graph

agents = [
    SwarmAgent(name="researcher", system_prompt="...", connections=["analyst"]),
    SwarmAgent(name="analyst", system_prompt="...", connections=["writer"]),
    SwarmAgent(name="writer", system_prompt="...", connections=[]),
]

graph = create_swarm_graph(llm, agents)
```

#### Handoff Pattern
```python
from langgraph_ollama_local.patterns.handoffs import create_handoff_tool

handoff_to_support = create_handoff_tool("support", "Transfer for technical issues")
```

#### Map-Reduce Pattern
```python
from langgraph_ollama_local.patterns.map_reduce import create_map_reduce_graph

graph = create_map_reduce_graph(llm, num_workers=5)
```

[Learn more →](/api/patterns)

---

## Common Use Cases

### RAG Applications

Build question-answering systems with document grounding:

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.rag import DocumentIndexer, DocumentGrader

config = LocalAgentConfig()
llm = config.create_chat_client()

# Index documents
indexer = DocumentIndexer()
indexer.index_directory("docs/")

# Create grader for quality control
grader = DocumentGrader(llm)
```

**Related tutorials:**
- [Basic RAG](/tutorials/rag/08-basic-rag)
- [Self-RAG](/tutorials/rag/09-self-rag)
- [Corrective RAG](/tutorials/rag/10-crag)
- [Adaptive RAG](/tutorials/rag/11-adaptive-rag)

---

### Multi-Agent Systems

Build teams of specialized agents:

```python
from langgraph_ollama_local.agents import create_multi_agent_graph

# Supervisor coordinates researcher, coder, and reviewer
graph = create_multi_agent_graph(
    llm,
    researcher_tools=[search_tool],
    coder_tools=[python_executor],
    reviewer_tools=[linter]
)
```

**Related tutorials:**
- [Multi-Agent Collaboration](/tutorials/multi-agent/14-multi-agent-collaboration)
- [Hierarchical Teams](/tutorials/multi-agent/15-hierarchical-teams)

---

### Evaluation & Testing

Automate agent testing with simulated users:

```python
from langgraph_ollama_local.patterns.evaluation import (
    SimulatedUser,
    create_evaluation_graph,
    run_multiple_evaluations
)

user_config = SimulatedUser(
    persona="Frustrated customer with billing issue",
    goals=["Get refund", "Express dissatisfaction"],
    behavior="impatient"
)

graph = create_evaluation_graph(llm, my_agent, user_config)
results = run_multiple_evaluations(graph, num_sessions=10)
```

**Related tutorial:**
- [Multi-Agent Evaluation](/tutorials/multi-agent/20-multi-agent-evaluation)

---

## API Structure

### By Category

#### Configuration & Setup
- [LocalAgentConfig](/api/configuration#localagentconfig)
- [OllamaConfig](/api/configuration#ollamaconfig)
- [LangGraphConfig](/api/configuration#langgraphconfig)
- [Environment Variables](/api/configuration#environment-variables)

#### Document Processing
- [DocumentLoader](/api/rag#documentloader)
- [DocumentIndexer](/api/rag#documentindexer)
- [LocalEmbeddings](/api/rag#localembeddings)

#### Quality Grading
- [DocumentGrader](/api/rag#documentgrader)
- [HallucinationGrader](/api/rag#hallucinationgrader)
- [AnswerGrader](/api/rag#answergrader)
- [QueryRouter](/api/rag#queryrouter)

#### Multi-Agent Graphs
- [create_multi_agent_graph()](/api/agents#create_multi_agent_graph)
- [create_team_graph()](/api/agents#create_team_graph)
- [create_hierarchical_graph()](/api/agents#create_hierarchical_graph)

#### Pattern Graphs
- [create_swarm_graph()](/api/patterns#create_swarm_graph)
- [create_handoff_graph()](/api/patterns#create_handoff_graph)
- [create_map_reduce_graph()](/api/patterns#create_map_reduce_graph)
- [create_evaluation_graph()](/api/patterns#create_evaluation_graph)

#### State Types
- [MultiAgentState](/api/types#multiagentstate)
- [SwarmState](/api/types#swarmstate)
- [HandoffState](/api/types#handoffstate)
- [MapReduceState](/api/types#mapreducestate)
- [EvaluationState](/api/types#evaluationstate)

---

## Utility Functions

### Model Management

```python
from langgraph_ollama_local import (
    pull_model,
    list_models,
    ensure_model,
    create_quick_client
)

# Pull a model
pull_model("llama3.2:3b")

# List available models
models = list_models()

# Ensure model is available
ensure_model("llama3.2:3b")

# Quick client for prototyping
llm = create_quick_client(model="llama3.2:1b")
```

[Learn more →](/api/configuration#convenience-functions)

---

## Environment Configuration

All settings can be configured via environment variables:

```bash
# Ollama settings
OLLAMA_HOST=192.168.1.100
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:7b
OLLAMA_TEMPERATURE=0.7

# LangGraph settings
LANGGRAPH_RECURSION_LIMIT=50
LANGGRAPH_CHECKPOINT_DIR=.checkpoints

# RAG settings
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=300
RAG_COLLECTION_NAME=my_docs
RAG_EMBEDDING_MODEL=all-mpnet-base-v2
```

Or use a `.env` file:

```python
from langgraph_ollama_local import LocalAgentConfig

# Automatically loads .env file
config = LocalAgentConfig()
```

[Learn more →](/api/configuration#configuration-hierarchy)

---

## Examples by Complexity

### Beginner: Simple Agent

```python
from langgraph_ollama_local import create_quick_client
from langgraph.prebuilt import create_react_agent

llm = create_quick_client()
agent = create_react_agent(llm, [])

result = agent.invoke({"messages": [("user", "Hello!")]})
```

### Intermediate: RAG System

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.rag import DocumentIndexer, DocumentGrader

config = LocalAgentConfig()
llm = config.create_chat_client()

indexer = DocumentIndexer()
indexer.index_directory("sources/")

grader = DocumentGrader(llm)
relevant, irrelevant = grader.grade_documents(docs, question)
```

### Advanced: Multi-Agent System

```python
from langgraph_ollama_local.agents import create_hierarchical_graph, create_team_graph

research_team = create_team_graph(llm, "research", [
    ("searcher", "Search information", [search_tool]),
    ("analyst", "Analyze findings", None)
])

dev_team = create_team_graph(llm, "development", [
    ("frontend", "Build UI", [ui_tool]),
    ("backend", "Build API", [db_tool])
])

graph = create_hierarchical_graph(llm, {
    "research": research_team,
    "development": dev_team
})
```

---

## Best Practices

### 1. Configuration Management

```python
# Good: Use unified configuration
from langgraph_ollama_local import LocalAgentConfig

config = LocalAgentConfig()
llm = config.create_chat_client()

# Good: Use environment variables for deployment
# Set OLLAMA_MODEL=llama3.2:70b in production
```

### 2. State Persistence

```python
# Good: Use appropriate backend for your use case
checkpointer = config.create_checkpointer(backend="sqlite")  # Development
checkpointer = config.create_checkpointer(backend="postgres")  # Production
```

### 3. Error Handling

```python
# Good: Handle failures gracefully
try:
    result = agent.invoke(input)
except Exception as e:
    logger.error(f"Agent failed: {e}")
    # Fallback logic
```

### 4. Resource Management

```python
# Good: Set appropriate limits
graph = create_multi_agent_graph(llm)
result = run_multi_agent_task(graph, task, max_iterations=10)
```

---

## Migration Guide

### From LangChain

If you're familiar with LangChain, this package provides opinionated wrappers:

```python
# LangChain way
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

# This package's way (with configuration management)
from langgraph_ollama_local import LocalAgentConfig
config = LocalAgentConfig()
llm = config.create_chat_client()  # Loads from .env automatically
```

### Adding Multi-Agent Support

```python
# Start with simple agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools)

# Upgrade to multi-agent
from langgraph_ollama_local.agents import create_multi_agent_graph
graph = create_multi_agent_graph(llm, researcher_tools=tools)
```

---

## Troubleshooting

### Common Issues

**Model not found:**
```python
from langgraph_ollama_local import ensure_model
ensure_model("llama3.2:3b")  # Pulls if needed
```

**ChromaDB issues:**
```bash
pip install langgraph-ollama-local[rag]
```

**Connection errors:**
```python
config = LocalAgentConfig(
    ollama=OllamaConfig(host="192.168.1.100", timeout=180)
)
```

---

## Further Reading

### Tutorials
- [Getting Started](/tutorials/) - Introduction and setup
- [Core Patterns](/tutorials/core/01-chatbot-basics) - Basic agent patterns
- [RAG Patterns](/tutorials/rag/08-basic-rag) - Document retrieval
- [Multi-Agent](/tutorials/multi-agent/14-multi-agent-collaboration) - Agent collaboration

### External Resources
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Documentation](https://python.langchain.com/)

---

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial/discussions)

<style>
.api-nav-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.api-nav-grid > div {
  padding: 1.5rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
}

.api-nav-grid h3 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  font-size: 1.25rem;
}

.api-nav-grid h3 a {
  text-decoration: none;
  color: var(--vp-c-brand);
}

.api-nav-grid p {
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}

.api-nav-grid ul {
  margin: 0.75rem 0 0 0;
  padding-left: 1.25rem;
}

.api-nav-grid li {
  margin: 0.25rem 0;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}
</style>
