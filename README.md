# LangGraph Ollama Local

**Local agent building at scale using Ollama** — A comprehensive collection of LangGraph patterns optimized for local LLM deployment.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides production-ready LangGraph agent patterns that run entirely on local hardware using [Ollama](https://ollama.ai). It's designed for developers who want to:

- Build AI agents without cloud API dependencies
- Run agents on LAN-accessible servers for team development
- Scale from laptop prototyping to multi-GPU production
- Learn LangGraph patterns with working local examples

**Powered by [ollama-local-serve](https://github.com/AbhinaavRamesh/ollama-local-serve)** — providing the infrastructure layer for Ollama connectivity, monitoring, and Kubernetes scaling.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    langgraph-ollama-local                        │
├─────────────────────────────────────────────────────────────────┤
│  agents/        ReAct, Multi-agent, Hierarchical teams          │
│  rag/           Self-RAG, CRAG, Adaptive RAG, Agentic RAG       │
│  patterns/      Plan-Execute, Reflection, LATS, ReWOO           │
│  memory/        Conversation history, Checkpointing             │
│  streaming/     Token streaming, Event streaming                │
├─────────────────────────────────────────────────────────────────┤
│                         DEPENDS ON                               │
├─────────────────────────────────────────────────────────────────┤
│  ollama-local-serve                                              │
│    ├── OllamaService, NetworkConfig                             │
│    ├── LangChain/LangGraph integration                          │
│    ├── OpenTelemetry metrics & tracing                          │
│    └── Kubernetes autoscaling                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.12+**
2. **Ollama** installed and running:
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama
   ollama serve

   # Pull a model
   ollama pull llama3.2:3b
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial.git
cd langgraph-ollama-tutorial

# Install in development mode
pip install -e ".[dev]"

# Verify connection
make ollama-check
```

### Your First Agent

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Define a simple tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Create configuration (loads from environment/.env)
config = LocalAgentConfig()

# Create a chat client connected to your local Ollama
llm = config.create_chat_client()

# Build a ReAct agent
agent = create_react_agent(llm, [multiply])

# Run the agent
result = agent.invoke({
    "messages": [("user", "What is 7 times 8?")]
})

print(result["messages"][-1].content)
```

## Configuration

All settings can be configured via environment variables or a `.env` file:

```bash
# Copy the example configuration
cp .env.example .env
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server address (use IP for LAN) |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model for agents |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `LANGGRAPH_RECURSION_LIMIT` | `25` | Max agent loop iterations |

### LAN Server Setup

To use a remote Ollama server on your network:

```bash
# In .env
OLLAMA_HOST=192.168.1.100
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:7b
```

```python
from langgraph_ollama_local import LocalAgentConfig

# Automatically loads from .env
config = LocalAgentConfig()
llm = config.create_chat_client()
```

## Project Structure

```
langgraph-ollama-tutorial/
├── langgraph_ollama_local/      # Main package
│   ├── config.py                # Configuration management
│   ├── cli.py                   # Command-line interface
│   ├── agents/                  # Agent implementations
│   │   ├── react.py             # ReAct agents
│   │   ├── react_memory.py      # ReAct with conversation memory
│   │   └── multi_agent.py       # Multi-agent collaboration
│   ├── rag/                     # RAG patterns
│   │   ├── self_rag.py          # Self-reflective RAG
│   │   ├── crag.py              # Corrective RAG
│   │   └── adaptive.py          # Adaptive RAG
│   ├── patterns/                # Advanced patterns
│   │   ├── plan_execute.py      # Plan-and-Execute
│   │   ├── reflection.py        # Self-reflection
│   │   └── lats.py              # Language Agent Tree Search
│   ├── memory/                  # Persistence
│   └── streaming/               # Streaming utilities
├── examples/                    # Jupyter notebooks
├── tests/                       # Test suite
├── Makefile                     # Development commands
└── pyproject.toml               # Package configuration
```

## Available Patterns

### Core Agents (Phase 2)
- **ReAct Agent** — Reasoning + Acting with tool use
- **ReAct with Memory** — Persistent conversation history
- **ReAct with HITL** — Human-in-the-loop approval
- **Tool Calling** — Robust error handling and retries

### RAG Patterns (Phase 3)
- **Self-RAG** — Self-reflective retrieval with grading
- **CRAG** — Corrective RAG with web fallback
- **Adaptive RAG** — Query-based strategy routing
- **Agentic RAG** — Multi-step retrieval planning

### Multi-Agent (Phase 4)
- **Collaboration** — Agents working together
- **Hierarchical Teams** — Nested team structures
- **Subgraphs** — Composable agent graphs

### Advanced Reasoning (Phase 5)
- **Plan-and-Execute** — Two-phase planning
- **Reflection** — Self-critique loops
- **Reflexion** — Learning from failures
- **LATS** — Monte Carlo Tree Search
- **ReWOO** — Reasoning Without Observations

## Development

```bash
# Install with all dependencies
make install-all

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Check Ollama connection
make ollama-check

# Show current configuration
make show-config
```

### Running Tests

```bash
# All tests
make test

# Quick tests (no coverage)
make test-quick

# Integration tests (requires Ollama)
make test-int

# With coverage report
make test-cov
```

## CLI Usage

```bash
# Check Ollama connectivity
langgraph-local check

# Show configuration
langgraph-local config

# List available examples
langgraph-local list

# Run an example
langgraph-local run react-agent
```

## Model Recommendations

| Use Case | Model | Notes |
|----------|-------|-------|
| Development/Testing | `llama3.2:1b` | Fast iteration, low memory |
| General Use | `llama3.2:3b` | Good balance |
| Production | `llama3.2:7b` | Better reasoning |
| Complex Tasks | `llama3.1:70b` | Requires significant GPU |

## Related Projects

- **[ollama-local-serve](https://github.com/AbhinaavRamesh/ollama-local-serve)** — Infrastructure layer powering this repository
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Framework for building stateful agents
- **[Ollama](https://ollama.ai)** — Local LLM runtime

## Roadmap

- [x] **Phase 1**: Repository foundation, configuration, tooling
- [ ] **Phase 2**: Core agent patterns (ReAct, memory, HITL)
- [ ] **Phase 3**: RAG patterns (Self-RAG, CRAG, Adaptive)
- [ ] **Phase 4**: Multi-agent patterns
- [ ] **Phase 5**: Advanced reasoning (LATS, ReWOO, Reflection)
- [ ] **Phase 6**: Streaming and persistence
- [ ] **Phase 7**: Docker and infrastructure integration
- [ ] **Phase 8**: Documentation and examples

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built for local-first AI development.**
