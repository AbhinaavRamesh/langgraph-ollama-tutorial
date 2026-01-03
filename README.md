# LangGraph Ollama Local

**Learn LangGraph by building agents that run entirely on your hardware.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What This Is

A hands-on tutorial series for building LangGraph agents with local LLMs via Ollama. Each notebook teaches a concept from scratch — no cloud APIs required.

**You'll learn:**
- LangGraph fundamentals: StateGraph, nodes, edges, reducers
- Tool calling and the ReAct pattern
- Memory and conversation persistence
- Human-in-the-loop workflows
- Self-reflection and critique patterns
- Plan-and-execute architectures

## Quick Start

### Prerequisites

1. **Python 3.12+**
2. **Ollama** running locally or on your LAN:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull a model
   ollama pull llama3.2:3b
   ```

### Installation

```bash
git clone https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial.git
cd langgraph-ollama-tutorial

pip install -e ".[dev]"

# Verify connection
langgraph-local check
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your Ollama server settings
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server address |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model |

### LAN Server with Monitoring

To host Ollama on a GPU machine accessible across your local network (with monitoring dashboard, GPU metrics, and model management), use **[ollama-local-serve](https://github.com/AbhinaavRamesh/ollama-local-serve)**:

```bash
pip install ollama-local-serve[all]

# Start the full stack (Ollama + Dashboard + Metrics)
make init && make up
# Dashboard at http://your-server:3000
```

Then configure this tutorial repo to use your LAN server:
```bash
# .env
OLLAMA_HOST=192.168.1.100
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:7b
```

**ollama-local-serve** provides:
- Real-time monitoring dashboard
- GPU utilization tracking
- Request/response logging
- Model management UI
- OpenTelemetry metrics

## Tutorials

| # | Notebook | Documentation | What You'll Learn |
|---|----------|---------------|-------------------|
| 01 | [Chatbot Basics](examples/01_chatbot_basics.ipynb) | [docs](docs/01-chatbot-basics.md) | StateGraph, nodes, edges, message handling |
| 02 | [Tool Calling](examples/02_tool_calling.ipynb) | [docs](docs/02-tool-calling.md) | Tools, ReAct loop from scratch |
| 03 | [Memory & Persistence](examples/03_memory_persistence.ipynb) | [docs](docs/03-memory-persistence.md) | Checkpointers, threads, conversation history |
| 04 | [Human-in-the-Loop](examples/04_human_in_the_loop.ipynb) | [docs](docs/04-human-in-the-loop.md) | Interrupts, approvals, resume |
| 05 | [Reflection](examples/05_reflection.ipynb) | [docs](docs/05-reflection.md) | Generate → Critique → Revise loops |
| 06 | [Plan & Execute](examples/06_plan_and_execute.ipynb) | [docs](docs/06-plan-and-execute.md) | Structured outputs, multi-step planning |
| 07 | [Research Assistant](examples/07_research_assistant.ipynb) | [docs](docs/07-research-assistant.md) | Capstone: all patterns combined |

Run any notebook:
```bash
jupyter lab examples/
```

## Project Structure

```
langgraph-ollama-tutorial/
├── examples/                    # Tutorial notebooks (start here!)
│   ├── 01_chatbot_basics.ipynb
│   ├── 02_tool_calling.ipynb
│   ├── 03_memory_persistence.ipynb
│   ├── 04_human_in_the_loop.ipynb
│   ├── 05_reflection.ipynb
│   ├── 06_plan_and_execute.ipynb
│   └── 07_research_assistant.ipynb
├── docs/                        # Detailed documentation
│   ├── 01-chatbot-basics.md
│   ├── ...
│   └── images/                  # Graph visualizations
├── langgraph_ollama_local/      # Helper library
│   ├── config.py                # Configuration + model management
│   └── cli.py                   # CLI tools
├── scripts/                     # Utility scripts
│   └── generate_diagrams.py     # Generate graph visualizations
├── tests/                       # Test suite (46 tests)
└── pyproject.toml
```

## CLI

```bash
langgraph-local check    # Verify Ollama connection
langgraph-local config   # Show current configuration
langgraph-local list     # List available examples
```

## Development

```bash
make test          # Run tests
make test-int      # Integration tests (requires Ollama)
make lint          # Linting
make format        # Format code
```

## Adapted From

These tutorials are adapted from the official [LangGraph documentation](https://langchain-ai.github.io/langgraph/) (MIT License), optimized for local Ollama deployment.

## License

MIT License
