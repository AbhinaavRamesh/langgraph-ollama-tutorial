# LangGraph Ollama Local

**Learn LangGraph by building agents that run entirely on your hardware.**

[![PyPI version](https://img.shields.io/pypi/v/langgraph-ollama-local.svg)](https://pypi.org/project/langgraph-ollama-local/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What This Is

A hands-on tutorial series for building LangGraph agents with local LLMs via Ollama. Each notebook teaches a concept from scratch - no cloud APIs required.

**You'll learn:**
- LangGraph fundamentals: StateGraph, nodes, edges, reducers
- Tool calling and the ReAct pattern
- Memory and conversation persistence
- Human-in-the-loop workflows
- Self-reflection and critique patterns
- **RAG patterns**: Basic RAG, Self-RAG, CRAG, Adaptive RAG, Agentic RAG
- Build a Perplexity-style research assistant

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

**Option 1: Install from PyPI (Recommended)**

```bash
# Install the package
pip install langgraph-ollama-local

# With all features (RAG, persistence, notebooks)
pip install langgraph-ollama-local[all]

# Verify connection
langgraph-local check
```

**Option 2: Install from source (for development)**

```bash
git clone https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial.git
cd langgraph-ollama-tutorial

# Install with all dependencies
pip install -e ".[all]"

# Verify connection
langgraph-local check
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server address |
| `OLLAMA_PORT` | `11434` | Ollama server port |
| `OLLAMA_MODEL` | `llama3.2:3b` | Default model |
| `TAVILY_API_KEY` | (optional) | For web search in CRAG tutorials |

### Web Search Setup (Optional)

For CRAG and Perplexity-style tutorials, get a free Tavily API key:

1. Sign up at https://tavily.com
2. Get your API key from the dashboard
3. Add to `.env`:
   ```
   TAVILY_API_KEY=tvly-your-key-here
   ```

### LAN Server with Monitoring

To host Ollama on a GPU machine accessible across your network, use **[ollama-local-serve](https://github.com/AbhinaavRamesh/ollama-local-serve)**:

```bash
pip install ollama-local-serve[all]
make init && make up
# Dashboard at http://your-server:3000
```

## Tutorials

### Core Patterns (Phase 2)

| # | Notebook | Documentation | What You'll Learn |
|---|----------|---------------|-------------------|
| 01 | [Chatbot Basics](examples/core_patterns/01_chatbot_basics.ipynb) | [docs](docs/core_patterns/01-chatbot-basics.md) | StateGraph, nodes, edges, message handling |
| 02 | [Tool Calling](examples/core_patterns/02_tool_calling.ipynb) | [docs](docs/core_patterns/02-tool-calling.md) | Tools, ReAct loop from scratch |
| 03 | [Memory & Persistence](examples/core_patterns/03_memory_persistence.ipynb) | [docs](docs/core_patterns/03-memory-persistence.md) | Checkpointers, threads, conversation history |
| 04 | [Human-in-the-Loop](examples/core_patterns/04_human_in_the_loop.ipynb) | [docs](docs/core_patterns/04-human-in-the-loop.md) | Interrupts, approvals, resume |
| 05 | [Reflection](examples/core_patterns/05_reflection.ipynb) | [docs](docs/core_patterns/05-reflection.md) | Generate, Critique, Revise loops |
| 06 | [Plan & Execute](examples/core_patterns/06_plan_and_execute.ipynb) | [docs](docs/core_patterns/06-plan-and-execute.md) | Structured outputs, multi-step planning |
| 07 | [Research Assistant](examples/core_patterns/07_research_assistant.ipynb) | [docs](docs/core_patterns/07-research-assistant.md) | Capstone: all patterns combined |

### RAG Patterns (Phase 3)

| # | Notebook | Documentation | What You'll Learn |
|---|----------|---------------|-------------------|
| 08 | [Basic RAG](examples/rag_patterns/08_basic_rag.ipynb) | [docs](docs/rag_patterns/08-basic-rag.md) | Document loading, chunking, embeddings, ChromaDB |
| 09 | [Self-RAG](examples/rag_patterns/09_self_rag.ipynb) | [docs](docs/rag_patterns/09-self-rag.md) | Document grading, hallucination detection, retry loops |
| 10 | [CRAG](examples/rag_patterns/10_crag.ipynb) | [docs](docs/rag_patterns/10-crag.md) | Web search fallback, corrective retrieval |
| 11 | [Adaptive RAG](examples/rag_patterns/11_adaptive_rag.ipynb) | [docs](docs/rag_patterns/11-adaptive-rag.md) | Query classification, strategy routing |
| 12 | [Agentic RAG](examples/rag_patterns/12_agentic_rag.ipynb) | [docs](docs/rag_patterns/12-agentic-rag.md) | Agent-controlled retrieval, multi-step |
| 13 | [Perplexity Clone](examples/rag_patterns/13_perplexity_clone.ipynb) | [docs](docs/rag_patterns/13-perplexity-clone.md) | Citations, source metadata, follow-ups |

Run any notebook:
```bash
jupyter lab examples/
```

## RAG Quick Start

Index your documents and start querying:

```python
from langgraph_ollama_local.rag import DocumentIndexer, LocalRetriever

# Index documents
indexer = DocumentIndexer()
indexer.index_directory("sources/")

# Query
retriever = LocalRetriever()
docs = retriever.retrieve_documents("What is Self-RAG?", k=4)
```

## Project Structure

```
langgraph-ollama-tutorial/
├── examples/
│   ├── core_patterns/          # Tutorials 01-07
│   │   ├── 01_chatbot_basics.ipynb
│   │   └── ...
│   └── rag_patterns/           # Tutorials 08-13
│       ├── 08_basic_rag.ipynb
│       └── ...
├── docs/
│   ├── core_patterns/          # Core pattern documentation
│   └── rag_patterns/           # RAG pattern documentation
├── sources/                    # PDF sources for RAG indexing
├── langgraph_ollama_local/
│   ├── config.py               # Configuration
│   ├── cli.py                  # CLI tools
│   └── rag/                    # RAG infrastructure
│       ├── embeddings.py       # Local embeddings
│       ├── indexer.py          # Document indexing
│       ├── retriever.py        # Document retrieval
│       └── graders.py          # Quality graders
├── tests/                      # Test suite
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
