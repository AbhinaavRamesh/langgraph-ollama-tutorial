---
layout: doc
title: Interactive Notebooks
description: Run tutorial notebooks directly in your browser with JupyterLite
---

# Interactive Notebooks

Run all 25 tutorial notebooks directly in your browser using [JupyterLite](https://jupyterlite.readthedocs.io/) — no installation required.

<a href="/notebooks/lab/index.html" target="_blank" class="doc-button brand">Open JupyterLite</a>

## Available Notebooks

### Phase 1: Core Patterns
| Notebook | Description |
|----------|-------------|
| `01_chatbot_basics.ipynb` | Build your first LangGraph chatbot |
| `02_tool_calling.ipynb` | Implement ReAct pattern with tools |
| `03_memory_persistence.ipynb` | Add conversation memory |
| `04_human_in_the_loop.ipynb` | Human approval workflows |
| `05_reflection.ipynb` | Self-improving agents |
| `06_plan_and_execute.ipynb` | Planning and execution patterns |
| `07_research_assistant.ipynb` | Multi-step research agent |

### Phase 2: RAG Patterns
| Notebook | Description |
|----------|-------------|
| `08_basic_rag.ipynb` | Basic retrieval-augmented generation |
| `09_self_rag.ipynb` | Self-correcting RAG |
| `10_crag.ipynb` | Corrective RAG with web fallback |
| `11_adaptive_rag.ipynb` | Query-adaptive retrieval |
| `12_agentic_rag.ipynb` | Agent-driven RAG |
| `13_perplexity_clone.ipynb` | Build a Perplexity-style search |

### Phase 3: Multi-Agent Patterns
| Notebook | Description |
|----------|-------------|
| `14_multi_agent_collaboration.ipynb` | Agent collaboration basics |
| `15_hierarchical_teams.ipynb` | Hierarchical agent teams |
| `16_subgraphs.ipynb` | Modular graph composition |
| `17_agent_handoffs.ipynb` | Agent-to-agent handoffs |
| `18_agent_swarm.ipynb` | Swarm-style agent systems |
| `19_map_reduce_agents.ipynb` | Parallel agent processing |
| `20_multi_agent_evaluation.ipynb` | Evaluating multi-agent systems |

### Phase 4: Advanced Reasoning
| Notebook | Description |
|----------|-------------|
| `21_plan_and_execute.ipynb` | Advanced planning patterns |
| `22_reflection.ipynb` | Advanced reflection techniques |
| `23_reflexion.ipynb` | Reflexion pattern implementation |
| `24_lats.ipynb` | Language Agent Tree Search |
| `25_rewoo.ipynb` | ReWOO reasoning pattern |

---

## How It Works

JupyterLite runs entirely in your browser using WebAssembly:

- **No server required** — Everything runs client-side
- **Instant startup** — No waiting for kernel initialization
- **Persistent storage** — Your work is saved in browser storage
- **Full Python** — Powered by Pyodide (Python 3.11)

::: warning Limitations
JupyterLite cannot make network requests to external services. Notebooks that require Ollama or other API calls will show simulated outputs. For full functionality, run notebooks locally with `langgraph-local serve`.
:::

## Running Locally

For the complete experience with actual LLM calls:

```bash
# Clone the repository
git clone https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial
cd langgraph-ollama-tutorial

# Install dependencies
pip install -e ".[dev]"

# Start the local environment
langgraph-local serve
```

This launches Jupyter with Ollama integration for real LLM interactions.
