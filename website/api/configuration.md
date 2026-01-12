---
title: Configuration API
description: Configuration classes for LocalGraph
---

# Configuration API

Complete reference for configuration classes and environment variables.

## LocalAgentConfig

Main configuration class for local agent development. Combines Ollama connection settings with LangGraph-specific settings.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ollama` | `OllamaConfig` | `OllamaConfig()` | Ollama server connection settings |
| `langgraph` | `LangGraphConfig` | `LangGraphConfig()` | LangGraph behavior settings |

### Methods

#### create_chat_client()

Create a LangChain chat client configured for this setup.

```python
def create_chat_client(
    self,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel
```

**Parameters:**
- `model` (str | None): Override the default model. If None, uses config.ollama.model
- `temperature` (float | None): Override the default temperature. If None, uses config.ollama.temperature
- `**kwargs`: Additional arguments passed to ChatOllama

**Returns:** A configured ChatOllama instance ready for use with LangGraph

**Example:**
```python
from langgraph_ollama_local import LocalAgentConfig

config = LocalAgentConfig()
llm = config.create_chat_client(model="llama3.2:7b", temperature=0.7)
response = llm.invoke("Hello!")
```

#### create_checkpointer()

Create a checkpoint saver for agent state persistence.

```python
def create_checkpointer(
    self,
    backend: Literal["memory", "sqlite", "postgres", "redis"] = "memory",
    **kwargs: Any,
) -> BaseCheckpointSaver
```

**Parameters:**
- `backend` (str): The persistence backend to use
  - `memory`: In-memory (default, lost on restart)
  - `sqlite`: SQLite file-based
  - `postgres`: PostgreSQL database
  - `redis`: Redis key-value store
- `**kwargs`: Backend-specific configuration

**Returns:** A configured checkpoint saver

**Raises:**
- `ImportError`: If the required backend package is not installed
- `ValueError`: If an unknown backend is specified

**Example:**
```python
# Memory checkpointer (default)
checkpointer = config.create_checkpointer(backend="memory")

# SQLite checkpointer
checkpointer = config.create_checkpointer(
    backend="sqlite",
    db_path="checkpoints.db"
)

# PostgreSQL checkpointer
checkpointer = config.create_checkpointer(
    backend="postgres",
    conn_string="postgresql://user:pass@localhost/db"
)
```

#### get_graph_config()

Get configuration dictionary for LangGraph graph execution.

```python
def get_graph_config(self) -> dict[str, Any]
```

**Returns:** Dictionary with recursion_limit and other graph settings

**Example:**
```python
config = LocalAgentConfig()
graph_config = config.get_graph_config()
result = graph.invoke(input, config=graph_config)
```

### Complete Example

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Create configuration
config = LocalAgentConfig()

# Create chat client
llm = config.create_chat_client()

# Create checkpointer
checkpointer = config.create_checkpointer(backend="memory")

# Create agent
agent = create_react_agent(llm, [add], checkpointer=checkpointer)

# Run agent
result = agent.invoke(
    {"messages": [("user", "What is 25 + 17?")]},
    config=config.get_graph_config()
)
```

---

## OllamaConfig

Ollama server connection configuration. All settings support environment variables with the `OLLAMA_` prefix.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"127.0.0.1"` | Ollama server hostname or IP address |
| `port` | `int` | `11434` | Ollama server port (1-65535) |
| `model` | `str` | `"llama3.1:8b"` | Default model for agents |
| `timeout` | `int` | `120` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Maximum retry attempts for failed requests |
| `temperature` | `float` | `0.0` | Default temperature for responses (0.0-2.0) |
| `num_ctx` | `int` | `4096` | Context window size |

### Computed Properties

#### base_url

Get the base URL for the Ollama service.

```python
@property
def base_url(self) -> str
```

**Returns:** Formatted URL string (e.g., "http://127.0.0.1:11434")

### Environment Variables

All parameters can be set via environment variables with the `OLLAMA_` prefix:

```bash
OLLAMA_HOST=192.168.1.100
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2:7b
OLLAMA_TIMEOUT=180
OLLAMA_MAX_RETRIES=5
OLLAMA_TEMPERATURE=0.7
OLLAMA_NUM_CTX=8192
```

### Example

```python
from langgraph_ollama_local.config import OllamaConfig

# Default configuration
config = OllamaConfig()
print(config.base_url)  # http://127.0.0.1:11434

# Custom configuration
config = OllamaConfig(
    host="192.168.1.100",
    model="llama3.2:7b",
    temperature=0.7,
    num_ctx=8192
)
```

---

## LangGraphConfig

LangGraph-specific configuration settings. All settings support environment variables with the `LANGGRAPH_` prefix.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recursion_limit` | `int` | `25` | Maximum recursion depth for agent loops (1-100) |
| `checkpoint_dir` | `Path` | `.checkpoints` | Directory for storing agent state checkpoints |
| `enable_streaming` | `bool` | `True` | Enable token streaming by default |
| `thread_id_prefix` | `str` | `"thread"` | Prefix for auto-generated thread IDs |

### Validation

The `ensure_checkpoint_dir` validator automatically creates the checkpoint directory if it doesn't exist.

### Environment Variables

All parameters can be set via environment variables with the `LANGGRAPH_` prefix:

```bash
LANGGRAPH_RECURSION_LIMIT=50
LANGGRAPH_CHECKPOINT_DIR=.my_checkpoints
LANGGRAPH_ENABLE_STREAMING=false
LANGGRAPH_THREAD_ID_PREFIX=my_thread
```

### Example

```python
from langgraph_ollama_local.config import LangGraphConfig
from pathlib import Path

# Default configuration
config = LangGraphConfig()

# Custom configuration
config = LangGraphConfig(
    recursion_limit=50,
    checkpoint_dir=Path(".my_checkpoints"),
    enable_streaming=True,
    thread_id_prefix="my_app"
)
```

---

## Convenience Functions

### pull_model()

Pull a model from Ollama registry.

```python
def pull_model(
    model: str,
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: int = 600,
) -> bool
```

**Parameters:**
- `model` (str): The model name to pull (e.g., 'llama3.2:3b')
- `host` (str): The Ollama server host
- `port` (int): The Ollama server port
- `timeout` (int): Timeout in seconds for the pull operation

**Returns:** True if the model was pulled successfully, False otherwise

**Example:**
```python
from langgraph_ollama_local import pull_model

# Pull a model
success = pull_model("llama3.2:3b")
if success:
    print("Model ready to use!")
```

### list_models()

List available models on the Ollama server.

```python
def list_models(
    host: str = "127.0.0.1",
    port: int = 11434,
) -> list[str]
```

**Parameters:**
- `host` (str): The Ollama server host
- `port` (int): The Ollama server port

**Returns:** List of model names available on the server

**Example:**
```python
from langgraph_ollama_local import list_models

models = list_models()
print(f"Available models: {models}")
```

### ensure_model()

Ensure a model is available, pulling it if necessary.

```python
def ensure_model(
    model: str,
    host: str = "127.0.0.1",
    port: int = 11434,
) -> bool
```

**Parameters:**
- `model` (str): The model name to ensure is available
- `host` (str): The Ollama server host
- `port` (int): The Ollama server port

**Returns:** True if the model is available (was already present or pulled)

**Example:**
```python
from langgraph_ollama_local import ensure_model

# Ensure model is available before running tutorial
ensure_model("llama3.2:3b")
```

### create_quick_client()

Quickly create a chat client with minimal configuration.

```python
def create_quick_client(
    model: str = "llama3.2:3b",
    host: str = "127.0.0.1",
    port: int = 11434,
) -> BaseChatModel
```

**Parameters:**
- `model` (str): The Ollama model to use
- `host` (str): The Ollama server host
- `port` (int): The Ollama server port

**Returns:** A configured ChatOllama instance

**Example:**
```python
from langgraph_ollama_local import create_quick_client

# Quick client for prototyping
llm = create_quick_client(model="llama3.2:1b")
response = llm.invoke("Hello!")
```

---

## Configuration Hierarchy

Settings are loaded in the following priority order (highest to lowest):

1. **Programmatic configuration** (explicit arguments)
2. **Environment variables**
3. **.env file**
4. **Default values**

### Example with .env file

```bash
# .env file
OLLAMA_HOST=192.168.1.100
OLLAMA_MODEL=llama3.2:7b
LANGGRAPH_RECURSION_LIMIT=50
```

```python
from langgraph_ollama_local import LocalAgentConfig

# Loads from .env file
config = LocalAgentConfig()
print(config.ollama.host)  # 192.168.1.100
print(config.ollama.model)  # llama3.2:7b
print(config.langgraph.recursion_limit)  # 50

# Override programmatically
config = LocalAgentConfig(
    ollama=OllamaConfig(model="llama3.1:8b")  # Overrides .env
)
```

---

## Related

- [Setup & Installation](/tutorials/setup) - Getting started guide
- [RAG API](/api/rag) - RAG configuration options
- [Tutorial: Chatbot Basics](/tutorials/core/01-chatbot-basics) - First tutorial using configuration
