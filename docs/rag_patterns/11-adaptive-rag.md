# Tutorial 11: Adaptive RAG

Adaptive RAG intelligently routes queries to the optimal retrieval strategy based on question type.

## Overview

Not all questions need the same approach:
- **Document questions** → Vector store
- **Current events** → Web search
- **Simple factual** → Direct LLM response

Adaptive RAG classifies and routes accordingly.

## Architecture

```mermaid
flowchart TB
    subgraph AdaptiveRAG["Adaptive RAG Flow"]
        START([START]) --> Classify[Query Router]
        Classify --> Route{Route Decision}
        Route -->|vectorstore| VectorStore[Vector Store]
        Route -->|websearch| WebSearch[Web Search]
        Route -->|direct| Direct[Direct LLM]
        VectorStore --> Generate[Generate]
        WebSearch --> Generate
        Direct --> END1([END])
        Generate --> END2([END])
    end

    style START fill:#e1f5fe,stroke:#01579b
    style END1 fill:#e8f5e9,stroke:#2e7d32
    style END2 fill:#e8f5e9,stroke:#2e7d32
    style Classify fill:#fff3e0,stroke:#ef6c00
    style Route fill:#f3e5f5,stroke:#7b1fa2
    style VectorStore fill:#e3f2fd,stroke:#1565c0
    style WebSearch fill:#e0f7fa,stroke:#00838f
    style Direct fill:#fff8e1,stroke:#f9a825
    style Generate fill:#fce4ec,stroke:#c2185b
```

## Query Router

The QueryRouter classifies questions:

```python
from langgraph_ollama_local.rag.graders import QueryRouter

router = QueryRouter(llm)

# Route examples
router.route("What is Self-RAG?")         # → "vectorstore"
router.route("Latest AI news today?")     # → "websearch"
router.route("What is 2 + 2?")            # → "direct"
```

## State Definition

```python
class AdaptiveRAGState(TypedDict):
    question: str
    query_type: Literal["vectorstore", "websearch", "direct"]
    documents: List[Document]
    generation: str
```

## Routing Logic

```python
def route_query(state: AdaptiveRAGState) -> str:
    """Route based on query classification."""
    return state["query_type"]

# In graph construction
graph.add_conditional_edges(
    "classify",
    route_query,
    {
        "vectorstore": "vectorstore",
        "websearch": "websearch",
        "direct": "direct",
    }
)
```

## Benefits

| Aspect | Fixed RAG | Adaptive RAG |
|--------|-----------|--------------|
| Simple questions | Full retrieval | Direct answer |
| Coverage | Local only | Multi-source |
| Efficiency | Same for all | Optimized per query |

## Next Steps

Continue to [Tutorial 12: Agentic RAG](12-agentic-rag.md)
