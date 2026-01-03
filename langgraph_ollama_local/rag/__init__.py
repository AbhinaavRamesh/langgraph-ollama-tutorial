"""
RAG (Retrieval-Augmented Generation) patterns for local Ollama deployment.

This module contains various RAG strategies optimized for local LLMs:
- Self-RAG: Self-reflective retrieval with document grading
- CRAG: Corrective RAG with web search fallback
- Adaptive RAG: Query-based retrieval strategy routing
- Agentic RAG: Multi-step retrieval with agent planning

All patterns are designed to work with local embedding models and vector stores.

Example:
    >>> from langgraph_ollama_local.rag import create_self_rag_agent
    >>> agent = create_self_rag_agent(retriever=my_retriever)
    >>> result = agent.invoke({"question": "What is LangGraph?"})
"""

__all__: list[str] = []
