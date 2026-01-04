"""
RAG (Retrieval-Augmented Generation) module for LangGraph Ollama Local.

This module provides infrastructure for building RAG applications with local LLMs,
including document loading, embedding, indexing, retrieval, and grading.

Components:
    - DocumentLoader: Load PDFs and text files
    - LocalEmbeddings: sentence-transformers embeddings
    - DocumentIndexer: Chunk and index documents into ChromaDB
    - LocalRetriever: Query documents from ChromaDB
    - DocumentGrader: LLM-based relevance grading
    - AnswerGrader: Hallucination and quality checks

Example:
    >>> from langgraph_ollama_local.rag import DocumentIndexer, LocalRetriever
    >>> indexer = DocumentIndexer()
    >>> indexer.index_directory("sources/")
    >>> retriever = LocalRetriever()
    >>> docs = retriever.retrieve("What is Self-RAG?")
"""

from langgraph_ollama_local.rag.document_loader import DocumentLoader
from langgraph_ollama_local.rag.embeddings import LocalEmbeddings, get_embedding_model
from langgraph_ollama_local.rag.graders import (
    AnswerGrader,
    DocumentGrader,
    HallucinationGrader,
    QueryRouter,
    QuestionRewriter,
    create_graders,
)
from langgraph_ollama_local.rag.indexer import DocumentIndexer
from langgraph_ollama_local.rag.retriever import LocalRetriever

__all__ = [
    # Document loading
    "DocumentLoader",
    # Embeddings
    "LocalEmbeddings",
    "get_embedding_model",
    # Indexing
    "DocumentIndexer",
    # Retrieval
    "LocalRetriever",
    # Grading
    "DocumentGrader",
    "AnswerGrader",
    "HallucinationGrader",
    "QueryRouter",
    "QuestionRewriter",
    "create_graders",
]
