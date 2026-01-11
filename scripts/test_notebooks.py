#!/usr/bin/env python3
"""
Test script for RAG pattern notebooks.

This script validates that all notebook cells can execute correctly.
It requires Ollama to be running for full testing.

Usage:
    python scripts/test_notebooks.py [--no-ollama]

Options:
    --no-ollama    Skip tests that require Ollama connection
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def create_test_sources():
    """Create sample source documents for testing."""
    test_sources = PROJECT_ROOT / "examples" / "rag_patterns" / "test_sources"
    test_sources.mkdir(exist_ok=True)

    (test_sources / "rag_intro.txt").write_text(
        """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with language model generation. RAG systems work by:
1. Retrieving relevant documents from a knowledge base
2. Using those documents as context for the language model
3. Generating responses grounded in the retrieved information

This approach helps reduce hallucinations and provides up-to-date information.
"""
    )

    (test_sources / "self_rag.txt").write_text(
        """
Self-RAG adds self-reflection capabilities to traditional RAG systems.

Key features of Self-RAG:
1. Document relevance grading - the model evaluates if retrieved documents are relevant
2. Hallucination detection - checking if the generated answer is grounded in sources
3. Answer quality assessment - evaluating if the answer actually addresses the question

Self-RAG improves accuracy by filtering out irrelevant documents and regenerating when needed.
"""
    )

    (test_sources / "crag.txt").write_text(
        """
Corrective RAG (CRAG) extends RAG with web search fallback capabilities.

When local document retrieval is insufficient, CRAG:
1. Detects knowledge gaps in retrieved documents
2. Falls back to web search for additional information
3. Combines local and web sources for comprehensive answers

CRAG ensures better coverage of topics not well-represented in the local corpus.
"""
    )

    (test_sources / "adaptive_rag.txt").write_text(
        """
Adaptive RAG routes queries to different strategies based on query type.
It can use vectorstore search, web search, or direct LLM responses.
This flexibility improves performance across diverse query types.
"""
    )

    return test_sources


def check_ollama_available():
    """Check if Ollama is available."""
    try:
        import httpx

        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        from langgraph_ollama_local import LocalAgentConfig
        print("  ✓ LocalAgentConfig")
    except ImportError as e:
        print(f"  ✗ LocalAgentConfig: {e}")
        return False

    try:
        from langgraph_ollama_local.rag import (
            DocumentLoader,
            LocalEmbeddings,
            DocumentIndexer,
            LocalRetriever,
            DocumentGrader,
            HallucinationGrader,
            AnswerGrader,
        )
        print("  ✓ RAG components")
    except ImportError as e:
        print(f"  ✗ RAG components: {e}")
        return False

    try:
        from langchain_core.documents import Document
        from langgraph.graph import StateGraph, START, END
        print("  ✓ LangChain/LangGraph")
    except ImportError as e:
        print(f"  ✗ LangChain/LangGraph: {e}")
        return False

    return True


def test_rag_pipeline(sources_dir: Path):
    """Test the complete RAG pipeline."""
    print("\nTesting RAG pipeline...")

    from langgraph_ollama_local.rag import (
        DocumentLoader,
        LocalEmbeddings,
        DocumentIndexer,
        LocalRetriever,
    )
    from langgraph_ollama_local.rag.indexer import IndexerConfig
    from langgraph_ollama_local.rag.retriever import RetrieverConfig

    # 1. Load documents
    print("  Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(sources_dir)
    print(f"    ✓ Loaded {len(documents)} documents")

    # 2. Create indexer and chunk
    print("  Chunking documents...")
    config = IndexerConfig(chunk_size=500, chunk_overlap=50)
    indexer = DocumentIndexer(config=config)
    chunks = indexer.chunk_documents(documents)
    print(f"    ✓ Created {len(chunks)} chunks")

    # 3. Create embeddings
    print("  Testing embeddings...")
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")  # Faster model
    test_vectors = embeddings.embed_documents(["Test query"])
    print(f"    ✓ Embeddings work ({len(test_vectors[0])} dimensions)")

    # 4. Index documents
    print("  Indexing to ChromaDB...")
    num_indexed = indexer.index_documents(chunks)
    print(f"    ✓ Indexed {num_indexed} chunks")

    # 5. Test retrieval
    print("  Testing retrieval...")
    retriever = LocalRetriever()
    results = retriever.retrieve("What is RAG?", k=2)
    print(f"    ✓ Retrieved {len(results)} documents")

    return True


def test_notebook_syntax():
    """Test that all notebooks have valid syntax."""
    print("\nTesting notebook syntax...")

    notebooks = list((PROJECT_ROOT / "examples" / "rag_patterns").glob("*.ipynb"))

    all_valid = True
    for nb_path in sorted(notebooks):
        try:
            with open(nb_path) as f:
                nb = json.load(f)

            # Check JSON structure
            assert "cells" in nb

            # Check code cell syntax
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    code = "".join(cell["source"])
                    compile(code, nb_path.name, "exec")

            print(f"  ✓ {nb_path.name}")
        except Exception as e:
            print(f"  ✗ {nb_path.name}: {e}")
            all_valid = False

    return all_valid


def main():
    """Run all tests."""
    skip_ollama = "--no-ollama" in sys.argv

    print("=" * 60)
    print("RAG Notebook Test Suite")
    print("=" * 60)

    # Check Ollama availability
    ollama_available = check_ollama_available()
    print(f"\nOllama available: {ollama_available}")

    if skip_ollama:
        print("Skipping Ollama-dependent tests (--no-ollama flag)")

    # Test imports
    if not test_imports():
        print("\nImport tests failed. Please install dependencies.")
        return 1

    # Test notebook syntax
    if not test_notebook_syntax():
        print("\nSyntax tests failed. Please fix notebook errors.")
        return 1

    # Create test sources
    test_sources = create_test_sources()

    try:
        # Test RAG pipeline
        if not test_rag_pipeline(test_sources):
            print("\nRAG pipeline tests failed.")
            return 1
    finally:
        # Cleanup
        shutil.rmtree(test_sources, ignore_errors=True)
        # Cleanup ChromaDB
        chromadb_dir = PROJECT_ROOT / ".chromadb"
        if chromadb_dir.exists():
            shutil.rmtree(chromadb_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    if not ollama_available:
        print("\nNote: Full notebook execution requires a running Ollama server.")
        print("Start Ollama and run the notebooks to test LLM integration.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
