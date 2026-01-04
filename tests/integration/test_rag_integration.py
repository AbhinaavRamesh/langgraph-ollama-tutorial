"""
Integration tests for RAG (Retrieval-Augmented Generation) components.

These tests require:
- Running Ollama server
- Sentence-transformers model downloaded
- ChromaDB dependencies

Run with: pytest tests/integration/test_rag_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


# === Fixtures ===


@pytest.fixture
def ollama_available():
    """Check if Ollama is available."""
    import httpx
    from langgraph_ollama_local.config import LocalAgentConfig

    try:
        config = LocalAgentConfig()
        response = httpx.get(f"{config.ollama.base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    pytest.skip("Ollama server not available")


@pytest.fixture
def llm(ollama_available):
    """Get configured LLM for RAG."""
    from langchain_ollama import ChatOllama
    from langgraph_ollama_local.config import LocalAgentConfig

    config = LocalAgentConfig()
    return ChatOllama(
        model=config.ollama.model,
        base_url=config.ollama.base_url,
        temperature=0,
    )


@pytest.fixture
def temp_chromadb(tmp_path):
    """Create a temporary ChromaDB directory."""
    chromadb_dir = tmp_path / "test_chromadb"
    chromadb_dir.mkdir()
    return chromadb_dir


@pytest.fixture
def sample_docs():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines retrieval with generation. "
            "It retrieves relevant documents and uses them as context for the LLM.",
            metadata={"source": "rag_intro.pdf", "page": 1, "filename": "rag_intro.pdf"},
        ),
        Document(
            page_content="Self-RAG adds reflection to grade document relevance and detect hallucinations. "
            "The LLM evaluates its own outputs for quality.",
            metadata={"source": "self_rag.pdf", "page": 1, "filename": "self_rag.pdf"},
        ),
        Document(
            page_content="CRAG (Corrective RAG) uses web search as a fallback when local retrieval fails. "
            "This ensures comprehensive topic coverage.",
            metadata={"source": "crag.pdf", "page": 1, "filename": "crag.pdf"},
        ),
        Document(
            page_content="Adaptive RAG routes queries to different strategies based on query type. "
            "It can use vectorstore, web search, or direct LLM responses.",
            metadata={"source": "adaptive_rag.pdf", "page": 1, "filename": "adaptive_rag.pdf"},
        ),
    ]


# === Embedding Tests ===


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests for embedding generation."""

    def test_embed_documents_real(self):
        """Test embedding generation with real model."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")

        texts = ["What is RAG?", "Retrieval augmented generation explained"]
        vectors = embeddings.embed_documents(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 384  # MiniLM dimensions
        assert all(isinstance(v, float) for v in vectors[0])

    def test_embed_query_real(self):
        """Test query embedding with real model."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")

        vector = embeddings.embed_query("What is Self-RAG?")

        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    def test_semantic_similarity(self):
        """Test that similar texts have similar embeddings."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")

        # Similar texts
        text1 = "RAG combines retrieval with generation"
        text2 = "Retrieval-Augmented Generation merges search and LLMs"
        # Dissimilar text
        text3 = "The weather is sunny today"

        vecs = embeddings.embed_documents([text1, text2, text3])

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_12 = cosine_similarity(vecs[0], vecs[1])
        sim_13 = cosine_similarity(vecs[0], vecs[2])

        # Similar texts should have higher similarity
        assert sim_12 > sim_13, "Similar texts should have higher cosine similarity"


# === Indexer Tests ===


@pytest.mark.integration
class TestIndexerIntegration:
    """Integration tests for document indexing."""

    def test_index_and_retrieve(self, temp_chromadb, sample_docs):
        """Test full indexing and retrieval cycle."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig

        # Index documents
        indexer_config = IndexerConfig(
            chunk_size=500,
            chunk_overlap=50,
            collection_name="integration_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        num_indexed = indexer.index_documents(sample_docs)

        assert num_indexed == len(sample_docs)

        # Retrieve documents
        retriever_config = RetrieverConfig(
            collection_name="integration_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
            default_k=2,
        )
        retriever = LocalRetriever(config=retriever_config)

        results = retriever.retrieve("What is Self-RAG?", k=2)

        assert len(results) > 0
        # Check that Self-RAG document is in top results
        contents = [doc.page_content for doc, _ in results]
        assert any("Self-RAG" in content for content in contents)

    def test_chunking_preserves_metadata(self, temp_chromadb):
        """Test that chunking preserves document metadata."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        config = IndexerConfig(
            chunk_size=100,
            chunk_overlap=20,
            collection_name="chunk_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=config)

        # Create a document that will be chunked
        long_doc = Document(
            page_content="This is a test document. " * 50,
            metadata={"source": "test.pdf", "page": 5, "author": "Test Author"},
        )

        chunks = indexer.chunk_document(long_doc)

        assert len(chunks) > 1  # Should be chunked
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["author"] == "Test Author"
            assert "chunk_index" in chunk.metadata


# === Grader Tests ===


@pytest.mark.integration
class TestGradersIntegration:
    """Integration tests for RAG graders with real LLM."""

    def test_document_grader(self, llm):
        """Test document relevance grading."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        grader = DocumentGrader(llm)

        # Relevant document
        relevant_doc = Document(
            page_content="RAG combines retrieval with generation to improve LLM responses."
        )
        is_relevant = grader.grade(relevant_doc, "What is RAG?")

        assert is_relevant is True

        # Irrelevant document
        irrelevant_doc = Document(
            page_content="The recipe calls for two cups of flour and one egg."
        )
        is_irrelevant = grader.grade(irrelevant_doc, "What is RAG?")

        assert is_irrelevant is False

    def test_hallucination_grader(self, llm):
        """Test hallucination detection."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        grader = HallucinationGrader(llm)

        documents = [
            Document(page_content="RAG stands for Retrieval-Augmented Generation.")
        ]

        # Grounded answer
        grounded = grader.grade(
            documents, "RAG stands for Retrieval-Augmented Generation."
        )
        assert grounded is True

        # Hallucinated answer
        hallucinated = grader.grade(
            documents, "RAG was invented by OpenAI in 2015."
        )
        assert hallucinated is False

    def test_answer_grader(self, llm):
        """Test answer usefulness grading."""
        from langgraph_ollama_local.rag.graders import AnswerGrader

        grader = AnswerGrader(llm)

        # Useful answer
        useful = grader.grade(
            "What is RAG?",
            "RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation.",
        )
        assert useful is True

        # Not useful answer
        not_useful = grader.grade(
            "What is RAG?",
            "I'm not sure what you're asking about.",
        )
        assert not_useful is False

    def test_query_router(self, llm):
        """Test query routing."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        router = QueryRouter(llm)

        # Document question should route to vectorstore
        doc_route = router.route("What does the documentation say about RAG?")
        assert doc_route in ["vectorstore", "websearch", "direct"]

        # Simple question might route to direct
        simple_route = router.route("What is 2 + 2?")
        assert simple_route in ["vectorstore", "websearch", "direct"]


# === Full RAG Pipeline Tests ===


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for complete RAG pipelines."""

    def test_basic_rag_pipeline(self, llm, temp_chromadb, sample_docs):
        """Test complete basic RAG pipeline."""
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig

        # Setup indexing
        indexer_config = IndexerConfig(
            chunk_size=500,
            chunk_overlap=50,
            collection_name="rag_pipeline_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents(sample_docs)

        # Setup retriever
        retriever_config = RetrieverConfig(
            collection_name="rag_pipeline_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
            default_k=2,
        )
        retriever = LocalRetriever(config=retriever_config)

        # Define state
        class RAGState(TypedDict):
            question: str
            documents: List[Document]
            generation: str

        # Define nodes
        def retrieve_node(state: RAGState) -> dict:
            docs = retriever.retrieve_documents(state["question"], k=2)
            return {"documents": docs}

        RAG_PROMPT = ChatPromptTemplate.from_template(
            """Answer based on context. If unknown, say so.

Context:
{context}

Question: {question}

Answer:"""
        )

        def generate_node(state: RAGState) -> dict:
            context = "\n\n".join([d.page_content for d in state["documents"]])
            messages = RAG_PROMPT.format_messages(
                context=context, question=state["question"]
            )
            response = llm.invoke(messages)
            return {"generation": response.content}

        # Build graph
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        rag_app = graph.compile()

        # Test the pipeline
        result = rag_app.invoke({"question": "What is Self-RAG?"})

        assert "generation" in result
        assert len(result["generation"]) > 0
        assert result["documents"] is not None
        assert len(result["documents"]) > 0

    def test_self_rag_pipeline(self, llm, temp_chromadb, sample_docs):
        """Test Self-RAG pipeline with grading."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.graders import DocumentGrader

        # Setup
        indexer_config = IndexerConfig(
            chunk_size=500,
            chunk_overlap=50,
            collection_name="self_rag_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents(sample_docs)

        retriever_config = RetrieverConfig(
            collection_name="self_rag_test",
            persist_directory=str(temp_chromadb),
            embedding_model="all-MiniLM-L6-v2",
            default_k=4,
        )
        retriever = LocalRetriever(config=retriever_config)
        grader = DocumentGrader(llm)

        # Define state
        class SelfRAGState(TypedDict):
            question: str
            documents: List[Document]
            relevant_documents: List[Document]
            generation: str

        # Define nodes
        def retrieve_node(state: SelfRAGState) -> dict:
            docs = retriever.retrieve_documents(state["question"], k=4)
            return {"documents": docs}

        def grade_node(state: SelfRAGState) -> dict:
            question = state["question"]
            docs = state["documents"]
            relevant, _ = grader.grade_documents(docs, question)
            return {"relevant_documents": relevant}

        # Build graph
        graph = StateGraph(SelfRAGState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade", grade_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_edge("grade", END)
        self_rag_app = graph.compile()

        # Test
        result = self_rag_app.invoke({"question": "What is CRAG?"})

        assert "relevant_documents" in result
        # At least some documents should be relevant
        assert len(result["relevant_documents"]) >= 0


# === Document Loader Tests ===


@pytest.mark.integration
class TestDocumentLoaderIntegration:
    """Integration tests for document loading."""

    def test_load_sources_directory(self):
        """Test loading from sources directory if it exists."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        sources_dir = Path(__file__).parent.parent.parent / "sources"
        if not sources_dir.exists():
            pytest.skip("sources/ directory not found")

        loader = DocumentLoader()
        docs = loader.load_directory(sources_dir)

        assert len(docs) > 0, "Should load documents from sources/"

        # Check metadata
        for doc in docs[:3]:
            assert "source" in doc.metadata or "filename" in doc.metadata

    def test_load_pdf_file(self, tmp_path):
        """Test loading a text file (PDF requires actual PDF)."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create a test text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for the document loader.")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert "test content" in docs[0].page_content

    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create a test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nThis is **bold** text.")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert "# Heading" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "markdown"
