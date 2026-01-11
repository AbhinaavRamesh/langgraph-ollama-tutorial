"""
Tests for RAG infrastructure and patterns.

These tests verify the RAG components work correctly with mocked LLMs
and embedding models where possible.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from langchain_core.documents import Document


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self, tmp_path: Path):
        """Test loading a text file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content.")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert docs[0].page_content == "This is test content."
        assert docs[0].metadata["file_type"] == "text"

    def test_load_markdown_file(self, tmp_path: Path):
        """Test loading a markdown file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nContent here.")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert "# Heading" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "markdown"

    def test_load_directory(self, tmp_path: Path):
        """Test loading multiple files from directory."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "ignored.xyz").write_text("Ignored")  # Unsupported

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 2

    def test_file_not_found(self):
        """Test handling of missing file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    def test_unsupported_extension(self, tmp_path: Path):
        """Test handling of unsupported file type."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        test_file = tmp_path / "test.xyz"
        test_file.write_text("Content")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(test_file)


class TestLocalEmbeddings:
    """Tests for LocalEmbeddings."""

    @patch("langgraph_ollama_local.rag.embeddings.SentenceTransformer")
    def test_embed_documents(self, mock_st_class):
        """Test embedding multiple documents."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        result = embeddings.embed_documents(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 3
        mock_model.encode.assert_called_once()

    @patch("langgraph_ollama_local.rag.embeddings.SentenceTransformer")
    def test_embed_query(self, mock_st_class):
        """Test embedding a single query."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        result = embeddings.embed_query("test query")

        assert len(result) == 3

    def test_embedding_dimensions(self):
        """Test embedding model info."""
        from langgraph_ollama_local.rag.embeddings import EMBEDDING_MODELS

        assert "all-mpnet-base-v2" in EMBEDDING_MODELS
        assert EMBEDDING_MODELS["all-mpnet-base-v2"]["dimensions"] == 768
        assert "all-MiniLM-L6-v2" in EMBEDDING_MODELS
        assert EMBEDDING_MODELS["all-MiniLM-L6-v2"]["dimensions"] == 384

    def test_local_embeddings_repr(self):
        """Test string representation."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

        # Don't load the model, just test repr
        embeddings = LocalEmbeddings.__new__(LocalEmbeddings)
        embeddings.model_name = "test-model"
        embeddings.dimensions = 768
        embeddings._model = None

        repr_str = repr(embeddings)
        assert "test-model" in repr_str
        assert "768" in repr_str


class TestDocumentIndexer:
    """Tests for DocumentIndexer."""

    def test_chunk_document(self, indexer_config):
        """Test document chunking."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)

        doc = Document(
            page_content="A" * 500,  # 500 characters
            metadata={"source": "test.txt"}
        )

        chunks = indexer.chunk_document(doc)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.page_content) <= indexer_config.chunk_size
            assert "chunk_index" in chunk.metadata

    def test_chunk_preserves_metadata(self, indexer_config):
        """Test that chunking preserves original metadata."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)

        doc = Document(
            page_content="Short content that will be chunked. " * 20,
            metadata={"source": "test.txt", "page": 1}
        )

        chunks = indexer.chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["page"] == 1

    def test_chunk_empty_document(self, indexer_config):
        """Test chunking empty document."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)

        doc = Document(page_content="", metadata={"source": "empty.txt"})
        chunks = indexer.chunk_document(doc)

        assert len(chunks) == 0

    def test_get_stats(self, indexer_config, mock_embeddings):
        """Test getting indexer statistics."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        with patch.object(DocumentIndexer, 'embeddings', mock_embeddings):
            indexer = DocumentIndexer(config=indexer_config)
            stats = indexer.get_stats()

            assert "collection_name" in stats
            assert "chunk_size" in stats
            assert stats["chunk_size"] == indexer_config.chunk_size


class TestDocumentGrader:
    """Tests for DocumentGrader."""

    def test_grade_relevant_document(self, mock_llm):
        """Test grading a relevant document."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        # Mock LLM to return "yes"
        mock_llm.invoke.return_value = Mock(content="yes")

        grader = DocumentGrader(mock_llm)
        doc = Document(page_content="RAG combines retrieval with generation.")

        result = grader.grade(doc, "What is RAG?")

        assert result is True

    def test_grade_irrelevant_document(self, mock_llm):
        """Test grading an irrelevant document."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = DocumentGrader(mock_llm)
        doc = Document(page_content="Recipe for chocolate cake.")

        result = grader.grade(doc, "What is RAG?")

        assert result is False

    def test_grade_documents_batch(self, mock_llm):
        """Test grading multiple documents."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        # Alternate yes/no responses
        mock_llm.invoke.side_effect = [
            Mock(content="yes"),
            Mock(content="no"),
            Mock(content="yes"),
        ]

        grader = DocumentGrader(mock_llm)
        docs = [
            Document(page_content="RAG info"),
            Document(page_content="Cake recipe"),
            Document(page_content="More RAG info"),
        ]

        relevant, irrelevant = grader.grade_documents(docs, "What is RAG?")

        assert len(relevant) == 2
        assert len(irrelevant) == 1

    def test_grade_with_string_input(self, mock_llm):
        """Test grading with string instead of Document."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = DocumentGrader(mock_llm)
        result = grader.grade("RAG is about retrieval", "What is RAG?")

        assert result is True


class TestHallucinationGrader:
    """Tests for HallucinationGrader."""

    def test_grounded_answer(self, mock_llm):
        """Test detecting a grounded answer."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = HallucinationGrader(mock_llm)
        docs = [Document(page_content="RAG stands for Retrieval-Augmented Generation.")]

        result = grader.grade(docs, "RAG stands for Retrieval-Augmented Generation.")

        assert result is True

    def test_hallucinated_answer(self, mock_llm):
        """Test detecting a hallucinated answer."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = HallucinationGrader(mock_llm)
        docs = [Document(page_content="RAG is about retrieval.")]

        result = grader.grade(docs, "RAG was invented in 1950.")

        assert result is False

    def test_grade_with_string_documents(self, mock_llm):
        """Test grading with string instead of document list."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = HallucinationGrader(mock_llm)
        result = grader.grade("RAG is about retrieval", "RAG uses retrieval.")

        assert result is True


class TestAnswerGrader:
    """Tests for AnswerGrader."""

    def test_useful_answer(self, mock_llm):
        """Test detecting a useful answer."""
        from langgraph_ollama_local.rag.graders import AnswerGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = AnswerGrader(mock_llm)
        result = grader.grade("What is RAG?", "RAG is Retrieval-Augmented Generation.")

        assert result is True

    def test_not_useful_answer(self, mock_llm):
        """Test detecting an answer that doesn't address the question."""
        from langgraph_ollama_local.rag.graders import AnswerGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = AnswerGrader(mock_llm)
        result = grader.grade("What is RAG?", "The weather is nice today.")

        assert result is False


class TestQueryRouter:
    """Tests for QueryRouter."""

    def test_route_to_vectorstore(self, mock_llm):
        """Test routing document questions to vectorstore."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="vectorstore")

        router = QueryRouter(mock_llm)
        result = router.route("What does the documentation say about RAG?")

        assert result == "vectorstore"

    def test_route_to_websearch(self, mock_llm):
        """Test routing current events to websearch."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="websearch")

        router = QueryRouter(mock_llm)
        result = router.route("What happened in the news today?")

        assert result == "websearch"

    def test_route_to_direct(self, mock_llm):
        """Test routing simple questions to direct."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="direct")

        router = QueryRouter(mock_llm)
        result = router.route("What is 2 + 2?")

        assert result == "direct"

    def test_route_unknown_defaults_to_vectorstore(self, mock_llm):
        """Test that unknown routes default to vectorstore."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="unknown_route")

        router = QueryRouter(mock_llm)
        result = router.route("Some question")

        assert result == "vectorstore"


class TestQuestionRewriter:
    """Tests for QuestionRewriter."""

    def test_rewrite_question(self, mock_llm):
        """Test question rewriting."""
        from langgraph_ollama_local.rag.graders import QuestionRewriter

        mock_llm.invoke.return_value = Mock(
            content="What is Retrieval-Augmented Generation (RAG)?"
        )

        rewriter = QuestionRewriter(mock_llm)
        result = rewriter.rewrite("what's rag")

        assert "Retrieval-Augmented Generation" in result

    def test_rewrite_returns_original_on_error(self, mock_llm):
        """Test that original question is returned on error."""
        from langgraph_ollama_local.rag.graders import QuestionRewriter

        mock_llm.invoke.side_effect = Exception("API Error")

        rewriter = QuestionRewriter(mock_llm)
        result = rewriter.rewrite("what's rag")

        assert result == "what's rag"


class TestIndexerConfig:
    """Tests for IndexerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from langgraph_ollama_local.rag.indexer import IndexerConfig

        config = IndexerConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.collection_name == "documents"
        assert config.embedding_model == "all-mpnet-base-v2"

    def test_custom_config(self):
        """Test custom configuration."""
        from langgraph_ollama_local.rag.indexer import IndexerConfig

        config = IndexerConfig(
            chunk_size=500,
            chunk_overlap=100,
            collection_name="my_docs",
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.collection_name == "my_docs"


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from langgraph_ollama_local.rag.retriever import RetrieverConfig

        config = RetrieverConfig()

        assert config.collection_name == "documents"
        assert config.default_k == 4
        assert config.score_threshold == 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        from langgraph_ollama_local.rag.retriever import RetrieverConfig

        config = RetrieverConfig(
            collection_name="custom_docs",
            default_k=10,
            score_threshold=0.5,
        )

        assert config.collection_name == "custom_docs"
        assert config.default_k == 10
        assert config.score_threshold == 0.5


class TestCreateGraders:
    """Tests for create_graders helper."""

    def test_create_graders(self, mock_llm):
        """Test creating all graders at once."""
        from langgraph_ollama_local.rag.graders import create_graders

        graders = create_graders(mock_llm)

        assert "document" in graders
        assert "hallucination" in graders
        assert "answer" in graders
        assert "rewriter" in graders
        assert "router" in graders
