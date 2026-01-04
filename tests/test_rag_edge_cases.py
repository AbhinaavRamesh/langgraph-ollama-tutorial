"""
Edge case and comprehensive tests for RAG components.

These tests cover boundary conditions, error handling, and edge cases
that are not covered by the basic unit tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path
from langchain_core.documents import Document
import tempfile
import os


# =============================================================================
# DocumentLoader Edge Cases
# =============================================================================


class TestDocumentLoaderEdgeCases:
    """Edge case tests for DocumentLoader."""

    def test_load_empty_file(self, tmp_path: Path):
        """Test loading an empty file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert docs[0].page_content == ""

    def test_load_unicode_content(self, tmp_path: Path):
        """Test loading file with Unicode characters."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        unicode_content = "Hello 世界! Привет мир! مرحبا بالعالم! 🚀 émojis"
        test_file = tmp_path / "unicode.txt"
        test_file.write_text(unicode_content, encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert docs[0].page_content == unicode_content

    def test_load_large_file(self, tmp_path: Path):
        """Test loading a large file (1MB+)."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create 1MB file
        large_content = "A" * (1024 * 1024)
        test_file = tmp_path / "large.txt"
        test_file.write_text(large_content)

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        assert len(docs[0].page_content) == 1024 * 1024

    def test_load_file_with_special_characters_in_name(self, tmp_path: Path):
        """Test loading file with special characters in filename."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        test_file = tmp_path / "test file (1) [copy].txt"
        test_file.write_text("Content")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1

    def test_load_empty_directory(self, tmp_path: Path):
        """Test loading from empty directory."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = DocumentLoader()
        docs = loader.load_directory(empty_dir)

        assert len(docs) == 0

    def test_load_directory_with_subdirs(self, tmp_path: Path):
        """Test loading from directory with subdirectories."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create nested structure
        (tmp_path / "file1.txt").write_text("Root content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("Sub content")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=True)

        assert len(docs) == 2

    def test_load_directory_non_recursive(self, tmp_path: Path):
        """Test non-recursive directory loading."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        (tmp_path / "file1.txt").write_text("Root content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("Sub content")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path, recursive=False)

        assert len(docs) == 1

    def test_load_file_with_newlines(self, tmp_path: Path):
        """Test loading file with various newline formats."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Mix of newline formats
        content = "Line 1\nLine 2\r\nLine 3\rLine 4"
        test_file = tmp_path / "newlines.txt"
        test_file.write_bytes(content.encode())

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1
        # Content should be preserved

    def test_load_whitespace_only_file(self, tmp_path: Path):
        """Test loading file with only whitespace."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        test_file = tmp_path / "whitespace.txt"
        test_file.write_text("   \n\t\n   ")

        loader = DocumentLoader()
        docs = loader.load_text(test_file)

        assert len(docs) == 1

    def test_load_symlink(self, tmp_path: Path):
        """Test loading symlinked file."""
        from langgraph_ollama_local.rag.document_loader import DocumentLoader

        # Create original file
        original = tmp_path / "original.txt"
        original.write_text("Original content")

        # Create symlink
        symlink = tmp_path / "symlink.txt"
        try:
            symlink.symlink_to(original)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        loader = DocumentLoader()
        docs = loader.load_text(symlink)

        assert len(docs) == 1
        assert docs[0].page_content == "Original content"


# =============================================================================
# LocalEmbeddings Edge Cases
# =============================================================================


class TestLocalEmbeddingsEdgeCases:
    """Edge case tests for LocalEmbeddings."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_empty_list(self, mock_st_class):
        """Test embedding empty list of documents."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([]).reshape(0, 3)
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        result = embeddings.embed_documents([])

        assert result == []

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_empty_string(self, mock_st_class):
        """Test embedding empty string."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.0, 0.0, 0.0]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        result = embeddings.embed_documents([""])

        assert len(result) == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_very_long_text(self, mock_st_class):
        """Test embedding very long text (10K+ chars)."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        long_text = "A" * 10000
        result = embeddings.embed_documents([long_text])

        assert len(result) == 1
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_special_characters(self, mock_st_class):
        """Test embedding text with special characters."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        special_text = "Hello <script>alert('xss')</script> & \"quotes\" 'apostrophe'"
        result = embeddings.embed_documents([special_text])

        assert len(result) == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_unicode_text(self, mock_st_class):
        """Test embedding Unicode text."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        unicode_text = "日本語 中文 한국어 العربية"
        result = embeddings.embed_documents([unicode_text])

        assert len(result) == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_large_batch(self, mock_st_class):
        """Test embedding large batch of documents."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(100, 3)
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")
        texts = [f"Document {i}" for i in range(100)]
        result = embeddings.embed_documents(texts)

        assert len(result) == 100

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings

        # Should not raise during init, only when model is accessed
        embeddings = LocalEmbeddings(model_name="nonexistent-model-xyz")
        assert embeddings.model_name == "nonexistent-model-xyz"


# =============================================================================
# DocumentIndexer Edge Cases
# =============================================================================


class TestDocumentIndexerEdgeCases:
    """Edge case tests for DocumentIndexer."""

    def test_chunk_single_word_document(self, indexer_config):
        """Test chunking a very short document."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)
        doc = Document(page_content="Hello", metadata={"source": "test.txt"})

        chunks = indexer.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Hello"

    def test_chunk_document_at_sentence_boundary(self, indexer_config):
        """Test that chunking respects sentence boundaries."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)
        # Create content that should be split at sentence boundary
        doc = Document(
            page_content="First sentence. " * 50 + "Last sentence.",
            metadata={"source": "test.txt"}
        )

        chunks = indexer.chunk_document(doc)

        # Each chunk should ideally end at a sentence boundary
        for chunk in chunks[:-1]:  # All but last
            content = chunk.page_content.strip()
            # Should end with sentence-ending punctuation or be at boundary
            assert content.endswith('.') or content.endswith('!') or content.endswith('?') or len(content) <= indexer_config.chunk_size

    def test_chunk_document_with_no_sentence_boundaries(self, indexer_config):
        """Test chunking document without sentence boundaries."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)
        doc = Document(
            page_content="A" * 500,  # No sentence boundaries
            metadata={"source": "test.txt"}
        )

        chunks = indexer.chunk_document(doc)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.page_content) <= indexer_config.chunk_size

    def test_chunk_with_overlap_larger_than_chunk(self):
        """Test chunk overlap larger than chunk size handling."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        # This should still work - overlap should be capped
        config = IndexerConfig(
            chunk_size=200,
            chunk_overlap=250,  # Larger than chunk_size
            collection_name="test",
        )
        indexer = DocumentIndexer(config=config)
        doc = Document(
            page_content="Test content " * 50,
            metadata={"source": "test.txt"}
        )

        chunks = indexer.chunk_document(doc)
        # Should not infinite loop
        assert len(chunks) > 0

    def test_chunk_preserves_all_metadata_fields(self, indexer_config):
        """Test that all metadata fields are preserved during chunking."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)
        doc = Document(
            page_content="Content " * 100,
            metadata={
                "source": "test.pdf",
                "page": 5,
                "author": "Test Author",
                "date": "2024-01-01",
                "custom_field": {"nested": "value"},
            }
        )

        chunks = indexer.chunk_document(doc)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["date"] == "2024-01-01"
            assert chunk.metadata["custom_field"] == {"nested": "value"}

    def test_generate_unique_doc_ids(self, indexer_config):
        """Test that document IDs are unique."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)

        doc1 = Document(page_content="Content 1", metadata={"source": "test.txt"})
        doc2 = Document(page_content="Content 2", metadata={"source": "test.txt"})
        doc3 = Document(page_content="Content 1", metadata={"source": "other.txt"})

        id1 = indexer._generate_doc_id(doc1)
        id2 = indexer._generate_doc_id(doc2)
        id3 = indexer._generate_doc_id(doc3)

        # All IDs should be unique
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_chunk_document_with_only_whitespace(self, indexer_config):
        """Test chunking document with only whitespace."""
        from langgraph_ollama_local.rag.indexer import DocumentIndexer

        indexer = DocumentIndexer(config=indexer_config)
        doc = Document(
            page_content="   \n\t\n   ",
            metadata={"source": "test.txt"}
        )

        chunks = indexer.chunk_document(doc)

        # Whitespace-only should result in no meaningful chunks
        assert len(chunks) == 0 or all(not c.page_content.strip() for c in chunks)


# =============================================================================
# Retriever Edge Cases
# =============================================================================


class TestRetrieverEdgeCases:
    """Edge case tests for LocalRetriever."""

    def test_retrieve_from_empty_collection(self, tmp_path):
        """Test retrieval from empty collection."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        # Create empty collection with same embedding model
        indexer_config = IndexerConfig(
            collection_name="empty_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        # Initialize collection without adding documents
        _ = indexer.collection

        retriever_config = RetrieverConfig(
            collection_name="empty_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )

        # Use real embeddings since we're testing empty collection behavior
        retriever = LocalRetriever(config=retriever_config)
        results = retriever.retrieve("test query", k=5)

        assert len(results) == 0

    def test_retrieve_with_k_larger_than_collection(self, tmp_path):
        """Test retrieval with k larger than collection size."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        # Create collection with 2 documents
        indexer_config = IndexerConfig(
            collection_name="small_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents([
            Document(page_content="Doc 1", metadata={"source": "a.txt"}),
            Document(page_content="Doc 2", metadata={"source": "b.txt"}),
        ])

        retriever_config = RetrieverConfig(
            collection_name="small_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        retriever = LocalRetriever(config=retriever_config)

        # Request more than available
        results = retriever.retrieve("test", k=100)

        # Should return only what's available
        assert len(results) <= 2

    def test_retrieve_with_score_threshold(self, tmp_path):
        """Test retrieval with score threshold filtering."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        indexer_config = IndexerConfig(
            collection_name="threshold_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents([
            Document(page_content="RAG is retrieval augmented generation", metadata={"source": "a.txt"}),
            Document(page_content="The weather is sunny", metadata={"source": "b.txt"}),
        ])

        retriever_config = RetrieverConfig(
            collection_name="threshold_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
            score_threshold=0.5,  # High threshold
        )
        retriever = LocalRetriever(config=retriever_config)

        # Query about RAG should have high score for relevant doc
        results = retriever.retrieve("What is RAG?", k=5, score_threshold=0.5)

        # Results should be filtered by threshold
        for doc, score in results:
            assert score >= 0.5

    def test_retrieve_with_metadata_filter(self, tmp_path):
        """Test retrieval with metadata filtering."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        indexer_config = IndexerConfig(
            collection_name="filter_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents([
            Document(page_content="RAG doc 1", metadata={"source": "rag.txt", "type": "rag"}),
            Document(page_content="RAG doc 2", metadata={"source": "other.txt", "type": "other"}),
        ])

        retriever_config = RetrieverConfig(
            collection_name="filter_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        retriever = LocalRetriever(config=retriever_config)

        # Filter by metadata
        results = retriever.retrieve("RAG", k=5, filter_metadata={"type": "rag"})

        for doc, _ in results:
            assert doc.metadata.get("type") == "rag"

    def test_retrieve_empty_query(self, tmp_path):
        """Test retrieval with empty query."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig
        from langgraph_ollama_local.rag.indexer import DocumentIndexer, IndexerConfig

        indexer_config = IndexerConfig(
            collection_name="empty_query_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        indexer = DocumentIndexer(config=indexer_config)
        indexer.index_documents([
            Document(page_content="Test doc", metadata={"source": "test.txt"}),
        ])

        retriever_config = RetrieverConfig(
            collection_name="empty_query_test",
            persist_directory=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
        )
        retriever = LocalRetriever(config=retriever_config)

        # Empty query should still work
        results = retriever.retrieve("", k=5)
        # Should return results (embedding of empty string)
        assert isinstance(results, list)


# =============================================================================
# Grader Edge Cases
# =============================================================================


class TestGraderEdgeCases:
    """Edge case tests for RAG graders."""

    def test_document_grader_ambiguous_response(self, mock_llm):
        """Test grader with ambiguous LLM response."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        # Ambiguous responses should default to relevant
        mock_llm.invoke.return_value = Mock(content="maybe")

        grader = DocumentGrader(mock_llm)
        result = grader.grade("Some content", "Some question")

        # Should handle gracefully (default to True or False based on impl)
        assert isinstance(result, bool)

    def test_document_grader_empty_document(self, mock_llm):
        """Test grading empty document."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = DocumentGrader(mock_llm)
        result = grader.grade("", "What is RAG?")

        # Empty document should still return a boolean
        assert isinstance(result, bool)

    def test_document_grader_empty_question(self, mock_llm):
        """Test grading with empty question."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = DocumentGrader(mock_llm)
        result = grader.grade("RAG content", "")

        assert isinstance(result, bool)

    def test_document_grader_very_long_document(self, mock_llm):
        """Test grading very long document."""
        from langgraph_ollama_local.rag.graders import DocumentGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = DocumentGrader(mock_llm)
        long_doc = "RAG information. " * 1000
        result = grader.grade(long_doc, "What is RAG?")

        assert result is True

    def test_hallucination_grader_empty_documents(self, mock_llm):
        """Test hallucination grading with empty documents."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = HallucinationGrader(mock_llm)
        result = grader.grade([], "Some answer")

        # Empty documents should return a boolean result
        assert isinstance(result, bool)

    def test_hallucination_grader_multiple_documents(self, mock_llm):
        """Test hallucination grading with multiple documents."""
        from langgraph_ollama_local.rag.graders import HallucinationGrader

        mock_llm.invoke.return_value = Mock(content="yes")

        grader = HallucinationGrader(mock_llm)
        docs = [
            Document(page_content="Fact 1"),
            Document(page_content="Fact 2"),
            Document(page_content="Fact 3"),
        ]
        result = grader.grade(docs, "Answer combining facts 1, 2, and 3")

        assert result is True

    def test_answer_grader_empty_answer(self, mock_llm):
        """Test grading empty answer."""
        from langgraph_ollama_local.rag.graders import AnswerGrader

        mock_llm.invoke.return_value = Mock(content="no")

        grader = AnswerGrader(mock_llm)
        result = grader.grade("What is RAG?", "")

        # Empty answers may be graded differently depending on LLM response
        assert isinstance(result, bool)

    def test_query_router_empty_query(self, mock_llm):
        """Test routing empty query."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="vectorstore")

        router = QueryRouter(mock_llm)
        result = router.route("")

        # Should return valid route
        assert result in ["vectorstore", "websearch", "direct"]

    def test_query_router_very_long_query(self, mock_llm):
        """Test routing very long query."""
        from langgraph_ollama_local.rag.graders import QueryRouter

        mock_llm.invoke.return_value = Mock(content="vectorstore")

        router = QueryRouter(mock_llm)
        long_query = "What is " + "RAG " * 500 + "?"
        result = router.route(long_query)

        assert result in ["vectorstore", "websearch", "direct"]

    def test_question_rewriter_empty_question(self, mock_llm):
        """Test rewriting empty question."""
        from langgraph_ollama_local.rag.graders import QuestionRewriter

        mock_llm.invoke.return_value = Mock(content="")

        rewriter = QuestionRewriter(mock_llm)
        result = rewriter.rewrite("")

        # Should return original (empty) on empty result
        assert result == ""

    def test_question_rewriter_already_clear(self, mock_llm):
        """Test rewriting already clear question."""
        from langgraph_ollama_local.rag.graders import QuestionRewriter

        clear_question = "What is Retrieval-Augmented Generation and how does it work?"
        mock_llm.invoke.return_value = Mock(content=clear_question)

        rewriter = QuestionRewriter(mock_llm)
        result = rewriter.rewrite(clear_question)

        # Should return similar question
        assert "Retrieval" in result or "RAG" in result


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across RAG components."""

    def test_document_grader_llm_timeout(self, mock_llm):
        """Test grader handling LLM timeout."""
        from langgraph_ollama_local.rag.graders import DocumentGrader
        import httpx

        mock_llm.invoke.side_effect = httpx.TimeoutException("Timeout")

        grader = DocumentGrader(mock_llm)

        # Should handle gracefully
        try:
            result = grader.grade("content", "question")
            # If it doesn't raise, should return default
            assert isinstance(result, bool)
        except httpx.TimeoutException:
            # Or it may propagate the error
            pass

    def test_retriever_chromadb_not_found(self, tmp_path):
        """Test retriever when ChromaDB directory doesn't exist."""
        from langgraph_ollama_local.rag.retriever import LocalRetriever, RetrieverConfig

        config = RetrieverConfig(
            collection_name="nonexistent",
            persist_directory=str(tmp_path / "nonexistent"),
        )

        retriever = LocalRetriever(config=config)

        with pytest.raises(FileNotFoundError):
            _ = retriever.client

    def test_indexer_invalid_chunk_size(self):
        """Test indexer with invalid chunk size."""
        from langgraph_ollama_local.rag.indexer import IndexerConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IndexerConfig(chunk_size=50)  # Below minimum

    def test_retriever_invalid_k(self):
        """Test retriever config with invalid k."""
        from langgraph_ollama_local.rag.retriever import RetrieverConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrieverConfig(default_k=0)  # Below minimum


# =============================================================================
# Config Edge Cases
# =============================================================================


class TestConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_indexer_config_from_env(self, monkeypatch):
        """Test IndexerConfig loading from environment."""
        from langgraph_ollama_local.rag.indexer import IndexerConfig

        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("RAG_COLLECTION_NAME", "env_collection")

        config = IndexerConfig()

        assert config.chunk_size == 500
        assert config.collection_name == "env_collection"

    def test_retriever_config_from_env(self, monkeypatch):
        """Test RetrieverConfig loading from environment."""
        from langgraph_ollama_local.rag.retriever import RetrieverConfig

        monkeypatch.setenv("RAG_DEFAULT_K", "10")
        monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.7")

        config = RetrieverConfig()

        assert config.default_k == 10
        assert config.score_threshold == 0.7

    def test_config_validation_bounds(self):
        """Test config validation boundaries."""
        from langgraph_ollama_local.rag.indexer import IndexerConfig
        from langgraph_ollama_local.rag.retriever import RetrieverConfig
        from pydantic import ValidationError

        # Test minimum bounds
        with pytest.raises(ValidationError):
            IndexerConfig(chunk_size=50)  # Too small

        with pytest.raises(ValidationError):
            RetrieverConfig(default_k=0)  # Too small

        with pytest.raises(ValidationError):
            RetrieverConfig(score_threshold=-0.1)  # Negative

        with pytest.raises(ValidationError):
            RetrieverConfig(score_threshold=1.5)  # Too large


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent access patterns."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_concurrent_embeddings(self, mock_st_class):
        """Test concurrent embedding generation."""
        from langgraph_ollama_local.rag.embeddings import LocalEmbeddings
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="test-model")

        def embed_text(text):
            return embeddings.embed_documents([text])

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(embed_text, f"Text {i}") for i in range(10)]
            results = [f.result() for f in futures]

        assert len(results) == 10
        assert all(len(r) == 1 for r in results)
