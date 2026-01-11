"""
Document loading utilities for RAG applications.

This module provides document loaders for various file formats including
PDF, text, and markdown files.

Example:
    >>> from langgraph_ollama_local.rag import DocumentLoader
    >>> loader = DocumentLoader()
    >>> docs = loader.load_pdf("sources/paper.pdf")
    >>> docs = loader.load_directory("sources/")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Multi-format document loader for RAG applications.

    Supports loading PDFs, text files, and markdown files with
    metadata extraction.

    Attributes:
        supported_extensions: List of supported file extensions.

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_pdf("paper.pdf")
        >>> print(docs[0].metadata["source"])
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self, extract_images: bool = False):
        """
        Initialize the document loader.

        Args:
            extract_images: Whether to extract images from PDFs (requires extra deps).
        """
        self.extract_images = extract_images

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF file and extract text by page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.

        Raises:
            ImportError: If pypdf is not installed.
            FileNotFoundError: If the file doesn't exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install with: pip install pypdf"
            )

        logger.info(f"Loading PDF: {file_path}")
        reader = PdfReader(str(file_path))

        documents = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Skip empty pages
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page": page_num,
                        "total_pages": len(reader.pages),
                        "file_type": "pdf",
                    },
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents

    def load_text(self, file_path: str | Path) -> list[Document]:
        """
        Load a text or markdown file.

        Args:
            file_path: Path to the text file.

        Returns:
            List containing a single Document.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading text file: {file_path}")
        text = file_path.read_text(encoding="utf-8")

        file_type = "markdown" if file_path.suffix in {".md", ".markdown"} else "text"

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_type,
                },
            )
        ]

    def load_file(self, file_path: str | Path) -> list[Document]:
        """
        Load a file based on its extension.

        Args:
            file_path: Path to the file.

        Returns:
            List of Document objects.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            return self.load_pdf(file_path)
        else:
            return self.load_text(file_path)

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> list[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            extensions: Specific extensions to load (default: all supported).

        Returns:
            List of all Document objects from the directory.

        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_directory("sources/", extensions={".pdf"})
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        extensions = extensions or self.SUPPORTED_EXTENSIONS

        pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in extensions
        ]

        logger.info(f"Found {len(files)} files in {directory}")

        all_documents = []
        for file_path in sorted(files):
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(all_documents)} documents total")
        return all_documents

    @staticmethod
    def extract_metadata_from_pdf(file_path: str | Path) -> dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary of PDF metadata (title, author, etc.).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            return {}

        file_path = Path(file_path)
        if not file_path.exists():
            return {}

        try:
            reader = PdfReader(str(file_path))
            metadata = reader.metadata
            if metadata:
                return {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                    "creation_date": str(metadata.get("/CreationDate", "")),
                }
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")

        return {}


def load_sources_directory(sources_dir: str = "sources") -> list[Document]:
    """
    Convenience function to load all documents from the sources directory.

    Args:
        sources_dir: Path to the sources directory.

    Returns:
        List of all loaded documents.

    Example:
        >>> from langgraph_ollama_local.rag.document_loader import load_sources_directory
        >>> docs = load_sources_directory()
        >>> print(f"Loaded {len(docs)} documents")
    """
    loader = DocumentLoader()
    return loader.load_directory(sources_dir)
