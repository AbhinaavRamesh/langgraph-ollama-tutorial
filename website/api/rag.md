---
title: RAG API
description: Document indexing, retrieval, and grading for RAG patterns
---

# RAG API

Complete reference for Retrieval-Augmented Generation (RAG) components including document loading, indexing, embeddings, and quality grading.

## DocumentLoader

Multi-format document loader for RAG applications. Supports PDFs, text files, and markdown with metadata extraction.

### Supported Extensions

- `.pdf` - PDF documents
- `.txt` - Plain text files
- `.md`, `.markdown` - Markdown files

### Constructor

```python
class DocumentLoader:
    def __init__(self, extract_images: bool = False)
```

**Parameters:**
- `extract_images` (bool): Whether to extract images from PDFs (requires extra dependencies)

### Methods

#### load_pdf()

Load a PDF file and extract text by page.

```python:line-numbers {1}
def load_pdf(self, file_path: str | Path) -> list[Document]
```

**Parameters:**
- `file_path` (str | Path): Path to the PDF file

**Returns:** List of Document objects, one per page

**Raises:**
- `ImportError`: If pypdf is not installed
- `FileNotFoundError`: If the file doesn't exist

**Example:**
```python
from langgraph_ollama_local.rag import DocumentLoader

loader = DocumentLoader()
docs = loader.load_pdf("sources/research_paper.pdf")

for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
```

#### load_text()

Load a text or markdown file.

```python
def load_text(self, file_path: str | Path) -> list[Document]
```

**Parameters:**
- `file_path` (str | Path): Path to the text file

**Returns:** List containing a single Document

**Example:**
```python
docs = loader.load_text("sources/notes.md")
print(docs[0].page_content)
```

#### load_file()

Load a file based on its extension (auto-detects format).

```python
def load_file(self, file_path: str | Path) -> list[Document]
```

**Parameters:**
- `file_path` (str | Path): Path to the file

**Returns:** List of Document objects

**Raises:**
- `ValueError`: If the file type is not supported

**Example:**
```python
# Auto-detects PDF, text, or markdown
docs = loader.load_file("sources/document.pdf")
```

#### load_directory()

Load all supported documents from a directory.

```python
def load_directory(
    self,
    directory: str | Path,
    recursive: bool = True,
    extensions: set[str] | None = None,
) -> list[Document]
```

**Parameters:**
- `directory` (str | Path): Path to the directory
- `recursive` (bool): Whether to search subdirectories (default: True)
- `extensions` (set[str] | None): Specific extensions to load (default: all supported)

**Returns:** List of all Document objects from the directory

**Example:**
```python
# Load all supported files
all_docs = loader.load_directory("sources/")

# Load only PDFs
pdf_docs = loader.load_directory("sources/", extensions={".pdf"})

# Non-recursive
docs = loader.load_directory("sources/", recursive=False)
```

### Convenience Function

```python
from langgraph_ollama_local.rag.document_loader import load_sources_directory

# Quick way to load all documents
docs = load_sources_directory("sources")
print(f"Loaded {len(docs)} documents")
```

---

## DocumentIndexer

Document indexing pipeline using ChromaDB for efficient similarity search.

### Constructor

```python
class DocumentIndexer:
    def __init__(
        self,
        config: IndexerConfig | None = None,
        embeddings: LocalEmbeddings | None = None,
    )
```

**Parameters:**
- `config` (IndexerConfig | None): Indexer configuration. Uses defaults if not provided
- `embeddings` (LocalEmbeddings | None): Embedding model. Creates one if not provided

### Configuration

#### IndexerConfig

```python
class IndexerConfig(BaseSettings):
    chunk_size: int = 1000  # Maximum chunk size in characters
    chunk_overlap: int = 200  # Overlap between chunks
    collection_name: str = "documents"  # ChromaDB collection name
    persist_directory: str = ".chromadb"  # ChromaDB persistence directory
    embedding_model: str = "all-mpnet-base-v2"  # Embedding model name
```

**Environment Variables:** All parameters support the `RAG_` prefix.

```bash
RAG_CHUNK_SIZE=1500
RAG_CHUNK_OVERLAP=300
RAG_COLLECTION_NAME=my_docs
RAG_PERSIST_DIRECTORY=.my_chromadb
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Methods

#### chunk_document()

Split a document into chunks with overlap.

```python
def chunk_document(self, document: Document) -> list[Document]
```

**Parameters:**
- `document` (Document): Document to split

**Returns:** List of chunked documents with updated metadata

**Example:**
```python
from langgraph_ollama_local.rag import DocumentIndexer, DocumentLoader

loader = DocumentLoader()
indexer = DocumentIndexer()

doc = loader.load_text("sources/article.txt")[0]
chunks = indexer.chunk_document(doc)
print(f"Split into {len(chunks)} chunks")
```

#### index_documents()

Index documents into ChromaDB.

```python
def index_documents(
    self,
    documents: list[Document],
    batch_size: int = 100,
) -> int
```

**Parameters:**
- `documents` (list[Document]): Documents to index (already chunked)
- `batch_size` (int): Number of documents to process at once (default: 100)

**Returns:** Number of documents indexed

**Example:**
```python
docs = loader.load_directory("sources/")
chunks = indexer.chunk_documents(docs)
count = indexer.index_documents(chunks)
print(f"Indexed {count} chunks")
```

#### index_file()

Load, chunk, and index a single file.

```python
def index_file(self, file_path: str | Path) -> int
```

**Parameters:**
- `file_path` (str | Path): Path to the file to index

**Returns:** Number of chunks indexed

**Example:**
```python
count = indexer.index_file("sources/paper.pdf")
print(f"Indexed {count} chunks from paper")
```

#### index_directory()

Load, chunk, and index all documents in a directory.

```python
def index_directory(
    self,
    directory: str | Path,
    recursive: bool = True,
) -> int
```

**Parameters:**
- `directory` (str | Path): Path to the directory
- `recursive` (bool): Whether to search subdirectories (default: True)

**Returns:** Number of chunks indexed

**Example:**
```python
from langgraph_ollama_local.rag import DocumentIndexer

indexer = DocumentIndexer()
count = indexer.index_directory("sources/")
print(f"Indexed {count} chunks")
```

#### get_stats()

Get statistics about the indexed documents.

```python
def get_stats(self) -> dict[str, Any]
```

**Returns:** Dictionary with collection statistics

**Example:**
```python
stats = indexer.get_stats()
print(f"Collection: {stats['collection_name']}")
print(f"Documents: {stats['document_count']}")
print(f"Model: {stats['embedding_model']}")
```

#### clear_collection()

Delete all documents from the collection.

```python
def clear_collection(self) -> None
```

#### delete_by_source()

Delete all chunks from a specific source file.

```python
def delete_by_source(self, source: str) -> int
```

**Parameters:**
- `source` (str): The source file path to delete

**Returns:** Number of documents deleted

### Complete Example

```python
from langgraph_ollama_local.rag import DocumentIndexer, IndexerConfig

# Configure indexer
config = IndexerConfig(
    chunk_size=1500,
    chunk_overlap=300,
    collection_name="my_docs",
    embedding_model="all-mpnet-base-v2"
)

# Create indexer
indexer = DocumentIndexer(config=config)

# Index directory
count = indexer.index_directory("sources/")
print(f"Indexed {count} chunks")

# Check stats
stats = indexer.get_stats()
print(stats)

# Clean up if needed
# indexer.clear_collection()
```

---

## LocalEmbeddings

Local embedding model using sentence-transformers. Keeps everything running locally without external API calls.

### Supported Models

| Model Name | Dimensions | Description | Size |
|-----------|-----------|-------------|------|
| `all-mpnet-base-v2` | 768 | High quality, best for semantic search | 420 MB |
| `all-MiniLM-L6-v2` | 384 | Fast, good balance of speed and quality | 90 MB |
| `paraphrase-MiniLM-L6-v2` | 384 | Optimized for paraphrase detection | 90 MB |

### Constructor

```python
class LocalEmbeddings:
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        normalize: bool = True,
        cache_folder: str | None = None,
    )
```

**Parameters:**
- `model_name` (str): The sentence-transformers model to use
- `device` (str): Device for inference ('cpu', 'cuda', 'mps')
- `normalize` (bool): Whether to normalize embeddings to unit length
- `cache_folder` (str | None): Directory to cache models

### Methods

#### embed_documents()

Embed a list of documents.

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]
```

**Parameters:**
- `texts` (list[str]): List of documents to embed

**Returns:** List of embedding vectors

**Example:**
```python
from langgraph_ollama_local.rag import LocalEmbeddings

embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
vectors = embeddings.embed_documents([
    "Retrieval-Augmented Generation combines search with LLMs",
    "RAG improves factual accuracy by grounding responses in documents"
])
print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")
```

#### embed_query()

Embed a single query.

```python
def embed_query(self, text: str) -> list[float]
```

**Parameters:**
- `text` (str): Query text to embed

**Returns:** Embedding vector

**Example:**
```python
query_vector = embeddings.embed_query("What is RAG?")
print(f"Query embedding dimension: {len(query_vector)}")
```

### Environment Variables

```bash
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true
EMBEDDING_CACHE_FOLDER=.embedding_cache
```

### Example: Custom Embedding Model

```python
from langgraph_ollama_local.rag import LocalEmbeddings

# Use faster, smaller model
embeddings = LocalEmbeddings(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize=True
)

# Embed documents
docs = ["Document 1", "Document 2", "Document 3"]
vectors = embeddings.embed_documents(docs)
```

---

## Graders

LLM-based graders for RAG quality assessment. Essential for Self-RAG and CRAG patterns.

### DocumentGrader

Grades document relevance to a query using an LLM.

#### Constructor

```python
class DocumentGrader:
    def __init__(self, llm: BaseChatModel)
```

**Parameters:**
- `llm` (BaseChatModel): Language model for grading

#### Methods

##### grade()

Grade whether a document is relevant to a question.

```python
def grade(self, document: Document | str, question: str) -> bool
```

**Parameters:**
- `document` (Document | str): Document to grade
- `question` (str): The user's question

**Returns:** True if relevant, False otherwise

**Example:**
```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.rag import DocumentGrader

config = LocalAgentConfig()
llm = config.create_chat_client()

grader = DocumentGrader(llm)
is_relevant = grader.grade(
    document="LangGraph is a framework for building agentic applications",
    question="What is LangGraph?"
)
print(f"Relevant: {is_relevant}")
```

##### grade_documents()

Grade multiple documents and separate relevant from irrelevant.

```python
def grade_documents(
    self,
    documents: list[Document],
    question: str,
) -> tuple[list[Document], list[Document]]
```

**Parameters:**
- `documents` (list[Document]): List of documents to grade
- `question` (str): The user's question

**Returns:** Tuple of (relevant_documents, irrelevant_documents)

**Example:**
```python
relevant, irrelevant = grader.grade_documents(docs, "What is RAG?")
print(f"{len(relevant)} relevant, {len(irrelevant)} irrelevant")
```

---

### HallucinationGrader

Grades whether an answer is grounded in the provided documents (detects hallucinations).

#### Constructor

```python
class HallucinationGrader:
    def __init__(self, llm: BaseChatModel)
```

#### Methods

##### grade()

Grade whether a generation is grounded in documents.

```python
def grade(
    self,
    documents: list[Document] | str,
    generation: str,
) -> bool
```

**Parameters:**
- `documents` (list[Document] | str): Source documents
- `generation` (str): The LLM's generated answer

**Returns:** True if grounded (no hallucination), False otherwise

**Example:**
```python
from langgraph_ollama_local.rag import HallucinationGrader

grader = HallucinationGrader(llm)
is_grounded = grader.grade(
    documents=retrieved_docs,
    generation="LangGraph is a framework for building stateful agents."
)
print(f"Grounded: {is_grounded}")
```

---

### AnswerGrader

Grades whether an answer adequately addresses the question.

#### Constructor

```python
class AnswerGrader:
    def __init__(self, llm: BaseChatModel)
```

#### Methods

##### grade()

Grade whether an answer addresses the question.

```python
def grade(self, question: str, generation: str) -> bool
```

**Parameters:**
- `question` (str): The user's question
- `generation` (str): The generated answer

**Returns:** True if the answer is useful, False otherwise

**Example:**
```python
from langgraph_ollama_local.rag import AnswerGrader

grader = AnswerGrader(llm)
is_useful = grader.grade(
    question="What is RAG?",
    generation="RAG stands for Retrieval-Augmented Generation..."
)
print(f"Useful: {is_useful}")
```

---

### QueryRouter

Routes queries to appropriate retrieval strategies (vectorstore, websearch, or direct).

#### Constructor

```python
class QueryRouter:
    def __init__(self, llm: BaseChatModel)
```

#### Methods

##### route()

Route a question to the appropriate strategy.

```python
def route(self, question: str) -> Literal["vectorstore", "websearch", "direct"]
```

**Parameters:**
- `question` (str): The user's question

**Returns:** Strategy name: 'vectorstore', 'websearch', or 'direct'

**Example:**
```python
from langgraph_ollama_local.rag import QueryRouter

router = QueryRouter(llm)

# Route different types of questions
route1 = router.route("What is in our company documentation?")
# Returns: "vectorstore"

route2 = router.route("Who won the 2024 election?")
# Returns: "websearch"

route3 = router.route("Hello, how are you?")
# Returns: "direct"
```

---

### QuestionRewriter

Rewrites questions for better retrieval by clarifying intent and removing ambiguity.

#### Constructor

```python
class QuestionRewriter:
    def __init__(self, llm: BaseChatModel)
```

#### Methods

##### rewrite()

Rewrite a question for better retrieval.

```python
def rewrite(self, question: str) -> str
```

**Parameters:**
- `question` (str): The original question

**Returns:** Improved version of the question

**Example:**
```python
from langgraph_ollama_local.rag import QuestionRewriter

rewriter = QuestionRewriter(llm)
better_question = rewriter.rewrite("what's rag")
# Returns: "What is Retrieval-Augmented Generation (RAG)?"
```

---

### create_graders()

Convenience function to create all graders with a single LLM instance.

```python
def create_graders(llm: BaseChatModel) -> dict[str, any]
```

**Parameters:**
- `llm` (BaseChatModel): Language model for grading

**Returns:** Dictionary of grader instances

**Example:**
```python
from langgraph_ollama_local.rag.graders import create_graders

graders = create_graders(llm)

# Use graders
is_relevant = graders["document"].grade(doc, question)
is_grounded = graders["hallucination"].grade(docs, answer)
is_useful = graders["answer"].grade(question, answer)
route = graders["router"].route(question)
better_q = graders["rewriter"].rewrite(question)
```

---

## Complete RAG Example

```python
from langgraph_ollama_local import LocalAgentConfig
from langgraph_ollama_local.rag import (
    DocumentLoader,
    DocumentIndexer,
    DocumentGrader,
    HallucinationGrader,
    AnswerGrader,
    QueryRouter,
)

# Setup
config = LocalAgentConfig()
llm = config.create_chat_client()

# 1. Load and index documents
loader = DocumentLoader()
indexer = DocumentIndexer()

docs = loader.load_directory("sources/")
chunks = indexer.chunk_documents(docs)
indexer.index_documents(chunks)

print(f"Indexed {indexer.get_stats()['document_count']} chunks")

# 2. Create graders
doc_grader = DocumentGrader(llm)
hallucination_grader = HallucinationGrader(llm)
answer_grader = AnswerGrader(llm)
router = QueryRouter(llm)

# 3. Process a query
question = "What is LangGraph?"

# Route the query
route = router.route(question)
print(f"Route: {route}")

# Grade documents (assuming retrieval happened)
relevant, irrelevant = doc_grader.grade_documents(retrieved_docs, question)

# Grade generation
is_grounded = hallucination_grader.grade(relevant, generated_answer)
is_useful = answer_grader.grade(question, generated_answer)

print(f"Grounded: {is_grounded}, Useful: {is_useful}")
```

---

## Related

- [Tutorial: Basic RAG](/tutorials/rag/08-basic-rag) - Introduction to RAG
- [Tutorial: Self-RAG](/tutorials/rag/09-self-rag) - Using DocumentGrader
- [Tutorial: Corrective RAG](/tutorials/rag/10-crag) - Using HallucinationGrader
- [Tutorial: Adaptive RAG](/tutorials/rag/11-adaptive-rag) - Using QueryRouter
- [Configuration API](/api/configuration) - RAG configuration options
