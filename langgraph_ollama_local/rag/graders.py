"""
LLM-based graders for RAG quality assessment.

This module provides graders for evaluating:
- Document relevance to queries
- Answer hallucination detection
- Answer quality and usefulness

These graders are essential for Self-RAG and CRAG patterns.

Example:
    >>> from langgraph_ollama_local.rag import DocumentGrader
    >>> grader = DocumentGrader(llm)
    >>> is_relevant = grader.grade(document, query)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Structured output schemas for grading
class GradeDocuments(BaseModel):
    """Binary score for document relevance."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination detection."""

    binary_score: str = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for answer usefulness."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# System prompts for graders
DOCUMENT_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

Respond with ONLY 'yes' or 'no', nothing else."""

HALLUCINATION_GRADER_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.
'no' means the answer contains information not supported by the facts (hallucination).

Respond with ONLY 'yes' or 'no', nothing else."""

ANSWER_GRADER_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question.

Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.
'no' means the answer does not adequately address the question.

Respond with ONLY 'yes' or 'no', nothing else."""

QUESTION_REWRITER_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized for web search or document retrieval.

Look at the input and try to reason about the underlying semantic intent / meaning.
Output ONLY the rewritten question, nothing else."""


class DocumentGrader:
    """
    Grades document relevance to a query using an LLM.

    This grader determines whether a retrieved document is relevant
    to the user's question, which is essential for Self-RAG.

    Attributes:
        llm: The language model for grading.

    Example:
        >>> from langgraph_ollama_local import LocalAgentConfig
        >>> config = LocalAgentConfig()
        >>> llm = config.create_chat_client()
        >>> grader = DocumentGrader(llm)
        >>> is_relevant = grader.grade(doc, "What is RAG?")
    """

    def __init__(self, llm: "BaseChatModel"):
        """
        Initialize the document grader.

        Args:
            llm: Language model for grading.
        """
        self.llm = llm

        # Create the grading chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DOCUMENT_GRADER_PROMPT),
            ("human", "Document:\n{document}\n\nQuestion: {question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def grade(self, document: Document | str, question: str) -> bool:
        """
        Grade whether a document is relevant to a question.

        Args:
            document: Document to grade (Document object or string).
            question: The user's question.

        Returns:
            True if relevant, False otherwise.
        """
        doc_text = document.page_content if isinstance(document, Document) else document

        try:
            result = self.chain.invoke({
                "document": doc_text,
                "question": question,
            })
            is_relevant = result.strip().lower() == "yes"
            logger.debug(f"Document relevance: {is_relevant}")
            return is_relevant
        except Exception as e:
            logger.warning(f"Grading failed: {e}, defaulting to relevant")
            return True

    def grade_documents(
        self,
        documents: list[Document],
        question: str,
    ) -> tuple[list[Document], list[Document]]:
        """
        Grade multiple documents and separate relevant from irrelevant.

        Args:
            documents: List of documents to grade.
            question: The user's question.

        Returns:
            Tuple of (relevant_documents, irrelevant_documents).

        Example:
            >>> relevant, irrelevant = grader.grade_documents(docs, question)
            >>> print(f"{len(relevant)} relevant, {len(irrelevant)} irrelevant")
        """
        relevant = []
        irrelevant = []

        for doc in documents:
            if self.grade(doc, question):
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        logger.info(f"Graded {len(documents)} docs: {len(relevant)} relevant, {len(irrelevant)} irrelevant")
        return relevant, irrelevant


class HallucinationGrader:
    """
    Grades whether an answer is grounded in the provided documents.

    Detects hallucinations by checking if the generated answer
    is supported by the retrieved facts.

    Example:
        >>> grader = HallucinationGrader(llm)
        >>> is_grounded = grader.grade(documents, answer)
    """

    def __init__(self, llm: "BaseChatModel"):
        """
        Initialize the hallucination grader.

        Args:
            llm: Language model for grading.
        """
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", HALLUCINATION_GRADER_PROMPT),
            ("human", "Facts:\n{documents}\n\nLLM Generation: {generation}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def grade(
        self,
        documents: list[Document] | str,
        generation: str,
    ) -> bool:
        """
        Grade whether a generation is grounded in documents.

        Args:
            documents: Source documents (list or combined string).
            generation: The LLM's generated answer.

        Returns:
            True if grounded (no hallucination), False otherwise.
        """
        if isinstance(documents, list):
            docs_text = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])
        else:
            docs_text = documents

        try:
            result = self.chain.invoke({
                "documents": docs_text,
                "generation": generation,
            })
            is_grounded = result.strip().lower() == "yes"
            logger.debug(f"Hallucination check: grounded={is_grounded}")
            return is_grounded
        except Exception as e:
            logger.warning(f"Hallucination grading failed: {e}, defaulting to grounded")
            return True


class AnswerGrader:
    """
    Grades whether an answer adequately addresses the question.

    Evaluates if the generated answer is useful and resolves
    the user's query.

    Example:
        >>> grader = AnswerGrader(llm)
        >>> is_useful = grader.grade(question, answer)
    """

    def __init__(self, llm: "BaseChatModel"):
        """
        Initialize the answer grader.

        Args:
            llm: Language model for grading.
        """
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_GRADER_PROMPT),
            ("human", "Question: {question}\n\nAnswer: {generation}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def grade(self, question: str, generation: str) -> bool:
        """
        Grade whether an answer addresses the question.

        Args:
            question: The user's question.
            generation: The generated answer.

        Returns:
            True if the answer is useful, False otherwise.
        """
        try:
            result = self.chain.invoke({
                "question": question,
                "generation": generation,
            })
            is_useful = result.strip().lower() == "yes"
            logger.debug(f"Answer usefulness: {is_useful}")
            return is_useful
        except Exception as e:
            logger.warning(f"Answer grading failed: {e}, defaulting to useful")
            return True


class QuestionRewriter:
    """
    Rewrites questions for better retrieval.

    Optimizes questions for semantic search by clarifying
    intent and removing ambiguity.

    Example:
        >>> rewriter = QuestionRewriter(llm)
        >>> better_question = rewriter.rewrite("what's rag")
        >>> # Returns: "What is Retrieval-Augmented Generation (RAG)?"
    """

    def __init__(self, llm: "BaseChatModel"):
        """
        Initialize the question rewriter.

        Args:
            llm: Language model for rewriting.
        """
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QUESTION_REWRITER_PROMPT),
            ("human", "Question: {question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """
        Rewrite a question for better retrieval.

        Args:
            question: The original question.

        Returns:
            Improved version of the question.
        """
        try:
            result = self.chain.invoke({"question": question})
            rewritten = result.strip()
            logger.debug(f"Rewritten: '{question}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Question rewriting failed: {e}, using original")
            return question


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies.

    Determines whether to use vector search, web search,
    or direct generation based on query type.

    Example:
        >>> router = QueryRouter(llm)
        >>> strategy = router.route("What is the capital of France?")
        >>> # Returns: "direct" (factual question, no retrieval needed)
    """

    ROUTER_PROMPT = """You are an expert at routing user questions to the appropriate data source.

Based on the question, determine the best approach:
- "vectorstore": For questions about specific documents, technical topics, or domain knowledge
- "websearch": For questions about current events, recent news, or general knowledge
- "direct": For simple factual questions or greetings that don't need retrieval

Output ONLY one of: vectorstore, websearch, direct"""

    def __init__(self, llm: "BaseChatModel"):
        """
        Initialize the query router.

        Args:
            llm: Language model for routing decisions.
        """
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.ROUTER_PROMPT),
            ("human", "Question: {question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, question: str) -> Literal["vectorstore", "websearch", "direct"]:
        """
        Route a question to the appropriate strategy.

        Args:
            question: The user's question.

        Returns:
            Strategy name: 'vectorstore', 'websearch', or 'direct'.
        """
        try:
            result = self.chain.invoke({"question": question})
            strategy = result.strip().lower()

            if strategy in ("vectorstore", "websearch", "direct"):
                logger.debug(f"Routed '{question[:50]}...' to {strategy}")
                return strategy
            else:
                logger.warning(f"Unknown route '{strategy}', defaulting to vectorstore")
                return "vectorstore"
        except Exception as e:
            logger.warning(f"Routing failed: {e}, defaulting to vectorstore")
            return "vectorstore"


def create_graders(llm: "BaseChatModel") -> dict[str, any]:
    """
    Create all graders with a single LLM instance.

    Args:
        llm: Language model for grading.

    Returns:
        Dictionary of grader instances.

    Example:
        >>> graders = create_graders(llm)
        >>> is_relevant = graders["document"].grade(doc, question)
        >>> is_grounded = graders["hallucination"].grade(docs, answer)
    """
    return {
        "document": DocumentGrader(llm),
        "hallucination": HallucinationGrader(llm),
        "answer": AnswerGrader(llm),
        "rewriter": QuestionRewriter(llm),
        "router": QueryRouter(llm),
    }
