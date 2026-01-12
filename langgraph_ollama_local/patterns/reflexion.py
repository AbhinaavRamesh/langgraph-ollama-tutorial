"""
Reflexion Pattern Module.

This module implements the Reflexion pattern for language agents with verbal
reinforcement learning. Unlike simple reflection that improves a single output,
Reflexion learns from multiple attempts with episodic memory.

Key concepts:
- **Episodic Memory**: Store all previous attempts to learn from failures
- **Self-Reflection**: Generate structured critique of own outputs
- **External Search**: Use tools to gather missing information
- **Iterative Improvement**: Revise answers based on reflection and search

Architecture:
    ```
    ┌───────────────────────────────────────────────────────────┐
    │                   Reflexion Loop                           │
    │                                                             │
    │  ┌──────────┐    ┌────────────┐    ┌─────────────┐       │
    │  │  Draft   │───►│  Execute   │───►│   Revise    │       │
    │  │ Answer + │    │  Search    │    │  Based on   │       │
    │  │ Reflect  │    │  Queries   │    │  Results    │       │
    │  └────┬─────┘    └────────────┘    └──────┬──────┘       │
    │       │                                     │               │
    │       │          Episodic Memory            │               │
    │       └─────────────────────────────────────┘               │
    │                    (Loop until max iterations)              │
    └───────────────────────────────────────────────────────────┘
    ```

Reference:
    Paper: Reflexion: Language Agents with Verbal Reinforcement Learning
    https://arxiv.org/abs/2303.11366

Example:
    >>> from langgraph_ollama_local.patterns.reflexion import (
    ...     create_reflexion_graph,
    ...     run_reflexion_task,
    ... )
    >>> from langchain_community.tools.tavily_search import TavilySearchResults
    >>>
    >>> search_tool = TavilySearchResults(max_results=3)
    >>> graph = create_reflexion_graph(llm, search_tool)
    >>> result = run_reflexion_task(
    ...     graph,
    ...     task="What are the latest developments in quantum computing?",
    ...     max_iterations=3
    ... )
    >>> print(result["current_attempt"])
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


# === State Definition ===


class ReflexionState(TypedDict):
    """
    State schema for Reflexion pattern with episodic memory.

    Key difference from Reflection pattern: maintains history of ALL attempts
    and reflections to learn from past failures across iterations.

    Attributes:
        messages: Message history (for tool execution)
        task: Original question or task to solve
        attempts: List of all attempts with answers and queries (episodic memory)
        current_attempt: The current answer being worked on
        reflections: List of all self-critiques from previous attempts
        current_reflection: Latest reflection on current attempt
        iteration: Current iteration number
        max_iterations: Maximum number of attempts allowed
        success_achieved: Whether task is considered complete
    """

    messages: Annotated[list, add_messages]
    task: str
    attempts: Annotated[list[dict], operator.add]
    current_attempt: str
    reflections: Annotated[list[str], operator.add]
    current_reflection: str
    iteration: int
    max_iterations: int
    success_achieved: bool


# === Pydantic Models for Structured Output ===


class Reflection(BaseModel):
    """
    Structured self-critique of an answer attempt.

    Identifies what information is missing and what is unnecessary
    to guide improvement in the next iteration.

    Attributes:
        missing: Description of information that's incomplete or absent
        superfluous: Description of information that's unnecessary or irrelevant
    """

    missing: str = Field(
        description="What critical information is missing or incomplete in this answer?"
    )
    superfluous: str = Field(
        description="What information is unnecessary, irrelevant, or distracting?"
    )


class AnswerQuestion(BaseModel):
    """
    Structured answer with self-critique and search queries.

    This model combines the initial answer attempt with immediate
    self-reflection and queries to improve the answer.

    Attributes:
        answer: The answer to the question (~250 words)
        reflection: Structured self-critique
        search_queries: List of search queries to fill knowledge gaps
    """

    answer: str = Field(
        description="Your answer to the question (approximately 250 words)"
    )
    reflection: Reflection = Field(
        description="Self-critique identifying gaps and excess"
    )
    search_queries: list[str] = Field(
        description="1-3 search queries to gather missing information",
        max_length=3,
    )


class ReviseAnswer(AnswerQuestion):
    """
    Revised answer that extends AnswerQuestion with citations.

    After search results are available, this model adds references
    to support the improved answer.

    Attributes:
        answer: Revised answer incorporating new information
        reflection: Updated self-critique
        search_queries: Additional queries if needed
        references: List of sources/citations used in the revised answer
    """

    references: list[str] = Field(
        description="Sources and citations used in this revised answer"
    )


# === Node Functions ===


def create_initial_responder(llm: "BaseChatModel"):
    """
    Create initial responder node that drafts answer with self-critique.

    This node generates the first attempt at answering, immediately reflects
    on its own output, and proposes search queries to improve.

    Args:
        llm: Language model for answer generation

    Returns:
        Node function that creates initial answer attempt

    Example:
        >>> responder = create_initial_responder(llm)
        >>> state = {"task": "What is quantum entanglement?", "iteration": 0}
        >>> result = responder(state)
        >>> print(result["current_attempt"])
    """
    # Try to use structured output
    try:
        structured_llm = llm.with_structured_output(AnswerQuestion)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def responder(state: ReflexionState) -> dict:
        """Generate initial answer with reflection and search queries."""
        task = state["task"]
        iteration = state.get("iteration", 0)

        # Build prompt with episodic memory if available
        previous_attempts = state.get("attempts", [])
        previous_reflections = state.get("reflections", [])

        prompt = f"""Answer the following question thoughtfully.

Question: {task}"""

        # Include previous attempts and reflections if this is not the first iteration
        if previous_attempts and previous_reflections:
            prompt += f"""

Previous attempts and what was wrong with them:
"""
            for i, (attempt, reflection) in enumerate(zip(previous_attempts, previous_reflections), 1):
                prompt += f"""
Attempt {i}:
Answer: {attempt.get('answer', 'N/A')[:200]}...
Reflection: {reflection[:200]}...
"""

            prompt += """

Learn from these previous attempts. Avoid repeating the same mistakes.
"""

        prompt += """

After answering:
1. Critique your answer - what's missing? What's unnecessary?
2. Suggest 1-3 search queries to fill knowledge gaps."""

        messages = [HumanMessage(content=prompt)]

        if use_structured:
            try:
                response = structured_llm.invoke(messages)
                answer = response.answer
                reflection_obj = response.reflection
                search_queries = response.search_queries
            except Exception:
                # Fallback if structured output fails
                response = llm.invoke(messages)
                answer = response.content
                reflection_obj = Reflection(
                    missing="Unable to determine with structured output",
                    superfluous="Unable to determine with structured output"
                )
                search_queries = [task]  # Default to original task
        else:
            # Fallback for models without structured output
            response = llm.invoke(messages)
            answer = response.content
            reflection_obj = Reflection(
                missing="Structured output not available - manual review needed",
                superfluous="Structured output not available - manual review needed"
            )
            # Extract simple queries from task
            search_queries = [task]

        reflection_text = f"Missing: {reflection_obj.missing}\nSuperfluous: {reflection_obj.superfluous}"

        return {
            "current_attempt": answer,
            "current_reflection": reflection_text,
            "attempts": [{
                "num": iteration + 1,
                "answer": answer,
                "search_queries": search_queries,
            }],
            "iteration": iteration + 1,
        }

    return responder


def create_tool_executor(search_tool: "BaseTool"):
    """
    Create tool executor node that runs search queries.

    Executes the search queries generated during reflection to gather
    external information that can improve the answer.

    Args:
        search_tool: Search tool (e.g., TavilySearchResults, DuckDuckGoSearch)

    Returns:
        Node function that executes search queries

    Example:
        >>> from langchain_community.tools.tavily_search import TavilySearchResults
        >>> search = TavilySearchResults(max_results=3)
        >>> executor = create_tool_executor(search)
    """
    def executor(state: ReflexionState) -> dict:
        """Execute search queries from the last attempt."""
        attempts = state.get("attempts", [])

        if not attempts:
            return {"messages": [ToolMessage(content="No attempts available", tool_call_id="search")]}

        last_attempt = attempts[-1]
        queries = last_attempt.get("search_queries", [])

        if not queries:
            return {"messages": [ToolMessage(content="No search queries generated", tool_call_id="search")]}

        # Execute search for each query
        all_results = []
        for query in queries:
            try:
                result = search_tool.invoke(query)
                all_results.append(f"Query: {query}\nResults: {result}\n")
            except Exception as e:
                all_results.append(f"Query: {query}\nError: {str(e)}\n")

        combined_results = "\n---\n".join(all_results)

        return {
            "messages": [ToolMessage(
                content=combined_results,
                tool_call_id="search"
            )],
        }

    return executor


def create_revisor(llm: "BaseChatModel"):
    """
    Create revisor node that improves answer using search results.

    Takes the search results and previous reflection to generate an
    improved answer with proper citations.

    Args:
        llm: Language model for answer revision

    Returns:
        Node function that revises the answer

    Example:
        >>> revisor = create_revisor(llm)
        >>> # Assumes state has current_attempt, current_reflection, and search results
        >>> result = revisor(state)
    """
    # Try to use structured output for ReviseAnswer
    try:
        structured_llm = llm.with_structured_output(ReviseAnswer)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def revisor(state: ReflexionState) -> dict:
        """Revise answer using search results and reflection."""
        task = state["task"]
        current_reflection = state.get("current_reflection", "")
        messages = state.get("messages", [])

        # Get search results from last message
        search_results = "No search results available"
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    search_results = msg.content
                    break

        prompt = f"""Revise your previous answer using the new information from search results.

Original question: {task}

Previous reflection on your answer:
{current_reflection}

Search results:
{search_results}

Instructions:
1. Incorporate relevant information from search results
2. Address the gaps identified in your reflection
3. Remove unnecessary information
4. Include citations/references for factual claims
5. Keep the answer around 250 words

Provide your revised answer with references."""

        if use_structured:
            try:
                response = structured_llm.invoke([HumanMessage(content=prompt)])
                revised_answer = response.answer
                new_reflection = response.reflection
                references = response.references
            except Exception:
                # Fallback
                response = llm.invoke([HumanMessage(content=prompt)])
                revised_answer = response.content
                new_reflection = Reflection(
                    missing="Unable to determine",
                    superfluous="Unable to determine"
                )
                references = ["Search results incorporated"]
        else:
            # Fallback for non-structured models
            response = llm.invoke([HumanMessage(content=prompt)])
            revised_answer = response.content
            new_reflection = Reflection(
                missing="Manual review needed",
                superfluous="Manual review needed"
            )
            references = ["See search results above"]

        reflection_text = f"Missing: {new_reflection.missing}\nSuperfluous: {new_reflection.superfluous}"
        iteration = state.get("iteration", 0)

        return {
            "current_attempt": revised_answer,
            "attempts": [{
                "num": iteration + 1,
                "answer": revised_answer,
                "references": references,
            }],
            "reflections": [current_reflection],
            "current_reflection": reflection_text,
        }

    return revisor


# === Graph Builder ===


def create_reflexion_graph(
    llm: "BaseChatModel",
    search_tool: "BaseTool",
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a Reflexion graph for iterative answer improvement.

    The graph follows this flow:
    1. Draft initial answer with self-critique and search queries
    2. Execute search queries to gather information
    3. Revise answer based on search results and reflection
    4. Repeat until max iterations or success

    Args:
        llm: Language model for answer generation and revision
        search_tool: Search tool for gathering external information
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled Reflexion graph

    Example:
        >>> from langchain_community.tools.tavily_search import TavilySearchResults
        >>> from langchain_ollama import ChatOllama
        >>>
        >>> llm = ChatOllama(model="llama3.2")
        >>> search = TavilySearchResults(max_results=3)
        >>> graph = create_reflexion_graph(llm, search)
        >>>
        >>> result = graph.invoke({
        ...     "task": "Explain quantum computing",
        ...     "messages": [],
        ...     "attempts": [],
        ...     "reflections": [],
        ...     "iteration": 0,
        ...     "max_iterations": 3,
        ...     "current_attempt": "",
        ...     "current_reflection": "",
        ...     "success_achieved": False,
        ... })
    """
    workflow = StateGraph(ReflexionState)

    # Add nodes
    workflow.add_node("draft", create_initial_responder(llm))
    workflow.add_node("execute_tools", create_tool_executor(search_tool))
    workflow.add_node("revise", create_revisor(llm))

    # Entry point: draft initial answer
    workflow.add_edge(START, "draft")

    # Draft -> Execute tools
    workflow.add_edge("draft", "execute_tools")

    # Execute tools -> Revise
    workflow.add_edge("execute_tools", "revise")

    # Conditional: continue or end
    def should_continue(state: ReflexionState) -> str:
        """Determine whether to continue iterating or end."""
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        success = state.get("success_achieved", False)

        # End if success or max iterations reached
        if success or iteration >= max_iterations:
            return END

        # Continue with new draft
        return "draft"

    workflow.add_conditional_edges(
        "revise",
        should_continue,
        {
            "draft": "draft",
            END: END,
        }
    )

    return workflow.compile(checkpointer=checkpointer)


# === Runner Function ===


def run_reflexion_task(
    graph: "CompiledStateGraph",
    task: str,
    max_iterations: int = 3,
    thread_id: str = "default",
) -> dict:
    """
    Run a Reflexion task with the given graph.

    Executes the full Reflexion loop: draft, search, revise, repeat.
    Returns the final state with the best answer and all attempts.

    Args:
        graph: Compiled Reflexion graph
        task: Question or task to solve
        max_iterations: Maximum number of improvement iterations (default: 3)
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - current_attempt: Best answer after all iterations
        - attempts: List of all attempts with episodic memory
        - reflections: All self-critiques generated
        - iteration: Total iterations performed

    Example:
        >>> result = run_reflexion_task(
        ...     graph,
        ...     task="What are the applications of CRISPR?",
        ...     max_iterations=3
        ... )
        >>> print(result["current_attempt"])
        >>> print(f"Total attempts: {len(result['attempts'])}")
        >>> for i, attempt in enumerate(result["attempts"], 1):
        ...     print(f"Attempt {i}: {attempt['answer'][:100]}...")
    """
    initial_state: ReflexionState = {
        "messages": [],
        "task": task,
        "attempts": [],
        "current_attempt": "",
        "reflections": [],
        "current_reflection": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "success_achieved": False,
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===


__all__ = [
    # State
    "ReflexionState",
    # Models
    "Reflection",
    "AnswerQuestion",
    "ReviseAnswer",
    # Node creators
    "create_initial_responder",
    "create_tool_executor",
    "create_revisor",
    # Graph builder
    "create_reflexion_graph",
    # Runner
    "run_reflexion_task",
]
