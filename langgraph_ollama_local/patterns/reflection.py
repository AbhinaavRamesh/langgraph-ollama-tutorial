"""
Reflection Pattern Module.

This module provides a reflection pattern where an LLM generates an output,
critiques it, and iteratively revises it until the output meets quality standards.
This pattern mirrors how humans improve their work through drafts and revisions.

Key concepts:
- **Generator**: Creates initial drafts and revisions
- **Critic**: Evaluates drafts and provides feedback or approval
- **Iteration Loop**: Cycles between generation and critique until approved
- **Quality Control**: Uses explicit approval signal or iteration limits

Architecture:
    ```
    ┌──────────────────────────────────────────────────────────┐
    │                   Reflection Loop                         │
    │                                                            │
    │  ┌───────────┐    ┌──────────┐    ┌──────────────┐      │
    │  │ Generator │───►│  Critic  │───►│Should        │      │
    │  │  (Draft)  │    │(Feedback)│    │Continue?     │      │
    │  └─────▲─────┘    └──────────┘    └─────┬────────┘      │
    │        │                                  │               │
    │        │                                  ▼               │
    │        │        ┌────────────────┬────────────┐          │
    │        │        │                │            │          │
    │        └────────┤ If Needs       │  If        ├────►END  │
    │         Revise  │ Revision       │  APPROVED  │          │
    │                 │                │  or Max    │          │
    │                 │                │  Iters     │          │
    │                 └────────────────┴────────────┘          │
    └──────────────────────────────────────────────────────────┘
    ```

Use Cases:
- **Writing**: Essays, reports, documentation
- **Code Generation**: Generate, review, refactor
- **Analysis**: Initial assessment, critique, refinement
- **Content Creation**: Drafts with quality improvement

Example:
    >>> from langgraph_ollama_local.patterns.reflection import (
    ...     create_reflection_graph,
    ...     run_reflection_task,
    ... )
    >>>
    >>> # Create graph with default settings
    >>> graph = create_reflection_graph(llm, max_iterations=3)
    >>>
    >>> # Run reflection on a task
    >>> result = run_reflection_task(
    ...     graph,
    ...     task="Write a brief explanation of quantum computing"
    ... )
    >>> print(result["draft"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# === State Definition ===

class ReflectionState(TypedDict):
    """
    State schema for reflection pattern.

    This state tracks the iterative refinement process where the generator
    creates drafts and the critic provides feedback until approval.

    Attributes:
        messages: Full conversation history (accumulates via add_messages reducer)
        task: The original task description
        draft: Current draft output from the generator
        critique: Latest critique from the critic
        iteration: Current iteration count
        max_iterations: Maximum iterations before forcing completion
    """

    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int
    max_iterations: int


# === Prompts ===

GENERATOR_PROMPT = """You are a skilled writer and content creator.

Your task is to write high-quality content that is:
- Clear and concise
- Accurate and complete
- Well-organized and structured
- Engaging and appropriate in tone

**Instructions:**
- If this is the first draft, write a complete response to the task
- If you have received critique, carefully revise your draft to address all feedback
- Focus on substantive improvements, not just cosmetic changes
- Ensure your revision directly addresses the specific concerns raised"""

CRITIC_PROMPT = """You are a thoughtful editor and critic.

Your job is to carefully evaluate drafts and provide constructive feedback.

**Evaluation Criteria:**
1. **Completeness**: Does it fully address the task?
2. **Clarity**: Is it clear and easy to understand?
3. **Accuracy**: Is the information correct?
4. **Organization**: Is it well-structured?
5. **Tone**: Is the tone appropriate?

**Instructions:**
- If the draft is excellent and needs no changes, respond with exactly: "APPROVED"
- Otherwise, provide specific, actionable feedback for improvement
- Focus on the most important issues first
- Be constructive and suggest concrete improvements"""


# === Structured Output Models ===

class MultiCriteriaFeedback(BaseModel):
    """
    Multi-criteria feedback for advanced reflection.

    Provides structured scores across multiple dimensions along with
    overall feedback and approval status.
    """

    clarity_score: int = Field(
        ge=1, le=10,
        description="Score for clarity and readability (1-10)"
    )
    accuracy_score: int = Field(
        ge=1, le=10,
        description="Score for accuracy and correctness (1-10)"
    )
    completeness_score: int = Field(
        ge=1, le=10,
        description="Score for completeness and thoroughness (1-10)"
    )
    overall_feedback: str = Field(
        description="Overall feedback and suggestions for improvement"
    )
    approved: bool = Field(
        description="Whether the draft is approved (True) or needs revision (False)"
    )


# === Node Functions ===

def create_generator_node(llm: "BaseChatModel"):
    """
    Create a generator node that creates or revises drafts.

    The generator:
    1. Creates an initial draft on first iteration
    2. Revises the draft based on critique on subsequent iterations
    3. Updates the draft and iteration count in state

    Args:
        llm: Language model for generation

    Returns:
        Node function for the generator

    Example:
        >>> generator = create_generator_node(llm)
        >>> # Generator will create drafts and revisions
    """
    def generator(state: ReflectionState) -> dict:
        """Generate or revise the draft based on critique."""
        iteration = state.get("iteration", 0)
        task = state["task"]
        draft = state.get("draft", "")
        critique = state.get("critique", "")

        if iteration == 0:
            # First draft
            user_message = f"""Write a complete response to this task:

Task: {task}

Provide your initial draft:"""
        else:
            # Revision based on critique
            user_message = f"""Revise your draft based on the critique provided.

ORIGINAL TASK:
{task}

CURRENT DRAFT:
{draft}

CRITIQUE:
{critique}

Please provide an improved version that addresses all the feedback:"""

        messages = [
            SystemMessage(content=GENERATOR_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        new_draft = response.content

        return {
            "draft": new_draft,
            "iteration": iteration + 1,
            "messages": [AIMessage(
                content=f"[Generator - Iteration {iteration + 1}] Generated draft ({len(new_draft)} chars)"
            )],
        }

    return generator


def create_critic_node(llm: "BaseChatModel"):
    """
    Create a critic node that evaluates drafts.

    The critic:
    1. Reviews the current draft against the original task
    2. Provides specific, actionable feedback
    3. Responds with "APPROVED" if the draft is excellent

    Args:
        llm: Language model for critique

    Returns:
        Node function for the critic

    Example:
        >>> critic = create_critic_node(llm)
        >>> # Critic will evaluate drafts and provide feedback
    """
    def critic(state: ReflectionState) -> dict:
        """Critique the current draft."""
        task = state["task"]
        draft = state["draft"]
        iteration = state["iteration"]

        user_message = f"""Review this draft for the given task.

TASK:
{task}

DRAFT:
{draft}

Provide your critique. If the draft is excellent and needs no changes, respond with exactly "APPROVED".
Otherwise, provide specific, actionable feedback for improvement:"""

        messages = [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = llm.invoke(messages)
        critique = response.content

        return {
            "critique": critique,
            "messages": [AIMessage(
                content=f"[Critic - Iteration {iteration}] {'APPROVED' if 'APPROVED' in critique.upper() else 'Feedback provided'}"
            )],
        }

    return critic


def create_multi_criteria_critic_node(
    llm: "BaseChatModel",
    approval_threshold: int = 7,
):
    """
    Create a multi-criteria critic node with structured scoring.

    This advanced critic provides:
    1. Individual scores for clarity, accuracy, and completeness
    2. Overall feedback
    3. Explicit approval decision

    Args:
        llm: Language model for critique
        approval_threshold: Minimum score for auto-approval (default: 7)

    Returns:
        Node function for multi-criteria critique

    Example:
        >>> critic = create_multi_criteria_critic_node(llm, approval_threshold=8)
        >>> # Critic will provide structured scores
    """
    # Try to use structured output
    try:
        structured_llm = llm.with_structured_output(MultiCriteriaFeedback)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    def multi_criteria_critic(state: ReflectionState) -> dict:
        """Critique with multi-criteria scoring."""
        task = state["task"]
        draft = state["draft"]
        iteration = state["iteration"]

        user_message = f"""Evaluate this draft using multiple criteria.

TASK:
{task}

DRAFT:
{draft}

Provide scores (1-10) for:
- Clarity: How clear and readable is it?
- Accuracy: How accurate and correct is the information?
- Completeness: How thoroughly does it address the task?

Also provide overall feedback and indicate if the draft should be approved."""

        messages = [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=user_message),
        ]

        if use_structured:
            feedback = structured_llm.invoke(messages)

            # Auto-approve if all scores meet threshold
            all_scores_good = all([
                feedback.clarity_score >= approval_threshold,
                feedback.accuracy_score >= approval_threshold,
                feedback.completeness_score >= approval_threshold,
            ])

            critique_text = f"""Clarity: {feedback.clarity_score}/10
Accuracy: {feedback.accuracy_score}/10
Completeness: {feedback.completeness_score}/10

{feedback.overall_feedback}

{'APPROVED' if feedback.approved or all_scores_good else 'NEEDS REVISION'}"""
        else:
            # Fallback: text-based critique
            response = structured_llm.invoke(messages)
            critique_text = response.content

        return {
            "critique": critique_text,
            "messages": [AIMessage(
                content=f"[Multi-Criteria Critic - Iteration {iteration}] {'APPROVED' if 'APPROVED' in critique_text.upper() else 'Scores provided'}"
            )],
        }

    return multi_criteria_critic


# === Routing Logic ===

def should_continue(state: ReflectionState) -> str:
    """
    Determine whether to continue refining or finish.

    The routing logic checks:
    1. If critique contains "APPROVED" -> END
    2. If max_iterations reached -> END
    3. Otherwise -> continue to generator for revision

    Args:
        state: Current reflection state

    Returns:
        "generator" to continue refining, END to finish
    """
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    critique = state.get("critique", "")

    # Stop if approved
    if "APPROVED" in critique.upper():
        return END

    # Stop if max iterations reached
    if iteration >= max_iterations:
        return END

    # Continue refining
    return "generator"


# === Graph Builders ===

def create_reflection_graph(
    llm: "BaseChatModel",
    max_iterations: int = 3,
    use_multi_criteria: bool = False,
    approval_threshold: int = 7,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a reflection graph with iterative improvement loop.

    This graph orchestrates:
    1. Generator creates initial draft
    2. Critic evaluates draft
    3. If not approved and under max iterations, generator revises
    4. Repeat until approved or max iterations

    Args:
        llm: Language model for generator and critic
        max_iterations: Maximum refinement iterations (default: 3)
        use_multi_criteria: Use structured multi-criteria scoring (default: False)
        approval_threshold: Score threshold for multi-criteria approval (default: 7)
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled reflection graph

    Example:
        >>> # Simple reflection
        >>> graph = create_reflection_graph(llm, max_iterations=3)
        >>>
        >>> # Multi-criteria reflection
        >>> graph = create_reflection_graph(
        ...     llm,
        ...     max_iterations=5,
        ...     use_multi_criteria=True,
        ...     approval_threshold=8
        ... )
    """
    workflow = StateGraph(ReflectionState)

    # Add nodes
    workflow.add_node("generator", create_generator_node(llm))

    if use_multi_criteria:
        workflow.add_node(
            "critic",
            create_multi_criteria_critic_node(llm, approval_threshold)
        )
    else:
        workflow.add_node("critic", create_critic_node(llm))

    # Entry point: start with generator
    workflow.add_edge(START, "generator")

    # Generator -> Critic
    workflow.add_edge("generator", "critic")

    # Critic -> Conditional (continue or end)
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "generator": "generator",
            END: END,
        }
    )

    return workflow.compile(checkpointer=checkpointer)


def create_multi_model_reflection_graph(
    generator_llm: "BaseChatModel",
    critic_llm: "BaseChatModel",
    max_iterations: int = 3,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create a reflection graph using different models for generation and critique.

    This pattern uses:
    - A fast model for generation (e.g., llama3.2:3b)
    - A stronger model for critique (e.g., llama3.1:70b)

    This can improve critique quality while keeping generation efficient.

    Args:
        generator_llm: Language model for generation (can be smaller/faster)
        critic_llm: Language model for critique (should be more capable)
        max_iterations: Maximum refinement iterations (default: 3)
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled reflection graph with multi-model setup

    Example:
        >>> from langchain_ollama import ChatOllama
        >>>
        >>> generator = ChatOllama(model="llama3.2:3b")
        >>> critic = ChatOllama(model="llama3.1:70b")
        >>>
        >>> graph = create_multi_model_reflection_graph(
        ...     generator_llm=generator,
        ...     critic_llm=critic,
        ...     max_iterations=3
        ... )
    """
    workflow = StateGraph(ReflectionState)

    # Add nodes with different models
    workflow.add_node("generator", create_generator_node(generator_llm))
    workflow.add_node("critic", create_critic_node(critic_llm))

    # Entry point: start with generator
    workflow.add_edge(START, "generator")

    # Generator -> Critic
    workflow.add_edge("generator", "critic")

    # Critic -> Conditional (continue or end)
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "generator": "generator",
            END: END,
        }
    )

    return workflow.compile(checkpointer=checkpointer)


# === Convenience Functions ===

def run_reflection_task(
    graph: "CompiledStateGraph",
    task: str,
    max_iterations: int = 3,
    thread_id: str = "default",
) -> dict:
    """
    Run a reflection task through the graph.

    This is a convenience function that sets up the initial state
    and invokes the graph with proper configuration.

    Args:
        graph: Compiled reflection graph
        task: Task description to work on
        max_iterations: Maximum iterations (default: 3)
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - draft: Final refined draft
        - critique: Final critique (should contain APPROVED if successful)
        - iteration: Final iteration count
        - messages: Full conversation history

    Example:
        >>> graph = create_reflection_graph(llm)
        >>> result = run_reflection_task(
        ...     graph,
        ...     task="Write a brief explanation of neural networks",
        ...     max_iterations=3
        ... )
        >>> print(result["draft"])
    """
    initial_state: ReflectionState = {
        "messages": [HumanMessage(content=f"Task: {task}")],
        "task": task,
        "draft": "",
        "critique": "",
        "iteration": 0,
        "max_iterations": max_iterations,
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


# === Module Exports ===

__all__ = [
    # State
    "ReflectionState",
    # Structured output
    "MultiCriteriaFeedback",
    # Node creators
    "create_generator_node",
    "create_critic_node",
    "create_multi_criteria_critic_node",
    # Routing
    "should_continue",
    # Graph builders
    "create_reflection_graph",
    "create_multi_model_reflection_graph",
    # Convenience
    "run_reflection_task",
]
