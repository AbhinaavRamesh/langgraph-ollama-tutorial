"""
Multi-Agent Evaluation Module.

This module provides patterns for evaluating agents through simulation
and automated scoring. It enables testing agents via simulated user
interactions and automated evaluator agents that score conversations.

Key concepts:
- **Simulated Users**: Agents that mimic real users with specific personas
- **Evaluator Agents**: Agents that score conversations on various metrics
- **Evaluation Sessions**: Run agent + simulated user with metrics collection
- **Metrics Aggregation**: Combine scores into summary statistics

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                  Evaluation Session                      │
    │                                                           │
    │  ┌──────────┐       ┌──────────┐       ┌──────────┐    │
    │  │  Agent   │◄─────►│Simulated │       │Evaluator │    │
    │  │ Under    │       │  User    │──────►│  Agent   │    │
    │  │  Test    │       │  Agent   │       │          │    │
    │  └──────────┘       └──────────┘       └────┬─────┘    │
    │                                              │           │
    │                                              ▼           │
    │                                        ┌──────────┐     │
    │                                        │  Scores  │     │
    │                                        │  Metrics │     │
    │                                        └──────────┘     │
    └─────────────────────────────────────────────────────────┘
    ```

Example:
    >>> from langgraph_ollama_local.patterns.evaluation import (
    ...     SimulatedUser,
    ...     create_evaluation_graph,
    ...     run_evaluation_session,
    ... )
    >>>
    >>> user_config = SimulatedUser(
    ...     persona="Frustrated customer with billing issue",
    ...     goals=["Get refund", "Express dissatisfaction"],
    ...     behavior="impatient",
    ... )
    >>>
    >>> graph = create_evaluation_graph(llm, agent_under_test, user_config)
    >>> results = run_evaluation_session(graph, max_turns=5)
    >>> print(results["metrics"])
"""

from __future__ import annotations

import operator
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

class EvaluationState(TypedDict):
    """
    State schema for agent evaluation sessions.

    Attributes:
        messages: Full conversation history between agent and simulated user
        conversation: Formatted conversation for evaluator review
        evaluator_scores: List of score dicts from evaluator agent
        turn_count: Current conversation turn number
        max_turns: Maximum number of conversation turns
        session_complete: Whether the evaluation session is done
        final_metrics: Aggregated metrics summary
    """

    messages: Annotated[list, add_messages]
    conversation: str
    evaluator_scores: Annotated[list[dict], operator.add]
    turn_count: int
    max_turns: int
    session_complete: bool
    final_metrics: dict[str, float]


# === Configuration Models ===

class SimulatedUser(BaseModel):
    """
    Configuration for a simulated user agent.

    The simulated user mimics real user behavior with specific persona,
    goals, and behavioral traits to test agent responses.

    Attributes:
        persona: Description of user's background and situation
        goals: List of objectives the user wants to achieve
        behavior: Behavioral style (friendly, impatient, confused, etc.)
        initial_message: Optional first message to start conversation
    """

    persona: str = Field(
        description="User's background, situation, and characteristics"
    )
    goals: list[str] = Field(
        description="What the user wants to accomplish in the conversation"
    )
    behavior: Literal[
        "friendly", "impatient", "confused", "technical", "casual"
    ] = Field(
        default="friendly",
        description="User's communication style"
    )
    initial_message: str | None = Field(
        default=None,
        description="First message to start conversation (auto-generated if None)"
    )


class EvaluationCriteria(BaseModel):
    """
    Criteria for evaluating conversations.

    Attributes:
        helpfulness: Score for how helpful the agent's responses are (1-5)
        accuracy: Score for factual accuracy of information (1-5)
        empathy: Score for empathy and understanding (1-5)
        efficiency: Score for conciseness and directness (1-5)
        goal_completion: Whether user's goals were achieved (0 or 1)
    """

    helpfulness: int = Field(
        ge=1, le=5,
        description="How helpful were the agent's responses? (1=not helpful, 5=very helpful)"
    )
    accuracy: int = Field(
        ge=1, le=5,
        description="How accurate was the information provided? (1=inaccurate, 5=accurate)"
    )
    empathy: int = Field(
        ge=1, le=5,
        description="How empathetic was the agent? (1=cold, 5=very empathetic)"
    )
    efficiency: int = Field(
        ge=1, le=5,
        description="How efficient was the conversation? (1=rambling, 5=concise)"
    )
    goal_completion: int = Field(
        ge=0, le=1,
        description="Were the user's goals completed? (0=no, 1=yes)"
    )
    reasoning: str = Field(
        description="Brief explanation of the scores"
    )


# === Node Functions ===

def create_simulated_user_node(
    llm: "BaseChatModel",
    user_config: SimulatedUser,
):
    """
    Create a simulated user agent node.

    The simulated user acts according to its persona, goals, and behavior
    to generate realistic user messages in the conversation.

    Args:
        llm: Language model for the simulated user
        user_config: Configuration defining user persona and behavior

    Returns:
        Node function for the simulated user

    Example:
        >>> user = SimulatedUser(
        ...     persona="Customer with broken product",
        ...     goals=["Get replacement", "Express frustration"],
        ...     behavior="impatient"
        ... )
        >>> node = create_simulated_user_node(llm, user)
    """
    system_prompt = f"""You are simulating a user in a conversation with a customer service agent.

**Your Persona:**
{user_config.persona}

**Your Goals:**
{chr(10).join(f"- {goal}" for goal in user_config.goals)}

**Your Behavior:**
You communicate in a {user_config.behavior} manner.

**Instructions:**
- Stay in character as this user throughout the conversation
- Work towards achieving your goals naturally
- Respond realistically based on what the agent says
- Keep responses concise (1-3 sentences typical for users)
- If your goals are met, you can express satisfaction and end the conversation
- Don't mention that you're simulating or testing anything"""

    def simulated_user(state: EvaluationState) -> dict:
        """Generate a user message based on conversation history."""
        # Check if this is the first turn
        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0)

        # Use initial message if provided and it's the first turn
        if turn_count == 0 and user_config.initial_message:
            user_message = user_config.initial_message
        else:
            # Generate response based on conversation
            prompt_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Based on the conversation so far, provide your next message as this user.

Conversation history:
{chr(10).join(f"{m.type}: {m.content}" for m in messages[-6:] if messages)}

Your next message:""")
            ]

            response = llm.invoke(prompt_messages)
            user_message = response.content

        return {
            "messages": [HumanMessage(content=user_message)],
            "turn_count": turn_count + 1,
        }

    return simulated_user


def create_evaluator_node(llm: "BaseChatModel"):
    """
    Create an evaluator agent node that scores conversations.

    The evaluator reviews the conversation and provides scores across
    multiple criteria: helpfulness, accuracy, empathy, efficiency, and
    goal completion.

    Args:
        llm: Language model for the evaluator

    Returns:
        Node function for the evaluator

    Example:
        >>> evaluator = create_evaluator_node(llm)
        >>> # Evaluator will score conversations on 5 criteria
    """
    # Try to use structured output for reliable scoring
    try:
        structured_llm = llm.with_structured_output(EvaluationCriteria)
        use_structured = True
    except (AttributeError, NotImplementedError):
        structured_llm = llm
        use_structured = False

    system_prompt = """You are an expert evaluator of customer service conversations.

Your job is to objectively score conversations on these criteria:
- **Helpfulness**: How helpful were the agent's responses? (1-5)
- **Accuracy**: How accurate was the information? (1-5)
- **Empathy**: How empathetic was the agent? (1-5)
- **Efficiency**: How concise and direct? (1-5)
- **Goal Completion**: Were the user's goals achieved? (0=no, 1=yes)

Be fair and objective. Consider the full conversation context."""

    def evaluator(state: EvaluationState) -> dict:
        """Evaluate the conversation and provide scores."""
        messages = state.get("messages", [])

        # Format conversation for evaluation
        conversation_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
            for m in messages
        ])

        eval_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Evaluate this conversation:

{conversation_text}

Provide scores for all criteria.""")
        ]

        if use_structured:
            criteria = structured_llm.invoke(eval_messages)
            scores = {
                "helpfulness": criteria.helpfulness,
                "accuracy": criteria.accuracy,
                "empathy": criteria.empathy,
                "efficiency": criteria.efficiency,
                "goal_completion": criteria.goal_completion,
                "reasoning": criteria.reasoning,
            }
        else:
            # Fallback: parse from text (simplified)
            response = structured_llm.invoke(eval_messages)
            scores = {
                "helpfulness": 3,
                "accuracy": 3,
                "empathy": 3,
                "efficiency": 3,
                "goal_completion": 0,
                "reasoning": response.content[:200],
            }

        return {
            "evaluator_scores": [scores],
            "conversation": conversation_text,
        }

    return evaluator


def create_check_completion_node():
    """
    Create a node that checks if the evaluation session should end.

    Returns:
        Node function that updates session_complete flag
    """
    def check_completion(state: EvaluationState) -> dict:
        """Check if session should complete."""
        turn_count = state.get("turn_count", 0)
        max_turns = state.get("max_turns", 10)

        # Check if max turns reached
        if turn_count >= max_turns:
            return {"session_complete": True}

        # Check if user seems satisfied (simple heuristic)
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1].content.lower()
            if any(phrase in last_message for phrase in [
                "thank you", "thanks", "goodbye", "bye",
                "that's all", "all set", "perfect"
            ]):
                return {"session_complete": True}

        return {"session_complete": False}

    return check_completion


def create_finalize_node():
    """
    Create a node that finalizes evaluation metrics.

    Aggregates all evaluator scores into summary statistics.

    Returns:
        Node function that computes final metrics
    """
    def finalize(state: EvaluationState) -> dict:
        """Aggregate scores into final metrics."""
        scores = state.get("evaluator_scores", [])

        if not scores:
            return {
                "final_metrics": {
                    "helpfulness_avg": 0.0,
                    "accuracy_avg": 0.0,
                    "empathy_avg": 0.0,
                    "efficiency_avg": 0.0,
                    "goal_completion_rate": 0.0,
                    "num_scores": 0,
                }
            }

        # Compute averages
        metrics = aggregate_scores(scores)

        return {"final_metrics": metrics}

    return finalize


# === Graph Builder ===

def create_evaluation_graph(
    llm: "BaseChatModel",
    agent_node: Any,
    user_config: SimulatedUser,
    evaluate_every_n_turns: int = 2,
    checkpointer: Any | None = None,
) -> "CompiledStateGraph":
    """
    Create an evaluation graph with agent, simulated user, and evaluator.

    This graph orchestrates:
    1. Simulated user sends message
    2. Agent under test responds
    3. Evaluator scores conversation periodically
    4. Repeat until max turns or completion

    Args:
        llm: Language model for simulated user and evaluator
        agent_node: The agent node being evaluated
        user_config: Configuration for simulated user behavior
        evaluate_every_n_turns: How often to run evaluator (default: 2)
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled evaluation graph

    Example:
        >>> def my_agent(state):
        ...     # Agent implementation
        ...     return {"messages": [AIMessage(content="Hello!")]}
        >>>
        >>> user_config = SimulatedUser(
        ...     persona="Frustrated customer",
        ...     goals=["Get refund"],
        ...     behavior="impatient"
        ... )
        >>>
        >>> graph = create_evaluation_graph(llm, my_agent, user_config)
    """
    workflow = StateGraph(EvaluationState)

    # Add nodes
    workflow.add_node("simulated_user", create_simulated_user_node(llm, user_config))
    workflow.add_node("agent", agent_node)
    workflow.add_node("evaluator", create_evaluator_node(llm))
    workflow.add_node("check_completion", create_check_completion_node())
    workflow.add_node("finalize", create_finalize_node())

    # Entry point: simulated user starts
    workflow.add_edge(START, "simulated_user")

    # User -> Agent
    workflow.add_edge("simulated_user", "agent")

    # Agent -> Check completion
    workflow.add_edge("agent", "check_completion")

    # Conditional: continue or finalize
    def route_after_check(state: EvaluationState) -> str:
        """Route based on session completion."""
        if state.get("session_complete", False):
            return "finalize"

        # Evaluate periodically
        turn_count = state.get("turn_count", 0)
        if turn_count % evaluate_every_n_turns == 0 and turn_count > 0:
            return "evaluator"

        return "simulated_user"

    workflow.add_conditional_edges(
        "check_completion",
        route_after_check,
        {
            "simulated_user": "simulated_user",
            "evaluator": "evaluator",
            "finalize": "finalize",
        }
    )

    # Evaluator loops back to simulated user
    workflow.add_edge("evaluator", "simulated_user")

    # Finalize ends
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)


# === Evaluation Runners ===

def run_evaluation_session(
    graph: "CompiledStateGraph",
    max_turns: int = 10,
    thread_id: str = "default",
) -> dict:
    """
    Run a complete evaluation session.

    Args:
        graph: Compiled evaluation graph
        max_turns: Maximum conversation turns (default: 10)
        thread_id: Thread ID for checkpointing (default: "default")

    Returns:
        Final state dict containing:
        - messages: Full conversation
        - evaluator_scores: All scores from evaluator
        - final_metrics: Aggregated metrics
        - turn_count: Total turns taken

    Example:
        >>> results = run_evaluation_session(graph, max_turns=5)
        >>> print(results["final_metrics"])
        {'helpfulness_avg': 4.2, 'accuracy_avg': 4.0, ...}
    """
    initial_state: EvaluationState = {
        "messages": [],
        "conversation": "",
        "evaluator_scores": [],
        "turn_count": 0,
        "max_turns": max_turns,
        "session_complete": False,
        "final_metrics": {},
    }

    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(initial_state, config=config)


def run_multiple_evaluations(
    graph: "CompiledStateGraph",
    num_sessions: int = 3,
    max_turns: int = 10,
) -> dict:
    """
    Run multiple evaluation sessions and aggregate results.

    Args:
        graph: Compiled evaluation graph
        num_sessions: Number of sessions to run (default: 3)
        max_turns: Maximum turns per session (default: 10)

    Returns:
        Dict with:
        - sessions: List of individual session results
        - aggregate_metrics: Overall metrics across all sessions

    Example:
        >>> results = run_multiple_evaluations(graph, num_sessions=5)
        >>> print(results["aggregate_metrics"])
    """
    sessions = []

    for i in range(num_sessions):
        result = run_evaluation_session(
            graph,
            max_turns=max_turns,
            thread_id=f"eval_session_{i}"
        )
        sessions.append(result)

    # Aggregate metrics across all sessions
    all_metrics = [s.get("final_metrics", {}) for s in sessions if s.get("final_metrics")]

    if not all_metrics:
        aggregate = {}
    else:
        aggregate = {
            "helpfulness_avg": sum(m.get("helpfulness_avg", 0) for m in all_metrics) / len(all_metrics),
            "accuracy_avg": sum(m.get("accuracy_avg", 0) for m in all_metrics) / len(all_metrics),
            "empathy_avg": sum(m.get("empathy_avg", 0) for m in all_metrics) / len(all_metrics),
            "efficiency_avg": sum(m.get("efficiency_avg", 0) for m in all_metrics) / len(all_metrics),
            "goal_completion_rate": sum(m.get("goal_completion_rate", 0) for m in all_metrics) / len(all_metrics),
            "num_sessions": len(all_metrics),
        }

    return {
        "sessions": sessions,
        "aggregate_metrics": aggregate,
    }


# === Metrics Aggregation ===

def aggregate_scores(scores: list[dict]) -> dict[str, float]:
    """
    Aggregate evaluator scores into summary metrics.

    Args:
        scores: List of score dicts from evaluator

    Returns:
        Dict with averaged metrics

    Example:
        >>> scores = [
        ...     {"helpfulness": 4, "accuracy": 5, "empathy": 3, "efficiency": 4, "goal_completion": 1},
        ...     {"helpfulness": 5, "accuracy": 4, "empathy": 4, "efficiency": 5, "goal_completion": 1},
        ... ]
        >>> metrics = aggregate_scores(scores)
        >>> print(metrics["helpfulness_avg"])
        4.5
    """
    if not scores:
        return {
            "helpfulness_avg": 0.0,
            "accuracy_avg": 0.0,
            "empathy_avg": 0.0,
            "efficiency_avg": 0.0,
            "goal_completion_rate": 0.0,
            "num_scores": 0,
        }

    num_scores = len(scores)

    return {
        "helpfulness_avg": sum(s.get("helpfulness", 0) for s in scores) / num_scores,
        "accuracy_avg": sum(s.get("accuracy", 0) for s in scores) / num_scores,
        "empathy_avg": sum(s.get("empathy", 0) for s in scores) / num_scores,
        "efficiency_avg": sum(s.get("efficiency", 0) for s in scores) / num_scores,
        "goal_completion_rate": sum(s.get("goal_completion", 0) for s in scores) / num_scores,
        "num_scores": num_scores,
    }


# === Module Exports ===

__all__ = [
    # State
    "EvaluationState",
    # Configuration
    "SimulatedUser",
    "EvaluationCriteria",
    # Node creators
    "create_simulated_user_node",
    "create_evaluator_node",
    "create_check_completion_node",
    "create_finalize_node",
    # Graph builder
    "create_evaluation_graph",
    # Runners
    "run_evaluation_session",
    "run_multiple_evaluations",
    # Utilities
    "aggregate_scores",
]
