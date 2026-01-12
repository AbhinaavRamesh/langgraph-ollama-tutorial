"""
Advanced Patterns Module.

This module provides implementations for advanced LangGraph patterns
including subgraph composition, state transformation, agent swarms,
map-reduce parallel execution, agent handoffs, evaluation, reflection,
reflexion, ReWOO, LATS, and reusable graph components.

Example:
    >>> from langgraph_ollama_local.patterns import create_subgraph_node
    >>> from langgraph.graph import StateGraph
    >>>
    >>> # Wrap a subgraph as a node in parent graph
    >>> node = create_subgraph_node(subgraph, state_in, state_out)
    >>> parent_graph.add_node("my_subgraph", node)
    >>>
    >>> # Create map-reduce graph for parallel execution
    >>> from langgraph_ollama_local.patterns import create_map_reduce_graph
    >>> graph = create_map_reduce_graph(llm, num_workers=3)
    >>>
    >>> # Create agent handoff system
    >>> from langgraph_ollama_local.patterns import create_handoff_graph
    >>> graph = create_handoff_graph(llm, agents={...})
    >>>
    >>> # Create evaluation graph for agent testing
    >>> from langgraph_ollama_local.patterns import create_evaluation_graph, SimulatedUser
    >>> user = SimulatedUser(persona="Customer", goals=["Get help"], behavior="friendly")
    >>> eval_graph = create_evaluation_graph(llm, agent, user)
    >>>
    >>> # Create reflection graph for iterative improvement
    >>> from langgraph_ollama_local.patterns import create_reflection_graph
    >>> graph = create_reflection_graph(llm, max_iterations=3)
    >>>
    >>> # Create ReWOO graph for token-efficient reasoning
    >>> from langgraph_ollama_local.patterns import create_rewoo_graph, run_rewoo_task
    >>> graph = create_rewoo_graph(llm, tools={"Google": search_tool})
    >>> result = run_rewoo_task(graph, "Who won the 2024 NBA championship?")
    >>>
    >>> # Create LATS graph for tree search reasoning
    >>> from langgraph_ollama_local.patterns import create_lats_graph, run_lats_task
    >>> graph = create_lats_graph(llm, tools=tools, max_depth=3, max_width=2)
    >>> result = run_lats_task(graph, "Complex reasoning task")
"""

from langgraph_ollama_local.patterns.evaluation import (
    EvaluationCriteria,
    EvaluationState,
    SimulatedUser,
    aggregate_scores,
    create_check_completion_node,
    create_evaluation_graph,
    create_evaluator_node,
    create_finalize_node,
    create_simulated_user_node,
    run_evaluation_session,
    run_multiple_evaluations,
)
from langgraph_ollama_local.patterns.handoffs import (
    create_handoff_agent_node,
    create_handoff_graph,
    create_handoff_tool,
    run_handoff_conversation,
)
from langgraph_ollama_local.patterns.map_reduce import (
    MapReduceState,
    create_custom_map_reduce_graph,
    create_map_reduce_graph,
    create_mapper_node,
    create_reducer_node,
    create_worker_node,
    run_map_reduce_task,
)
from langgraph_ollama_local.patterns.plan_execute import (
    Act,
    Plan,
    PlanExecuteState,
    Response,
    create_executor_node,
    create_plan_execute_graph,
    create_planner_node as create_plan_execute_planner_node,
    create_replanner_node,
    route_after_executor,
    route_after_replanner,
    run_plan_execute_task,
)
from langgraph_ollama_local.patterns.reflection import (
    MultiCriteriaFeedback,
    ReflectionState,
    create_critic_node,
    create_generator_node,
    create_multi_criteria_critic_node,
    create_multi_model_reflection_graph,
    create_reflection_graph,
    run_reflection_task,
    should_continue,
)
from langgraph_ollama_local.patterns.reflexion import (
    AnswerQuestion,
    Reflection,
    ReflexionState,
    ReviseAnswer,
    create_initial_responder,
    create_reflexion_graph,
    create_revisor,
    create_tool_executor as create_reflexion_tool_executor,
    run_reflexion_task,
)
from langgraph_ollama_local.patterns.rewoo import (
    PLAN_REGEX,
    ReWOOState,
    create_planner_node,
    create_rewoo_graph,
    create_solver_node,
    create_tool_executor,
    format_tool_descriptions,
    parse_plan,
    route_rewoo,
    run_rewoo_task,
)
from langgraph_ollama_local.patterns.lats import (
    Node,
    Reflection as LATSReflection,
    TreeState,
    create_expansion_node,
    create_lats_graph,
    get_best_solution,
    run_lats_task,
    select,
    should_loop,
)
from langgraph_ollama_local.patterns.subgraphs import (
    create_subgraph_node,
    create_subgraph_node_async,
)
from langgraph_ollama_local.patterns.swarm import (
    SwarmAgent,
    SwarmState,
    create_swarm_graph,
    run_swarm_task,
)

__all__ = [
    # Subgraph patterns
    "create_subgraph_node",
    "create_subgraph_node_async",
    # Swarm patterns
    "SwarmAgent",
    "SwarmState",
    "create_swarm_graph",
    "run_swarm_task",
    # Map-reduce patterns
    "MapReduceState",
    "create_mapper_node",
    "create_worker_node",
    "create_reducer_node",
    "create_map_reduce_graph",
    "create_custom_map_reduce_graph",
    "run_map_reduce_task",
    # Handoff patterns
    "create_handoff_tool",
    "create_handoff_agent_node",
    "create_handoff_graph",
    "run_handoff_conversation",
    # Evaluation patterns
    "EvaluationState",
    "SimulatedUser",
    "EvaluationCriteria",
    "create_simulated_user_node",
    "create_evaluator_node",
    "create_check_completion_node",
    "create_finalize_node",
    "create_evaluation_graph",
    "run_evaluation_session",
    "run_multiple_evaluations",
    "aggregate_scores",
    # Reflection patterns
    "ReflectionState",
    "MultiCriteriaFeedback",
    "create_generator_node",
    "create_critic_node",
    "create_multi_criteria_critic_node",
    "should_continue",
    "create_reflection_graph",
    "create_multi_model_reflection_graph",
    "run_reflection_task",
    # Reflexion patterns
    "ReflexionState",
    "Reflection",
    "AnswerQuestion",
    "ReviseAnswer",
    "create_initial_responder",
    "create_reflexion_tool_executor",
    "create_revisor",
    "create_reflexion_graph",
    "run_reflexion_task",
    # ReWOO patterns
    "ReWOOState",
    "PLAN_REGEX",
    "parse_plan",
    "create_planner_node",
    "create_tool_executor",
    "create_solver_node",
    "route_rewoo",
    "create_rewoo_graph",
    "run_rewoo_task",
    "format_tool_descriptions",
    # LATS patterns
    "Node",
    "LATSReflection",
    "TreeState",
    "select",
    "create_expansion_node",
    "should_loop",
    "get_best_solution",
    "create_lats_graph",
    "run_lats_task",
]
