"""
LangGraph workflow definition.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState

from .nodes import (
    router_node,
    retrieval_node,
    context_node,
    prompt_node,
    generation_node,
    validation_node,
)

from .formatter_node import formatting_node


def should_retry(state):
    """
    Determine whether the workflow should retry generation.

    Returns:
        "success" -> formatting
        "retry"   -> rebuild prompt and regenerate
        "fail"    -> stop workflow
    """

    retry_count = state.get("retry_count", 0)
    success = state.get("success", False)

    # Successful generation
    if success:
        return "success"

    # Maximum number of retries reached
    if retry_count >= 2:
        return "fail"

    # Retry from prompt construction
    return "retry"


def create_graph(
    retriever,
    prompt_manager,
    llm_client,
    router=None,
    context_manager=None,
    response_formatter=None,
):
    """
    Create and compile the LangGraph workflow.
    """

    graph = StateGraph(AgentState)

    # =========================================================
    # Nodes
    # =========================================================

    graph.add_node(
        "router",
        lambda state: router_node(
            state,
            router,
        ),
    )

    graph.add_node(
        "retrieval",
        lambda state: retrieval_node(
            state,
            retriever,
        ),
    )

    graph.add_node(
        "context",
        lambda state: context_node(
            state,
            context_manager,
        ),
    )

    graph.add_node(
        "prompt",
        lambda state: prompt_node(
            state,
            prompt_manager,
        ),
    )

    graph.add_node(
        "generation",
        lambda state: generation_node(
            state,
            llm_client,
        ),
    )

    graph.add_node(
        "validation",
        validation_node,
    )

    graph.add_node(
        "format",
        lambda state: formatting_node(
            state,
            response_formatter,
        ),
    )

    # =========================================================
    # Main flow
    # =========================================================

    graph.set_entry_point("router")

    graph.add_edge(
        "router",
        "retrieval",
    )

    graph.add_edge(
        "retrieval",
        "context",
    )

    graph.add_edge(
        "context",
        "prompt",
    )

    graph.add_edge(
        "prompt",
        "generation",
    )

    graph.add_edge(
        "generation",
        "validation",
    )

    # =========================================================
    # Validation routing
    # =========================================================

    graph.add_conditional_edges(
        "validation",
        should_retry,
        {
            "success": "format",
            "retry": "prompt",
            "fail": END,
        },
    )

    # =========================================================
    # Final formatting
    # =========================================================

    graph.add_edge(
        "format",
        END,
    )

    return graph.compile()