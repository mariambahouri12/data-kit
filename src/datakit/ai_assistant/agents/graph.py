"""
LangGraph workflow definition.
"""

from langgraph.graph import (
    StateGraph,
    END
)

from .state import AgentState

from .nodes import (
    router_node,
    retrieval_node,
    context_node,
    prompt_node,
    generation_node,
    validation_node
)

from .formatter_node import formatting_node


def should_retry(state):
    """
    Determine if we should retry generation.
    """
    retry_count = state.get("retry_count", 0)
    success = state.get("success", False)
    
    # If success, stop
    if success:
        return "success"
    
    # If too many retries, stop
    if retry_count >= 2:
        return "fail"
    
    # Retry - go back to prompt to rebuild with improved instructions
    return "retry"


def create_graph(
    retriever,
    prompt_manager,
    llm_client,
    router=None,
    context_manager=None,
    response_formatter=None
):
    graph = StateGraph(
        AgentState
    )

    # --------------------------
    # Nodes
    # --------------------------

    graph.add_node(
        "router",
        lambda state: router_node(
            state,
            router
        )
    )

    graph.add_node(
        "retrieval",
        lambda state: retrieval_node(
            state,
            retriever
        )
    )

    graph.add_node(
        "context",
        lambda state: context_node(
            state,
            context_manager
        )
    )

    graph.add_node(
        "prompt",
        lambda state: prompt_node(
            state,
            prompt_manager
        )
    )

    graph.add_node(
        "generation",
        lambda state: generation_node(
            state,
            llm_client
        )
    )

    # Validation BEFORE formatting
    graph.add_node(
        "validation",
        validation_node
    )

    # Formatting AFTER validation
    graph.add_node(
        "format",
        lambda state: formatting_node(
            state,
            response_formatter
        )
    )

    # --------------------------
    # Flow
    # --------------------------

    graph.set_entry_point(
        "router"
    )

    graph.add_edge(
        "router",
        "retrieval"
    )

    graph.add_edge(
        "retrieval",
        "context"
    )

    graph.add_edge(
        "context",
        "prompt"
    )

    graph.add_edge(
        "prompt",
        "generation"
    )

    # Generation -> Validation
    graph.add_edge(
        "generation",
        "validation"
    )

    # Conditional edge from validation
    graph.add_conditional_edges(
        "validation",
        should_retry,
        {
            "success": "format",  # Go to formatting on success
            "retry": "prompt",    # Retry from prompt (rebuild with better instructions)
            "fail": END          # Stop on too many retries
        }
    )

    # Formatting -> END
    graph.add_edge(
        "format",
        END
    )

    return graph.compile()