"""
Shared state definition for LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.
    """

    # User question
    question: str

    # Dataset information
    dataset_context: Optional[str]

    # Router output
    selected_files: List[str]

    # Retrieved documents
    documents: List[Dict[str, Any]]

    # Final prompt
    prompt: str

    # Generated answer
    answer: str

    # Formatted answer (after formatting node)
    formatted_answer: str

    # Structured data (if JSON)
    structured: Optional[Dict[str, Any]]
    
    # Recommendation format
    recommendation: Optional[Dict[str, Any]]

    # Execution status
    success: bool

    # Formatting success
    format_success: bool

    # Errors
    error: str

    # Retry counter (incrémenté dans validation_node)
    retry_count: int