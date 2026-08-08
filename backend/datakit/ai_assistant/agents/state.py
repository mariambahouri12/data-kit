"""
Shared state definition for LangGraph workflow.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.
    """

    # User question
    question: str

    # DataKit dataset/project context
    dataset_context: Optional[str]

    # Files selected by DocumentRouter
    selected_files: List[str]

    # Documents retrieved from FAISS
    documents: List[Dict[str, Any]]

    # Final LLM prompt
    prompt: str

    # Raw LLM answer
    answer: str

    # Formatted answer
    formatted_answer: str

    # Parsed structured response
    structured: Optional[Dict[str, Any]]

    # UI recommendation structure
    recommendation: Optional[Dict[str, Any]]

    # Workflow status
    success: bool

    # Formatting status
    format_success: bool

    # Error information
    error: str

    # Number of generation retries
    retry_count: int