"""
Formatting node for LangGraph workflow.

Formats LLM responses for UI consumption.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def formatting_node(state, formatter=None) -> dict:
    """
    Format the LLM response using ResponseFormatter.

    Args:
        state: Current agent state.
        formatter: ResponseFormatter instance.

    Returns:
        Formatted state updates.
    """

    answer = state.get("answer", "")
    success = state.get("success", False)

    if not answer or not success:
        return {
            "formatted_answer": "No valid response generated.",
            "format_success": False,
        }

    try:
        # ---------------------------------------------------------
        # Use ResponseFormatter when available
        # ---------------------------------------------------------

        if formatter is not None:

            formatted = formatter.format_text(answer)

            structured = formatter.try_json_format(answer)

            if structured is not None:
                recommendation = formatter.format_recommendation(answer)

                return {
                    "formatted_answer": formatted,
                    "format_success": True,
                    "structured": structured,
                    "recommendation": recommendation,
                }

            return {
                "formatted_answer": formatted,
                "format_success": True,
                "structured": None,
                "recommendation": None,
            }

        # ---------------------------------------------------------
        # Fallback formatting
        # ---------------------------------------------------------

        formatted = answer.strip()
        formatted = " ".join(formatted.split())

        try:
            structured = json.loads(formatted)

            if not isinstance(structured, dict):
                structured = None

        except (json.JSONDecodeError, TypeError):
            structured = None

        return {
            "formatted_answer": formatted,
            "format_success": True,
            "structured": structured,
            "recommendation": None,
        }

    except Exception as e:
        logger.exception("Formatting failed")

        return {
            "formatted_answer": answer,
            "format_success": False,
            "structured": None,
            "recommendation": None,
            "error": str(e),
        }