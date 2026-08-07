"""
Formatting node for LangGraph workflow.
Formats LLM responses for UI consumption.
"""

from .state import AgentState


def formatting_node(
    state,
    formatter=None
) -> dict:
    """
    Format the LLM response using ResponseFormatter.
    
    Args:
        state: Current agent state
        formatter: ResponseFormatter instance
    
    Returns:
        Formatted state updates
    """
    
    answer = state.get("answer", "")
    success = state.get("success", False)
    
    if not answer or not success:
        return {
            "formatted_answer": "No valid response generated.",
            "format_success": False
        }
    
    try:
        # Use ResponseFormatter if available
        if formatter is not None:
            formatted = formatter.format_text(answer)
            
            # Try to extract structured data
            structured = formatter.try_json_format(answer)
            
            # Check if it's a recommendation
            if isinstance(structured, dict) and "answer" in structured:
                recommendation = formatter.format_recommendation(answer)
                return {
                    "formatted_answer": formatted,
                    "format_success": True,
                    "structured": structured,
                    "recommendation": recommendation
                }
            
            return {
                "formatted_answer": formatted,
                "format_success": True,
                "structured": structured if isinstance(structured, dict) else None
            }
        
        # Fallback formatting
        formatted = answer.strip()
        formatted = " ".join(formatted.split())
        
        import json
        try:
            structured = json.loads(formatted)
            return {
                "formatted_answer": formatted,
                "format_success": True,
                "structured": structured
            }
        except:
            return {
                "formatted_answer": formatted,
                "format_success": True,
                "structured": None
            }
            
    except Exception as e:
        return {
            "formatted_answer": answer,
            "format_success": False,
            "error": str(e)
        }