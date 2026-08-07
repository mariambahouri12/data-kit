"""
LangGraph nodes for DataKit AI Assistant.

Each function represents an agent step.
"""

from typing import Dict, Any

# =====================================================
# Router Agent
# =====================================================

def router_node(
    state,
    router=None
):
    question = state.get("question", "")

    if router is None:
        return {
            "selected_files": []
        }

    try:
        files = router.route(question)

        return {
            "selected_files": files
        }

    except Exception as e:
        return {
            "selected_files": [],
            "error": str(e)
        }

# =====================================================
# Retrieval Agent
# =====================================================

def retrieval_node(
    state,
    retriever
):
    question = state.get("question", "")

    selected_files = (
        state.get(
            "selected_files",
            None
        )
    )

    try:
        documents = retriever.retrieve(
            query=question,
            selected_files=selected_files
        )

        # If documents is a list with a fake message, don't use it
        if documents and len(documents) == 1:
            content = documents[0].get("content", "")
            if "Knowledge base not ready" in content:
                return {
                    "documents": []
                }

        return {
            "documents": documents
        }

    except Exception as e:
        return {
            "documents": [],
            "error": str(e)
        }

# =====================================================
# Context Agent
# =====================================================

def context_node(
    state,
    context_manager=None
):
    if context_manager is None:
        return {
            "dataset_context": ""
        }

    try:
        # Use to_markdown_context which handles empty dataset internally
        context = context_manager.to_markdown_context()
        return {
            "dataset_context": context
        }

    except Exception as e:
        return {
            "dataset_context": "",
            "error": str(e)
        }

# =====================================================
# Prompt Construction Agent
# =====================================================

def prompt_node(
    state,
    prompt_manager
):
    # Get retry count to adjust prompt quality
    retry_count = state.get("retry_count", 0)
    
    # Base prompt
    prompt = (
        prompt_manager
        .build_prompt(
            user_question=state.get("question", ""),
            dataset_context=state.get("dataset_context", ""),
            retrieved_documents=state.get("documents", [])
        )
    )
    
    # Add quality improvement instructions on retry
    if retry_count > 0:
        prompt += """
        
IMPORTANT - Improve your answer quality:
- Provide a more detailed and complete explanation
- Include specific examples or steps
- Structure your response clearly
- Add practical recommendations
- Be thorough and comprehensive
"""
    
    return {
        "prompt": prompt
    }

# =====================================================
# LLM Generation Agent
# =====================================================

def generation_node(
    state,
    llm_client
):
    try:
        prompt = state.get("prompt", "")
        
        answer = (
            llm_client
            .generate_response(prompt)
        )

        return {
            "answer": answer,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "success": False,
            "error": str(e)
        }

# =====================================================
# Validation Agent (avec bad_answers moins agressif)
# =====================================================

def validation_node(
    state
):
    """
    Validate the generated answer BEFORE formatting.
    Returns:
        - success: True if answer is valid
        - error: Error message if invalid
        - retry_count: Incremented if validation fails
    """
    answer = state.get("answer", "")
    success = state.get("success", False)
    
    # Only reject answers that are explicitly unhelpful
    bad_answers = [
        "i don't know",
        "i cannot answer",
        "i am unable to answer",
        "no answer available",
        "i don't have enough information"
    ]

    if not success:
        retry_count = state.get("retry_count", 0) + 1
        return {
            "success": False,
            "error": state.get("error", "Generation failed"),
            "retry_count": retry_count
        }

    if not answer or len(answer.strip()) < 5:
        retry_count = state.get("retry_count", 0) + 1
        return {
            "success": False,
            "answer": "",
            "error": "Empty or insufficient response",
            "retry_count": retry_count
        }

    # Check for explicit unhelpful responses only
    answer_lower = answer.lower().strip()
    if any(bad in answer_lower for bad in bad_answers):
        retry_count = state.get("retry_count", 0) + 1
        return {
            "success": False,
            "error": "Unhelpful response detected",
            "retry_count": retry_count
        }

    # Validation passes
    return {
        "success": True
    }