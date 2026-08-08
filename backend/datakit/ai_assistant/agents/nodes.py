"""
LangGraph nodes for DataKit AI Assistant.

Each function represents one agent step.
"""

from typing import Any


# =============================================================
# Router Agent
# =============================================================

def router_node(state, router=None):
    """
    Select relevant knowledge-base files.
    """

    question = state.get("question", "")

    if router is None:
        return {
            "selected_files": [],
        }

    try:
        files = router.route(question)

        if not isinstance(files, list):
            return {
                "selected_files": [],
                "error": "Document router returned an invalid format.",
            }

        return {
            "selected_files": files,
        }

    except Exception as e:
        return {
            "selected_files": [],
            "error": str(e),
        }


# =============================================================
# Retrieval Agent
# =============================================================

def retrieval_node(state, retriever):
    """
    Retrieve relevant documents using the Retriever.
    """

    question = state.get("question", "")
    selected_files = state.get("selected_files", [])

    try:
        documents = retriever.retrieve(
            query=question,
            selected_files=selected_files,
        )

        return {
            "documents": documents,
        }

    except Exception as e:
        return {
            "documents": [],
            "error": str(e),
        }


# =============================================================
# Context Agent
# =============================================================

def context_node(state, context_manager=None):
    """
    Build the DataKit project/dataset context.
    """

    if context_manager is None:
        return {
            "dataset_context": "",
        }

    try:
        context = context_manager.to_markdown_context()

        return {
            "dataset_context": context,
        }

    except Exception as e:
        return {
            "dataset_context": "",
            "error": str(e),
        }


# =============================================================
# Prompt Construction Agent
# =============================================================

def prompt_node(state, prompt_manager):
    """
    Build the final prompt sent to the LLM.
    """

    retry_count = state.get("retry_count", 0)

    prompt = prompt_manager.build_prompt(
        user_question=state.get("question", ""),
        dataset_context=state.get("dataset_context", ""),
        retrieved_documents=state.get("documents", []),
    )

    # Add stronger instructions after a failed generation.
    if retry_count > 0:
        prompt += """

IMPORTANT - Improve your answer quality:

- Provide a more detailed and complete explanation.
- Include specific examples or steps when relevant.
- Structure your response clearly.
- Provide practical recommendations.
- Be precise and avoid unsupported claims.
"""

    return {
        "prompt": prompt,
    }


# =============================================================
# LLM Generation Agent
# =============================================================

def generation_node(state, llm_client):
    """
    Generate an answer using the configured LLM client.
    """

    try:
        prompt = state.get("prompt", "")

        answer = llm_client.generate_response(prompt)

        return {
            "answer": answer,
            "success": True,
        }

    except Exception as e:
        return {
            "answer": "",
            "success": False,
            "error": str(e),
        }


# =============================================================
# Validation Agent
# =============================================================

_ERROR_MARKERS = (
    "❌ error",
    "⚠️ the llm model is not available",
    "⏱️ the model took too long to respond",
)

_UNHELPFUL_PHRASES = (
    "i don't know",
    "i cannot answer",
    "i am unable to answer",
    "no answer available",
    "i don't have enough information",
)


def validation_node(state):
    """
    Validate the generated answer before formatting.

    Validation checks:
    - generation success
    - minimum answer length
    - known backend error messages
    - unhelpful responses

    If validation fails, retry_count is incremented.
    """

    answer = state.get("answer", "")
    success = state.get("success", False)

    # ---------------------------------------------------------
    # Generation failed
    # ---------------------------------------------------------

    if not success:
        retry_count = state.get("retry_count", 0) + 1

        return {
            "success": False,
            "error": state.get(
                "error",
                "Generation failed",
            ),
            "retry_count": retry_count,
        }

    # ---------------------------------------------------------
    # Empty / insufficient response
    # ---------------------------------------------------------

    if not answer or len(answer.strip()) < 5:
        retry_count = state.get("retry_count", 0) + 1

        return {
            "success": False,
            "answer": "",
            "error": "Empty or insufficient response",
            "retry_count": retry_count,
        }

    answer_lower = answer.lower().strip()

    # ---------------------------------------------------------
    # Backend error markers
    # ---------------------------------------------------------

    if any(
        marker in answer_lower
        for marker in _ERROR_MARKERS
    ):
        retry_count = state.get("retry_count", 0) + 1

        return {
            "success": False,
            "error": "LLM backend error detected",
            "retry_count": retry_count,
        }

    # ---------------------------------------------------------
    # Unhelpful response
    # ---------------------------------------------------------

    if any(
        phrase in answer_lower
        for phrase in _UNHELPFUL_PHRASES
    ):
        retry_count = state.get("retry_count", 0) + 1

        return {
            "success": False,
            "error": "Unhelpful response detected",
            "retry_count": retry_count,
        }

    # ---------------------------------------------------------
    # Valid response
    # ---------------------------------------------------------

    return {
        "success": True,
    }