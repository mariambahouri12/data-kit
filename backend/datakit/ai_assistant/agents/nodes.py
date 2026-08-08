"""
LangGraph nodes for DataKit AI Assistant.

Each function represents an agent step.
"""

from typing import Dict, Any

# =====================================================
# Router Agent
# =====================================================

def router_node(state, router=None):
    question = state.get("question", "")

    if router is None:
        return {"selected_files": []}

    try:
        files = router.route(question)
        return {"selected_files": files}
    except Exception as e:
        return {"selected_files": [], "error": str(e)}

# =====================================================
# Retrieval Agent
# =====================================================

def retrieval_node(state, retriever):
    """
    FIX : suppression du check "Knowledge base not ready" — ce texte
    n'est produit par aucune implémentation actuelle de Retriever.retrieve()
    (qui renvoie [] directement si non initialisé, jamais un faux document).
    C'était du code mort reliquat d'une ancienne version.
    """
    question = state.get("question", "")
    selected_files = state.get("selected_files", None)

    try:
        documents = retriever.retrieve(
            query=question,
            selected_files=selected_files
        )
        return {"documents": documents}
    except Exception as e:
        return {"documents": [], "error": str(e)}

# =====================================================
# Context Agent
# =====================================================

def context_node(state, context_manager=None):
    if context_manager is None:
        return {"dataset_context": ""}

    try:
        context = context_manager.to_markdown_context()
        return {"dataset_context": context}
    except Exception as e:
        return {"dataset_context": "", "error": str(e)}

# =====================================================
# Prompt Construction Agent
# =====================================================

def prompt_node(state, prompt_manager):
    retry_count = state.get("retry_count", 0)

    prompt = prompt_manager.build_prompt(
        user_question=state.get("question", ""),
        dataset_context=state.get("dataset_context", ""),
        retrieved_documents=state.get("documents", [])
    )

    if retry_count > 0:
        prompt += """

IMPORTANT - Improve your answer quality:
- Provide a more detailed and complete explanation
- Include specific examples or steps
- Structure your response clearly
- Add practical recommendations
- Be thorough and comprehensive
"""

    return {"prompt": prompt}

# =====================================================
# LLM Generation Agent
# =====================================================

def generation_node(state, llm_client):
    try:
        prompt = state.get("prompt", "")
        answer = llm_client.generate_response(prompt)
        return {"answer": answer, "success": True}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "success": False, "error": str(e)}

# =====================================================
# Validation Agent
# =====================================================

# FIX (#8) : OllamaClient catche ses propres exceptions et encode les
# erreurs directement dans le texte de la réponse ("❌ Error: ...",
# "⚠️ The LLM model is not available..."). generation_node reçoit donc
# toujours success=True dans ces cas. On détecte maintenant ces motifs
# ici pour déclencher un retry au lieu de renvoyer l'erreur brute à
# l'utilisateur comme si c'était une réponse valide.
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
    Validate the generated answer BEFORE formatting.
    Returns:
        - success: True if answer is valid
        - error: Error message if invalid
        - retry_count: Incremented if validation fails
    """
    answer = state.get("answer", "")
    success = state.get("success", False)

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

    answer_lower = answer.lower().strip()

    if any(marker in answer_lower for marker in _ERROR_MARKERS):
        retry_count = state.get("retry_count", 0) + 1
        return {
            "success": False,
            "error": "LLM backend error detected",
            "retry_count": retry_count
        }

    if any(phrase in answer_lower for phrase in _UNHELPFUL_PHRASES):
        retry_count = state.get("retry_count", 0) + 1
        return {
            "success": False,
            "error": "Unhelpful response detected",
            "retry_count": retry_count
        }

    return {"success": True}