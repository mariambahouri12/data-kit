# app/ui/llm/diagnostic.py

"""
Diagnostic display for LLM assistant.
"""

import streamlit as st


def render_diagnostic(assistant: dict) -> bool:
    """Render diagnostic panel and return if LLM is available."""
    llm = assistant.get("llm_client")
    
    try:
        connection_status = llm.check_connection()
        is_available = connection_status.get("status", False)
        status_message = connection_status.get("message", "")
    except Exception as e:
        is_available = False
        status_message = f"Erreur: {e}"
    
    if not is_available:
        st.warning(f"⚠️ Ollama n'est pas disponible: {status_message}")
        
        with st.expander("🔧 Diagnostic", expanded=True):
            st.markdown("**Informations :**")
            st.code(f"""
Ollama URL: http://localhost:11434
Modèle attendu: mistral
Statut: {status_message}

Commandes utiles :
1. Vérifier qu'Ollama tourne : ollama list
2. Télécharger Mistral : ollama pull mistral
3. Vérifier les modèles : ollama ps
4. Démarrer Ollama : ollama serve
            """)
        
        if st.button("🔄 Réessayer la connexion"):
            if "ai_assistant" in st.session_state:
                del st.session_state.ai_assistant
            st.rerun()
        return False
    
    return True