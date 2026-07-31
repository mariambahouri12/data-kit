# app/ui/llm/assistant_loader.py

"""
Assistant initialization and loading.
"""

import streamlit as st

from datakit.ai_assistant import create_assistant, logger


def load_assistant() -> dict:
    """Load or create AI assistant."""
    if "ai_assistant" not in st.session_state:
        with st.spinner("🔄 Initialisation de l'assistant..."):
            try:
                st.session_state.ai_assistant = create_assistant()
                logger.info("Assistant initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize assistant: {e}")
                st.error(f"❌ Erreur lors de l'initialisation: {e}")
                st.session_state.ai_assistant = None
    
    return st.session_state.ai_assistant


def get_assistant_or_show_error() -> dict:
    """Get assistant or show error message."""
    assistant = load_assistant()
    
    if assistant is None:
        st.warning("⚠️ L'assistant n'a pas pu être initialisé.")
        st.info("""
        **Causes possibles :**
        1. Ollama n'est pas installé
        2. Ollama n'est pas en cours d'exécution
        3. Le modèle 'mistral' n'est pas téléchargé
        
        **Solutions :**
        1. Installez Ollama depuis [ollama.ai](https://ollama.ai)
        2. Lancez `ollama serve` dans le terminal
        3. Téléchargez Mistral : `ollama pull mistral`
        """)
        return None
    
    return assistant