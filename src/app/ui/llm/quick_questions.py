# app/ui/llm/quick_questions.py

"""
Quick questions shortcuts for LLM assistant.
"""

import streamlit as st


def render_quick_questions(assistant: dict) -> None:
    """Render quick questions buttons."""
    context_manager = assistant.get("context_manager")
    
    if not (context_manager and context_manager.dataset_context):
        return
    
    with st.expander("💡 Questions rapides", expanded=False):
        st.caption("Cliquez sur une question pour la poser directement :")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Quelle méthode de scaling choisir ?"):
                st.session_state.llm_quick_question = "Quelle méthode de scaling recommandez-vous pour mon dataset ?"
                st.rerun()
            
            if st.button("🔍 Comment gérer les valeurs manquantes ?"):
                st.session_state.llm_quick_question = "Quelle stratégie de traitement des valeurs manquantes me conseillez-vous ?"
                st.rerun()
            
            if st.button("📈 Quelles colonnes sont importantes ?"):
                st.session_state.llm_quick_question = "Quelles sont les colonnes les plus importantes de mon dataset ?"
                st.rerun()
        
        with col2:
            if st.button("⚖️ Y a-t-il un déséquilibre ?"):
                st.session_state.llm_quick_question = "Mon dataset est-il déséquilibré ? Que me conseillez-vous ?"
                st.rerun()
            
            if st.button("🧹 Comment nettoyer mes données ?"):
                st.session_state.llm_quick_question = "Quelles sont les meilleures pratiques pour nettoyer mon dataset ?"
                st.rerun()
            
            if st.button("📊 Quelle visualisation choisir ?"):
                st.session_state.llm_quick_question = "Quelles visualisations recommandez-vous pour mon dataset ?"
                st.rerun()