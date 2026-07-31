# app/ui/llm/context_panel.py

"""
Dataset context display for LLM assistant.
"""

import streamlit as st


def render_context_panel(assistant: dict) -> None:
    """Render dataset context panel."""
    context_manager = assistant.get("context_manager")
    
    if context_manager and context_manager.dataset_context:
        with st.expander("📊 Contexte du dataset", expanded=False):
            st.json(context_manager.dataset_context)
            
            if st.button("🔄 Mettre à jour le contexte"):
                if st.session_state.current_data is not None:
                    context_manager.update_dataset(
                        st.session_state.current_data,
                        "Dataset actuel"
                    )
                    st.success("✅ Contexte mis à jour !")
                    st.rerun()
                else:
                    st.warning("⚠️ Aucune donnée chargée.")
    else:
        with st.expander("📊 Contexte du dataset", expanded=False):
            st.info("Aucun dataset chargé. Chargez des données pour obtenir des recommandations contextuelles.")
            
            if st.session_state.current_data is not None and context_manager:
                if st.button("📤 Charger le dataset actuel"):
                    context_manager.update_dataset(
                        st.session_state.current_data,
                        "Dataset actuel"
                    )
                    st.success("✅ Contexte chargé !")
                    st.rerun()