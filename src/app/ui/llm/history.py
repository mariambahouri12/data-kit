# app/ui/llm/history.py

"""
History controls for LLM assistant.
"""

import streamlit as st


def render_history_controls() -> None:
    """Render history controls buttons."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.llm_chat_history = []
            st.rerun()
    
    with col2:
        # Mettre à jour le contexte
        if "context_manager" in st.session_state and st.session_state.current_data is not None:
            if st.button("🔄 Actualiser le contexte", use_container_width=True):
                context_manager = st.session_state.context_manager
                context_manager.update_dataset(
                    st.session_state.current_data,
                    "Dataset actuel"
                )
                st.success("✅ Contexte actualisé !")
                st.rerun()
    
    with col3:
        message_count = len(st.session_state.llm_chat_history)
        if message_count == 0:
            st.caption("💬 Aucun message")
        elif message_count == 1:
            st.caption("💬 1 message")
        else:
            st.caption(f"💬 {message_count} messages")