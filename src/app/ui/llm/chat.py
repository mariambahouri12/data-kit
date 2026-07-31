# app/ui/llm/chat.py

"""
Chat display and interaction for LLM assistant.
"""

import streamlit as st


def render_chat_history() -> None:
    """Display chat history."""
    for message in st.session_state.llm_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_input() -> str:
    """Render chat input and return prompt if submitted."""
    return st.chat_input("Posez une question sur vos données...")


def render_assistant_response(response: dict) -> None:
    """Render assistant response with sources."""
    if response.get("success", False):
        st.markdown(response["answer"])
        
        # Show sources
        if response.get("context"):
            with st.expander("📚 Sources utilisées", expanded=False):
                for doc in response["context"]:
                    if isinstance(doc, dict):
                        content = doc.get("content", "")
                    else:
                        content = str(doc)
                    if len(content) > 200:
                        st.caption(f"- {content[:200]}...")
                    else:
                        st.caption(f"- {content}")
    else:
        st.error(f"❌ {response.get('answer', 'Erreur inconnue')}")