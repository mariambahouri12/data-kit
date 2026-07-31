# app/ui/llm_assistant_page.py

"""
LLM Assistant page for DataKit.

Provides chat interface with RAG-powered AI assistance.
"""

import streamlit as st

from .llm.assistant_loader import get_assistant_or_show_error
from .llm.chat import render_chat_history, render_chat_input, render_assistant_response
from .llm.context_panel import render_context_panel
from .llm.quick_questions import render_quick_questions
from .llm.diagnostic import render_diagnostic
from .llm.history import render_history_controls


def render_llm_assistant_page() -> None:
    """Render the LLM assistant chat interface."""
    
    st.markdown('<p class="sub-header">🤖 Assistant IA</p>', unsafe_allow_html=True)
    
    # Load assistant
    assistant = get_assistant_or_show_error()
    if assistant is None:
        return
    
    # Diagnostic
    if not render_diagnostic(assistant):
        return
    
    # Initialize chat history
    if "llm_chat_history" not in st.session_state:
        st.session_state.llm_chat_history = []
    
    # Context panel
    render_context_panel(assistant)
    
    # Quick questions
    render_quick_questions(assistant)
    
    # Chat history
    render_chat_history()
    
    # Chat input
    prompt = render_chat_input()
    
    # Check for quick question
    if "llm_quick_question" in st.session_state and st.session_state.llm_quick_question:
        prompt = st.session_state.llm_quick_question
        st.session_state.llm_quick_question = None
    
    # Process prompt
    if prompt:
        _process_prompt(prompt, assistant)
    
    # History controls
    render_history_controls()


def _process_prompt(prompt: str, assistant: dict) -> None:
    """Process user prompt and generate response."""
    # Add user message
    st.session_state.llm_chat_history.append(
        {"role": "user", "content": prompt}
    )
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Réflexion en cours..."):
            try:
                # Prepare context
                context_manager = assistant.get("context_manager")
                context = context_manager.to_markdown_context() if context_manager else None
                
                # Get response
                pipeline = assistant.get("pipeline")
                response = pipeline.ask(prompt, dataset_context=context)
                
                # Render response
                render_assistant_response(response)
                
                # Add to history
                st.session_state.llm_chat_history.append(
                    {"role": "assistant", "content": response.get("answer", "Erreur")}
                )
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Erreur lors de la génération: {error_msg}")
                st.session_state.llm_chat_history.append(
                    {"role": "assistant", "content": f"❌ Erreur: {error_msg}"}
                )
    
    # Rerun to update chat
    st.rerun()