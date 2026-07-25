# state.py
"""Initialisation du session state Streamlit utilisé par l'application."""
import streamlit as st

from datakit.data.loader import FileLoader


def initialize_session_state() -> None:
    """Crée les clés du session state si elles n'existent pas déjà."""
    # File loader
    if "file_loader" not in st.session_state:
        st.session_state.file_loader = FileLoader()
    
    # Données
    for key in ("current_data", "processed_data"):
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Modèles
    for key in (
        "trained_model",
        "model_metrics",
        "model_cv_results",
        "model_name",
        "model_task"
    ):
        if key not in st.session_state:
            st.session_state[key] = None