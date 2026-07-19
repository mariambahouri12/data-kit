"""Initialisation du session state Streamlit utilisé par l'application."""
import streamlit as st

from datakit.data.loader import FileLoader 


def initialize_session_state() -> None:
    """Crée les clés du session state si elles n'existent pas déjà."""
    if "file_loader" not in st.session_state:
        st.session_state.file_loader = FileLoader()
    for key in ("current_data", "processed_data"):
        if key not in st.session_state:
            st.session_state[key] = None