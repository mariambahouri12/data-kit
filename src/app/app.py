# app.py
"""
Point d'entrée de l'application Streamlit.

Ce fichier ne fait plus qu'orchestrer : la configuration de page, le CSS,
l'état de session et chaque section fonctionnelle sont délégués à des
modules dédiés (ui/, services/, state.py).
"""
import os
import sys

import streamlit as st

# Ce fichier vit dans src/app/. Deux dossiers doivent être sur sys.path :
# - src/app/  (ce dossier)  -> pour les imports plats "state", "services.xxx", "ui.xxx"
# - src/      (le parent)   -> pour que "datakit" (voisin de app/) soit importable
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in (_CURRENT_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from state import initialize_session_state
from ui.styles import inject_css
from ui.sidebar import render_sidebar
from ui.upload_page import render_upload_page
from ui.preview_page import render_preview_page
from ui.preprocessing_page import render_preprocessing_page
from ui.processed_page import render_processed_page
from ui.visualization_page import render_visualization_page
from ui.models_page import render_models_page
from ui.llm_assistant_page import render_llm_assistant_page  # ← NOUVEAU


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Experimentation Platform - Data Preprocessing",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.markdown('<p class="main-header">📊 AI Experimentation Platform</p>', unsafe_allow_html=True)
    st.markdown("### Data Upload & Preprocessing Module")
    st.divider()


def render_main_tabs() -> None:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(  # ← AJOUT tab7
        ["📤 Upload", "🔍 Preview", "⚙️ Preprocess", "📊 Processed", "📈 Visualize", "🤖 Models", "💬 Assistant"]
    )
    with tab1:
        render_upload_page()
    with tab2:
        render_preview_page()
    with tab3:
        render_preprocessing_page()
    with tab4:
        render_processed_page()
    with tab5:
        render_visualization_page()
    with tab6:
        render_models_page()
    with tab7:  # ← NOUVEAU
        render_llm_assistant_page()


def main() -> None:
    configure_page()
    inject_css()
    initialize_session_state()
    render_sidebar()
    render_header()
    render_main_tabs()


if __name__ == "__main__":
    main()