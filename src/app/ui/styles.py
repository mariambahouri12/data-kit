"""CSS personnalisé de l'application."""
import streamlit as st

_CUSTOM_CSS = """
<style>
    .main-header { font-size: 2.5rem; color: #4ECDC4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.5rem; color: #2C3E50; margin-top: 1rem; margin-bottom: 0.5rem; }
    .info-box { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border-left: 5px solid #4ECDC4; margin-bottom: 15px; }
    .warning-box { background-color: #FFF3CD; border-radius: 10px; padding: 15px; border-left: 5px solid #FFC107; margin-bottom: 15px; }
    .success-box { background-color: #D4EDDA; border-radius: 10px; padding: 15px; border-left: 5px solid #28A745; margin-bottom: 15px; }
    .metric-card { background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F8F9FA; border-radius: 5px 5px 0 0; gap: 1px; padding: 10px 20px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #4ECDC4; color: white; }
</style>
"""


def inject_css() -> None:
    """Injecte le CSS personnalisé dans la page Streamlit courante."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
