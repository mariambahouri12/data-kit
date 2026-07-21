"""Page de visualisation des données."""
import numpy as np
import pandas as pd
import streamlit as st


from datakit.profiling.visualizers import DataVisualizer
from datakit.preprocessing.utils.target_detection import detect_target_column



def render_visualization_page() -> None:
    st.markdown('<p class="sub-header">📈 Visualisation</p>', unsafe_allow_html=True)

    df = (
        st.session_state.processed_data
        if st.session_state.processed_data is not None
        else st.session_state.current_data
    )
    if df is None:
        st.info("ℹ️ Chargez des données pour visualiser")
        return

    visualizer = DataVisualizer()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🔍 Corrélation", "📦 Outliers", "📉 PCA"])
    with tab1:
        _render_distribution_tab(df, visualizer)
    with tab2:
        st.pyplot(visualizer.plot_correlation_matrix(df))
    with tab3:
        _render_outliers_tab(df, visualizer)
    with tab4:
        _render_pca_tab(df, visualizer)


def _render_distribution_tab(df: pd.DataFrame, visualizer: DataVisualizer) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("Aucune colonne numérique")
        return
    selected_cols = st.multiselect(
        "Sélectionner les colonnes", options=numeric_cols,
        default=list(numeric_cols[: min(6, len(numeric_cols))]), key="dist_cols",
    )
    if selected_cols:
        st.pyplot(visualizer.plot_distribution(df[selected_cols], n_cols=3))
    else:
        st.warning("Sélectionnez des colonnes")


def _render_outliers_tab(df: pd.DataFrame, visualizer: DataVisualizer) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("Aucune colonne numérique")
        return
    selected_cols = st.multiselect(
        "Sélectionner les colonnes pour les outliers", options=numeric_cols,
        default=list(numeric_cols[: min(3, len(numeric_cols))]), key="outlier_cols",
    )
    if selected_cols:
        st.pyplot(visualizer.plot_outliers(df[selected_cols]))
    else:
        st.warning("Sélectionnez des colonnes")


def _render_pca_tab(df: pd.DataFrame, visualizer: DataVisualizer) -> None:
    if len(df.select_dtypes(include=[np.number]).columns) < 2:
        st.warning("Besoin d'au moins 2 colonnes numériques pour la PCA")
        return
    try:
        target_col = detect_target_column(df)
        y = df[target_col] if target_col else None
        st.pyplot(visualizer.plot_pca(df, y))
    except Exception as e:
        st.error(f"Erreur PCA: {e}")
