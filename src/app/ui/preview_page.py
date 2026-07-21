"""Page de prévisualisation des données."""
import numpy as np
import pandas as pd
import streamlit as st

from datakit.validation.data_validator import DataValidator
from .dataframe_display import safe_display_dataframe
from datakit.preprocessing.utils.target_detection import detect_target_column


def render_preview_page() -> None:
    st.markdown('<p class="sub-header">🔍 Data Preview</p>', unsafe_allow_html=True)

    if st.session_state.current_data is None:
        st.info("ℹ️ Chargez d'abord des données pour afficher un aperçu")
        return

    df = st.session_state.current_data
    _render_metrics(df)

    tabs = st.tabs(["📊 Aperçu", "📈 Statistiques", "📋 Types", "🔍 Détection"])
    with tabs[0]:
        _render_overview_tab(df)
    with tabs[1]:
        _render_stats_tab(df)
    with tabs[2]:
        _render_types_tab(df)
    with tabs[3]:
        _render_detection_tab(df)


def _render_metrics(df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", f"{len(df.columns)}")
    with col3:
        missing_pct = df.isnull().sum().sum() / df.size * 100
        st.metric("🔍 Manquantes", f"{missing_pct:.1f}%")
    with col4:
        st.metric("🔄 Doublons", f"{df.duplicated().sum():,}")
    with col5:
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Mémoire", f"{memory:.1f} MB")


def _render_overview_tab(df: pd.DataFrame) -> None:
    st.dataframe(safe_display_dataframe(df), width="stretch", height=400)
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Affichage des 100 premières lignes sur {len(df)}")
    with col2:
        if st.button("📊 Voir les types", key="view_types_btn"):
            dtype_df = pd.DataFrame({"Colonne": df.columns, "Type": df.dtypes.astype(str)})
            st.dataframe(safe_display_dataframe(dtype_df), width="stretch")


def _render_stats_tab(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.info("Aucune colonne numérique pour les statistiques")
        return
    st.dataframe(safe_display_dataframe(df[numeric_cols].describe()), width="stretch")


def _render_types_tab(df: pd.DataFrame) -> None:
    st.bar_chart(df.dtypes.value_counts())
    df_info = pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str),
        "Non-Null": df.count(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2),
        "Uniques": df.nunique(),
    })
    st.dataframe(safe_display_dataframe(df_info), width="stretch")


def _render_detection_tab(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Cible potentielle**")
        target_col = detect_target_column(df)
        if target_col:
            st.write(f"- `{target_col}` (détectée par nom de colonne)")
        else:
            st.info("Aucune cible potentielle détectée (colonnes attendues: target, y, label, class)")

    with col2:
        st.markdown("**⚠️ Problèmes détectés**")
        try:
            results = DataValidator().validate(df)
            for err in results.get("errors", [])[:5]:
                st.warning(f"🔴 {err}")
            for warn in results.get("warnings", [])[:5]:
                st.warning(f"🟡 {warn}")
            if not results.get("errors") and not results.get("warnings"):
                st.success("✅ Aucun problème majeur détecté")
        except Exception as e:
            st.error(f"Erreur de validation: {e}")
