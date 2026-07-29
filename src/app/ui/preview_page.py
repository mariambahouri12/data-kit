"""Page de prévisualisation des données."""
import numpy as np
import pandas as pd
import streamlit as st

from datakit.validation.data_validator import DataValidator
from .dataframe_display import safe_display_dataframe
from datakit.preprocessing.utils.target_detection import detect_target_column


def render_preview_page() -> None:
    """Rendre la page de prévisualisation des données."""
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
    """Afficher les métriques principales du DataFrame."""
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
    """Afficher l'aperçu des données."""
    st.dataframe(safe_display_dataframe(df), width="stretch", height=400)
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Affichage des 100 premières lignes sur {len(df)}")
    with col2:
        if st.button("📊 Voir les types", key="view_types_btn"):
            dtype_df = pd.DataFrame({"Colonne": df.columns, "Type": df.dtypes.astype(str)})
            st.dataframe(safe_display_dataframe(dtype_df), width="stretch")


def _render_stats_tab(df: pd.DataFrame) -> None:
    """Afficher les statistiques descriptives."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.info("Aucune colonne numérique pour les statistiques")
        return
    st.dataframe(safe_display_dataframe(df[numeric_cols].describe()), width="stretch")


def _render_types_tab(df: pd.DataFrame) -> None:
    """Afficher les types de données."""
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
    """
    Afficher les détections de problèmes par colonne.
    """
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
            # Exécuter la validation
            results = DataValidator().validate(df)
            
            # Récupérer les statistiques
            total_problems = results.get("summary", {}).get("total_problems", 0)
            
            if total_problems > 0:
                st.warning(f"🔴 **{total_problems}** problème(s) détecté(s)")
                
                st.divider()
                
                # Afficher les problèmes par colonne
                st.markdown("**📝 Description des problèmes :**")
                
                problems_by_col = results.get("problems_by_column", {})
                
                if problems_by_col:
                    # Sélecteur de colonne pour filtrer les problèmes
                    selected_col = st.selectbox(
                        "Filtrer par colonne :",
                        ["Toutes les colonnes"] + list(problems_by_col.keys())
                    )
                    
                    # Afficher les problèmes
                    all_problems = []
                    for col, problems in problems_by_col.items():
                        if selected_col == "Toutes les colonnes" or selected_col == col:
                            for problem in problems:
                                all_problems.append({
                                    "Colonne": col,
                                    "Description": problem.get("description", ""),
                                })
                    
                    if all_problems:
                        problems_df = pd.DataFrame(all_problems)
                        st.dataframe(
                            problems_df,
                            width="stretch",
                            use_container_width=True,
                            height=400,
                            column_config={
                                "Colonne": st.column_config.TextColumn("Colonne"),
                                "Description": st.column_config.TextColumn("Description"),
                            }
                        )
                    else:
                        st.info("Aucun problème détaillé à afficher")
                else:
                    st.info("Aucun problème spécifique détecté par colonne")
            else:
                st.success("✅ **Aucun problème détecté**")
                st.info("Toutes les colonnes semblent avoir une bonne qualité de données")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la validation : {str(e)}")
            st.exception(e)