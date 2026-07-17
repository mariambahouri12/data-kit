# app.py
"""
Application Streamlit pour l'upload et le preprocessing de données tabulaires.
"""
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.datakit.modules.uploads import UploadManager
from src.datakit.modules.preprocessing.tabular.config import PreprocessingConfig, TaskType, BalancingMethod
from src.datakit.modules.preprocessing.tabular.pipeline_builder import PipelineBuilder
from src.datakit.modules.preprocessing.tabular.balancers import ClassBalancer
from src.datakit.modules.preprocessing.tabular.balance_analyzer import ImbalanceAnalyzer
from src.datakit.modules.preprocessing.utils.validators import DataValidator
from src.datakit.modules.preprocessing.utils.visualizers import DataVisualizer
from src.datakit.modules.preprocessing.factory import PreprocessingPresets
from src.datakit.modules.preprocessing.utils.arrow_fix import fix_dataframe_for_arrow, safe_display_dataframe, fix_dataframe_complete

TARGET_COLUMN_CANDIDATES = ("target", "y", "label", "class")

st.set_page_config(
    page_title="AI Experimentation Platform - Data Preprocessing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
""", unsafe_allow_html=True)


# -- Helpers partagés ---------------------------------------------------------

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Détecte une colonne cible probable par son nom (heuristique simple)."""
    for col in df.columns:
        if col.lower() in TARGET_COLUMN_CANDIDATES:
            return col
    return None


def _as_dataframe(X, columns) -> pd.DataFrame:
    """Garantit un pd.DataFrame, quelle que soit la version d'imblearn utilisée
    (certaines versions retournent un ndarray brut plutôt qu'un DataFrame)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)


def _as_series(y, name: str, index) -> pd.Series:
    """Garantit un pd.Series, même si imblearn a retourné un ndarray brut."""
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y, name=name, index=index)


# -- État de session -----------------------------------------------------------

def initialize_session_state() -> None:
    defaults = {
        "upload_manager": UploadManager,
        "current_data": lambda: None,
        "current_metadata": lambda: None,
        "processed_data": lambda: None,
        "selected_file_id": lambda: None,
    }
    if "upload_manager" not in st.session_state:
        st.session_state.upload_manager = UploadManager()
    for key in ("current_data", "current_metadata", "processed_data", "selected_file_id"):
        if key not in st.session_state:
            st.session_state[key] = None


def render_header() -> None:
    st.markdown('<p class="main-header">📊 AI Experimentation Platform</p>', unsafe_allow_html=True)
    st.markdown("### Data Upload & Preprocessing Module")
    st.divider()


# -- Upload -----------------------------------------------------------------

def render_upload_section() -> None:
    st.markdown('<p class="sub-header">📤 Upload Data</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV, Excel ou Parquet",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Taille max: 100MB",
        )
        if uploaded_file is not None:
            _handle_file_upload(uploaded_file)

    with col2:
        _render_existing_uploads()


def _handle_file_upload(uploaded_file) -> None:
    try:
        df = _read_uploaded_file(uploaded_file)
        df = fix_dataframe_complete(df)

        upload_manager = st.session_state.upload_manager
        metadata = upload_manager.upload(
            file_content=uploaded_file.getvalue(),
            filename=uploaded_file.name,
            user_id="streamlit_user",
            tags={"source": "streamlit"},
        )

        st.session_state.current_data = df
        st.session_state.current_metadata = metadata
        st.session_state.selected_file_id = metadata["file_id"]

        st.markdown(
            f'<div class="success-box">✅ Fichier uploadé avec succès !<br>'
            f'ID: <code>{metadata["file_id"]}</code></div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    extension = Path(uploaded_file.name).suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
    }
    reader = readers.get(extension)
    if reader is None:
        raise ValueError(f"Format non supporté: {extension}")
    return reader(uploaded_file)


def _render_existing_uploads() -> None:
    st.markdown("### 📋 Fichiers uploadés")
    upload_manager = st.session_state.upload_manager
    uploads = upload_manager.list_uploads()

    if not uploads:
        st.info("Aucun fichier uploadé")
        return

    file_options = {
        u["file_id"]: f"{u['filename']} ({u['rows']} rows, {u['columns']} cols)" for u in uploads
    }
    selected_id = st.selectbox(
        "Sélectionner un fichier existant",
        options=list(file_options.keys()),
        format_func=lambda x: file_options[x],
        key="file_selector",
    )

    if selected_id and selected_id != st.session_state.selected_file_id:
        try:
            df, metadata = upload_manager.load_data(selected_id)
            df = fix_dataframe_complete(df)
            st.session_state.current_data = df
            st.session_state.current_metadata = metadata
            st.session_state.selected_file_id = selected_id
            st.success(f"✅ Fichier chargé: {metadata['filename']}")
        except Exception as e:
            st.error(f"Erreur: {e}")

    st.markdown("---")
    st.markdown("**📊 Statistiques:**")
    st.write(f"📁 Total: {len(uploads)} fichiers")
    total_size = sum(u.get("file_size_mb", 0) for u in uploads)
    st.write(f"💾 Taille totale: {total_size:.1f} MB")


# -- Preview -----------------------------------------------------------------

def render_data_preview() -> None:
    st.markdown('<p class="sub-header">🔍 Data Preview</p>', unsafe_allow_html=True)

    if st.session_state.current_data is None:
        st.info("ℹ️ Chargez d'abord des données pour afficher un aperçu")
        return

    df = st.session_state.current_data
    _render_preview_metrics(df)

    tabs = st.tabs(["📊 Aperçu", "📈 Statistiques", "📋 Types", "🔍 Détection"])
    with tabs[0]:
        _render_preview_overview_tab(df)
    with tabs[1]:
        _render_preview_stats_tab(df)
    with tabs[2]:
        _render_preview_types_tab(df)
    with tabs[3]:
        _render_preview_detection_tab(df)


def _render_preview_metrics(df: pd.DataFrame) -> None:
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


def _render_preview_overview_tab(df: pd.DataFrame) -> None:
    st.dataframe(safe_display_dataframe(df), width="stretch", height=400)
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Affichage des 100 premières lignes sur {len(df)}")
    with col2:
        if st.button("📊 Voir les types", key="view_types_btn"):
            dtype_df = pd.DataFrame({"Colonne": df.columns, "Type": df.dtypes.astype(str)})
            st.dataframe(safe_display_dataframe(dtype_df), width="stretch")


def _render_preview_stats_tab(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.info("Aucune colonne numérique pour les statistiques")
        return
    st.dataframe(safe_display_dataframe(df[numeric_cols].describe()), width="stretch")


def _render_preview_types_tab(df: pd.DataFrame) -> None:
    st.bar_chart(df.dtypes.value_counts())
    df_info = pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str),
        "Non-Null": df.count(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2),
        "Uniques": df.nunique(),
    })
    st.dataframe(safe_display_dataframe(df_info), width="stretch")


def _render_preview_detection_tab(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Cibles potentielles**")
        analysis = (st.session_state.current_metadata or {}).get("analysis", {})
        targets = analysis.get("potential_targets", [])
        if targets:
            for t in targets[:5]:
                st.write(f"- `{t['column']}` (score: {t['score']})")
        else:
            st.info("Aucune cible potentielle détectée")

    with col2:
        st.markdown("**⚠️ Problèmes détectés**")
        try:
            results = DataValidator(verbose=False).validate(df)
            for err in results.get("errors", [])[:5]:
                st.warning(f"🔴 {err}")
            for warn in results.get("warnings", [])[:5]:
                st.warning(f"🟡 {warn}")
            if not results.get("errors") and not results.get("warnings"):
                st.success("✅ Aucun problème majeur détecté")
        except Exception as e:
            st.error(f"Erreur de validation: {e}")


# -- Configuration du preprocessing ------------------------------------------

def render_preprocessing_config() -> None:
    st.markdown('<p class="sub-header">⚙️ Configuration</p>', unsafe_allow_html=True)

    if st.session_state.current_data is None:
        st.warning("⚠️ Chargez d'abord des données")
        return

    _render_preset_selector()

    # Les valeurs des widgets sont collectées ici, puis PreprocessingConfig est
    # construite UNE SEULE FOIS à la fin (voir plus bas), pour garantir que la
    # normalisation str -> Enum de PreprocessingConfig.__post_init__ s'applique
    # bien à tous les champs, plutôt que de muter un objet déjà construit.
    settings: Dict[str, Any] = {}

    tabs = st.tabs([
        "🎯 Général", "🧹 Nettoyage", "📐 Scaling & Encoding",
        "⚖️ Balancing", "🔧 Feature Engineering", "📉 Réduction",
    ])

    with tabs[0]:
        _config_tab_general(settings)
    with tabs[1]:
        _config_tab_cleaning(settings)
    with tabs[2]:
        _config_tab_scaling_encoding(settings)
    with tabs[3]:
        _config_tab_balancing(settings)
    with tabs[4]:
        _config_tab_feature_engineering(settings)
    with tabs[5]:
        _config_tab_reduction(settings)

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Appliquer le preprocessing", type="primary", key="apply_preprocessing"):
            config = PreprocessingConfig(**settings)
            process_data(config)


def _render_preset_selector() -> None:
    st.markdown("**📋 Présélections**")
    presets = PreprocessingPresets.list_presets()
    selected_preset = st.selectbox(
        "Charger une configuration prédéfinie", options=[""] + presets, key="preset_selector"
    )
    if selected_preset:
        preset_values = PreprocessingPresets.get_preset(selected_preset)
        for field_name, value in preset_values.items():
            # Pré-remplit le widget correspondant AVANT son rendu, en supposant
            # que la clé du widget porte le même nom que le champ de config.
            st.session_state[field_name] = value
        st.info(f"Configuration '{selected_preset}' chargée")
        st.rerun()    


def _config_tab_general(settings: Dict[str, Any]) -> None:
    st.markdown("### 🎯 Configuration Générale")
    col1, col2 = st.columns(2)
    with col1:
        settings["task_type"] = st.selectbox(
            "Type de tâche", options=["classification", "regression"], index=0, key="task_type"
        )
    with col2:
        settings["random_state"] = st.number_input(
            "Random State", min_value=0, max_value=9999, value=42, key="random_state"
        )


def _config_tab_cleaning(settings: Dict[str, Any]) -> None:
    st.markdown("### 🧹 Nettoyage des Données")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Valeurs manquantes**")
        settings["imputation_method"] = st.selectbox(
            "Méthode d'imputation",
            options=["mean", "median", "most_frequent", "knn", "drop"],
            index=1, key="imputation_method",
        )
        if settings["imputation_method"] == "knn":
            settings["imputation_knn_neighbors"] = st.slider(
                "Nombre de voisins KNN", min_value=1, max_value=10, value=5, key="knn_neighbors"
            )

    with col2:
        st.markdown("**Outliers**")
        settings["outlier_method"] = st.selectbox(
            "Méthode de détection", options=["iqr", "zscore", "isolation_forest", "none"],
            index=0, key="outlier_method",
        )
        if settings["outlier_method"] != "none":
            settings["outlier_threshold"] = st.slider(
                "Seuil", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="outlier_threshold"
            )
            settings["outlier_action"] = st.selectbox(
                "Action", options=["winsorize", "drop"], index=0, key="outlier_action"
            )

    settings["drop_duplicates"] = st.checkbox("Supprimer les doublons", value=True, key="drop_duplicates")
    settings["drop_high_missing"] = st.checkbox(
        "Supprimer les colonnes avec trop de valeurs manquantes", value=True, key="drop_high_missing"
    )
    if settings["drop_high_missing"]:
        settings["high_missing_threshold"] = st.slider(
            "Seuil de suppression (%)", min_value=0.0, max_value=1.0, value=0.8, key="high_missing_threshold"
        )


def _config_tab_scaling_encoding(settings: Dict[str, Any]) -> None:
    st.markdown("### 📐 Normalisation & Encodage")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Scaling**")
        settings["scaling_method"] = st.selectbox(
            "Méthode de normalisation",
            options=["standard", "minmax", "robust", "maxabs", "quantile", "power", "none"],
            index=0, key="scaling_method",
        )

    with col2:
        st.markdown("**Encodage**")
        settings["encoding_method"] = st.selectbox(
            "Méthode d'encodage",
            options=["onehot", "label", "target", "frequency", "ordinal", "none"],
            index=0, key="encoding_method",
        )
        if settings["encoding_method"] != "none":
            settings["encoding_max_categories"] = st.number_input(
                "Nombre max de catégories", min_value=5, max_value=200, value=50, key="encoding_max_categories"
            )

    st.markdown("**Transformations de distribution**")
    col1, col2, col3 = st.columns(3)
    with col1:
        settings["apply_log_transform"] = st.checkbox("Log Transform", value=False, key="apply_log_transform")
    with col2:
        settings["apply_boxcox"] = st.checkbox("Box-Cox Transform", value=False, key="apply_boxcox")
    with col3:
        settings["apply_yeojohnson"] = st.checkbox("Yeo-Johnson Transform", value=False, key="apply_yeojohnson")


def _config_tab_balancing(settings: Dict[str, Any]) -> None:
    st.markdown("### ⚖️ Rééquilibrage des Classes")
    settings["balancing_method"] = st.selectbox(
        "Méthode de rééquilibrage",
        options=["smote", "adasyn", "random_over", "random_under", "tomek", "none"],
        index=4, key="balancing_method",
    )

    if settings["balancing_method"] == "none":
        return

    settings["balancing_random_state"] = st.number_input(
        "Random State", min_value=0, max_value=9999, value=42, key="balancing_random_state"
    )
    settings["balancing_apply_before_pipeline"] = st.checkbox(
        "Appliquer avant le pipeline (recommandé)", value=True, key="balancing_apply_before"
    )

    if st.button("📊 Analyser le déséquilibre", key="analyze_balance"):
        df_local = st.session_state.current_data
        target_col = detect_target_column(df_local)
        if target_col is None:
            st.warning("Aucune colonne cible détectée. Définissez 'target', 'y', 'label' ou 'class'")
            return

        y = df_local[target_col]
        col1, col2 = st.columns(2)
        with col1:
            st.json(ImbalanceAnalyzer.get_class_distribution(y))
        with col2:
            st.json(ImbalanceAnalyzer.suggest_method(y))


def _config_tab_feature_engineering(settings: Dict[str, Any]) -> None:
    st.markdown("### 🔧 Feature Engineering")
    col1, col2 = st.columns(2)

    with col1:
        settings["create_polynomial"] = st.checkbox(
            "Créer des features polynomiales", value=False, key="create_polynomial"
        )
        if settings["create_polynomial"]:
            settings["polynomial_degree"] = st.slider(
                "Degré", min_value=2, max_value=4, value=2, key="polynomial_degree"
            )

    with col2:
        settings["create_interactions"] = st.checkbox(
            "Créer des interactions", value=False, key="create_interactions"
        )

    settings["create_ratios"] = st.checkbox("Créer des ratios", value=False, key="create_ratios")
    if settings["create_ratios"]:
        settings["ratios_max_pairs"] = st.number_input(
            "Nombre max de paires", min_value=10, max_value=500, value=100, key="ratios_max_pairs"
        )


def _config_tab_reduction(settings: Dict[str, Any]) -> None:
    """Réduction de dimension (PCA/LDA) et sélection de features : deux
    mécanismes indépendants, exposés via deux contrôles séparés plutôt
    qu'un seul selectbox qui laissait croire qu'ils étaient exclusifs."""
    st.markdown("### 📉 Réduction de Dimensionnalité")

    reduction_method = st.selectbox(
        "Méthode de réduction (PCA / LDA)", options=["none", "pca", "lda"], index=0, key="reduction_method"
    )
    settings["reduction_method"] = None if reduction_method == "none" else reduction_method

    if reduction_method == "pca":
        col1, col2 = st.columns(2)
        with col1:
            settings["reduction_components"] = st.number_input(
                "Nombre de composantes", min_value=1, max_value=50, value=5, key="reduction_components"
            )
        with col2:
            settings["reduction_variance_ratio"] = st.slider(
                "Variance expliquée", min_value=0.5, max_value=1.0, value=0.95, key="reduction_variance_ratio"
            )
    elif reduction_method == "lda":
        settings["reduction_components"] = st.number_input(
            "Nombre de composantes", min_value=1, max_value=50, value=5, key="reduction_components_lda"
        )

    st.markdown("---")
    enable_feature_selection = st.checkbox(
        "Activer la sélection de features (indépendant de PCA/LDA)", value=False, key="enable_feature_selection"
    )
    if enable_feature_selection:
        settings["feature_selection_method"] = st.selectbox(
            "Méthode de sélection", options=["variance", "correlation", "importance"],
            index=2, key="feature_selection_method",
        )
        settings["feature_selection_k"] = st.number_input(
            "Nombre de features à sélectionner", min_value=1, max_value=100, value=20, key="feature_selection_k"
        )


# -- Traitement ---------------------------------------------------------------

def process_data(config: PreprocessingConfig) -> None:
    if st.session_state.current_data is None:
        st.error("❌ Aucune données à traiter")
        return

    df = st.session_state.current_data

    with st.spinner("⏳ Prétraitement en cours..."):
        try:
            builder = PipelineBuilder(config)
            df_processed = _apply_balancing_if_needed(builder, config, df)

            pipeline = builder.build_pipeline()
            df_transformed = pipeline.fit_transform(df_processed)
            df_transformed = fix_dataframe_complete(df_transformed)

            st.session_state.processed_data = df_transformed

            if st.session_state.selected_file_id:
                st.session_state.upload_manager.save_processed(
                    df_transformed, st.session_state.selected_file_id, format="csv"
                )

            st.markdown(
                f'<div class="success-box">✅ Preprocessing terminé !<br>'
                f'Données: {len(df)} → {len(df_transformed)} lignes, '
                f'{len(df.columns)} → {len(df_transformed.columns)} colonnes</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ Erreur lors du preprocessing: {e}")
            st.exception(e)


def _apply_balancing_if_needed(builder: PipelineBuilder, config: PreprocessingConfig, df: pd.DataFrame) -> pd.DataFrame:
    if config.balancing_method == BalancingMethod.NONE or not config.balancing_apply_before_pipeline:
        return df

    target_col = detect_target_column(df)
    if target_col is None:
        st.warning("⚠️ Aucune colonne cible trouvée pour le rééquilibrage")
        return df

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_balanced, y_balanced = builder.apply_balancing(X, y)
    X_balanced = _as_dataframe(X_balanced, X.columns)
    y_balanced = _as_series(y_balanced, name=target_col, index=X_balanced.index)

    df_processed = pd.concat([X_balanced, y_balanced], axis=1)
    st.info(f"⚖️ Rééquilibrage appliqué: {len(df)} → {len(df_processed)} lignes")
    return df_processed


# -- Résultats -----------------------------------------------------------------

def render_processed_data() -> None:
    st.markdown('<p class="sub-header">📊 Données Transformées</p>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.info("ℹ️ Aucune donnée transformée. Lancez le preprocessing.")
        return

    df = st.session_state.processed_data

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Lignes transformées", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes transformées", f"{len(df.columns)}")
    with col3:
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Mémoire", f"{memory:.1f} MB")

    st.dataframe(safe_display_dataframe(df), width="stretch", height=400)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger CSV", data=csv, file_name="processed_data.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        st.download_button(
            "📥 Télécharger Parquet", data=parquet_buffer.getvalue(), file_name="processed_data.parquet",
            mime="application/octet-stream", use_container_width=True,
        )


def render_visualization() -> None:
    st.markdown('<p class="sub-header">📈 Visualisation</p>', unsafe_allow_html=True)

    df = (
    st.session_state.processed_data
    if st.session_state.processed_data is not None
    else st.session_state.current_data
)
    if df is None:
        st.info("ℹ️ Chargez des données pour visualiser")
        return

    df = fix_dataframe_for_arrow(df)
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


# -- Sidebar -------------------------------------------------------------------

def render_sidebar() -> None:
    """Panneau d'information. Les onglets principaux (st.tabs) gèrent déjà
    la navigation ; ce panneau n'essaie plus de la dupliquer avec des
    boutons qui n'avaient aucun effet."""
    with st.sidebar:
        st.markdown("### 📊 AI Experimentation Platform")
        st.markdown("---")

        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            st.markdown("### 📊 Données chargées")
            st.write(f"**Lignes:** {len(df):,}")
            st.write(f"**Colonnes:** {len(df.columns)}")

            if st.session_state.processed_data is not None:
                df_processed = st.session_state.processed_data
                st.markdown("### ✅ Données traitées")
                st.write(f"**Lignes:** {len(df_processed):,}")
                st.write(f"**Colonnes:** {len(df_processed.columns)}")

        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.caption("Version 1.0.0")
        st.caption("Made with ❤️ using Streamlit")


# -- Point d'entrée --------------------------------------------------------------

def main() -> None:
    initialize_session_state()
    render_sidebar()
    render_header()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📤 Upload", "🔍 Preview", "⚙️ Preprocess", "📊 Processed", "📈 Visualize"]
    )
    with tab1:
        render_upload_section()
    with tab2:
        render_data_preview()
    with tab3:
        render_preprocessing_config()
    with tab4:
        render_processed_data()
    with tab5:
        render_visualization()


if __name__ == "__main__":
    main()