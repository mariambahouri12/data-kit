"""Page de configuration et d'exécution du preprocessing."""
from typing import Any, Dict

import streamlit as st

from datakit.preprocessing.tabular.config import PreprocessingConfig
from datakit.preprocessing.presets import PreprocessingPresets
from datakit.preprocessing.tabular.balancers.balance_analyzer import ImbalanceAnalyzer
from datakit.preprocessing.orchestrator import run_preprocessing
from datakit.preprocessing.utils.target_detection import detect_target_column


def render_preprocessing_page() -> None:
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
        _tab_general(settings)
    with tabs[1]:
        _tab_cleaning(settings)
    with tabs[2]:
        _tab_scaling_encoding(settings)
    with tabs[3]:
        _tab_balancing(settings)
    with tabs[4]:
        _tab_feature_engineering(settings)
    with tabs[5]:
        _tab_reduction(settings)

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Appliquer le preprocessing", type="primary", key="apply_preprocessing"):
            config = PreprocessingConfig(**settings)
            _run_and_store(config)


def _run_and_store(config: PreprocessingConfig) -> None:
    """Délègue l'exécution du pipeline au service métier et affiche le résultat.
    L'UI ne fait plus que récupérer les valeurs, appeler le service et afficher."""
    df = st.session_state.current_data

    with st.spinner("⏳ Prétraitement en cours..."):
        try:
            result = run_preprocessing(df, config)
            st.session_state.processed_data = result.df

            if result.balancing_message:
                st.info(result.balancing_message)

            st.markdown(
                f'<div class="success-box">✅ Preprocessing terminé !<br>'
                f'Données: {len(df)} → {len(result.df)} lignes, '
                f'{len(df.columns)} → {len(result.df.columns)} colonnes</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"❌ Erreur lors du preprocessing: {e}")
            st.exception(e)


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


def _tab_general(settings: Dict[str, Any]) -> None:
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


def _tab_cleaning(settings: Dict[str, Any]) -> None:
    st.markdown("### 🧹 Nettoyage des Données")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Valeurs manquantes**")
        settings["imputation_method"] = st.selectbox(
            "Méthode d'imputation par défaut",
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

    # --- NOUVEAU: Stratégies par colonne (avancé) ---
    st.markdown("---")
    st.markdown("**🎯 Stratégies d'imputation par colonne (avancé)**")
    
    use_column_strategies = st.checkbox(
        "Définir des stratégies spécifiques par colonne",
        value=False,
        key="use_column_strategies"
    )
    
    if use_column_strategies:
        df = st.session_state.current_data
        if df is not None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            
            if numeric_cols:
                st.info(f"Configurez une stratégie pour chaque colonne numérique. Les colonnes non spécifiées utiliseront la stratégie par défaut.")
                
                column_strategies = {}
                
                # Utiliser des colonnes pour un affichage compact
                for i, col in enumerate(numeric_cols):
                    cols = st.columns([0.3, 0.7])
                    with cols[0]:
                        st.caption(f"`{col}`")
                    with cols[1]:
                        strategy = st.selectbox(
                            f"Stratégie pour {col}",
                            options=["default", "mean", "median", "most_frequent", "constant", "knn", "drop"],
                            index=0,
                            key=f"col_strategy_{col}",
                            label_visibility="collapsed"
                        )
                        if strategy != "default":
                            column_strategies[col] = strategy
                
                if column_strategies:
                    settings["column_strategies"] = column_strategies
                    
                    # Afficher un résumé des stratégies définies
                    st.markdown("**Résumé des stratégies personnalisées :**")
                    for col, strat in column_strategies.items():
                        st.caption(f"- `{col}` → **{strat}**")
                else:
                    st.caption("Aucune stratégie personnalisée définie. La stratégie par défaut sera utilisée.")
            else:
                st.info("Aucune colonne numérique détectée pour les stratégies personnalisées.")


def _tab_scaling_encoding(settings: Dict[str, Any]) -> None:
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


def _tab_balancing(settings: Dict[str, Any]) -> None:
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


def _tab_feature_engineering(settings: Dict[str, Any]) -> None:
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


def _tab_reduction(settings: Dict[str, Any]) -> None:
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