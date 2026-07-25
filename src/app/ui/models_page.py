# ui/models_page.py
"""Page de gestion des modèles - entrainement, évaluation et tracking."""

import pandas as pd
import streamlit as st

from datakit.models.models.factory import ModelFactory
from datakit.models.models.registry import ModelRegistry
from datakit.models.tracking.experiment_tracker import ExperimentTracker
from datakit.preprocessing.utils.target_detection import detect_target_column


def render_models_page() -> None:
    """Affiche la page de gestion des modèles."""
    st.markdown('<p class="sub-header">🤖 Entraînement de modèles</p>', unsafe_allow_html=True)

    # Vérifier que les données sont disponibles
    # CORRECTION : Utiliser une vérification explicite au lieu de l'opérateur 'or'
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
    elif st.session_state.current_data is not None:
        df = st.session_state.current_data
    else:
        st.warning("⚠️ Chargez et prétraitez des données d'abord")
        return

    # Détection de la colonne cible
    target_col = detect_target_column(df)
    if target_col is None:
        st.warning("Aucune colonne cible détectée. Ajoutez 'target', 'y', 'label' ou 'class'")
        return

    # Séparer X et y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Configuration du modèle
    with st.expander("⚙️ Configuration du modèle", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Sélection du modèle
            registry = ModelRegistry()
            available_models = registry.get_available_models()
            
            # Vérifier que des modèles sont disponibles
            if not available_models:
                st.error("❌ Aucun modèle disponible dans le registre")
                return
                
            model_name = st.selectbox("Modèle", available_models)

            # Type de tâche (détection auto ou manuelle)
            if y.dtype in ['int64', 'float64'] and y.nunique() > 10:
                default_task = "regression"
            else:
                default_task = "classification"
            task = st.selectbox(
                "Type de tâche",
                ["classification", "regression"],
                index=0 if default_task == "classification" else 1
            )

        with col2:
            # Split des données
            test_size = st.slider(
                "Test size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                format="%.2f"
            )
            random_state = st.number_input("Random state", value=42, min_value=0)

    # Paramètres du modèle (générés dynamiquement)
    with st.expander("🔧 Paramètres du modèle", expanded=False):
        try:
            schema = registry.get_parameter_schema(model_name)
        except ValueError:
            st.error(f"❌ Schéma de paramètres non trouvé pour {model_name}")
            return
            
        params = {}
        cols = st.columns(2)

        for idx, (param_name, param_def) in enumerate(schema.items()):
            with cols[idx % 2]:
                param_type = param_def.get("type", "str")
                default = param_def.get("default")
                description = param_def.get("description", param_name)
                choices = param_def.get("choices", [])

                if param_type == "int":
                    min_val = param_def.get("min", 0)
                    max_val = param_def.get("max", 1000)
                    # Si default est None, utiliser min_val
                    if default is None:
                        default = min_val
                    params[param_name] = st.number_input(
                        f"{param_name}",
                        value=int(default),
                        min_value=int(min_val),
                        max_value=int(max_val),
                        step=1,
                        help=description
                    )
                elif param_type == "float":
                    min_val = param_def.get("min", 0.0)
                    max_val = param_def.get("max", 1000.0)
                    step = param_def.get("step", 0.01)
                    if default is None:
                        default = min_val
                    params[param_name] = st.number_input(
                        f"{param_name}",
                        value=float(default),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        step=step,
                        help=description
                    )
                elif param_type == "str" and choices:
                    # S'assurer que default est dans choices
                    if default not in choices and choices:
                        default = choices[0]
                    params[param_name] = st.selectbox(
                        f"{param_name}",
                        options=choices,
                        index=choices.index(default) if default in choices else 0,
                        help=description
                    )
                elif param_type == "bool":
                    params[param_name] = st.checkbox(
                        f"{param_name}",
                        value=bool(default) if default is not None else False,
                        help=description
                    )
                else:
                    # Fallback pour les autres types (None, etc.)
                    if default is None:
                        default = ""
                    params[param_name] = st.text_input(
                        f"{param_name}",
                        value=str(default),
                        help=description
                    )

    # Entraînement
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Entraîner le modèle", type="primary", use_container_width=True):
            _train_and_display(df, X, y, model_name, task, params, test_size, random_state)


def _train_and_display(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                       model_name: str, task: str, params: dict,
                       test_size: float, random_state: int) -> None:
    """Entraîne le modèle et affiche les résultats."""
    with st.spinner("Entraînement en cours..."):
        try:
            # Créer le modèle
            factory = ModelFactory()
            model = factory.create_model(model_name, task=task, **params)

            # Entraîner et évaluer
            metrics = model.train(X, y, test_size=test_size, random_state=random_state)

            # Cross-validation
            cv_results = model.cross_validate(X, y, cv=5)

            # Enregistrer dans session state
            st.session_state.trained_model = model
            st.session_state.model_metrics = metrics
            st.session_state.model_cv_results = cv_results
            st.session_state.model_name = model_name
            st.session_state.model_task = task

            # Tracker (optionnel - pour la persistance)
            try:
                tracker = ExperimentTracker()
                run_id = tracker.start_run(model_name, task, params)
                tracker.log_metrics(run_id, metrics)
                tracker.finish_run(run_id)
            except Exception as e:
                st.warning(f"⚠️ Erreur de tracking (ignorée): {e}")

            # Afficher les résultats
            _display_results(metrics, cv_results, model_name, model)

        except Exception as e:
            st.error(f"❌ Erreur d'entraînement: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_results(metrics: dict, cv_results: dict, model_name: str, model) -> None:
    """Affiche les résultats d'entraînement."""
    # Métriques
    st.subheader("📊 Métriques")

    # Afficher les métriques dans des colonnes
    metric_items = list(metrics.items())
    # Filtrer les métriques None
    metric_items = [(k, v) for k, v in metric_items if v is not None]
    
    if metric_items:
        cols = st.columns(min(len(metric_items), 5))
        
        for idx, (metric_name, value) in enumerate(metric_items):
            if idx < len(cols):
                with cols[idx]:
                    display_name = metric_name.replace("_", " ").title()
                    if isinstance(value, (int, float)):
                        if metric_name in ["accuracy", "f1_macro", "f1_weighted", "r2"]:
                            st.metric(display_name, f"{value:.3f}")
                        elif metric_name in ["mse", "rmse"]:
                            st.metric(display_name, f"{value:.3f}")
                        else:
                            st.metric(display_name, f"{value:.3f}")
                    else:
                        st.metric(display_name, str(value))
    else:
        st.info("Aucune métrique disponible")

    # Cross-validation
    st.subheader("📈 Cross-Validation (5 folds)")
    if cv_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            mean_score = cv_results.get('mean_score', 0)
            st.metric("Moyenne CV", f"{mean_score:.3f}")
        with col2:
            std_score = cv_results.get('std_score', 0)
            st.metric("Écart-type CV", f"{std_score:.3f}")
        with col3:
            train_scores = cv_results.get('train_scores', 0)
            st.metric("Score entraînement", f"{train_scores:.3f}" if isinstance(train_scores, (int, float)) else "N/A")

    # Feature importance (si disponible)
    if hasattr(model, 'model') and model.model is not None:
        if hasattr(model.model, "feature_importances_"):
            st.subheader("🔍 Importance des features")
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                # Récupérer les noms des features
                feature_names = None
                if hasattr(model.model, "feature_names_in_"):
                    feature_names = model.model.feature_names_in_
                elif hasattr(model, '_feature_names'):
                    feature_names = model._feature_names
                
                if feature_names is None:
                    # Utiliser les noms des colonnes du dataframe original
                    df = st.session_state.current_data or st.session_state.processed_data
                    if df is not None:
                        # Exclure la colonne cible
                        target_col = detect_target_column(df)
                        if target_col:
                            feature_names = [col for col in df.columns if col != target_col]
                        else:
                            feature_names = df.columns.tolist()

                fig, ax = plt.subplots(figsize=(10, 6))
                importance = model.model.feature_importances_
                
                # S'assurer que nous avons assez de noms
                if feature_names is None or len(feature_names) != len(importance):
                    feature_names = [f"Feature_{i}" for i in range(len(importance))]
                
                # Trier par importance
                indices = np.argsort(importance)[::-1]
                top_n = min(15, len(indices))
                indices = indices[:top_n]
                
                ax.barh(range(top_n), importance[indices])
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Importance")
                ax.set_title(f"Top {top_n} feature importances")
                ax.invert_yaxis()
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Impossible d'afficher l'importance des features: {e}")

    # Informations sur l'expérience
    st.subheader("📝 Informations sur l'expérience")
    
    # Créer un dataframe pour un affichage propre
    info_data = {
        "Propriété": ["Modèle", "Tâche", "Run ID", "Timestamp", "Entraîné"],
        "Valeur": [
            model_name,
            getattr(model, 'task', 'N/A'),
            getattr(model, 'run_id', 'N/A')[:8] + "..." if hasattr(model, 'run_id') else "N/A",
            getattr(model, 'timestamp', 'N/A'),
            "✅ Oui" if getattr(model, 'is_trained', False) else "❌ Non"
        ]
    }
    info_df = pd.DataFrame(info_data)
    st.dataframe(info_df, hide_index=True, use_container_width=True)

    # Afficher les paramètres
    with st.expander("🔧 Paramètres utilisés", expanded=False):
        if hasattr(model, 'get_params'):
            st.json(model.get_params())
        else:
            st.info("Paramètres non disponibles")