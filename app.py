# app.py
"""
Application Streamlit pour l'upload et le preprocessing de données tabulaires.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import io
import time

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uploads import UploadManager
from preprocessing.tabular.config import PreprocessingConfig, TaskType
from preprocessing.tabular.pipeline_builder import PipelineBuilder
from preprocessing.tabular.detectors import (
    MissingValueDetector, OutlierDetector, CorrelationDetector,
    CardinalityDetector, DuplicateDetector
)
from preprocessing.tabular.cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner
from preprocessing.tabular.encoders import CategoricalEncoder
from preprocessing.tabular.scalers import FeatureScaler
from preprocessing.tabular.balancers import ClassBalancer
from preprocessing.tabular.reducers import PCAReducer, FeatureSelector
from preprocessing.utils.validators import DataValidator
from preprocessing.utils.visualizers import DataVisualizer
from preprocessing.factory import PreprocessingFactory, PreprocessingPresets
from preprocessing.utils.arrow_fix import fix_dataframe_for_arrow, safe_display_dataframe, fix_dataframe_complete

# Configuration de la page
st.set_page_config(
    page_title="AI Experimentation Platform - Data Preprocessing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4ECDC4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2C3E50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4ECDC4;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FFC107;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: #D4EDDA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #28A745;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialiser les variables de session"""
    if 'upload_manager' not in st.session_state:
        st.session_state.upload_manager = UploadManager()
    
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    if 'current_metadata' not in st.session_state:
        st.session_state.current_metadata = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'pipeline_builder' not in st.session_state:
        st.session_state.pipeline_builder = None
    
    if 'selected_file_id' not in st.session_state:
        st.session_state.selected_file_id = None


def render_header():
    """Afficher l'en-tête"""
    st.markdown('<p class="main-header">📊 AI Experimentation Platform</p>', unsafe_allow_html=True)
    st.markdown("### Data Upload & Preprocessing Module")
    st.divider()


def render_upload_section():
    """Section d'upload de fichiers"""
    st.markdown('<p class="sub-header">📤 Upload Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV, Excel ou Parquet",
            type=['csv', 'xlsx', 'xls', 'parquet'],
            help="Taille max: 100MB"
        )
        
        if uploaded_file is not None:
            # Lire le fichier
            try:
                file_extension = Path(uploaded_file.name).suffix.lower()
                if file_extension == '.csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == '.parquet':
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error(f"Format non supporté: {file_extension}")
                    return
                
                # Corriger les types du DataFrame
                df = fix_dataframe_complete(df)
                
                # Uploader le fichier
                upload_manager = st.session_state.upload_manager
                file_content = uploaded_file.getvalue()
                
                metadata = upload_manager.upload(
                    file_content=file_content,
                    filename=uploaded_file.name,
                    user_id="streamlit_user",
                    tags={'source': 'streamlit'}
                )
                
                st.session_state.current_data = df
                st.session_state.current_metadata = metadata
                st.session_state.selected_file_id = metadata['file_id']
                
                st.markdown(
                    f'<div class="success-box">✅ Fichier uploadé avec succès !<br>'
                    f'ID: <code>{metadata["file_id"]}</code></div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Fichiers uploadés")
        
        upload_manager = st.session_state.upload_manager
        uploads = upload_manager.list_uploads()
        
        if uploads:
            # Sélecteur de fichiers
            file_options = {
                u['file_id']: f"{u['filename']} ({u['rows']} rows, {u['columns']} cols)"
                for u in uploads
            }
            
            selected_id = st.selectbox(
                "Sélectionner un fichier existant",
                options=list(file_options.keys()),
                format_func=lambda x: file_options[x],
                key="file_selector"
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
                    st.error(f"Erreur: {str(e)}")
            
            # Statistiques rapides
            st.markdown("---")
            st.markdown("**📊 Statistiques:**")
            st.write(f"📁 Total: {len(uploads)} fichiers")
            total_size = sum(u.get('file_size_mb', 0) for u in uploads)
            st.write(f"💾 Taille totale: {total_size:.1f} MB")
        else:
            st.info("Aucun fichier uploadé")


def render_data_preview():
    """Afficher un aperçu des données"""
    st.markdown('<p class="sub-header">🔍 Data Preview</p>', unsafe_allow_html=True)
    
    if st.session_state.current_data is None:
        st.info("ℹ️ Chargez d'abord des données pour afficher un aperçu")
        return
    
    df = st.session_state.current_data
    
    # Métriques rapides
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", f"{len(df.columns)}")
    with col3:
        missing_pct = df.isnull().sum().sum() / df.size * 100
        st.metric("🔍 Manquantes", f"{missing_pct:.1f}%")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("🔄 Doublons", f"{duplicates:,}")
    with col5:
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Mémoire", f"{memory:.1f} MB")
    
    # Aperçu des données
    tabs = st.tabs(["📊 Aperçu", "📈 Statistiques", "📋 Types", "🔍 Détection"])
    
    with tabs[0]:
        st.dataframe(
            safe_display_dataframe(df),
            width='stretch',
            height=400
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Affichage des 100 premières lignes sur {len(df)}")
        with col2:
            if st.button("📊 Voir les types", key="view_types_btn"):
                dtype_df = pd.DataFrame({
                    'Colonne': df.columns,
                    'Type': df.dtypes.astype(str)
                })
                st.dataframe(safe_display_dataframe(dtype_df), width='stretch')
    
    with tabs[1]:
        # Statistiques descriptives
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(
                safe_display_dataframe(df[numeric_cols].describe()),
                width='stretch'
            )
        else:
            st.info("Aucune colonne numérique pour les statistiques")
    
    with tabs[2]:
        # Types de colonnes
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
        
        df_info = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Uniques': df.nunique()
        })
        st.dataframe(safe_display_dataframe(df_info), width='stretch')
    
    with tabs[3]:
        # Détection automatique
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Cibles potentielles**")
            if st.session_state.current_metadata:
                analysis = st.session_state.current_metadata.get('analysis', {})
                targets = analysis.get('potential_targets', [])
                if targets:
                    for t in targets[:5]:
                        st.write(f"- `{t['column']}` (score: {t['score']})")
                else:
                    st.info("Aucune cible potentielle détectée")
        
        with col2:
            st.markdown("**⚠️ Problèmes détectés**")
            try:
                validator = DataValidator(verbose=False)
                results = validator.validate(df)
                
                if results.get('errors'):
                    for err in results['errors'][:5]:
                        st.warning(f"🔴 {err}")
                if results.get('warnings'):
                    for warn in results['warnings'][:5]:
                        st.warning(f"🟡 {warn}")
                if not results.get('errors') and not results.get('warnings'):
                    st.success("✅ Aucun problème majeur détecté")
            except Exception as e:
                st.error(f"Erreur de validation: {str(e)}")


def render_preprocessing_config():
    """Configuration du preprocessing"""
    st.markdown('<p class="sub-header">⚙️ Configuration</p>', unsafe_allow_html=True)
    
    if st.session_state.current_data is None:
        st.warning("⚠️ Chargez d'abord des données")
        return
    
    df = st.session_state.current_data
    
    # Onglets de configuration
    tabs = st.tabs([
        "🎯 Général", 
        "🧹 Nettoyage", 
        "📐 Scaling & Encoding",
        "⚖️ Balancing",
        "🔧 Feature Engineering",
        "📉 Réduction"
    ])
    
    # Initialiser la configuration
    config = PreprocessingConfig()
    
    with tabs[0]:  # Général
        st.markdown("### 🎯 Configuration Générale")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task_type = st.selectbox(
                "Type de tâche",
                options=['classification', 'regression'],
                index=0,
                key="task_type"
            )
            config.task_type = TaskType(task_type)
        
        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=9999,
                value=42,
                key="random_state"
            )
            config.random_state = random_state
        
        # Présélections
        st.markdown("**📋 Présélections**")
        presets = PreprocessingPresets.list_presets()
        selected_preset = st.selectbox(
            "Charger une configuration prédéfinie",
            options=[''] + presets,
            key="preset_selector"
        )
        
        if selected_preset:
            preset_config = PreprocessingPresets.get_preset(selected_preset)
            st.info(f"Configuration '{selected_preset}' chargée")
            for key, value in preset_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            st.rerun()
    
    with tabs[1]:  # Nettoyage
        st.markdown("### 🧹 Nettoyage des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Valeurs manquantes**")
            imputation_method = st.selectbox(
                "Méthode d'imputation",
                options=['mean', 'median', 'most_frequent', 'knn', 'drop'],
                index=1,
                key="imputation_method"
            )
            config.imputation_method = imputation_method
            
            if imputation_method == 'knn':
                config.imputation_knn_neighbors = st.slider(
                    "Nombre de voisins KNN",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="knn_neighbors"
                )
        
        with col2:
            st.markdown("**Outliers**")
            outlier_method = st.selectbox(
                "Méthode de détection",
                options=['iqr', 'zscore', 'isolation_forest', 'none'],
                index=0,
                key="outlier_method"
            )
            config.outlier_method = outlier_method
            
            if outlier_method != 'none':
                config.outlier_threshold = st.slider(
                    "Seuil",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.5,
                    key="outlier_threshold"
                )
                
                config.outlier_action = st.selectbox(
                    "Action",
                    options=['winsorize', 'drop'],
                    index=0,
                    key="outlier_action"
                )
        
        # Doublons
        config.drop_duplicates = st.checkbox(
            "Supprimer les doublons",
            value=True,
            key="drop_duplicates"
        )
        
        config.drop_high_missing = st.checkbox(
            "Supprimer les colonnes avec trop de valeurs manquantes",
            value=True,
            key="drop_high_missing"
        )
        
        if config.drop_high_missing:
            config.high_missing_threshold = st.slider(
                "Seuil de suppression (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                key="high_missing_threshold"
            )
    
    with tabs[2]:  # Scaling & Encoding
        st.markdown("### 📐 Normalisation & Encodage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scaling**")
            scaling_method = st.selectbox(
                "Méthode de normalisation",
                options=['standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'none'],
                index=0,
                key="scaling_method"
            )
            config.scaling_method = scaling_method
        
        with col2:
            st.markdown("**Encodage**")
            encoding_method = st.selectbox(
                "Méthode d'encodage",
                options=['onehot', 'label', 'target', 'frequency', 'ordinal', 'none'],
                index=0,
                key="encoding_method"
            )
            config.encoding_method = encoding_method
            
            if encoding_method != 'none':
                config.encoding_max_categories = st.number_input(
                    "Nombre max de catégories",
                    min_value=5,
                    max_value=200,
                    value=50,
                    key="encoding_max_categories"
                )
        
        # Transformations
        st.markdown("**Transformations de distribution**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            config.apply_log_transform = st.checkbox(
                "Log Transform",
                value=False,
                key="apply_log_transform"
            )
        
        with col2:
            config.apply_boxcox = st.checkbox(
                "Box-Cox Transform",
                value=False,
                key="apply_boxcox"
            )
        
        with col3:
            config.apply_yeojohnson = st.checkbox(
                "Yeo-Johnson Transform",
                value=False,
                key="apply_yeojohnson"
            )
    
    with tabs[3]:  # Balancing
        st.markdown("### ⚖️ Rééquilibrage des Classes")
        
        balancing_method = st.selectbox(
            "Méthode de rééquilibrage",
            options=['smote', 'adasyn', 'random_over', 'random_under', 'tomek', 'none'],
            index=4,
            key="balancing_method"
        )
        config.balancing_method = balancing_method
        
        if balancing_method != 'none':
            config.balancing_random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=9999,
                value=42,
                key="balancing_random_state"
            )
            
            config.balancing_apply_before_pipeline = st.checkbox(
                "Appliquer avant le pipeline (recommandé)",
                value=True,
                key="balancing_apply_before"
            )
            
            if st.button("📊 Analyser le déséquilibre", key="analyze_balance"):
                df_local = st.session_state.current_data
                target_cols = [c for c in df_local.columns if c.lower() in ['target', 'y', 'label', 'class']]
                if target_cols:
                    target_col = target_cols[0]
                    y = df_local[target_col]
                    
                    from preprocessing.tabular.balancers import ClassBalancer
                    balancer = ClassBalancer()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(balancer.get_class_distribution(y))
                    with col2:
                        st.json(balancer.suggest_balancing_method(y))
                else:
                    st.warning("Aucune colonne cible détectée. Définissez 'target', 'y', 'label' ou 'class'")
    
    with tabs[4]:  # Feature Engineering
        st.markdown("### 🔧 Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config.create_polynomial = st.checkbox(
                "Créer des features polynomiales",
                value=False,
                key="create_polynomial"
            )
            
            if config.create_polynomial:
                config.polynomial_degree = st.slider(
                    "Degré",
                    min_value=2,
                    max_value=4,
                    value=2,
                    key="polynomial_degree"
                )
        
        with col2:
            config.create_interactions = st.checkbox(
                "Créer des interactions",
                value=False,
                key="create_interactions"
            )
        
        config.create_ratios = st.checkbox(
            "Créer des ratios",
            value=False,
            key="create_ratios"
        )
        
        if config.create_ratios:
            config.ratios_max_pairs = st.number_input(
                "Nombre max de paires",
                min_value=10,
                max_value=500,
                value=100,
                key="ratios_max_pairs"
            )
    
    with tabs[5]:  # Réduction
        st.markdown("### 📉 Réduction de Dimensionnalité")
        
        reduction_method = st.selectbox(
            "Méthode de réduction",
            options=['pca', 'lda', 'feature_selection', 'none'],
            index=3,
            key="reduction_method"
        )
        config.reduction_method = reduction_method
        
        if reduction_method == 'pca':
            col1, col2 = st.columns(2)
            with col1:
                config.reduction_components = st.number_input(
                    "Nombre de composantes",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="reduction_components"
                )
            with col2:
                config.reduction_variance_ratio = st.slider(
                    "Variance expliquée",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.95,
                    key="reduction_variance_ratio"
                )
        
        elif reduction_method == 'feature_selection':
            config.feature_selection_method = st.selectbox(
                "Méthode de sélection",
                options=['variance', 'correlation', 'importance'],
                index=2,
                key="feature_selection_method"
            )
            
            config.feature_selection_k = st.number_input(
                "Nombre de features à sélectionner",
                min_value=1,
                max_value=100,
                value=20,
                key="feature_selection_k"
            )
    
    # Bouton de validation
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🚀 Appliquer le preprocessing", type="primary", key="apply_preprocessing"):
            process_data(config)


def process_data(config: PreprocessingConfig):
    """Appliquer le preprocessing"""
    if st.session_state.current_data is None:
        st.error("❌ Aucune données à traiter")
        return
    
    df = st.session_state.current_data
    
    with st.spinner("⏳ Prétraitement en cours..."):
        try:
            builder = PipelineBuilder(config)
            
            if config.balancing_method != 'none' and config.balancing_apply_before_pipeline:
                target_cols = [c for c in df.columns if c.lower() in ['target', 'y', 'label', 'class']]
                if target_cols:
                    target_col = target_cols[0]
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    X_balanced, y_balanced = builder.apply_balancing(X, y)
                    df_processed = pd.concat([X_balanced, y_balanced.to_frame()], axis=1)
                    st.info(f"⚖️ Rééquilibrage appliqué: {len(df)} → {len(df_processed)} lignes")
                else:
                    df_processed = df
                    st.warning("⚠️ Aucune colonne cible trouvée pour le rééquilibrage")
            else:
                df_processed = df
            
            pipeline = builder.build_pipeline(df_processed, None)
            df_transformed = pipeline.fit_transform(df_processed)
            
            df_transformed = fix_dataframe_complete(df_transformed)
            
            st.session_state.processed_data = df_transformed
            
            upload_manager = st.session_state.upload_manager
            if st.session_state.selected_file_id:
                upload_manager.save_processed(
                    df_transformed,
                    st.session_state.selected_file_id,
                    format='csv'
                )
            
            st.markdown(
                f'<div class="success-box">✅ Preprocessing terminé !<br>'
                f'Données: {len(df)} → {len(df_transformed)} lignes, '
                f'{len(df.columns)} → {len(df_transformed.columns)} colonnes</div>',
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors du preprocessing: {str(e)}")
            st.exception(e)


def render_processed_data():
    """Afficher les données transformées"""
    st.markdown('<p class="sub-header">📊 Données Transformées</p>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.info("ℹ️ Aucune donnée transformée. Lancez le preprocessing.")
        return
    
    df = st.session_state.processed_data
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Lignes transformées", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes transformées", f"{len(df.columns)}")
    with col3:
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Mémoire", f"{memory:.1f} MB")
    
    # Aperçu
    st.dataframe(
        safe_display_dataframe(df),
        width='stretch',
        height=400
    )
    
    # Téléchargement
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        import io
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_data = parquet_buffer.getvalue()
        st.download_button(
            label="📥 Télécharger Parquet",
            data=parquet_data,
            file_name="processed_data.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )


def render_visualization():
    """Visualisation des données"""
    st.markdown('<p class="sub-header">📈 Visualisation</p>', unsafe_allow_html=True)
    
    df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.current_data
    
    if df is None:
        st.info("ℹ️ Chargez des données pour visualiser")
        return
    
    # Corriger le DataFrame pour la visualisation
    df = fix_dataframe_for_arrow(df)
    
    visualizer = DataVisualizer()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution",
        "🔍 Corrélation",
        "📦 Outliers",
        "📉 PCA"
    ])
    
    with tab1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_cols = st.multiselect(
                "Sélectionner les colonnes",
                options=numeric_cols,
                default=list(numeric_cols[:min(6, len(numeric_cols))]),
                key="dist_cols"
            )
            
            if selected_cols:
                fig = visualizer.plot_distribution(df[selected_cols], n_cols=3)
                st.pyplot(fig)
            else:
                st.warning("Sélectionnez des colonnes")
        else:
            st.warning("Aucune colonne numérique")
    
    with tab2:
        fig = visualizer.plot_correlation_matrix(df)
        st.pyplot(fig)
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_cols = st.multiselect(
                "Sélectionner les colonnes pour les outliers",
                options=numeric_cols,
                default=list(numeric_cols[:min(3, len(numeric_cols))]),
                key="outlier_cols"
            )
            
            if selected_cols:
                fig = visualizer.plot_outliers(df[selected_cols])
                st.pyplot(fig)
            else:
                st.warning("Sélectionnez des colonnes")
        else:
            st.warning("Aucune colonne numérique")
    
    with tab4:
        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
            try:
                target_cols = [c for c in df.columns if c.lower() in ['target', 'y', 'label', 'class']]
                y = df[target_cols[0]] if target_cols else None
                
                fig = visualizer.plot_pca(df, y)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur PCA: {str(e)}")
        else:
            st.warning("Besoin d'au moins 2 colonnes numériques pour la PCA")


def render_sidebar():
    """Sidebar avec informations"""
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
        
        st.markdown("### 🧭 Navigation")
        pages = {
            "📤 Upload": "upload",
            "🔍 Preview": "preview",
            "⚙️ Config": "config",
            "📊 Processed": "processed",
            "📈 Visualize": "visualize"
        }
        
        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Info")
        st.caption("Version 1.0.0")
        st.caption("Made with ❤️ using Streamlit")


def main():
    """Fonction principale"""
    initialize_session_state()
    
    render_sidebar()
    
    render_header()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "🔍 Preview",
        "⚙️ Preprocess",
        "📊 Processed",
        "📈 Visualize"
    ])
    
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