import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datakit.preprocessing.tabular.reducers import FeatureSelector, PCAReducer, LDAReducer
from datakit.preprocessing.tabular.config import TaskType


class TestFeatureSelector:
    
    def test_variance_selection(self, sample_data_without_nan):
        """Tester la sélection par variance"""
        selector = FeatureSelector(
            method='variance',
            threshold=0.1
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        selector.fit(X)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0
        assert transformed.shape[1] <= X.shape[1]
    
    def test_variance_selection_no_numeric(self):
        """Tester la sélection par variance sans colonnes numériques"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'Z']
        })
        
        selector = FeatureSelector(
            method='variance',
            threshold=0.1
        )
        
        selector.fit(data)
        transformed = selector.transform(data)
        
        assert transformed.shape[1] == data.shape[1]
    
    def test_correlation_selection_classification(self, sample_data_without_nan):
        """Tester la sélection par corrélation pour la classification"""
        selector = FeatureSelector(
            method='correlation',
            threshold=0.01,
            task_type='classification'
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        selector.fit(X, y)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0
        assert transformed.shape[1] <= X.shape[1]
        assert len(selector.feature_importances) > 0
    
    def test_correlation_selection_regression(self, sample_data_regression):
        """Tester la sélection par corrélation pour la régression"""
        selector = FeatureSelector(
            method='correlation',
            threshold=0.01,
            task_type='regression'
        )
        
        X = sample_data_regression.drop('target', axis=1)
        y = sample_data_regression['target']
        
        selector.fit(X, y)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0
        assert transformed.shape[1] <= X.shape[1]
    
    def test_correlation_selection_with_k(self, sample_data_without_nan):
        """Tester la sélection par corrélation avec k features"""
        selector = FeatureSelector(
            method='correlation',
            k=2,
            task_type='classification'
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        selector.fit(X, y)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] == 2 + X.select_dtypes(exclude=[np.number]).shape[1]
    
    def test_importance_selection_classification(self, sample_data_without_nan):
        """Tester la sélection par importance pour la classification"""
        selector = FeatureSelector(
            method='importance',
            threshold=0.01,
            task_type='classification'
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        selector.fit(X, y)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0
        assert transformed.shape[1] <= X.shape[1]
        assert len(selector.feature_importances) > 0
    
    def test_importance_selection_regression(self, sample_data_regression):
        """Tester la sélection par importance pour la régression"""
        selector = FeatureSelector(
            method='importance',
            threshold=0.01,
            task_type='regression'
        )
        
        X = sample_data_regression.drop('target', axis=1)
        y = sample_data_regression['target']
        
        selector.fit(X, y)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0
        assert transformed.shape[1] <= X.shape[1]
    
    def test_rfe_selection_classification(self, sample_data_without_nan):
        """Tester la sélection par RFE pour la classification"""
        selector = FeatureSelector(
            method='rfe',
            k=3,
            task_type='classification'
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        selector.fit(X_encoded, y)
        transformed = selector.transform(X_encoded)
        
        assert transformed.shape[1] == 3
    
    def test_rfe_selection_regression(self, sample_data_regression):
        """Tester la sélection par RFE pour la régression"""
        selector = FeatureSelector(
            method='rfe',
            k=2,
            task_type='regression'
        )
        
        X = sample_data_regression.drop('target', axis=1)
        y = sample_data_regression['target']
        
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        selector.fit(X_encoded, y)
        transformed = selector.transform(X_encoded)
        
        assert transformed.shape[1] == 2
    
    def test_selection_with_columns_subset(self, sample_data_without_nan):
        """Tester la sélection sur un sous-ensemble de colonnes"""
        selector = FeatureSelector(
            method='variance',
            threshold=0.1,
            columns=['numeric_1', 'numeric_2']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        selector.fit(X)
        transformed = selector.transform(X)
        
        assert 'numeric_1' in transformed.columns or 'numeric_2' in transformed.columns
    
    def test_selection_without_y(self, sample_data_without_nan):
        """Tester la sélection sans target"""
        selector = FeatureSelector(
            method='variance',
            threshold=0.1
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        selector.fit(X)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] > 0


class TestPCAReducer:
    
    def test_pca_reduction(self, sample_data_without_nan):
        """Tester la réduction par PCA"""
        reducer = PCAReducer(
            n_components=2,
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        reducer.fit(X)
        transformed = reducer.transform(X)
        
        assert transformed.shape[1] == X.shape[1] - 3 + 2
        assert 'PC1' in transformed.columns
        assert 'PC2' in transformed.columns
        assert 'numeric_1' not in transformed.columns
    
    def test_pca_variance_ratio(self, sample_data_without_nan):
        """Tester la réduction par PCA avec ratio de variance"""
        reducer = PCAReducer(
            variance_ratio=0.95,
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        reducer.fit(X)
        transformed = reducer.transform(X)
        
        assert transformed.shape[1] > 0
        explained_variance = reducer.get_explained_variance()
        assert explained_variance.get('total_variance', 0) >= 0.95
    
    def test_pca_no_numeric_columns(self):
        """Tester PCA sans colonnes numériques"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'Z']
        })
        
        reducer = PCAReducer(n_components=2)
        reducer.fit(data)
        transformed = reducer.transform(data)
        
        assert transformed.shape[1] == data.shape[1]
    
    def test_pca_explained_variance(self, sample_data_without_nan):
        """Tester l'obtention de la variance expliquée"""
        reducer = PCAReducer(
            n_components=2,
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        reducer.fit(X)
        variance = reducer.get_explained_variance()
        
        assert 'explained_variance_ratio' in variance
        assert 'cumulative_variance' in variance
        assert 'total_variance' in variance
        assert len(variance['explained_variance_ratio']) == 2
    
    def test_pca_with_none_components(self, sample_data_without_nan):
        """Tester PCA sans spécifier n_components"""
        reducer = PCAReducer(
            variance_ratio=0.9,
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        reducer.fit(X)
        transformed = reducer.transform(X)
        
        assert transformed.shape[1] > 0


class TestLDAReducer:
    
    def test_lda_reduction_classification(self, sample_data_without_nan):
        """Tester la réduction par LDA pour la classification"""
        reducer = LDAReducer(
            n_components=1,
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        reducer.fit(X, y)
        transformed = reducer.transform(X)
        
        assert 'LD1' in transformed.columns
        assert transformed.shape[1] == X.shape[1] - 3 + 1
    
    def test_lda_reduction_multiple_components(self, sample_data):
        """Tester la réduction par LDA avec plusieurs composantes"""
        np.random.seed(42)
        n_samples = 300
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.choice([0, 1, 2], n_samples))
        
        reducer = LDAReducer(
            n_components=2,
            columns=['feature_1', 'feature_2', 'feature_3', 'feature_4']
        )
        
        reducer.fit(data, y)
        transformed = reducer.transform(data)
        
        assert 'LD1' in transformed.columns
        assert 'LD2' in transformed.columns
    
    def test_lda_requires_y(self, sample_data_without_nan):
        """Tester que LDA nécessite une target"""
        reducer = LDAReducer(
            n_components=1,
            columns=['numeric_1', 'numeric_2']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        
        with pytest.raises(ValueError):
            reducer.fit(X)
    
    def test_lda_not_enough_classes(self):
        """Tester LDA avec pas assez de classes"""
        data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        y = pd.Series(np.zeros(100))
        
        reducer = LDAReducer(
            n_components=1,
            columns=['feature_1', 'feature_2']
        )
        
        with pytest.raises(ValueError):
            reducer.fit(data, y)
    
    def test_lda_no_numeric_columns(self, sample_data):
        """Tester LDA sans colonnes numériques"""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'Z']
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        reducer = LDAReducer(n_components=1)
        reducer.fit(data, y)
        transformed = reducer.transform(data)
        
        assert transformed.shape[1] == data.shape[1]
    
    def test_lda_with_auto_components(self, sample_data_without_nan):
        """Tester LDA avec détermination automatique du nombre de composantes"""
        reducer = LDAReducer(
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        y = sample_data_without_nan['target']
        
        reducer.fit(X, y)
        transformed = reducer.transform(X)
        
        assert 'LD1' in transformed.columns
        assert transformed.shape[1] == X.shape[1] - 3 + 1


class TestFeatureSelectorEdgeCases:
    
    def test_selector_with_empty_columns(self, sample_data_without_nan):
        """Tester le sélecteur avec des colonnes vides"""
        selector = FeatureSelector(
            method='variance',
            threshold=0.1,
            columns=[]
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        selector.fit(X)
        transformed = selector.transform(X)
        
        assert transformed.shape[1] == X.shape[1]
    
    def test_selector_with_high_threshold(self, sample_data_without_nan):
        """
        Tester le sélecteur avec un seuil très élevé.
        Aucune colonne ne dépasse le seuil → ValueError attendue.
        """
        selector = FeatureSelector(
            method='variance',
            threshold=100.0
        )
        
        X = sample_data_without_nan.drop('target', axis=1)
        
        # ✅ Vérifier que l'exception est bien levée
        with pytest.raises(ValueError) as exc_info:
            selector.fit(X)
        
        # ✅ Vérifier que le message d'erreur est correct
        assert "No feature in X meets the variance threshold" in str(exc_info.value)