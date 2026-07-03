# tests/test_balancers.py
import pytest
import pandas as pd
import numpy as np
from preprocessing.tabular.balancers import ClassBalancer
from preprocessing.tabular.config import BalancingMethod


class TestClassBalancer:
    
    def test_smote_balancing(self, sample_data_imbalanced):
        """Tester SMOTE pour le rééquilibrage"""
        balancer = ClassBalancer(
            method='smote',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        # Vérifier le déséquilibre initial
        initial_counts = y.value_counts()
        assert initial_counts[0] > initial_counts[1]
        
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        # Vérifier l'équilibrage
        resampled_counts = y_resampled.value_counts()
        assert abs(resampled_counts[0] - resampled_counts[1]) <= 1
    
    def test_random_over_sampling(self, sample_data_imbalanced):
        """Tester le sur-échantillonnage aléatoire"""
        balancer = ClassBalancer(
            method='random_over',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        assert len(X_resampled) > len(X)
        counts = y_resampled.value_counts()
        assert abs(counts[0] - counts[1]) <= 1
    
    def test_random_under_sampling(self, sample_data_imbalanced):
        """Tester le sous-échantillonnage aléatoire"""
        balancer = ClassBalancer(
            method='random_under',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        assert len(X_resampled) < len(X)
        counts = y_resampled.value_counts()
        assert abs(counts[0] - counts[1]) <= 1
    
    def test_smote_enn(self, sample_data_imbalanced):
        """Tester SMOTE-ENN"""
        balancer = ClassBalancer(
            method='smote_enn',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        counts = y_resampled.value_counts()
        assert len(counts) == 2
        assert len(X_resampled) > 0
    
    def test_smote_tomek(self, sample_data_imbalanced):
        """Tester SMOTE-Tomek"""
        balancer = ClassBalancer(
            method='smote_tomek',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        counts = y_resampled.value_counts()
        assert len(counts) == 2
        assert len(X_resampled) > 0
    
    def test_get_balance_report(self, sample_data_imbalanced):
        """Tester l'obtention du rapport d'équilibrage"""
        balancer = ClassBalancer(
            method='smote',
            random_state=42
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        balancer.fit_resample(X, y)
        report = balancer.get_balance_report()
        
        assert 'method' in report
        assert 'original_shape' in report
        assert 'balanced_shape' in report
        assert report['method'] == 'smote'
    
    def test_get_class_distribution(self, sample_data_imbalanced):
        """Tester l'obtention de la distribution des classes"""
        balancer = ClassBalancer()
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        distribution = balancer.get_class_distribution(y)
        
        assert 'n_classes' in distribution
        assert 'counts' in distribution
        assert 'imbalance_ratio' in distribution
        assert distribution['n_classes'] == 2
    
    def test_suggest_balancing_method(self, sample_data_imbalanced):
        """Tester la suggestion de méthode d'équilibrage"""
        balancer = ClassBalancer()
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        suggestions = balancer.suggest_balancing_method(y)
        
        assert 'imbalance_ratio' in suggestions
        assert 'severity' in suggestions
        assert 'suggestions' in suggestions
        assert len(suggestions['suggestions']) > 0
    
    def test_balancer_with_categorical_target(self, sample_data_imbalanced):
        """Tester avec une cible catégorielle"""
        balancer = ClassBalancer(
            method='smote',
            random_state=42
        )
        
        # Créer une target catégorielle
        np.random.seed(42)
        y_categorical = pd.Series(
            np.random.choice(['class_0', 'class_1'], 1000, p=[0.9, 0.1]),
            dtype='object'  # ✅ Forcer en object pour éviter StringDtype
        )
        
        X = sample_data_imbalanced.drop('target', axis=1)
        
        X_resampled, y_resampled = balancer.fit_resample(X, y_categorical)
        
        assert len(X_resampled) > 0
        # ✅ Vérifier que c'est un objet (pandas 2.0)
        assert y_resampled.dtype == 'object' or y_resampled.dtype.name == 'object'