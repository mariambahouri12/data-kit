# tests/test_detectors.py
import pytest
import pandas as pd
import numpy as np
from preprocessing.tabular.detectors import (
    MissingValueDetector,
    OutlierDetector,
    CorrelationDetector,
    CardinalityDetector,
    DuplicateDetector
)


class TestMissingValueDetector:
    
    def test_detect_missing_values(self, sample_data):
        """Tester la détection des valeurs manquantes"""
        detector = MissingValueDetector(threshold=0.01)
        
        detector.fit(sample_data)
        
        # Vérifier que des problèmes sont détectés
        assert len(detector.problems) > 0
        
        # Vérifier le rapport
        report = detector.get_report()
        assert 'problems' in report
        assert 'summary' in report
    
    def test_no_missing_values(self):
        """Tester avec des données sans valeurs manquantes"""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': ['x', 'y', 'z', 'w']
        })
        
        detector = MissingValueDetector()
        detector.fit(data)
        
        assert len(detector.problems) == 0


class TestOutlierDetector:
    
    def test_detect_outliers_iqr(self, sample_data):
        """Tester la détection des outliers avec IQR"""
        detector = OutlierDetector(method='iqr', threshold=1.5)
        
        detector.fit(sample_data)
        
        # Devrait détecter des outliers dans numeric_2
        assert len(detector.problems) > 0
        
        # Vérifier qu'il y a des statistiques
        assert len(detector.outlier_stats) > 0
    
    def test_detect_outliers_zscore(self, sample_data):
        """Tester la détection des outliers avec z-score"""
        detector = OutlierDetector(method='zscore', threshold=3)
        
        detector.fit(sample_data)
        
        # Les outliers extrêmes sont détectés
        assert len(detector.problems) > 0


class TestCorrelationDetector:
    
    def test_detect_correlations(self):
        """Tester la détection des corrélations"""
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100) * 2,
            'c': np.random.randn(100) * 0.5,
            'd': np.random.randn(100)  # Colonne non corrélée
        })
        
        # Ajouter une corrélation
        data['e'] = data['a'] * 2 + np.random.randn(100) * 0.1
        
        detector = CorrelationDetector(threshold=0.8)
        detector.fit(data)
        
        # Devrait détecter la corrélation entre a et e
        assert len(detector.problems) > 0
    
    def test_no_correlations(self):
        """Tester sans corrélations"""
        data = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })
        
        detector = CorrelationDetector(threshold=0.9)
        detector.fit(data)
        
        assert len(detector.problems) == 0


class TestCardinalityDetector:
    
    def test_detect_high_cardinality(self, sample_data):
        """Tester la détection de cardinalité élevée"""
        detector = CardinalityDetector(max_categories=3)
        
        detector.fit(sample_data)
        
        # categorical_1 a 4 catégories, devrait être détecté
        assert len(detector.problems) > 0
        
        # Vérifier que la colonne problématique est identifiée
        problem_columns = [p.get('column') for p in detector.problems]
        assert 'categorical_1' in problem_columns


class TestDuplicateDetector:
    
    def test_detect_duplicates(self):
        """Tester la détection des doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3, 3],
            'b': ['x', 'y', 'y', 'z', 'z']
        })
        
        detector = DuplicateDetector()
        detector.fit(data)
        
        assert len(detector.problems) > 0
        assert detector.duplicate_count == 2
    
    def test_no_duplicates(self):
        """Tester sans doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': ['x', 'y', 'z', 'w']
        })
        
        detector = DuplicateDetector()
        detector.fit(data)
        
        assert len(detector.problems) == 0
        assert detector.duplicate_count == 0