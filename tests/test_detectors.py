# tests/test_detectors.py
import numpy as np
import pandas as pd
import pytest

from preprocessing.tabular.detectors import (
    CardinalityDetector,
    CorrelationDetector,
    DuplicateDetector,
    MissingValueDetector,
    OutlierDetector,
)


class TestMissingValueDetector:

    def test_detect_missing_values(self, sample_data):
        """Tester la détection des valeurs manquantes"""
        detector = MissingValueDetector(threshold=0.01)
        detector.fit(sample_data)

        assert len(detector.problems) > 0

        report = detector.get_report()
        assert 'problems' in report
        assert 'summary' in report

    def test_no_missing_values(self):
        """Tester avec des données sans valeurs manquantes"""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': ['x', 'y', 'z', 'w'],
        })

        detector = MissingValueDetector()
        detector.fit(data)

        assert len(detector.problems) == 0


class TestOutlierDetector:

    def test_detect_outliers_iqr(self, sample_data):
        """Tester la détection des outliers avec IQR"""
        detector = OutlierDetector(method='iqr', threshold=1.5)
        detector.fit(sample_data)

        assert len(detector.problems) > 0
        assert len(detector.outlier_stats) > 0

    def test_detect_outliers_zscore(self, sample_data):
        """Tester la détection des outliers avec z-score"""
        detector = OutlierDetector(method='zscore', threshold=3)
        detector.fit(sample_data)

        assert len(detector.problems) > 0

    def test_zscore_ignores_constant_column_without_crashing(self):
        """Une colonne à variance nulle (std=0) ne doit pas faire planter le
        détecteur en z-score (division par zéro) : le code source gère ce cas
        en retournant None et en ignorant la colonne. Non vérifié jusqu'ici."""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5, 5],
            'normal': [1, 2, 3, 100, 2, 3],
        })

        detector = OutlierDetector(method='zscore', threshold=3)
        detector.fit(data)  # ne doit lever aucune exception

        assert 'constant' not in detector.outlier_stats


class TestCorrelationDetector:

    def test_detect_correlations(self):
        """Tester la détection des corrélations.
        Générateur local (default_rng) plutôt que np.random.seed() global,
        pour un résultat reproductible indépendant de l'ordre d'exécution
        des autres tests."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            'a': rng.standard_normal(100),
            'b': rng.standard_normal(100) * 2,
            'c': rng.standard_normal(100) * 0.5,
            'd': rng.standard_normal(100),
        })
        data['e'] = data['a'] * 2 + rng.standard_normal(100) * 0.1

        detector = CorrelationDetector(threshold=0.8)
        detector.fit(data)

        assert len(detector.problems) > 0

    def test_no_correlations(self):
        """Tester sans corrélations (colonnes indépendantes, seed fixe)."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            'a': rng.standard_normal(100),
            'b': rng.standard_normal(100),
            'c': rng.standard_normal(100),
        })

        detector = CorrelationDetector(threshold=0.9)
        detector.fit(data)

        assert len(detector.problems) == 0

    def test_fewer_than_two_numeric_columns_does_not_crash(self):
        """Avec moins de 2 colonnes numériques, le détecteur ne peut rien
        corréler : il doit sortir proprement plutôt que planter."""
        data = pd.DataFrame({
            'only_numeric': [1, 2, 3, 4],
            'categorical': ['a', 'b', 'c', 'd'],
        })

        detector = CorrelationDetector(threshold=0.8)
        detector.fit(data)  # ne doit lever aucune exception

        assert len(detector.problems) == 0


class TestCardinalityDetector:

    def test_detect_high_cardinality(self, sample_data):
        """Tester la détection de cardinalité élevée"""
        detector = CardinalityDetector(max_categories=3)
        detector.fit(sample_data)

        assert len(detector.problems) > 0

        problem_columns = [p.get('column') for p in detector.problems]
        assert 'categorical_1' in problem_columns

    def test_column_within_limit_not_flagged(self, sample_data):
        """categorical_2 a 3 catégories (X, Y, Z) : avec max_categories=3,
        elle ne doit pas être signalée (limite non dépassée, `>` strict)."""
        detector = CardinalityDetector(max_categories=3)
        detector.fit(sample_data)

        problem_columns = [p.get('column') for p in detector.problems]
        assert 'categorical_2' not in problem_columns


class TestDuplicateDetector:

    def test_detect_duplicates(self):
        """Tester la détection des doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3, 3],
            'b': ['x', 'y', 'y', 'z', 'z'],
        })

        detector = DuplicateDetector()
        detector.fit(data)

        assert len(detector.problems) > 0
        assert detector.duplicate_count == 2

    def test_no_duplicates(self):
        """Tester sans doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': ['x', 'y', 'z', 'w'],
        })

        detector = DuplicateDetector()
        detector.fit(data)

        assert len(detector.problems) == 0
        assert detector.duplicate_count == 0