# tests/test_cleaners.py
import pandas as pd
import pytest

from preprocessing.tabular.cleaners import (
    DuplicateCleaner,
    MissingValueCleaner,
    OutlierCleaner,
)

TOLERANCE = 0.01


def _fit_transform(cleaner, data: pd.DataFrame) -> pd.DataFrame:
    """Raccourci fit + transform, pour éviter de répéter ces deux lignes
    dans chaque test."""
    cleaner.fit(data)
    return cleaner.transform(data)


class TestMissingValueCleaner:

    def test_mean_imputation(self, sample_data):
        """Tester l'imputation par la moyenne"""
        cleaner = MissingValueCleaner(strategy='mean', columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['numeric_1'].isnull().sum() == 0
        assert transformed['numeric_1'].mean() == pytest.approx(
            sample_data['numeric_1'].mean(), abs=TOLERANCE
        )

    def test_median_imputation(self, sample_data):
        """Tester l'imputation par la médiane"""
        cleaner = MissingValueCleaner(strategy='median', columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['numeric_1'].isnull().sum() == 0
        assert transformed['numeric_1'].median() == pytest.approx(
            sample_data['numeric_1'].median(), abs=TOLERANCE
        )

    def test_constant_imputation(self, sample_data):
        """Tester l'imputation par une constante"""
        cleaner = MissingValueCleaner(strategy='constant', fill_value=999, columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)

        nan_mask = sample_data['numeric_1'].isna()
        assert (transformed.loc[nan_mask, 'numeric_1'] == 999).all()

    def test_constant_imputation_default(self, sample_data):
        """Tester l'imputation par constante avec valeur par défaut (0, avec warning)"""
        cleaner = MissingValueCleaner(strategy='constant', columns=['numeric_1'])

        with pytest.warns(RuntimeWarning):
            transformed = _fit_transform(cleaner, sample_data)

        nan_mask = sample_data['numeric_1'].isna()
        assert (transformed.loc[nan_mask, 'numeric_1'] == 0).all()

    def test_drop_rows(self, sample_data):
        """Tester la suppression des lignes avec valeurs manquantes"""
        cleaner = MissingValueCleaner(strategy='drop', columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['numeric_1'].isnull().sum() == 0
        assert len(transformed) < len(sample_data)

    def test_categorical_imputation(self, sample_data):
        """Tester l'imputation des colonnes catégorielles (most_frequent)"""
        cleaner = MissingValueCleaner()
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['categorical_1'].isnull().sum() == 0

    def test_knn_imputation(self, sample_data):
        """Tester l'imputation par KNN"""
        cleaner = MissingValueCleaner(strategy='knn', columns=['numeric_1', 'numeric_2'])
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['numeric_1'].isnull().sum() == 0
        assert transformed['numeric_2'].isnull().sum() == 0

    def test_columns_outside_scope_are_left_untouched(self, sample_data):
        """Si `columns` restreint le traitement à numeric_1, les NaN de
        categorical_1 (hors périmètre) ne doivent pas être touchés."""
        cleaner = MissingValueCleaner(strategy='mean', columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)

        assert transformed['categorical_1'].isnull().sum() == sample_data['categorical_1'].isnull().sum()

    @pytest.mark.xfail(
        reason=(
            "ImputationMethod.MICE est déclaré dans config.py mais n'est pas "
            "implémenté dans MissingValueCleaner._fit_numeric_imputer : il "
            "tombe dans la branche SimpleImputer par défaut, qui ne supporte "
            "pas strategy='mice' et lève une erreur sklearn peu explicite. "
            "Ce test doit passer en XPASS le jour où MICE sera implémenté, "
            "signalant qu'il faut retirer ce xfail."
        ),
        strict=False,
    )
    def test_mice_imputation_not_actually_supported(self, sample_data):
        cleaner = MissingValueCleaner(strategy='mice', columns=['numeric_1'])
        transformed = _fit_transform(cleaner, sample_data)
        assert transformed['numeric_1'].isnull().sum() == 0


class TestOutlierCleaner:

    def test_winsorize_outliers_iqr(self, sample_data):
        """Tester le winsorizing des outliers (méthode IQR)"""
        cleaner = OutlierCleaner(method='iqr', threshold=1.5, action='winsorize', columns=['numeric_2'])
        transformed = _fit_transform(cleaner, sample_data)

        q1 = sample_data['numeric_2'].quantile(0.25)
        q3 = sample_data['numeric_2'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        assert transformed['numeric_2'].max() <= upper_bound

    def test_drop_outliers_iqr(self, sample_data):
        """Tester la suppression des outliers (méthode IQR)"""
        cleaner = OutlierCleaner(method='iqr', threshold=1.5, action='drop', columns=['numeric_2'])
        transformed = _fit_transform(cleaner, sample_data)

        q1 = sample_data['numeric_2'].quantile(0.25)
        q3 = sample_data['numeric_2'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        assert (transformed['numeric_2'] >= lower_bound).all()
        assert (transformed['numeric_2'] <= upper_bound).all()

    def test_drop_outliers_zscore(self, sample_data):
        """Tester la suppression des outliers (méthode Z-score, non couverte jusqu'ici)"""
        cleaner = OutlierCleaner(method='zscore', threshold=3.0, action='drop', columns=['numeric_2'])
        transformed = _fit_transform(cleaner, sample_data)

        mean = sample_data['numeric_2'].mean()
        std = sample_data['numeric_2'].std()
        lower_bound = mean - 3.0 * std
        upper_bound = mean + 3.0 * std

        assert (transformed['numeric_2'] >= lower_bound).all()
        assert (transformed['numeric_2'] <= upper_bound).all()

    def test_unsupported_method_raises_value_error(self):
        """Contrairement à MICE dans MissingValueCleaner, une méthode de
        détection d'outliers inconnue doit échouer explicitement (ValueError)."""
        data = pd.DataFrame({'a': [1, 2, 3, 100]})
        cleaner = OutlierCleaner(method='isolation_forest', columns=['a'])

        with pytest.raises(ValueError):
            cleaner.fit(data)


class TestDuplicateCleaner:

    def test_remove_duplicates(self):
        """Tester la suppression des doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3, 3, 3],
            'b': ['x', 'y', 'y', 'z', 'z', 'z'],
        })

        transformed = _fit_transform(DuplicateCleaner(), data)

        assert len(transformed) == 3

    def test_keep_first(self):
        """Tester keep='first'"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': ['x', 'y', 'y', 'z'],
        })

        transformed = _fit_transform(DuplicateCleaner(keep='first'), data)

        assert transformed.iloc[1]['a'] == 2
        assert len(transformed) == 3

    def test_keep_last(self):
        """Tester keep='last'"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': ['x', 'y', 'y', 'z'],
        })

        transformed = _fit_transform(DuplicateCleaner(keep='last'), data)

        assert transformed.iloc[1]['a'] == 2
        assert len(transformed) == 3

    def test_subset(self):
        """Tester la suppression des doublons sur un sous-ensemble de colonnes"""
        data = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': ['x', 'y', 'x', 'y'],
            'c': [1, 2, 3, 4],
        })

        transformed = _fit_transform(DuplicateCleaner(subset=['a']), data)

        assert len(transformed) == 2