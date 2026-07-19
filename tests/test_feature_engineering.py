# tests/test_feature_engineering.py
import numpy as np
import pandas as pd
import pytest

from datakit.preprocessing.tabular.feature_engineering import (
    AggregationFeatureCreator,
    DateFeatureCreator,
    InteractionFeatureCreator,
    PolynomialFeatureCreator,
    RatioFeatureCreator,
)


class TestPolynomialFeatureCreator:

    def test_polynomial_creation(self, sample_data_without_nan):
        """Tester la création de features polynomiales"""
        creator = PolynomialFeatureCreator(degree=2, columns=['numeric_1', 'numeric_2'])

        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        assert 'numeric_1' in transformed.columns
        assert 'numeric_2' in transformed.columns
        assert 'numeric_1^2' in transformed.columns
        assert 'numeric_1 numeric_2' in transformed.columns
        assert 'numeric_2^2' in transformed.columns

    def test_polynomial_max_features_guard(self, sample_data_without_nan):
        """Tester la limite sur le NOMBRE de colonnes en entrée.
        Sans `columns` explicite, toutes les colonnes numériques du fixture
        sont candidates (numeric_1/2/3 + target, qui est un int) : 4 > 2."""
        creator = PolynomialFeatureCreator(degree=2, max_features=2, max_output_features=10)

        with pytest.raises(ValueError):
            creator.fit(sample_data_without_nan)

    def test_polynomial_max_output_features_guard(self):
        """Tester la limite sur le NOMBRE de features EN SORTIE, un garde-fou
        distinct du précédent (peu de colonnes en entrée, mais un degré élevé
        qui fait exploser combinatoirement le nombre de sorties)."""
        data = pd.DataFrame({
            'a': np.random.randn(10),
            'b': np.random.randn(10),
            'c': np.random.randn(10),
        })
        creator = PolynomialFeatureCreator(degree=5, max_features=50, max_output_features=5)

        with pytest.raises(ValueError):
            creator.fit(data)

    def test_interaction_only(self, sample_data_without_nan):
        """Tester la création de features d'interaction uniquement"""
        creator = PolynomialFeatureCreator(
            degree=2, columns=['numeric_1', 'numeric_2'], interaction_only=True
        )

        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        assert 'numeric_1^2' not in transformed.columns
        assert 'numeric_2^2' not in transformed.columns
        assert 'numeric_1 numeric_2' in transformed.columns


class TestInteractionFeatureCreator:

    def test_interaction_creation(self, sample_data_without_nan):
        """Tester la création de features d'interaction"""
        creator = InteractionFeatureCreator(columns=['numeric_1', 'numeric_2', 'numeric_3'])

        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        assert 'numeric_1*numeric_2' in transformed.columns
        assert 'numeric_1*numeric_3' in transformed.columns
        assert 'numeric_2*numeric_3' in transformed.columns

    def test_max_interactions(self, sample_data_without_nan):
        """Tester la limite de niveau d'interaction"""
        creator = InteractionFeatureCreator(
            columns=['numeric_1', 'numeric_2', 'numeric_3'], max_interactions=3
        )

        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        assert 'numeric_1*numeric_2*numeric_3' in transformed.columns

    def test_max_output_features_guard(self):
        """Tester la limite sur le nombre total de combinaisons produites,
        jamais exercée jusqu'ici. Avec 4 colonnes et max_interactions=4 :
        C(4,2)+C(4,3)+C(4,4) = 6+4+1 = 11 combinaisons > max_output_features=5."""
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12],
        })
        creator = InteractionFeatureCreator(max_interactions=4, max_output_features=5)

        with pytest.raises(ValueError):
            creator.fit(data)


class TestRatioFeatureCreator:

    def test_ratio_creation(self, sample_data_without_nan):
        """Tester la création de ratios"""
        creator = RatioFeatureCreator(columns=['numeric_1', 'numeric_2'])

        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        assert 'numeric_1_over_numeric_2' in transformed.columns
        assert 'numeric_2_over_numeric_1' in transformed.columns
        assert not transformed['numeric_1_over_numeric_2'].isna().all()

    def test_max_pairs_truncates_and_warns(self, sample_data_without_nan):
        """Tester la limite de paires. 3 colonnes -> 3 paires possibles ;
        max_pairs=2 doit en garder exactement 2 (donc 4 colonnes de ratio,
        pas 'au plus 4') et émettre un avertissement explicite."""
        creator = RatioFeatureCreator(
            columns=['numeric_1', 'numeric_2', 'numeric_3'], max_pairs=2
        )

        with pytest.warns(RuntimeWarning):
            creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)

        ratio_cols = [c for c in transformed.columns if '_over_' in c]
        assert len(ratio_cols) == 4


class TestAggregationFeatureCreator:

    def test_aggregation_creation(self, sample_data_without_nan):
        """Tester la création de features d'agrégation (présence des colonnes)"""
        data = sample_data_without_nan.copy()
        data['group'] = np.random.choice(['A', 'B', 'C'], len(data))

        creator = AggregationFeatureCreator(
            group_column='group',
            agg_columns=['numeric_1', 'numeric_2'],
            aggregations=['mean', 'sum'],
        )

        creator.fit(data)
        transformed = creator.transform(data)

        assert 'numeric_1_mean' in transformed.columns
        assert 'numeric_1_sum' in transformed.columns
        assert 'numeric_2_mean' in transformed.columns

    def test_aggregation_values_match_manual_groupby(self, sample_data_without_nan):
        """Au-delà de la présence des colonnes : vérifie que les VALEURS
        produites correspondent à un groupby pandas manuel sur les mêmes
        données. Un mapping groupe->valeur mal câblé passerait inaperçu avec
        seulement le test précédent."""
        data = sample_data_without_nan.copy()
        data['group'] = np.random.choice(['A', 'B', 'C'], len(data))

        creator = AggregationFeatureCreator(
            group_column='group', agg_columns=['numeric_1'], aggregations=['mean']
        )
        creator.fit(data)
        transformed = creator.transform(data)

        expected = data.groupby('group')['numeric_1'].transform('mean')
        pd.testing.assert_series_equal(
            transformed['numeric_1_mean'], expected.rename('numeric_1_mean'), check_dtype=False
        )

    def test_group_column_required(self, sample_data_without_nan):
        """Tester que group_column est requis"""
        creator = AggregationFeatureCreator()

        with pytest.raises(ValueError):
            creator.fit(sample_data_without_nan)


class TestDateFeatureCreator:

    def test_date_features(self):
        """Tester la création de features à partir de dates déjà typées datetime64"""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
        })

        creator = DateFeatureCreator(
            date_columns=['date'],
            create_year=True,
            create_month=True,
            create_day=True,
            create_dayofweek=True,
            create_quarter=True,
            create_is_weekend=True,
        )

        creator.fit(data)
        transformed = creator.transform(data)

        assert 'date_year' in transformed.columns
        assert 'date_month' in transformed.columns
        assert 'date_day' in transformed.columns
        assert 'date_dayofweek' in transformed.columns
        assert 'date_quarter' in transformed.columns
        assert 'date_is_weekend' in transformed.columns

        assert transformed['date_year'].iloc[0] == 2020
        assert transformed['date_month'].iloc[0] == 1
        assert transformed['date_day'].iloc[0] == 1

    def test_string_date_column_is_parsed(self):
        """Une colonne de dates fournie en string (pas encore convertie en
        datetime64) doit être parsée automatiquement au transform, via
        _as_datetime / pd.to_datetime. Jamais testé jusqu'ici."""
        data = pd.DataFrame({
            'date': ['2021-06-01', '2021-06-02', '2021-06-03'],
            'value': [1, 2, 3],
        })

        creator = DateFeatureCreator(date_columns=['date'], create_year=True, create_month=True)
        creator.fit(data)
        transformed = creator.transform(data)

        assert transformed['date_year'].iloc[0] == 2021
        assert transformed['date_month'].iloc[0] == 6

    def test_unparsable_date_column_is_skipped_with_warning(self):
        """Une colonne déclarée comme date mais dont le contenu n'est pas
        convertible doit être ignorée proprement (aucune feature créée) et
        signalée par un warning, plutôt que de planter."""
        data = pd.DataFrame({
            'date': ['not_a_date', 'still_not_a_date', 'nope'],
            'value': [1, 2, 3],
        })

        creator = DateFeatureCreator(date_columns=['date'], create_year=True)
        creator.fit(data)

        with pytest.warns(RuntimeWarning):
            transformed = creator.transform(data)

        assert 'date_year' not in transformed.columns

    def test_auto_detect_enabled_finds_datetime_columns(self):
        """auto_detect=True doit repérer automatiquement les colonnes
        datetime64 sans qu'on liste `date_columns` explicitement. Jamais
        testé jusqu'ici (seul le cas désactivé l'était)."""
        data = pd.DataFrame({
            'date': pd.date_range('2022-03-01', periods=5, freq='D'),
            'value': np.random.randn(5),
        })

        creator = DateFeatureCreator(auto_detect=True, create_year=True)
        creator.fit(data)
        transformed = creator.transform(data)

        assert 'date_year' in transformed.columns
        assert transformed['date_year'].iloc[0] == 2022

    def test_auto_detect_disabled_warns_and_does_nothing(self):
        """Tester que l'auto-détection désactivée par défaut n'ajoute aucune
        colonne ET émet l'avertissement documenté dans le code source."""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'value': np.random.randn(10),
        })

        creator = DateFeatureCreator(auto_detect=False)

        with pytest.warns(RuntimeWarning):
            creator.fit(data)
        transformed = creator.transform(data)

        assert len(transformed.columns) == len(data.columns)