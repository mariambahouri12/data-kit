import pytest
import pandas as pd
import numpy as np
from preprocessing.tabular.feature_engineering import (
    PolynomialFeatureCreator,
    InteractionFeatureCreator,
    RatioFeatureCreator,
    AggregationFeatureCreator,
    DateFeatureCreator
)


class TestPolynomialFeatureCreator:
    
    def test_polynomial_creation(self, sample_data_without_nan):
        """Tester la création de features polynomiales"""
        creator = PolynomialFeatureCreator(
            degree=2,
            columns=['numeric_1', 'numeric_2']
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        # Les colonnes originales sont conservées
        assert 'numeric_1' in transformed.columns
        assert 'numeric_2' in transformed.columns
        
        # Vérifier que les features polynomiales sont créées
        assert 'numeric_1^2' in transformed.columns
        assert 'numeric_1 numeric_2' in transformed.columns
        assert 'numeric_2^2' in transformed.columns
    
    def test_polynomial_max_features(self, sample_data_without_nan):
        """Tester la limite de features"""
        creator = PolynomialFeatureCreator(
            degree=2,
            max_features=2,
            max_output_features=10
        )
        
        with pytest.raises(ValueError):
            creator.fit(sample_data_without_nan)
    
    def test_interaction_only(self, sample_data_without_nan):
        """Tester la création de features d'interaction uniquement"""
        creator = PolynomialFeatureCreator(
            degree=2,
            columns=['numeric_1', 'numeric_2'],
            interaction_only=True
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        assert 'numeric_1^2' not in transformed.columns
        assert 'numeric_2^2' not in transformed.columns
        assert 'numeric_1 numeric_2' in transformed.columns


class TestInteractionFeatureCreator:
    
    def test_interaction_creation(self, sample_data_without_nan):
        """Tester la création de features d'interaction"""
        creator = InteractionFeatureCreator(
            columns=['numeric_1', 'numeric_2', 'numeric_3']
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        assert 'numeric_1*numeric_2' in transformed.columns
        assert 'numeric_1*numeric_3' in transformed.columns
        assert 'numeric_2*numeric_3' in transformed.columns
    
    def test_max_interactions(self, sample_data_without_nan):
        """Tester la limite de niveau d'interaction"""
        creator = InteractionFeatureCreator(
            columns=['numeric_1', 'numeric_2', 'numeric_3'],
            max_interactions=3
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        assert 'numeric_1*numeric_2*numeric_3' in transformed.columns


class TestRatioFeatureCreator:
    
    def test_ratio_creation(self, sample_data_without_nan):
        """Tester la création de ratios"""
        creator = RatioFeatureCreator(
            columns=['numeric_1', 'numeric_2']
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        assert 'numeric_1_over_numeric_2' in transformed.columns
        assert 'numeric_2_over_numeric_1' in transformed.columns
        assert not transformed['numeric_1_over_numeric_2'].isna().all()
    
    def test_max_pairs(self, sample_data_without_nan):
        """Tester la limite de paires"""
        creator = RatioFeatureCreator(
            columns=['numeric_1', 'numeric_2', 'numeric_3'],
            max_pairs=2
        )
        
        creator.fit(sample_data_without_nan)
        transformed = creator.transform(sample_data_without_nan)
        
        ratio_cols = [c for c in transformed.columns if '_over_' in c]
        assert len(ratio_cols) <= 4


class TestAggregationFeatureCreator:
    
    def test_aggregation_creation(self, sample_data_without_nan):
        """Tester la création de features d'agrégation"""
        data = sample_data_without_nan.copy()
        data['group'] = np.random.choice(['A', 'B', 'C'], len(data))
        
        creator = AggregationFeatureCreator(
            group_column='group',
            agg_columns=['numeric_1', 'numeric_2'],
            aggregations=['mean', 'sum']
        )
        
        creator.fit(data)
        transformed = creator.transform(data)
        
        assert 'numeric_1_mean' in transformed.columns
        assert 'numeric_1_sum' in transformed.columns
        assert 'numeric_2_mean' in transformed.columns
    
    def test_group_column_required(self, sample_data_without_nan):
        """Tester que group_column est requis"""
        creator = AggregationFeatureCreator()
        
        with pytest.raises(ValueError):
            creator.fit(sample_data_without_nan)


class TestDateFeatureCreator:
    
    def test_date_features(self):
        """Tester la création de features à partir de dates"""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100)
        })
        
        creator = DateFeatureCreator(
            date_columns=['date'],
            create_year=True,
            create_month=True,
            create_day=True,
            create_dayofweek=True,
            create_quarter=True,
            create_is_weekend=True
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
    
    def test_auto_detect_disabled(self):
        """Tester que l'auto-détection est désactivée par défaut"""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'value': np.random.randn(10)
        })
        
        creator = DateFeatureCreator(auto_detect=False)
        
        creator.fit(data)
        transformed = creator.transform(data)
        
        assert len(transformed.columns) == len(data.columns)