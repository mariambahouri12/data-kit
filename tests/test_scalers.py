# tests/test_scalers.py
import pytest
import pandas as pd
import numpy as np
from datakit.preprocessing.tabular.transformers.scalers import FeatureScaler, PowerTransformerWrapper


class TestFeatureScaler:
    
    def test_standard_scaler(self, sample_data):
        """Tester StandardScaler"""
        scaler = FeatureScaler(
            method='standard',
            columns=['numeric_1', 'numeric_2']
        )
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Vérifier la moyenne ~0
        assert abs(transformed['numeric_1'].mean()) < 0.1
        assert abs(transformed['numeric_2'].mean()) < 0.1
        
        # Vérifier l'écart-type ~1
        assert abs(transformed['numeric_1'].std() - 1) < 0.1
        assert abs(transformed['numeric_2'].std() - 1) < 0.1
    
    def test_minmax_scaler(self, sample_data):
        """Tester MinMaxScaler"""
        scaler = FeatureScaler(
            method='minmax',
            columns=['numeric_1', 'numeric_2']
        )
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Vérifier les bornes [0,1]
        assert transformed['numeric_1'].min() >= 0
        assert transformed['numeric_1'].max() <= 1
        assert transformed['numeric_2'].min() >= 0
        assert transformed['numeric_2'].max() <= 1
    
    def test_robust_scaler(self, sample_data):
        """Tester RobustScaler"""
        scaler = FeatureScaler(
            method='robust',
            columns=['numeric_1', 'numeric_2']
        )
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Vérifier la médiane ~0
        assert abs(transformed['numeric_1'].median()) < 0.1
        assert abs(transformed['numeric_2'].median()) < 0.1
    
    def test_inverse_transform(self, sample_data):
        """Tester l'inversion de transformation"""
        scaler = FeatureScaler(
            method='standard',
            columns=['numeric_1']
        )
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        inverted = scaler.inverse_transform(transformed)
        
        # Comparer les valeurs originales (en ignorant les NaN)
        mask = ~sample_data['numeric_1'].isna()
        pd.testing.assert_series_equal(
            inverted.loc[mask, 'numeric_1'],
            sample_data.loc[mask, 'numeric_1'],
            rtol=1e-10
        )
    
    def test_select_columns_auto(self, sample_data):
        """Tester la sélection automatique des colonnes"""
        scaler = FeatureScaler(method='standard')  # Pas de columns spécifiées
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Seules les colonnes numériques doivent être scalées
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert col in transformed.columns
    
    def test_get_scale_params(self, sample_data):
        """Tester l'obtention des paramètres d'échelle"""
        scaler = FeatureScaler(
            method='standard',
            columns=['numeric_1']
        )
        
        scaler.fit(sample_data)
        params = scaler.get_scale_params()
        
        assert 'mean' in params
        assert 'scale' in params


class TestPowerTransformerWrapper:
    
    def test_yeo_johnson(self, sample_data):
        """Tester Yeo-Johnson transformation"""
        transformer = PowerTransformerWrapper(
            method='yeo-johnson',
            columns=['numeric_3']
        )
        
        transformer.fit(sample_data)
        transformed = transformer.transform(sample_data)
        
        # Vérifier que la transformation a eu lieu
        assert not transformed['numeric_3'].isna().all()
        assert transformed['numeric_3'].dtype in ['float64', 'int64']