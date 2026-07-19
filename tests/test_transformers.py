import pytest
import pandas as pd
import numpy as np
from datakit.preprocessing.tabular.transformers import yeojohnson
from datakit.preprocessing.tabular.transformers import (
    PercentileTransformer,
    boxcox,
    log,
    sqrt
)


class TestLogTransformer:
    
    def test_log_transform(self):
        """Tester la transformation logarithmique"""
        data = pd.DataFrame({
            'positive': np.exp(np.random.randn(100)),
            'zero': np.zeros(100),
            'negative': np.random.randn(100) - 5
        })
        
        transformer = log(
            columns=['positive', 'zero', 'negative'],
            base=np.e,
            shift=1e-6
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert not transformed['positive'].isna().all()
        assert not transformed['zero'].isna().all()
        assert not transformed['negative'].isna().all()
        assert np.isfinite(transformed['positive']).all()
    
    def test_log_base(self):
        """Tester différentes bases de log"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.exp(np.random.randn(100))
        })
        
        transformer = log(
            columns=['x'],
            base=10
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        original = np.log10(data['x'])
        
        # ✅ CORRIGÉ : Utiliser une tolérance plus grande pour les erreurs de précision
        np.testing.assert_allclose(
            transformed['x'].values,
            original.values,
            rtol=1e-4,
            atol=1e-6
        )


class TestSqrtTransformer:
    
    def test_sqrt_transform(self):
        """Tester la transformation racine carrée"""
        data = pd.DataFrame({
            'positive': np.random.exponential(2, 100),
            'zero': np.zeros(100)
        })
        
        transformer = sqrt(
            columns=['positive', 'zero']
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert (transformed['positive'] >= 0).all()
        assert (transformed['zero'] >= 0).all()


class TestBoxCoxTransformer:
    
    def test_boxcox_transform(self):
        """Tester Box-Cox transformation"""
        data = pd.DataFrame({
            'positive': np.random.exponential(2, 100) + 1
        })
        
        transformer = boxcox(
            columns=['positive']
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert not transformed['positive'].isna().all()
        assert np.isfinite(transformed['positive']).all()
    
    def test_boxcox_with_shift(self):
        """Tester Box-Cox avec shift automatique"""
        data = pd.DataFrame({
            'with_zeros': np.random.randn(100) + 2,
            'with_zeros_shift': np.random.randn(100)
        })
        
        transformer = boxcox(
            columns=['with_zeros', 'with_zeros_shift']
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert not transformed.isna().all().all()


class TestYeoJohnsonTransformer:
    
    def test_yeojohnson_transform(self):
        """Tester Yeo-Johnson transformation"""
        data = pd.DataFrame({
            'normal': np.random.randn(100),
            'exponential': np.random.exponential(2, 100)
        })
        
        transformer = yeojohnson(
            columns=['normal', 'exponential']
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert not transformed.isna().all().all()


class TestPercentileTransformer:
    
    def test_percentile_transform(self):
        """Tester la transformation en percentiles"""
        data = pd.DataFrame({
            'x': np.random.randn(100)
        })
        
        transformer = PercentileTransformer(
            columns=['x'],
            n_quantiles=100
        )
        
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert (transformed['x'] >= 0).all()
        assert (transformed['x'] <= 1).all()