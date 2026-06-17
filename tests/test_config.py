# tests/test_config.py
import pytest
import json
import tempfile
from preprocessing.tabular.config import (
    PreprocessingConfig,
    ImputationMethod,
    ScalingMethod,
    EncodingMethod,
    BalancingMethod,
    TaskType
)


class TestPreprocessingConfig:
    
    def test_default_config(self):
        """Tester la configuration par défaut"""
        config = PreprocessingConfig()
        
        assert config.imputation_method == ImputationMethod.MEDIAN
        assert config.scaling_method == ScalingMethod.STANDARD
        assert config.encoding_method == EncodingMethod.ONE_HOT
        assert config.balancing_method == BalancingMethod.NONE
        assert config.task_type == TaskType.CLASSIFICATION
    
    def test_custom_config(self):
        """Tester la configuration personnalisée"""
        config = PreprocessingConfig(
            imputation_method='knn',
            scaling_method='robust',
            encoding_method='target',
            balancing_method='smote',
            task_type='regression',
            drop_duplicates=True,
            create_polynomial=True,
            polynomial_degree=3
        )
        
        assert config.imputation_method == ImputationMethod.KNN
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encoding_method == EncodingMethod.TARGET
        assert config.balancing_method == BalancingMethod.SMOTE
        assert config.task_type == TaskType.REGRESSION
        assert config.drop_duplicates is True
        assert config.create_polynomial is True
        assert config.polynomial_degree == 3
    
    def test_to_dict(self):
        """Tester la conversion en dictionnaire"""
        config = PreprocessingConfig(
            imputation_method='median',
            scaling_method='standard'
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['imputation_method'] == 'median'
        assert config_dict['scaling_method'] == 'standard'
    
    def test_from_dict(self):
        """Tester la création depuis un dictionnaire"""
        data = {
            'imputation_method': 'knn',
            'scaling_method': 'robust',
            'encoding_method': 'onehot',
            'task_type': 'classification'
        }
        
        config = PreprocessingConfig.from_dict(data)
        
        assert config.imputation_method == ImputationMethod.KNN
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encoding_method == EncodingMethod.ONE_HOT
        assert config.task_type == TaskType.CLASSIFICATION
    
    def test_save_and_load(self):
        """Tester la sauvegarde et le chargement"""
        config = PreprocessingConfig(
            imputation_method='knn',
            scaling_method='robust'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            loaded = PreprocessingConfig.load(f.name)
        
        assert loaded.imputation_method == config.imputation_method
        assert loaded.scaling_method == config.scaling_method
    
    def test_get_active_steps(self):
        """Tester l'obtention des étapes actives"""
        config = PreprocessingConfig(
            imputation_method='median',
            scaling_method='standard',
            encoding_method='onehot',
            outlier_method='iqr',
            balancing_method='smote',
            balancing_apply_before_pipeline=True,  # Le balancing est avant le pipeline
            drop_duplicates=True,
            drop_high_missing=True
        )
        
        steps = config.get_active_steps()
        
        assert 'drop_duplicates' in steps
        assert 'drop_high_missing' in steps
        assert 'imputation' in steps
        assert 'outlier_handling' in steps
        assert 'encoding' in steps
        assert 'scaling' in steps
        # Le balancing n'est PAS dans le pipeline car apply_before=True
        assert 'balancing' not in steps
    
    def test_enum_values(self):
        """Tester les valeurs des enums"""
        assert ImputationMethod.MEAN.value == 'mean'
        assert ImputationMethod.MEDIAN.value == 'median'
        assert ImputationMethod.CONSTANT.value == 'constant'
        
        assert ScalingMethod.STANDARD.value == 'standard'
        assert ScalingMethod.MINMAX.value == 'minmax'
        
        assert EncodingMethod.ONE_HOT.value == 'onehot'
        assert EncodingMethod.TARGET.value == 'target'
        
        assert BalancingMethod.SMOTE.value == 'smote'
        assert BalancingMethod.SMOTE_ENN.value == 'smote_enn'