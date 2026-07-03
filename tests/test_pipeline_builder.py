import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from preprocessing.tabular.pipeline_builder import PipelineBuilder, SimplePipelineBuilder
from preprocessing.tabular.config import PreprocessingConfig, TaskType, ScalingMethod, EncodingMethod


class TestPipelineBuilder:
    
    def test_build_default_pipeline(self, sample_data_without_nan):
        """Tester la construction du pipeline par défaut"""
        config = PreprocessingConfig()
        builder = PipelineBuilder(config)
        pipeline = builder.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0
        
        X_transformed = pipeline.fit_transform(sample_data_without_nan.drop('target', axis=1))
        assert X_transformed.shape[0] > 0
    
    def test_build_with_imputation(self, sample_data_with_nan):
        """Tester le pipeline avec imputation"""
        config = PreprocessingConfig(
            imputation_method='median',
            drop_duplicates=True
        )
        builder = PipelineBuilder(config)
        pipeline = builder.build_pipeline()
        
        X_transformed = pipeline.fit_transform(sample_data_with_nan.drop('target', axis=1))
        
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_build_with_encoding(self, sample_data_without_nan):
        """Tester le pipeline avec encodage"""
        config = PreprocessingConfig(
            encoding_method='onehot',
            encoding_columns=['categorical_1', 'categorical_2']
        )
        builder = PipelineBuilder(config)
        pipeline = builder.build_pipeline()
        
        X_transformed = pipeline.fit_transform(sample_data_without_nan.drop('target', axis=1))
        
        assert 'categorical_1' not in X_transformed.columns
        assert 'categorical_2' not in X_transformed.columns
    
    def test_build_with_scaling(self, sample_data_without_nan):
        """Tester le pipeline avec scaling"""
        config = PreprocessingConfig(
            scaling_method='standard'
        )
        builder = PipelineBuilder(config)
        pipeline = builder.build_pipeline()
        
        X_transformed = pipeline.fit_transform(sample_data_without_nan.drop('target', axis=1))
        
        numeric_cols = sample_data_without_nan.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target' and not X_transformed[col].isna().all():
                assert abs(X_transformed[col].mean()) < 0.1
    
    def test_apply_balancing(self, sample_data_imbalanced):
        """Tester l'application du rééquilibrage"""
        config = PreprocessingConfig(
            balancing_method='smote',
            balancing_random_state=42
        )
        builder = PipelineBuilder(config)
        
        X = sample_data_imbalanced.drop('target', axis=1)
        y = sample_data_imbalanced['target']
        
        X_resampled, y_resampled = builder.apply_balancing(X, y)
        
        original_counts = y.value_counts()
        resampled_counts = y_resampled.value_counts()
        
        imbalance_ratio_original = max(original_counts) / min(original_counts)
        imbalance_ratio_resampled = max(resampled_counts) / min(resampled_counts)
        
        assert imbalance_ratio_resampled < imbalance_ratio_original
    
    def test_build_detection_pipeline(self, sample_data_without_nan):
        """Tester le pipeline de détection"""
        builder = PipelineBuilder()
        detection_pipeline = builder.build_detection_pipeline()
        
        assert isinstance(detection_pipeline, Pipeline)
        
        detection_pipeline.fit(sample_data_without_nan.drop('target', axis=1))
    
    def test_get_step_names(self, sample_data_without_nan):
        """Tester l'obtention des noms des étapes"""
        config = PreprocessingConfig(
            imputation_method='median',
            scaling_method='standard',
            encoding_method='onehot',
            drop_duplicates=True
        )
        builder = PipelineBuilder(config)
        steps = builder.get_step_names()
        
        assert 'drop_duplicates' in steps
        assert 'imputation' in steps
        assert 'encoding' in steps
        assert 'scaling' in steps
    
    def test_get_pipeline_summary(self, sample_data_without_nan):
        """Tester le résumé du pipeline"""
        builder = PipelineBuilder()
        summary = builder.get_pipeline_summary()
        
        assert 'steps' in summary
        assert 'n_steps' in summary
        assert 'config' in summary


class TestSimplePipelineBuilder:
    
    def test_create_default(self):
        """Tester la création du pipeline par défaut"""
        builder = SimplePipelineBuilder.create_default()
        pipeline = builder.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0
    
    def test_create_robust(self):
        """Tester la création du pipeline robuste"""
        builder = SimplePipelineBuilder.create_robust()
        pipeline = builder.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        
        encoding_method = builder.config.encoding_method
        if hasattr(encoding_method, 'value'):
            assert encoding_method.value == 'target'
        else:
            assert str(encoding_method) == 'target'
        
        scaling_method = builder.config.scaling_method
        if hasattr(scaling_method, 'value'):
            assert scaling_method.value == 'robust'
        else:
            assert str(scaling_method) == 'robust'
    
    def test_create_high_performance(self):
        """Tester la création du pipeline haute performance"""
        builder = SimplePipelineBuilder.create_high_performance()
        pipeline = builder.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        
        assert builder.config.create_polynomial is True
        assert builder.config.apply_boxcox is True
        
        encoding_method = builder.config.encoding_method
        if hasattr(encoding_method, 'value'):
            assert encoding_method.value == 'catboost'
        else:
            assert str(encoding_method) == 'catboost'
    
    def test_create_minimal(self):
        """Tester la création du pipeline minimal"""
        builder = SimplePipelineBuilder.create_minimal()
        pipeline = builder.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        
        assert builder.config.drop_duplicates is False
        assert builder.config.drop_high_missing is False
        
        outlier_method = builder.config.outlier_method
        if hasattr(outlier_method, 'value'):
            assert outlier_method.value == 'none'
        else:
            assert str(outlier_method) == 'none'