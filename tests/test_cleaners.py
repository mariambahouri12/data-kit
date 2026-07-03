import pytest
import pandas as pd
import numpy as np
from preprocessing.tabular.cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner


class TestMissingValueCleaner:
    
    def test_mean_imputation(self, sample_data):
        """Tester l'imputation par la moyenne"""
        cleaner = MissingValueCleaner(
            strategy='mean',
            columns=['numeric_1']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert transformed['numeric_1'].isnull().sum() == 0
        original_mean = sample_data['numeric_1'].mean()
        imputed_mean = transformed['numeric_1'].mean()
        assert abs(original_mean - imputed_mean) < 0.01
    
    def test_median_imputation(self, sample_data):
        """Tester l'imputation par la médiane"""
        cleaner = MissingValueCleaner(
            strategy='median',
            columns=['numeric_1']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert transformed['numeric_1'].isnull().sum() == 0
        original_median = sample_data['numeric_1'].median()
        imputed_median = transformed['numeric_1'].median()
        assert abs(original_median - imputed_median) < 0.01
    
    def test_constant_imputation(self, sample_data):
        """Tester l'imputation par une constante"""
        cleaner = MissingValueCleaner(
            strategy='constant',
            fill_value=999,
            columns=['numeric_1']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        nan_mask = sample_data['numeric_1'].isna()
        assert (transformed.loc[nan_mask, 'numeric_1'] == 999).all()
    
    def test_constant_imputation_default(self, sample_data):
        """Tester l'imputation par constante avec valeur par défaut"""
        cleaner = MissingValueCleaner(
            strategy='constant',
            columns=['numeric_1']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        nan_mask = sample_data['numeric_1'].isna()
        assert (transformed.loc[nan_mask, 'numeric_1'] == 0).all()
    
    def test_drop_rows(self, sample_data):
        """Tester la suppression des lignes avec valeurs manquantes"""
        cleaner = MissingValueCleaner(
            strategy='drop',
            columns=['numeric_1']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert transformed['numeric_1'].isnull().sum() == 0
        assert len(transformed) <= len(sample_data)
        assert len(transformed) < len(sample_data)
    
    def test_categorical_imputation(self, sample_data):
        """Tester l'imputation des colonnes catégorielles"""
        cleaner = MissingValueCleaner()
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert transformed['categorical_1'].isnull().sum() == 0
    
    def test_knn_imputation(self, sample_data):
        """Tester l'imputation par KNN"""
        cleaner = MissingValueCleaner(
            strategy='knn',
            columns=['numeric_1', 'numeric_2']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert transformed['numeric_1'].isnull().sum() == 0
        assert transformed['numeric_2'].isnull().sum() == 0


class TestOutlierCleaner:
    
    def test_winsorize_outliers(self, sample_data):
        """Tester le winsorizing des outliers"""
        cleaner = OutlierCleaner(
            method='iqr',
            threshold=1.5,
            action='winsorize',
            columns=['numeric_2']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        Q1 = sample_data['numeric_2'].quantile(0.25)
        Q3 = sample_data['numeric_2'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        assert transformed['numeric_2'].max() <= upper_bound
    
    def test_drop_outliers(self, sample_data):
        """Tester la suppression des outliers"""
        cleaner = OutlierCleaner(
            method='iqr',
            threshold=1.5,
            action='drop',
            columns=['numeric_2']
        )
        
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        Q1 = sample_data['numeric_2'].quantile(0.25)
        Q3 = sample_data['numeric_2'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        assert (transformed['numeric_2'] >= lower_bound).all()
        assert (transformed['numeric_2'] <= upper_bound).all()


class TestDuplicateCleaner:
    
    def test_remove_duplicates(self):
        """Tester la suppression des doublons"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3, 3, 3],
            'b': ['x', 'y', 'y', 'z', 'z', 'z']
        })
        
        cleaner = DuplicateCleaner()
        cleaner.fit(data)
        transformed = cleaner.transform(data)
        
        assert len(transformed) == 3
        
    def test_keep_first(self):
        """Tester keep='first'"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': ['x', 'y', 'y', 'z']
        })
        
        cleaner = DuplicateCleaner(keep='first')
        cleaner.fit(data)
        transformed = cleaner.transform(data)
        
        assert transformed.iloc[1]['a'] == 2
        assert len(transformed) == 3
    
    def test_keep_last(self):
        """Tester keep='last'"""
        data = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': ['x', 'y', 'y', 'z']
        })
        
        cleaner = DuplicateCleaner(keep='last')
        cleaner.fit(data)
        transformed = cleaner.transform(data)
        
        assert transformed.iloc[1]['a'] == 2
        assert len(transformed) == 3
    
    def test_subset(self):
        """Tester la suppression des doublons sur un sous-ensemble"""
        data = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': ['x', 'y', 'x', 'y'],
            'c': [1, 2, 3, 4]
        })
        
        cleaner = DuplicateCleaner(subset=['a'])
        cleaner.fit(data)
        transformed = cleaner.transform(data)
        
        assert len(transformed) == 2