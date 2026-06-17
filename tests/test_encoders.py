# tests/test_encoders.py
import pytest
import pandas as pd
import numpy as np
from preprocessing.tabular.encoders import CategoricalEncoder, OrdinalEncoderWrapper
from preprocessing.tabular.config import EncodingMethod


class TestCategoricalEncoder:
    
    def test_onehot_encoding(self, sample_data):
        """Tester One-Hot Encoding"""
        encoder = CategoricalEncoder(
            method='onehot',
            columns=['categorical_1', 'categorical_2'],
            sparse=False
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        # Vérifier que les colonnes originales sont supprimées
        assert 'categorical_1' not in transformed.columns
        assert 'categorical_2' not in transformed.columns
        
        # Vérifier que les nouvelles colonnes sont créées
        assert 'categorical_1_A' in transformed.columns
        assert 'categorical_1_B' in transformed.columns
        assert 'categorical_2_X' in transformed.columns
        
        # Vérifier la forme
        expected_cols = sample_data.shape[1] - 2 + 3 + 2  # -2 originales + (3+2) encodées
        assert transformed.shape[1] == expected_cols
        
        # Vérifier que les valeurs sont binaires
        cat_cols = [c for c in transformed.columns if c.startswith('categorical_')]
        for col in cat_cols:
            assert transformed[col].isin([0, 1]).all()
    
    def test_ordinal_encoding(self, sample_data):
        """Tester Ordinal Encoding"""
        encoder = OrdinalEncoderWrapper(
            columns=['categorical_1', 'categorical_2']
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        # Vérifier que les colonnes existent
        assert 'categorical_1' in transformed.columns
        assert 'categorical_2' in transformed.columns
        
        # Vérifier que les valeurs sont encodées
        assert transformed['categorical_1'].dtype in ['int64', 'float64']
        
        # Vérifier que les valeurs inconnues sont gérées
        test_data = sample_data.copy()
        test_data.loc[0, 'categorical_1'] = 'UNKNOWN'
        transformed = encoder.transform(test_data)
        assert transformed.loc[0, 'categorical_1'] == -1
    
    def test_frequency_encoding(self, sample_data):
        """Tester Frequency Encoding"""
        encoder = CategoricalEncoder(
            method='frequency',
            columns=['categorical_1']
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        # Vérifier que la colonne est transformée en numérique
        assert transformed['categorical_1'].dtype in ['float64', 'int64']
        
        # Vérifier que les fréquences sont entre 0 et 1
        assert (transformed['categorical_1'] >= 0).all()
        assert (transformed['categorical_1'] <= 1).all()
    
    def test_binary_encoding(self, sample_data):
        """Tester Binary Encoding"""
        encoder = CategoricalEncoder(
            method='binary',
            columns=['categorical_1']
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        # Vérifier que la colonne originale est supprimée
        assert 'categorical_1' not in transformed.columns
        
        # Vérifier que les bits sont binaires
        bit_cols = [c for c in transformed.columns if c.startswith('categorical_1_bit_')]
        assert len(bit_cols) > 0
        
        for col in bit_cols:
            assert transformed[col].isin([0, 1]).all()
    
    def test_hash_encoding_reproducible(self, sample_data):
        """Tester que le hash encoding est reproductible"""
        encoder = CategoricalEncoder(
            method='hash',
            columns=['categorical_1']
        )
        
        encoder.fit(sample_data)
        transformed_1 = encoder.transform(sample_data)
        
        # Re-créer un encodeur identique
        encoder2 = CategoricalEncoder(
            method='hash',
            columns=['categorical_1']
        )
        encoder2.fit(sample_data)
        transformed_2 = encoder2.transform(sample_data)
        
        # Les résultats doivent être identiques
        pd.testing.assert_series_equal(
            transformed_1['categorical_1'],
            transformed_2['categorical_1']
        )
    
    def test_target_encoding_warning(self, sample_data):
        """Tester que target encoding fonctionne avec warning"""
        encoder = CategoricalEncoder(
            method='target',
            columns=['categorical_1'],
            target=sample_data['target']
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        assert 'categorical_1' in transformed.columns
        assert transformed['categorical_1'].dtype in ['float64', 'int64']
    
    def test_max_categories_filtering(self, sample_data):
        """Tester le filtrage des colonnes avec trop de catégories"""
        encoder = CategoricalEncoder(
            method='onehot',
            max_categories=2,  # Seuil bas pour tester
            sparse=False
        )
        
        encoder.fit(sample_data)
        transformed = encoder.transform(sample_data)
        
        # categorical_1 a 4 catégories donc devrait être encodé en frequency
        assert 'categorical_1' in transformed.columns
    
    def test_handle_unknown_ignore(self, sample_data):
        """Tester la gestion des valeurs inconnues en mode ignore"""
        encoder = CategoricalEncoder(
            method='onehot',
            columns=['categorical_1'],
            handle_unknown='ignore',
            sparse=False
        )
        
        encoder.fit(sample_data)
        
        # Ajouter une valeur inconnue
        test_data = sample_data.copy()
        test_data.loc[0, 'categorical_1'] = 'UNKNOWN'
        
        transformed = encoder.transform(test_data)
        
        # Vérifier que toutes les colonnes one-hot sont 0 pour la valeur inconnue
        cat_cols = [c for c in transformed.columns if c.startswith('categorical_1_')]
        for col in cat_cols:
            assert transformed.loc[0, col] == 0
    
    def test_get_feature_names(self, sample_data):
        """Tester l'obtention des noms de features"""
        encoder = CategoricalEncoder(
            method='onehot',
            columns=['categorical_1']
        )
        
        encoder.fit(sample_data)
        feature_names = encoder.get_feature_names()
        
        assert len(feature_names) == sample_data['categorical_1'].nunique()
        assert all(name.startswith('categorical_1_') for name in feature_names)