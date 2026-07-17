# tests/test_encoders.py
import pandas as pd
import pytest

from src.datakit.modules.preprocessing.tabular.encoders.encoders import CategoricalEncoder


class TestCategoricalEncoder:

    def test_onehot_encoding(self, sample_data_without_nan):
        """Tester One-Hot Encoding.
        Note : ce test suppose que categorical_1 (4 catégories, la plus rare
        à 10%) et categorical_2 (3 catégories) sont toutes au-dessus de
        min_frequency=0.01 (1%) — sinon sklearn regrouperait les catégories
        rares en 'infrequent_sklearn' et le compte de colonnes ci-dessous
        ne correspondrait plus."""
        encoder = CategoricalEncoder(
            method='onehot',
            columns=['categorical_1', 'categorical_2'],
            sparse=False,
        )

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        # Les colonnes originales sont supprimées
        assert 'categorical_1' not in transformed.columns
        assert 'categorical_2' not in transformed.columns

        # Les nouvelles colonnes sont créées
        assert 'categorical_1_A' in transformed.columns
        assert 'categorical_1_B' in transformed.columns
        assert 'categorical_2_X' in transformed.columns

        # categorical_1 a 4 catégories (A,B,C,D), categorical_2 en a 3 (X,Y,Z)
        original_cols = sample_data_without_nan.shape[1]
        expected_cols = original_cols - 2 + 4 + 3
        assert transformed.shape[1] == expected_cols

        cat_cols = [c for c in transformed.columns if c.startswith('categorical_')]
        for col in cat_cols:
            assert transformed[col].isin([0, 1]).all()

    def test_ordinal_encoding(self, sample_data_without_nan):
        """Tester Ordinal Encoding (via CategoricalEncoder(method='ordinal'),
        et non une classe OrdinalEncoderWrapper séparée qui n'existe pas
        dans encoders.py)."""
        encoder = CategoricalEncoder(method='ordinal', columns=['categorical_1', 'categorical_2'])

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        assert 'categorical_1' in transformed.columns
        assert 'categorical_2' in transformed.columns
        assert transformed['categorical_1'].dtype in ['int64', 'float64']

        # Catégorie inconnue au transform -> -1 (handle_unknown="use_encoded_value")
        unseen = sample_data_without_nan.copy()
        unseen.loc[0, 'categorical_1'] = 'UNKNOWN'
        transformed_unseen = encoder.transform(unseen)
        assert transformed_unseen.loc[0, 'categorical_1'] == -1

    def test_label_encoding(self, sample_data_without_nan):
        """Tester Label Encoding (méthode distincte de 'ordinal', non testée
        jusqu'ici : gestion manuelle des catégories inconnues, contrairement
        à sklearn.OrdinalEncoder utilisé pour 'ordinal')."""
        encoder = CategoricalEncoder(method='label', columns=['categorical_1'])

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        assert transformed['categorical_1'].dtype.kind in 'iu'

        unseen = sample_data_without_nan.copy()
        unseen.loc[0, 'categorical_1'] = 'UNKNOWN_CAT'
        transformed_unseen = encoder.transform(unseen)
        assert transformed_unseen.loc[0, 'categorical_1'] == -1

    def test_frequency_encoding(self, sample_data_without_nan):
        """Tester Frequency Encoding"""
        encoder = CategoricalEncoder(method='frequency', columns=['categorical_1'])

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        assert transformed['categorical_1'].dtype in ['float64', 'int64']
        assert (transformed['categorical_1'] >= 0).all()
        assert (transformed['categorical_1'] <= 1).all()

    def test_frequency_encoding_unknown_category_maps_to_zero(self, sample_data_without_nan):
        """Chemin distinct du test onehot 'handle_unknown' : ici la logique
        passe par _transform_mapped_column (mapping dict), pas par sklearn."""
        encoder = CategoricalEncoder(method='frequency', columns=['categorical_1'])
        encoder.fit(sample_data_without_nan)

        unseen = sample_data_without_nan.copy()
        unseen.loc[0, 'categorical_1'] = 'UNKNOWN_CAT'
        transformed = encoder.transform(unseen)

        assert transformed.loc[0, 'categorical_1'] == 0

    def test_binary_encoding(self, sample_data_without_nan):
        """Tester Binary Encoding"""
        encoder = CategoricalEncoder(method='binary', columns=['categorical_1'])

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        assert 'categorical_1' not in transformed.columns

        bit_cols = [c for c in transformed.columns if c.startswith('categorical_1_bit_')]
        # categorical_1 a 4 catégories -> 4.bit_length() == 3 bits nécessaires
        assert len(bit_cols) == 3
        for col in bit_cols:
            assert transformed[col].isin([0, 1]).all()

    def test_hash_encoding_reproducible(self, sample_data_without_nan):
        """Tester que le hash encoding est reproductible (MD5, pas hash()
        natif, qui n'est pas stable entre runs Python)."""
        encoder = CategoricalEncoder(method='hash', columns=['categorical_1'])
        encoder.fit(sample_data_without_nan)
        transformed_1 = encoder.transform(sample_data_without_nan)

        encoder2 = CategoricalEncoder(method='hash', columns=['categorical_1'])
        encoder2.fit(sample_data_without_nan)
        transformed_2 = encoder2.transform(sample_data_without_nan)

        pd.testing.assert_series_equal(
            transformed_1['categorical_1'],
            transformed_2['categorical_1'],
        )

    def test_target_encoding(self, sample_data_without_nan):
        """Tester target encoding. Le target encoding a besoin de y au fit,
        pas d'un paramètre 'target' au constructeur (qui n'existe pas)."""
        X = sample_data_without_nan.drop(columns=['target'])
        y = sample_data_without_nan['target']

        encoder = CategoricalEncoder(method='target', columns=['categorical_1'])
        encoder.fit(X, y)
        transformed = encoder.transform(X)

        assert 'categorical_1' in transformed.columns
        assert transformed['categorical_1'].dtype in ['float64', 'int64']

    def test_target_encoding_without_y_raises_value_error(self, sample_data_without_nan):
        """Le target encoding doit échouer explicitement sans y, plutôt que
        de produire un résultat silencieusement incorrect."""
        X = sample_data_without_nan.drop(columns=['target'])
        encoder = CategoricalEncoder(method='target', columns=['categorical_1'])

        with pytest.raises(ValueError):
            encoder.fit(X)

    def test_catboost_encoding(self, sample_data_without_nan):
        """Tester CatBoost encoding (méthode distincte non couverte jusqu'ici,
        nécessite aussi un y)."""
        X = sample_data_without_nan.drop(columns=['target'])
        y = sample_data_without_nan['target']

        encoder = CategoricalEncoder(method='catboost', columns=['categorical_1'])
        encoder.fit(X, y)
        transformed = encoder.transform(X)

        assert 'categorical_1' in transformed.columns
        assert transformed['categorical_1'].dtype in ['float64', 'int64']

    def test_max_categories_filtering_falls_back_to_frequency(self, sample_data_without_nan):
        """Les colonnes dont la cardinalité dépasse max_categories doivent
        basculer en frequency encoding, PAS rester en one-hot. L'ancienne
        version de ce test se contentait de vérifier la présence de la
        colonne, ce qui aurait aussi été vrai sans aucun fallback réel."""
        encoder = CategoricalEncoder(method='onehot', max_categories=2, sparse=False)

        encoder.fit(sample_data_without_nan)
        transformed = encoder.transform(sample_data_without_nan)

        assert 'categorical_1' in transformed.columns
        assert 'categorical_1_A' not in transformed.columns  # pas de one-hot
        assert transformed['categorical_1'].dtype == 'float64'
        assert (transformed['categorical_1'] >= 0).all()
        assert (transformed['categorical_1'] <= 1).all()

    def test_handle_unknown_ignore(self, sample_data_without_nan):
        """Tester la gestion des valeurs inconnues en mode ignore (one-hot)"""
        encoder = CategoricalEncoder(
            method='onehot',
            columns=['categorical_1'],
            handle_unknown='ignore',
            sparse=False,
        )
        encoder.fit(sample_data_without_nan)

        test_data = sample_data_without_nan.copy()
        test_data.loc[0, 'categorical_1'] = 'UNKNOWN'
        transformed = encoder.transform(test_data)

        cat_cols = [c for c in transformed.columns if c.startswith('categorical_1_')]
        for col in cat_cols:
            assert transformed.loc[0, col] == 0

    def test_get_feature_names(self, sample_data_without_nan):
        """Tester l'obtention des noms de features"""
        encoder = CategoricalEncoder(method='onehot', columns=['categorical_1'])

        encoder.fit(sample_data_without_nan)
        feature_names = encoder.get_feature_names()

        # categorical_1 a 4 catégories (A,B,C,D) ; le OneHotEncoder par défaut
        # ne crée pas de colonne pour les NaN
        assert len(feature_names) == 4
        assert all(name.startswith('categorical_1_') for name in feature_names)

    def test_invalid_method_raises_value_error(self):
        """Une méthode d'encodage inconnue doit échouer dès la construction
        (EncodingMethod est un Enum), pas silencieusement plus tard."""
        with pytest.raises(ValueError):
            CategoricalEncoder(method='not_a_real_encoding_method')