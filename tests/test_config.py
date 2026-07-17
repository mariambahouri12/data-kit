# tests/test_config.py
import pytest

from preprocessing.tabular.config import (
    BalancingMethod,
    EncodingMethod,
    ImputationMethod,
    PreprocessingConfig,
    ScalingMethod,
    TaskType,
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
        """Tester la configuration personnalisée. Les valeurs choisies
        diffèrent volontairement des défauts, pour garantir que le test
        échouerait si un paramètre était silencieusement ignoré."""
        config = PreprocessingConfig(
            imputation_method='knn',
            scaling_method='robust',
            encoding_method='target',
            balancing_method='smote',
            task_type='regression',
            drop_duplicates=False,  # défaut = True
            create_polynomial=True,  # défaut = False
            polynomial_degree=3,     # défaut = 2
        )

        assert config.imputation_method == ImputationMethod.KNN
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encoding_method == EncodingMethod.TARGET
        assert config.balancing_method == BalancingMethod.SMOTE
        assert config.task_type == TaskType.REGRESSION
        assert config.drop_duplicates is False
        assert config.create_polynomial is True
        assert config.polynomial_degree == 3

    def test_invalid_enum_value_raises_value_error(self):
        """Une valeur de méthode inconnue doit échouer explicitement plutôt
        que d'être acceptée silencieusement (normalisation faite dans
        __post_init__ via ImputationMethod(value))."""
        with pytest.raises(ValueError):
            PreprocessingConfig(imputation_method='not_a_real_strategy')

    def test_to_dict(self):
        """Tester la conversion en dictionnaire"""
        config = PreprocessingConfig(imputation_method='median', scaling_method='standard')

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
            'task_type': 'classification',
        }

        config = PreprocessingConfig.from_dict(data)

        assert config.imputation_method == ImputationMethod.KNN
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encoding_method == EncodingMethod.ONE_HOT
        assert config.task_type == TaskType.CLASSIFICATION

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict ne doit pas planter si le dictionnaire contient des clés
        qui ne correspondent à aucun champ du dataclass (ex: config sauvegardée
        avec une version antérieure/postérieure du module)."""
        data = {
            'imputation_method': 'mean',
            'this_field_does_not_exist': 'some_value',
        }

        config = PreprocessingConfig.from_dict(data)

        assert config.imputation_method == ImputationMethod.MEAN
        assert not hasattr(config, 'this_field_does_not_exist')

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict() puis from_dict() doit reproduire une config équivalente."""
        original = PreprocessingConfig(
            imputation_method='knn',
            scaling_method='robust',
            encoding_method='target',
            balancing_method='smote',
        )

        restored = PreprocessingConfig.from_dict(original.to_dict())

        assert restored == original

    def test_save_and_load(self, tmp_path):
        """Tester la sauvegarde et le chargement (fichier nettoyé
        automatiquement par le fixture tmp_path de pytest)."""
        config = PreprocessingConfig(imputation_method='knn', scaling_method='robust')

        filepath = tmp_path / "config.json"
        config.save(str(filepath))
        loaded = PreprocessingConfig.load(str(filepath))

        assert loaded == config

    def test_get_active_steps(self):
        """Tester l'obtention des étapes actives (cas balancing hors pipeline)"""
        config = PreprocessingConfig(
            imputation_method='median',
            scaling_method='standard',
            encoding_method='onehot',
            outlier_method='iqr',
            balancing_method='smote',
            balancing_apply_before_pipeline=True,  # le balancing est avant le pipeline
            drop_duplicates=True,
            drop_high_missing=True,
        )

        steps = config.get_active_steps()

        assert 'drop_duplicates' in steps
        assert 'drop_high_missing' in steps
        assert 'imputation' in steps
        assert 'outlier_handling' in steps
        assert 'encoding' in steps
        assert 'scaling' in steps
        # Le balancing n'est PAS dans le pipeline car apply_before_pipeline=True
        assert 'balancing' not in steps

    def test_get_active_steps_includes_balancing_when_applied_in_pipeline(self):
        """Cas complémentaire du test précédent : quand
        balancing_apply_before_pipeline=False, l'étape doit apparaître."""
        config = PreprocessingConfig(
            balancing_method='smote',
            balancing_apply_before_pipeline=False,
        )

        steps = config.get_active_steps()

        assert 'balancing' in steps

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