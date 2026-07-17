# tests/conftest.py
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RANDOM_SEED = 42
N_SAMPLES_DEFAULT = 1000
N_SAMPLES_REGRESSION = 500


def _build_classification_frame(n_samples: int = N_SAMPLES_DEFAULT,
                                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Construit un DataFrame de classification de base (sans NaN), réutilisé
    par les fixtures `sample_data*`. Centraliser cette construction évite de
    dupliquer le même schéma de colonnes à chaque fixture."""
    np.random.seed(seed)
    return pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(5, 2, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(
            ['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]
        ).astype('object'),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples).astype('object'),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    })


@pytest.fixture
def sample_data():
    """Jeu de données de test avec quelques NaN épars (tests de détection)."""
    data = _build_classification_frame()
    data.loc[0:10, 'numeric_1'] = np.nan
    data.loc[20:25, 'categorical_1'] = np.nan
    data.loc[100:105, 'numeric_2'] = np.nan
    return data


@pytest.fixture
def sample_data_with_nan():
    """Jeu de données avec une proportion plus élevée de NaN (tests d'imputation)."""
    data = _build_classification_frame()
    data.loc[0:50, 'numeric_1'] = np.nan
    data.loc[20:70, 'categorical_1'] = np.nan
    data.loc[100:120, 'numeric_2'] = np.nan
    return data


@pytest.fixture
def sample_data_without_nan():
    """Jeu de données sans aucune valeur manquante."""
    return _build_classification_frame()


@pytest.fixture
def sample_data_regression():
    """Jeu de données pour les tests de régression (target continue)."""
    np.random.seed(RANDOM_SEED)
    n_samples = N_SAMPLES_REGRESSION

    feature_1 = np.random.normal(0, 1, n_samples)
    feature_2 = np.random.normal(5, 2, n_samples)
    feature_3 = np.random.exponential(2, n_samples)

    target = (
        2 * feature_1 + 0.5 * feature_2 - 1.5 * feature_3
        + np.random.normal(0, 0.5, n_samples)
    )

    return pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples).astype('object'),
        'target': target,
    })


@pytest.fixture
def sample_data_imbalanced():
    """Jeu de données avec une target binaire déséquilibrée (90/10)."""
    np.random.seed(RANDOM_SEED)
    n_samples = N_SAMPLES_DEFAULT

    target = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples) + target * 2,
        'feature_2': np.random.normal(5, 2, n_samples) - target * 3,
        'feature_3': np.random.exponential(2, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples).astype('object'),
    })
    data['target'] = target
    return data


@pytest.fixture
def temp_storage_dir():
    """Dossier temporaire nettoyé automatiquement après le test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)