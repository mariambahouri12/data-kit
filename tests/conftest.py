import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_data():
    """Créer un jeu de données de test avec NaN"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(5, 2, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]).astype('object'),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples).astype('object'),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    # Ajouter des NaN pour les tests de détection
    data.loc[0:10, 'numeric_1'] = np.nan
    data.loc[20:25, 'categorical_1'] = np.nan
    data.loc[100:105, 'numeric_2'] = np.nan
    
    return data


@pytest.fixture
def sample_data_with_nan():
    """Créer un jeu de données avec NaN (pour les tests d'imputation)"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(5, 2, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]).astype('object'),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples).astype('object'),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    # Ajouter des valeurs manquantes
    data.loc[0:50, 'numeric_1'] = np.nan
    data.loc[20:70, 'categorical_1'] = np.nan
    data.loc[100:120, 'numeric_2'] = np.nan
    
    return data


@pytest.fixture
def sample_data_without_nan():
    """Créer un jeu de données SANS NaN"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(5, 2, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]).astype('object'),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples).astype('object'),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    return data


@pytest.fixture
def sample_data_regression():
    """Créer un jeu de données pour la régression"""
    np.random.seed(42)
    n_samples = 500
    
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(5, 2, n_samples)
    X3 = np.random.exponential(2, n_samples)
    
    y = 2 * X1 + 0.5 * X2 - 1.5 * X3 + np.random.normal(0, 0.5, n_samples)
    
    data = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples).astype('object'),
        'target': y
    })
    
    return data


@pytest.fixture
def sample_data_imbalanced():
    """Créer un jeu de données déséquilibré"""
    np.random.seed(42)
    n_samples = 1000
    
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples) + y * 2,
        'feature_2': np.random.normal(5, 2, n_samples) - y * 3,
        'feature_3': np.random.exponential(2, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples).astype('object')
    })
    
    X['target'] = y
    return X


@pytest.fixture
def temp_storage_dir():
    """Créer un dossier temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)