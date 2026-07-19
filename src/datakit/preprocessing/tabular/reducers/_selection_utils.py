
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..config import TaskType


def make_importance_model(task_type: TaskType):
    """Modèle utilisé pour les méthodes basées sur l'importance des features (importance, RFE)."""
    if task_type == TaskType.CLASSIFICATION:
        return RandomForestClassifier(n_estimators=100, random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes catégorielles en codes entiers (pour modèles sklearn qui l'exigent)."""
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes
    return X_encoded
