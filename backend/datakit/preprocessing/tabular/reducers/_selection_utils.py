
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from ..config import TaskType


def make_importance_model(task_type: TaskType):
    """Model used for feature-importance-based methods (importance, RFE)."""
    if task_type == TaskType.CLASSIFICATION:
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    return RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical columns into integer codes (for sklearn models that require it)."""
    X_encoded = X.copy()

    cat_cols = X_encoded.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in cat_cols:
        X_encoded[col] = (
            X_encoded[col]
            .astype("category")
            .cat.codes
        )

    return X_encoded
