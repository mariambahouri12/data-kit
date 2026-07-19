"""
Logique métier du preprocessing : détection de cible, rééquilibrage des
classes et exécution du pipeline. Ce module ne dépend pas de Streamlit,
il peut donc être testé et réutilisé indépendamment de l'UI.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from datakit.preprocessing.tabular.config import PreprocessingConfig, BalancingMethod
from datakit.preprocessing.tabular.pipeline_builder import PipelineBuilder
from datakit.preprocessing.utils.arrow_fix import fix_dataframe_complete

TARGET_COLUMN_CANDIDATES = ("target", "y", "label", "class")


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Détecte une colonne cible probable par son nom (heuristique simple)."""
    for col in df.columns:
        if col.lower() in TARGET_COLUMN_CANDIDATES:
            return col
    return None


def _as_dataframe(X, columns) -> pd.DataFrame:
    """Garantit un pd.DataFrame, quelle que soit la version d'imblearn utilisée
    (certaines versions retournent un ndarray brut plutôt qu'un DataFrame)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)


def _as_series(y, name: str, index) -> pd.Series:
    """Garantit un pd.Series, même si imblearn a retourné un ndarray brut."""
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y, name=name, index=index)


@dataclass
class BalancingResult:
    df: pd.DataFrame
    applied: bool
    message: Optional[str] = None


def apply_balancing_if_needed(
    builder: PipelineBuilder, config: PreprocessingConfig, df: pd.DataFrame
) -> BalancingResult:
    """Applique le rééquilibrage des classes si demandé dans la config."""
    if config.balancing_method == BalancingMethod.NONE or not config.balancing_apply_before_pipeline:
        return BalancingResult(df=df, applied=False)

    target_col = detect_target_column(df)
    if target_col is None:
        return BalancingResult(
            df=df, applied=False, message="⚠️ Aucune colonne cible trouvée pour le rééquilibrage"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_balanced, y_balanced = builder.apply_balancing(X, y)
    X_balanced = _as_dataframe(X_balanced, X.columns)
    y_balanced = _as_series(y_balanced, name=target_col, index=X_balanced.index)

    df_processed = pd.concat([X_balanced, y_balanced], axis=1)
    message = f"⚖️ Rééquilibrage appliqué: {len(df)} → {len(df_processed)} lignes"
    return BalancingResult(df=df_processed, applied=True, message=message)


@dataclass
class PreprocessingResult:
    df: pd.DataFrame
    balancing_message: Optional[str] = None


def run_preprocessing(df: pd.DataFrame, config: PreprocessingConfig) -> PreprocessingResult:
    """Exécute le pipeline de preprocessing complet (balancing + transformations)."""
    builder = PipelineBuilder(config)
    balancing_result = apply_balancing_if_needed(builder, config, df)

    pipeline = builder.build_pipeline()
    df_transformed = pipeline.fit_transform(balancing_result.df)
    df_transformed = fix_dataframe_complete(df_transformed)

    return PreprocessingResult(df=df_transformed, balancing_message=balancing_result.message)
