import warnings
from typing import Optional, Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import ImputationMethod


class MissingValueCleaner(BasePreprocessor):
    """
    Impute ou supprime les valeurs manquantes.
    
    Permet de spécifier une stratégie différente par colonne via `column_strategies`.
    """

    def __init__(
        self,
        strategy: Union[str, ImputationMethod] = ImputationMethod.MEDIAN,
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        knn_neighbors: int = 5,
        column_strategies: Optional[Dict[str, Union[str, ImputationMethod]]] = None,  # ← NOUVEAU
        **kwargs
    ):
        """
        Args:
            strategy: stratégie d'imputation par défaut pour les colonnes numériques.
            fill_value: valeur utilisée si strategy == CONSTANT.
            columns: colonnes à traiter (None = toutes).
            knn_neighbors: nombre de voisins si strategy == KNN.
            column_strategies: dictionnaire {col: strategy} pour des stratégies spécifiques.
                               Ex: {'age': 'median', 'salaire': 'mean'}
        """
        super().__init__(**kwargs)
        self.strategy = ImputationMethod(strategy) if isinstance(strategy, str) else strategy
        self.fill_value = fill_value
        self.columns = columns
        self.knn_neighbors = knn_neighbors
        self.column_strategies = column_strategies or {}  # ← NOUVEAU

        self.imputers: Dict[str, Any] = {}  # ← NOUVEAU: un imputer par colonne
        self.cat_imputer = None
        self.column_types: Dict[str, List[str]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_impute = select_columns(X, self.columns)

        numeric_cols = X[cols_to_impute].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X[cols_to_impute].select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        self.column_types = {"numeric": numeric_cols, "categorical": categorical_cols}

        # ← NOUVEAU: Fit pour chaque colonne numérique avec sa stratégie
        for col in numeric_cols:
            strategy = self._get_strategy_for_column(col)
            self._fit_column_imputer(X, col, strategy)

        if categorical_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent", fill_value="unknown")
            self.cat_imputer.fit(X[categorical_cols])

    def _get_strategy_for_column(self, col: str) -> ImputationMethod:
        """Retourne la stratégie pour une colonne donnée."""
        if col in self.column_strategies:
            return ImputationMethod(self.column_strategies[col])
        return self.strategy

    def _fit_column_imputer(self, X: pd.DataFrame, col: str, strategy: ImputationMethod) -> None:
        """Fit un imputer pour une colonne spécifique."""
        
        if strategy == ImputationMethod.DROP:
            self.imputers[col] = None
            return

        if strategy == ImputationMethod.KNN:
            # KNN nécessite plusieurs colonnes, on garde l'approche globale
            # Sinon, on peut utiliser KNN sur toutes les colonnes
            if not hasattr(self, '_knn_imputer'):
                self._knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                self._knn_imputer.fit(X[self.column_types["numeric"]])
            return

        fill_value = self.fill_value
        if strategy == ImputationMethod.CONSTANT and fill_value is None:
            fill_value = 0
            warnings.warn(
                f"fill_value is None avec strategy=CONSTANT pour {col}, utilisation de 0 par défaut.",
                RuntimeWarning,
            )

        imputer = SimpleImputer(strategy=strategy.value, fill_value=fill_value)
        imputer.fit(X[[col]])
        self.imputers[col] = imputer

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        numeric_cols = self.column_types.get("numeric", [])
        
        for col in numeric_cols:
            strategy = self._get_strategy_for_column(col)
            
            if strategy == ImputationMethod.DROP:
                X_copy = X_copy.dropna(subset=[col])
            elif col in self.imputers and self.imputers[col] is not None:
                X_copy[col] = self.imputers[col].transform(X_copy[[col]])
            elif hasattr(self, '_knn_imputer'):
                # Cas KNN global
                X_copy[numeric_cols] = self._knn_imputer.transform(X_copy[numeric_cols])
                break
            elif self.imputers.get(col) is None:
                continue

        categorical_cols = self.column_types.get("categorical", [])
        if categorical_cols and self.cat_imputer is not None:
            X_copy[categorical_cols] = self.cat_imputer.transform(X_copy[categorical_cols])

        return X_copy