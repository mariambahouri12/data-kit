from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """Base class for all preprocessing, compatible with sklearn Pipeline."""

    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.id = str(uuid.uuid4())[:8]
        self.params = kwargs
        self.is_fitted = False
        self.created_at = datetime.now().isoformat()
        self._validate_params()

    def _validate_params(self) -> None:
        """To override if validation needed."""
        return None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":
        self._fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        return self._transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Implementation of the adaptation."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implementation of the transformation."""

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params) -> "BasePreprocessor":
        self.params.update(params)
        self._validate_params()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "class": self.__class__.__name__,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class BaseDetector(BasePreprocessor):
    """Detector: identifies issues without changing the data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problems: List[Dict[str, Any]] = []
        self.report: Dict[str, Any] = {}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def get_report(self) -> Dict[str, Any]:
        return {
            "problems": self.problems,
            "report": self.report,
            "summary": self._generate_summary(),
        }


    def _generate_summary(self) -> str:
        if not self.problems:
            return "✅ No problem detected"
