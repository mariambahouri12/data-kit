# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ...base import BasePreprocessor
from .._column_utils import select_columns

from ._base import _LambdaFamilyTransformer


class YeoJohnsonTransformer(_LambdaFamilyTransformer):
    """Transformée Yeo-Johnson : comme Box-Cox mais accepte les valeurs
    négatives, donc aucun shift n'est nécessaire."""

    def _estimate_column(self, series: pd.Series) -> Dict[str, float]:
        lambda_val = self.lambda_ if self.lambda_ is not None else stats.yeojohnson(series.values)[1]
        return {"lambda": lambda_val}

    def _transform_column(self, series: pd.Series, params: Dict[str, float]) -> pd.Series:
        return stats.yeojohnson(series.values, lmbda=params["lambda"])