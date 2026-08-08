
# preprocessing/tabular/transformers/boxcox.py

from typing import  Dict

import numpy as np
import pandas as pd
from scipy import stats



from ._base import _LambdaFamilyTransformer


class BoxCoxTransformer(_LambdaFamilyTransformer):
    """
    Box-Cox transformation. Requires strictly positive values:
    a shift is automatically added if the column contains values <= 0.
    """

    def _estimate_column(
        self,
        series: pd.Series
    ) -> Dict[str, float]:
        params: Dict[str, float] = {}
        shifted = series

        if (series <= 0).any():
            shift = abs(series.min()) + 1
            params["shift"] = shift
            shifted = series + shift

        params["lambda"] = (
            self.lambda_
            if self.lambda_ is not None
            else stats.boxcox(shifted)[1]
        )

        return params

    def _transform_column(
        self,
        series: pd.Series,
        params: Dict[str, float]
    ) -> pd.Series:
        shift = params.get("shift", 0)
        lambda_val = params["lambda"]
        shifted = series + shift

        if lambda_val == 0:
            return np.log(shifted)

        return (
            shifted ** lambda_val - 1
        ) / lambda_val
