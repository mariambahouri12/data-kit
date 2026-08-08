
# tabular/balancers/balancers.py

import warnings
from typing import Optional, Dict, Any, Tuple, Union

import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import LabelEncoder

from ...base import BasePreprocessor
from ..config import BalancingMethod


_SAMPLERS_WITH_RANDOM_STATE = {
    BalancingMethod.SMOTE: SMOTE,
    BalancingMethod.ADASYN: ADASYN,
    BalancingMethod.RANDOM_OVER: RandomOverSampler,
    BalancingMethod.RANDOM_UNDER: RandomUnderSampler,
    BalancingMethod.SMOTE_TOMEK: SMOTETomek,
    BalancingMethod.SMOTE_ENN: SMOTEENN,
}

_SAMPLERS_WITHOUT_RANDOM_STATE = {
    BalancingMethod.TOMEK: TomekLinks,
    BalancingMethod.ENN: EditedNearestNeighbours,
}


class ClassBalancer(BasePreprocessor):
    """
    Rebalances an imbalanced categorical target.
    Supported methods: SMOTE, ADASYN, Random Over/Under, Tomek, ENN,
    SMOTE+Tomek, SMOTE+ENN.

    API note: do not use fit()/transform(). Rebalancing changes
    the number of rows of both X *and* y simultaneously, which the
    standard sklearn API cannot express cleanly. Use fit_resample().

    LSP note: this class inherits from BasePreprocessor to benefit from
    the common infrastructure (init, is_fitted...), but intentionally
    breaks the fit()/transform() contract of the base class — see
    _fit/_transform below.

    It is effectively excluded from any generic code that would iterate
    over BasePreprocessor instances and call fit()/transform() on them
    (PipelineBuilder already explicitly excludes it from the sklearn
    pipeline, see apply_balancing()).
    """

    def __init__(
        self,
        method: Union[str, BalancingMethod] = BalancingMethod.SMOTE,
        sampling_strategy: Union[str, Dict] = "auto",
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method = BalancingMethod(method) if isinstance(method, str) else method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.balancer = None
        self.encoder: Optional[LabelEncoder] = None
        self.original_shape: Optional[Tuple[int, int]] = None
        self.balanced_shape: Optional[Tuple[int, int]] = None

    # -- Inherited sklearn API: not relevant here, explicitly fails --
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        raise NotImplementedError(
            "ClassBalancer uses fit_resample(X, y), not fit()/transform()."
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "ClassBalancer uses fit_resample(X, y), not fit()/transform()."
        )

    # -- Sampler construction --------------------------------------------
    def _build_sampler(self):
        if self.method in _SAMPLERS_WITH_RANDOM_STATE:
            cls = _SAMPLERS_WITH_RANDOM_STATE[self.method]
            return cls(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )

        if self.method in _SAMPLERS_WITHOUT_RANDOM_STATE:
            cls = _SAMPLERS_WITHOUT_RANDOM_STATE[self.method]
            return cls(sampling_strategy=self.sampling_strategy)

        raise ValueError(f"Unsupported rebalancing method: {self.method}")

    # -- Target encoding / decoding --------------------------------------
    def _encode_target(self, y: pd.Series):
        if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):
            self.encoder = LabelEncoder()
            return self.encoder.fit_transform(y)

        self.encoder = None
        return y

    def _decode_target(self, y_encoded):
        return (
            self.encoder.inverse_transform(y_encoded)
            if self.encoder is not None
            else y_encoded
        )

    # -- Public API -------------------------------------------------------
    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit and rebalance (X, y). Returns (X_resampled, y_resampled)."""
        self.original_shape = X.shape

        if self.method == BalancingMethod.NONE:
            self.balanced_shape = X.shape
            self.is_fitted = True
            return X, y

        y_encoded = self._encode_target(y)
        self.balancer = self._build_sampler()

        try:
            X_resampled, y_resampled = self.balancer.fit_resample(
                X,
                y_encoded
            )

        except ValueError as e:
            warnings.warn(
                f"{self.method} failed ({e}). Falling back to RandomOverSampler.",
                RuntimeWarning,
            )
            fallback = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = fallback.fit_resample(X, y_encoded)

        y_resampled = self._decode_target(y_resampled)
        self.balanced_shape = X_resampled.shape
        self.is_fitted = True

        return X_resampled, y_resampled

    def get_balance_report(self) -> Dict[str, Any]:
        """Report on the last performed rebalancing."""
        if self.original_shape is None:
            return {}

        return {
            "method": self.method.value,
            "original_shape": self.original_shape,
            "balanced_shape": self.balanced_shape,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

