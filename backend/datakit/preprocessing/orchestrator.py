
"""
Business logic for preprocessing: target detection, class balancing,
and pipeline execution.

This module does not depend on Streamlit, so it can be tested and
reused independently of the UI.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from datakit.preprocessing.tabular.config import (
    PreprocessingConfig,
    BalancingMethod,
)
from datakit.preprocessing.tabular.pipeline_builder import PipelineBuilder
from datakit.preprocessing.utils.target_detection import detect_target_column
from datakit.preprocessing.utils.compatibility import _as_dataframe, _as_series


TARGET_COLUMN_CANDIDATES = ("target", "y", "label", "class")


@dataclass
class BalancingResult:
    df: pd.DataFrame
    applied: bool
    message: Optional[str] = None
    report: Optional[Dict[str, Any]] = None


def apply_balancing_if_needed(
    builder: PipelineBuilder,
    config: PreprocessingConfig,
    df: pd.DataFrame,
) -> BalancingResult:
    """Apply class balancing if requested in the configuration."""

    if (
        config.balancing_method == BalancingMethod.NONE
        or not config.balancing_apply_before_pipeline
    ):
        return BalancingResult(df=df, applied=False)

    target_col = detect_target_column(df)

    if target_col is None:
        return BalancingResult(
            df=df,
            applied=False,
            message="⚠️ No target column found for class balancing",
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    from datakit.preprocessing.tabular.balancers.balancers import ClassBalancer

    balancer = ClassBalancer(
        method=config.balancing_method,
        sampling_strategy=config.balancing_sampling_strategy,
        random_state=config.balancing_random_state,
    )

    X_balanced, y_balanced = balancer.fit_resample(X, y)

    X_balanced = _as_dataframe(X_balanced, X.columns)
    y_balanced = _as_series(
        y_balanced,
        name=target_col,
        index=X_balanced.index,
    )

    df_processed = pd.concat([X_balanced, y_balanced], axis=1)

    message = (
        f"⚖️ Class balancing applied: "
        f"{len(df)} → {len(df_processed)} rows"
    )

    return BalancingResult(
        df=df_processed,
        applied=True,
        message=message,
        report=balancer.get_balance_report(),
    )


@dataclass
class PreprocessingResult:
    df: pd.DataFrame
    balancing_message: Optional[str] = None
    balancing_report: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    pipeline: Optional[Any] = None
    builder: Optional[PipelineBuilder] = None


def _ensure_dataframe(data, reference_index=None) -> pd.DataFrame:
    """
    Ensure that the pipeline output is a pandas DataFrame.

    This acts as a safety net for cases where a preprocessing step
    returns another data structure.
    """
    if isinstance(data, pd.DataFrame):
        return data

    df = _as_dataframe(data)

    if reference_index is not None and len(df) == len(reference_index):
        df.index = reference_index

    return df


def _json_safe(value: Any) -> Any:
    """
    Recursively convert numpy/pandas types that are not directly
    JSON serializable by json.dumps into native Python types.

    Supported conversions include numpy.int64, numpy.float64,
    numpy.bool_, ndarray, DataFrame, Series, and nested structures.

    Critical fix:
    GET /detect previously returned a raw HTTP 500 instead of a JSON
    response because MissingValueDetector, CardinalityDetector, and
    CorrelationDetector could expose numpy.int64/float64 values and
    even complete DataFrames (correlations['matrix']) in their stats.

    json.dumps() cannot serialize these objects. The error therefore
    occurred during response encoding by Starlette, after the route
    function had already returned and outside the application's
    try/except blocks.

    Rather than modifying every detector individually, and potentially
    missing a future detector, serialization is sanitized at the
    single output boundary.
    """

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict())

    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float) and (
        value != value
        or value in (float("inf"), float("-inf"))
    ):
        # Normalize Python NaN / Infinity values to None.
        # This produces valid standard JSON.
        return None

    return value


def run_preprocessing(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """
    Run the complete preprocessing pipeline
    (class balancing + transformations).
    """

    builder = PipelineBuilder(config)

    balancing_result = apply_balancing_if_needed(
        builder,
        config,
        df,
    )

    pipeline = builder.build_pipeline()

    target_col = detect_target_column(balancing_result.df)

    if target_col is not None:
        X = balancing_result.df.drop(columns=[target_col])
        y = balancing_result.df[target_col]

        X_transformed = pipeline.fit_transform(X, y)

        X_transformed = _ensure_dataframe(
            X_transformed,
            reference_index=X.index,
        )

        y_aligned = y.loc[
            y.index.intersection(X_transformed.index)
        ]

        df_transformed = pd.concat(
            [X_transformed, y_aligned],
            axis=1,
        )

    else:
        df_transformed = pipeline.fit_transform(
            balancing_result.df
        )

        df_transformed = _ensure_dataframe(
            df_transformed
        )

    return PreprocessingResult(
        df=df_transformed,
        balancing_message=balancing_result.message,
        balancing_report=(
            _json_safe(balancing_result.report)
            if balancing_result.report
            else None
        ),
        target_column=target_col,
        pipeline=pipeline,
        builder=builder,
    )


def run_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run diagnostic detectors only.
    No preprocessing transformations are applied.
    """

    builder = PipelineBuilder()
    detection_pipeline = builder.build_detection_pipeline()

    detection_pipeline.fit(df)

    report: Dict[str, Any] = {}

    for detector in builder.detectors:
        name = type(detector).__name__

        report[name] = {
            "problems": detector.problems,
        }

        for attr in (
            "missing_stats",
            "outlier_stats",
            "correlations",
            "cardinality_stats",
            "duplicate_count",
        ):
            if hasattr(detector, attr):
                value = getattr(detector, attr)

                if (
                    attr == "correlations"
                    and isinstance(value, dict)
                    and "matrix" in value
                ):
                    value = {
                        **value,
                        "matrix": value["matrix"].to_dict(),
                    }

                report[name][attr] = value


    return _json_safe(report)

