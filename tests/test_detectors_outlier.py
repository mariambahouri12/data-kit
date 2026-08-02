import numpy as np
import pandas as pd
import pytest

from src.datakit.preprocessing.tabular.detectors.outlier import OutlierDetector
from src.datakit.preprocessing.tabular.config import OutlierMethod


def test_no_outliers_iqr():
    """A normal dataset should not detect outliers with IQR."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 14],
        "B": ["a", "b", "c", "d", "e"]
    })

    detector = OutlierDetector(
        method=OutlierMethod.IQR
    )

    detector._fit(df)

    assert detector.problems == []
    assert detector.outlier_stats["A"]["n_outliers"] == 0


def test_detect_outliers_iqr():
    """IQR method should detect extreme values."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100]
    })

    detector = OutlierDetector(
        method=OutlierMethod.IQR,
        threshold=1.5
    )

    detector._fit(df)

    assert detector.outlier_stats["A"]["n_outliers"] == 1

    assert len(detector.problems) == 1

    assert detector.problems[0]["column"] == "A"
    assert detector.problems[0]["description"] == "1 outliers (20.0%)"


def test_detect_outliers_zscore():
    """Z-score method should detect extreme values."""

    df = pd.DataFrame({
        "A": [
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            100
        ]
    })

    detector = OutlierDetector(
        method=OutlierMethod.ZSCORE,
        threshold=2
    )

    detector._fit(df)

    assert detector.outlier_stats["A"]["n_outliers"] == 1
    assert detector.problems[0]["column"] == "A"


def test_ignore_non_numeric_columns():
    """Only numeric columns should be analyzed."""

    df = pd.DataFrame({
        "A": [1, 2, 3, 100],
        "B": ["x", "y", "z", "w"]
    })

    detector = OutlierDetector()

    detector._fit(df)

    assert "B" not in detector.outlier_stats


def test_ignore_empty_numeric_column():
    """A numeric column containing only NaN should be ignored."""

    df = pd.DataFrame({
        "A": [np.nan, np.nan, np.nan]
    })

    detector = OutlierDetector()

    detector._fit(df)

    assert detector.outlier_stats == {}
    assert detector.problems == []


def test_zscore_with_zero_standard_deviation():
    """No outlier computation when standard deviation is zero."""

    df = pd.DataFrame({
        "A": [5, 5, 5, 5]
    })

    detector = OutlierDetector(
        method=OutlierMethod.ZSCORE
    )

    detector._fit(df)

    assert detector.outlier_stats == {}
    assert detector.problems == []


def test_invalid_method_raises_error():
    """Unsupported method should raise ValueError."""

    with pytest.raises(ValueError):
        OutlierDetector(method="INVALID")


def test_missing_values_are_ignored():
    """NaN values should not affect outlier calculation."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100, np.nan]
    })

    detector = OutlierDetector()

    detector._fit(df)

    assert detector.outlier_stats["A"]["n_outliers"] == 1