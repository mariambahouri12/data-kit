import numpy as np
import pandas as pd
import pytest

from src.datakit.preprocessing.tabular.detectors.correlation import CorrelationDetector


def test_detect_high_correlation():
    """Highly correlated columns should be detected."""

    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [10, 20, 30, 40, 50]
    })

    detector = CorrelationDetector(
        threshold=0.8
    )

    detector._fit(df)

    assert len(detector.correlations["high_corr_pairs"]) > 0

    assert len(detector.problems) > 0

    assert detector.correlations["high_corr_pairs"][0]["col1"] == "A"
    assert detector.correlations["high_corr_pairs"][0]["col2"] == "B"


def test_no_high_correlation():
    """Columns with low correlation should not produce problems."""

    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 3, 8, 1, 6],
        "C": [7, 2, 9, 4, 5]
    })

    detector = CorrelationDetector(
        threshold=0.8
    )

    detector._fit(df)

    assert detector.problems == []

    assert detector.correlations["high_corr_pairs"] == []


def test_ignore_non_numeric_columns():
    """Only numeric columns should be used for correlation."""

    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [2, 4, 6, 8],
        "Text": ["a", "b", "c", "d"]
    })

    detector = CorrelationDetector()

    detector._fit(df)

    pairs = detector.correlations["high_corr_pairs"]

    for pair in pairs:
        assert pair["col1"] != "Text"
        assert pair["col2"] != "Text"


def test_not_enough_numeric_columns():
    """Detector should stop if less than two numeric columns exist."""

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "Text": ["a", "b", "c"]
    })

    detector = CorrelationDetector()

    detector._fit(df)

    assert detector.correlations == {}
    assert detector.problems == []


def test_not_enough_data_after_nan_removal():
    """Should warn when NaNs leave insufficient data."""

    df = pd.DataFrame({
        "A": [1, np.nan, np.nan],
        "B": [2, np.nan, np.nan]
    })

    detector = CorrelationDetector()

    with pytest.warns(
        RuntimeWarning,
        match="Not enough data after dropping NaNs for correlation"
    ):
        detector._fit(df)

    assert detector.correlations == {}
    assert detector.problems == []


def test_correlation_threshold():
    """Only correlations above threshold should be detected."""

    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 10, 15, 20, 25]
    })

    detector = CorrelationDetector(
        threshold=0.99
    )

    detector._fit(df)

    assert len(
        detector.correlations["high_corr_pairs"]
    ) == 1


def test_correlation_matrix_is_created():
    """Correlation matrix should be stored."""

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [3, 6, 9]
    })

    detector = CorrelationDetector()

    detector._fit(df)

    assert "matrix" in detector.correlations

    assert isinstance(
        detector.correlations["matrix"],
        pd.DataFrame
    )


def test_find_high_corr_pairs_directly():
    """Internal method should return correct correlated pairs."""

    detector = CorrelationDetector(
        threshold=0.8
    )

    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.95, 0.2],
            [0.95, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ],
        columns=["A", "B", "C"],
        index=["A", "B", "C"]
    )

    result = detector._find_high_corr_pairs(
        corr_matrix,
        ["A", "B", "C"]
    )

    assert len(result) == 1

    assert result[0]["col1"] == "A"
    assert result[0]["col2"] == "B"
    assert result[0]["correlation"] == 0.95