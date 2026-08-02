import numpy as np
import pandas as pd

from src.datakit.preprocessing.tabular.detectors.cardinality import CardinalityDetector


def test_detect_high_cardinality_column():
    """Columns with too many categories should be detected."""

    df = pd.DataFrame({
        "Category": [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J"
        ]
    })

    detector = CardinalityDetector(
        max_categories=5
    )

    detector._fit(df)

    assert detector.cardinality_stats["Category"] == 10

    assert len(detector.problems) == 1

    assert detector.problems[0]["column"] == "Category"

    assert (
        detector.problems[0]["description"]
        == "10 categories (recommanded: < 5)"
    )


def test_no_high_cardinality():
    """Columns below the threshold should not produce problems."""

    df = pd.DataFrame({
        "Category": [
            "A",
            "B",
            "C",
            "A",
            "B"
        ]
    })

    detector = CardinalityDetector(
        max_categories=5
    )

    detector._fit(df)

    assert detector.cardinality_stats["Category"] == 3

    assert detector.problems == []


def test_ignore_numeric_columns():
    """Numeric columns should not be analyzed."""

    df = pd.DataFrame({
        "Numbers": [1, 2, 3, 4, 5],
        "Category": ["A", "B", "C", "D", "E"]
    })

    detector = CardinalityDetector(
        max_categories=3
    )

    detector._fit(df)

    assert "Numbers" not in detector.cardinality_stats

    assert "Category" in detector.cardinality_stats


def test_detect_category_dtype():
    """Category pandas dtype should be detected."""

    df = pd.DataFrame({
        "Category": pd.Series(
            ["A", "B", "C", "D"],
            dtype="category"
        )
    })

    detector = CardinalityDetector(
        max_categories=2
    )

    detector._fit(df)

    assert detector.cardinality_stats["Category"] == 4

    assert len(detector.problems) == 1


def test_empty_dataframe():
    """Empty dataframe should not produce errors."""

    df = pd.DataFrame()

    detector = CardinalityDetector()

    detector._fit(df)

    assert detector.cardinality_stats == {}

    assert detector.problems == []


def test_missing_values_not_counted_as_categories():
    """NaN values should not increase cardinality."""

    df = pd.DataFrame({
        "Category": [
            "A",
            "B",
            np.nan,
            "A",
            "B"
        ]
    })

    detector = CardinalityDetector(
        max_categories=2
    )

    detector._fit(df)

    # pandas nunique() ignores NaN by default
    assert detector.cardinality_stats["Category"] == 2

    assert detector.problems == []


def test_threshold_boundary():
    """Exactly max_categories should not be considered high cardinality."""

    df = pd.DataFrame({
        "Category": [
            "A",
            "B",
            "C",
            "D",
            "E"
        ]
    })

    detector = CardinalityDetector(
        max_categories=5
    )

    detector._fit(df)

    assert detector.cardinality_stats["Category"] == 5

    assert detector.problems == []


def test_multiple_categorical_columns():
    """Several categorical columns should be analyzed independently."""

    df = pd.DataFrame({
        "LowCardinality": [
            "A", "B", "A",
            "C", "A", "B"
        ],
        "HighCardinality": [
            "A", "B", "C",
            "D", "E", "F"
        ]
    })

    detector = CardinalityDetector(
        max_categories=3
    )

    detector._fit(df)

    assert detector.cardinality_stats["LowCardinality"] == 3

    assert detector.cardinality_stats["HighCardinality"] == 6

    assert len(detector.problems) == 1

    assert detector.problems[0]["column"] == "HighCardinality"