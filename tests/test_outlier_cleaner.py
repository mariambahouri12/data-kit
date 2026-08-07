import numpy as np
import pandas as pd
import pytest

from src.datakit.preprocessing.tabular.cleaners.outlier import OutlierCleaner
from src.datakit.preprocessing.tabular.config import (
    OutlierMethod,
    OutlierAction
)


def test_iqr_compute_bounds():
    """IQR method should compute lower and upper bounds."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100]
    })

    cleaner = OutlierCleaner(
        method=OutlierMethod.IQR,
        threshold=1.5
    )

    cleaner._fit(df)

    assert "A" in cleaner.bounds

    assert "lower" in cleaner.bounds["A"]
    assert "upper" in cleaner.bounds["A"]

    assert cleaner.bounds["A"]["upper"] < 100



def test_winsorize_outliers():
    """Winsorization should replace extreme values by bounds."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100]
    })

    cleaner = OutlierCleaner(
        method=OutlierMethod.IQR,
        threshold=1.5,
        action=OutlierAction.WINSORIZE
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    upper_bound = cleaner.bounds["A"]["upper"]

    assert result["A"].max() == upper_bound

    assert result["A"].iloc[-1] == upper_bound



def test_drop_outliers():
    """DROP action should remove rows containing outliers."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100]
    })

    cleaner = OutlierCleaner(
        method=OutlierMethod.IQR,
        threshold=1.5,
        action=OutlierAction.DROP
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert len(result) == 4

    assert 100 not in result["A"].values



def test_zscore_method():
    """Z-score method should compute bounds."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 1000]
    })

    cleaner = OutlierCleaner(
        method=OutlierMethod.ZSCORE,
        threshold=2
    )

    cleaner._fit(df)

    assert "A" in cleaner.bounds

    assert cleaner.bounds["A"]["upper"] > 0



def test_specific_columns_only():
    """Only selected columns should be processed."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 100],
        "B": [100, 101, 102, 103, 104]
    })

    cleaner = OutlierCleaner(
        columns=["B"]
    )

    cleaner._fit(df)

    assert "B" in cleaner.bounds

    assert "A" not in cleaner.bounds



def test_ignore_non_numeric_columns():
    """Non numeric columns should not be included."""

    df = pd.DataFrame({
        "A": [10, 11, 12],
        "B": ["x", "y", "z"]
    })

    cleaner = OutlierCleaner()

    cleaner._fit(df)

    assert "A" in cleaner.bounds

    assert "B" not in cleaner.bounds



def test_no_outlier_dataframe():
    """Normal data should remain unchanged with winsorization."""

    df = pd.DataFrame({
        "A": [10, 11, 12, 13, 14]
    })

    cleaner = OutlierCleaner()

    cleaner._fit(df)

    result = cleaner._transform(df)

    pd.testing.assert_frame_equal(
        result,
        df
    )



def test_empty_numeric_column():
    """Empty numeric columns should be ignored."""

    df = pd.DataFrame({
        "A": [np.nan, np.nan]
    })

    cleaner = OutlierCleaner()

    cleaner._fit(df)

    assert cleaner.bounds == {}



def test_invalid_method():
    """Unsupported method should raise ValueError."""

    with pytest.raises(ValueError):

        OutlierCleaner(
            method="INVALID"
        )



def test_invalid_action():
    """Unsupported action should raise ValueError."""

    with pytest.raises(ValueError):

        OutlierCleaner(
            action="INVALID"
        )