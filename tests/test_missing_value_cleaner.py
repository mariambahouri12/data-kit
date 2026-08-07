import numpy as np
import pandas as pd
import pytest

from src.datakit.preprocessing.tabular.cleaners.missing_value import MissingValueCleaner
from src.datakit.preprocessing.tabular.config import ImputationMethod


def test_median_imputation_numeric_column():
    """Missing numeric values should be replaced by median."""

    df = pd.DataFrame({
        "Age": [10, 20, np.nan, 40]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.MEDIAN
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert result["Age"].isna().sum() == 0

    # Median of [10,20,40] = 20
    assert result.loc[2, "Age"] == 20


def test_mean_imputation_numeric_column():
    """Missing values should be replaced by mean."""

    df = pd.DataFrame({
        "Score": [10, 20, np.nan, 30]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.MEAN
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert result["Score"].isna().sum() == 0

    assert result.loc[2, "Score"] == 20


def test_constant_imputation():
    """Constant strategy should fill missing values with given value."""

    df = pd.DataFrame({
        "Value": [1, np.nan, 3]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.CONSTANT,
        fill_value=-1
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert result["Value"].isna().sum() == 0

    assert result.loc[1, "Value"] == -1


def test_constant_without_fill_value_uses_zero():
    """Constant strategy without fill_value should use 0."""

    df = pd.DataFrame({
        "Value": [1, np.nan, 3]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.CONSTANT
    )

    with pytest.warns(RuntimeWarning):
        cleaner._fit(df)

    result = cleaner._transform(df)

    assert result.loc[1, "Value"] == 0


def test_categorical_imputation():
    """Missing categorical values should be replaced by most frequent value."""

    df = pd.DataFrame({
        "City": [
            "Paris",
            "Paris",
            np.nan,
            "London"
        ]
    })

    cleaner = MissingValueCleaner()

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert result["City"].isna().sum() == 0

    assert result.loc[2, "City"] == "Paris"


def test_column_specific_strategy():
    """Different columns can use different strategies."""

    df = pd.DataFrame({
        "Age": [10, 20, np.nan, 40],
        "Salary": [1000, 2000, np.nan, 4000]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.MEDIAN,
        column_strategies={
            "Salary": ImputationMethod.MEAN
        }
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    # Median Age = 20
    assert result.loc[2, "Age"] == 20

    # Mean Salary = 2333.33
    assert result.loc[2, "Salary"] == pytest.approx(
        2333.33,
        rel=1e-2
    )


def test_selected_columns_only():
    """Only selected columns should be imputed."""

    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [10, np.nan, 30]
    })

    cleaner = MissingValueCleaner(
        columns=["A"]
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert result["A"].isna().sum() == 0

    assert result["B"].isna().sum() == 1


def test_drop_strategy():
    """DROP strategy should remove rows containing missing values."""

    df = pd.DataFrame({
        "Age": [10, np.nan, 30]
    })

    cleaner = MissingValueCleaner(
        strategy=ImputationMethod.DROP
    )

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert len(result) == 2

    assert result["Age"].isna().sum() == 0


def test_no_missing_values_keeps_dataframe():
    """Data without missing values should keep same values."""

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

    cleaner = MissingValueCleaner()

    cleaner._fit(df)

    result = cleaner._transform(df)

    pd.testing.assert_frame_equal(
        result,
        df,
        check_dtype=False
    )


def test_fit_stores_column_types():
    """Fit should correctly identify numeric and categorical columns."""

    df = pd.DataFrame({
        "Age": [10, 20, 30],
        "City": [
            "Paris",
            "London",
            "Rome"
        ]
    })

    cleaner = MissingValueCleaner()

    cleaner._fit(df)

    assert "Age" in cleaner.column_types["numeric"]

    assert "City" in cleaner.column_types["categorical"]