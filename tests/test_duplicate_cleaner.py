import pandas as pd

from src.datakit.preprocessing.tabular.cleaners.duplicate import DuplicateCleaner


def test_remove_duplicate_rows():
    """Duplicate rows should be removed."""

    df = pd.DataFrame({
        "A": [1, 2, 2, 3],
        "B": ["x", "y", "y", "z"]
    })

    cleaner = DuplicateCleaner()

    cleaner._fit(df)

    result = cleaner._transform(df)

    assert len(result) == 3

    assert result.duplicated().sum() == 0


def test_keep_first_duplicate():
    """First identical row should be kept."""

    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": ["x", "x", "test"]
    })

    cleaner = DuplicateCleaner(
        keep="first"
    )

    result = cleaner._transform(df)

    assert len(result) == 2

    assert result.iloc[0]["B"] == "x"


def test_keep_last_duplicate():
    """Last identical row should be kept."""

    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": ["x", "x", "test"]
    })

    cleaner = DuplicateCleaner(
        keep="last"
    )

    result = cleaner._transform(df)

    assert len(result) == 2

    assert result.iloc[0]["A"] == 1


def test_remove_duplicates_with_subset():
    """Duplicates can be detected using selected columns."""

    df = pd.DataFrame({
        "ID": [1, 1, 2],
        "Value": ["old", "new", "test"]
    })

    cleaner = DuplicateCleaner(
        subset=["ID"]
    )

    result = cleaner._transform(df)

    assert len(result) == 2

    assert list(result["ID"]) == [1, 2]


def test_no_duplicates_keeps_dataframe():
    """Data without duplicates should remain unchanged."""

    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

    cleaner = DuplicateCleaner()

    result = cleaner._transform(df)

    pd.testing.assert_frame_equal(
        result,
        df
    )


def test_empty_dataframe():
    """Empty dataframe should be handled."""

    df = pd.DataFrame()

    cleaner = DuplicateCleaner()

    result = cleaner._transform(df)

    assert result.empty


def test_fit_is_stateless():
    """Fit should not modify state."""

    df = pd.DataFrame({
        "A": [1, 1, 2]
    })

    cleaner = DuplicateCleaner()

    result = cleaner._fit(df)

    assert result is None

    assert cleaner.subset is None

    assert cleaner.keep == "first"


def test_original_dataframe_is_not_modified():
    """Transform should not modify original dataframe."""

    df = pd.DataFrame({
        "A": [1, 1, 2]
    })

    original = df.copy()

    cleaner = DuplicateCleaner()

    cleaner._transform(df)

    pd.testing.assert_frame_equal(
        df,
        original
    )