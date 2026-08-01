import pandas as pd
import pytest
from types import SimpleNamespace
from io import StringIO

from src.datakit.data.loader import FileLoader
from src.datakit.exceptions import EmptyFileError


def test_csv_load():

    csv_content = StringIO(
        "name,age\n"
        "Alice,20\n"
        "Ali,30"
    )

    csv_content.name = "employees.csv"

    loader = FileLoader()

    df = loader.load(csv_content)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,2)
    assert list(df.columns) == ["name", "age"]
    assert df.iloc[0]["name"] == "Alice"


def test_invalid_extension():

    fake_file = SimpleNamespace(name="employees.txt")
    loader = FileLoader()

    with pytest.raises(ValueError):

        loader.load(fake_file)

def test_load_empty():

    empty_file = StringIO("")
    empty_file.name = "empty_file.csv"

    loader = FileLoader()

    with pytest.raises(EmptyFileError, match="Uploaded file is empty"):
        loader.load(empty_file)



       