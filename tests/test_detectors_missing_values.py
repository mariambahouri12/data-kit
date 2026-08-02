import pandas as pd
import pytest

from src.datakit.preprocessing.tabular.detectors.missing_value import MissingValueDetector

def test_fit_no_missing_values():

                
    """ No missing value should produce no problem"""
    df = pd.DataFrame({
        "A" : [1, 2, 3],
        "B" : [4, 5, 6]
    }
    )
    detector = MissingValueDetector(threshold = 0.05)
    detector._fit(df)

    assert detector.missing_stats["total_missing"] == 0
    assert detector.missing_stats["total_cells"] == 6
    assert detector.missing_stats["missing_percentage"] == 0
    assert detector.problems == []

def test_fit_detects_missing_values_above_threshold():
    """columns above the threshold or equal to thee threshold should be reported"""
    df = pd.DataFrame(
        {
            "A" : [1, 3, None, 3],
            "B" : [1, 2, 3, 4]
        }
    )
    detector = MissingValueDetector(threshold = 0.25)
    detector._fit(df)

    assert detector.missing_stats["total_missing"] == 1
    assert detector.missing_stats["columns"]["A"]["missing_count"] == 1
    assert detector.missing_stats["columns"]["A"]["missing_percentage"] == 25.0

    assert (len(detector.problems)) == 1
    assert detector.problems[0]["column"] == "A"
    assert detector.problems[0]["description"] == "25.0% missing values"

def test_fit_missing_below_threshold():

    """Columns below the threshold should not be reported."""
    df = pd.DataFrame({
        "A": [1, 2, 3, None],     
        "B": [1, 2, 3, 4]
    })

    detector = MissingValueDetector(threshold=0.30)  
    detector._fit(df)

    assert detector.problems == []


def test_missing_statistics_are_correct():
    """Global statistics should be correctly computed."""
    df = pd.DataFrame({
        "A": [1, None],
        "B": [None, 2]
    })

    detector = MissingValueDetector()
    detector._fit(df)

    assert detector.missing_stats["total_missing"] == 2
    assert detector.missing_stats["total_cells"] == 4
    assert detector.missing_stats["missing_percentage"] == pytest.approx(50.0)

def test_column_statistics_are_correct():
    """Each column statistics should be stored."""
    df = pd.DataFrame({
        "A": [1, None, 3],      
        "B": [None, None, 3]    
    })

    detector = MissingValueDetector()
    detector._fit(df)

    stats = detector.missing_stats["columns"]

    assert stats["A"]["missing_count"] == 1
    assert stats["A"]["missing_percentage"] == pytest.approx(33.333333)

    assert stats["B"]["missing_count"] == 2
    assert stats["B"]["missing_percentage"] == pytest.approx(66.666666)

    









