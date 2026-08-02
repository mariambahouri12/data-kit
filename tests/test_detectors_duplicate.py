import pandas as pd

from src.datakit.preprocessing.tabular.detectors.duplicate import DuplicateDetector


def test_no_duplicate_lines():
    """ if there is no duplicate lines so no problme should be raised"""


    df = pd.DataFrame(
        {
            "A" : [1, 2, 3],
            "B" : ["x", "y", "z"]
        }
    )

    detector = DuplicateDetector()
    detector._fit(df)

    assert detector.duplicate_count == 0
    assert detector.problems == []

def detect_duplicate_lines():
    """duplicate rows should be detected"""

    df = pd.DataFrame(
        {
            "A" : [1, 2, 2, 4],
            "B" : ["x","y","y", "z" ]
        }
    )

    detector = DuplicateDetector()
    detector._fit(df)

    assert detector.duplicate_count == 2
    assert detector.problems[0]["description"] == "2 duplicate lines"


def test_empty_dataframe():
    """Empty dataframe should not create problems."""

    df = pd.DataFrame()

    detector = DuplicateDetector()
    detector._fit(df)

    assert detector.duplicate_count == 0
    assert detector.problems == []



