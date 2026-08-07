# Detectors

## Overview

The `detectors` package provides a collection of reusable data quality detectors for tabular datasets. Each detector focuses on identifying a specific type of data quality issue without modifying the input data.

These detectors are designed to help users analyze datasets before applying preprocessing or machine learning pipelines.

---

## Purpose

The main objectives of this package are to:

- Detect common data quality issues.
- Provide detailed statistics and diagnostics.
- Generate actionable recommendations.
- Offer a consistent API across all detectors.
- Integrate seamlessly with the library's preprocessing workflow.

---

## Available Detectors

| Detector               | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `MissingValueDetector` | Detects columns with excessive missing values.           |
| `OutlierDetector`      | Detects numerical outliers using IQR or Z-Score methods. |
| `CorrelationDetector`  | Identifies highly correlated numerical features.         |
| `CardinalityDetector`  | Detects categorical features with high cardinality.      |
| `DuplicateDetector`    | Detects duplicated rows in a dataset.                    |

---

## Package Structure

```text
detectors/
│
├── __init__.py
├── missing_value.py
├── outlier.py
├── correlation.py
├── cardinality.py
└── duplicate.py
```

Each detector is implemented in its own module to improve readability, maintainability, and extensibility.

---

## Common Interface

All detectors inherit from `BaseDetector` and expose a consistent interface.

Typical workflow:

```python
from preprocessing.tabular.detectors import MissingValueDetector

detector = MissingValueDetector()

detector.fit(df)

print(detector.problems)
```

---

## Output

After calling `fit()`, each detector populates the following attributes:

| Attribute    | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `problems`   | List of detected issues.                                               |
| `statistics` | Detector-specific statistics (attribute name depends on the detector). |

For example:

```python
detector.problems
```

```python
[
    {
        "column": "Age",
        "description": "23.5% missing values",
        "severity": "high",
        "suggestion": "Consider median imputation"
    }
]
```

---

## Severity Levels

Detected issues are classified into severity levels to help prioritize corrective actions.

Typical levels include:

- **Low** – Minor issue with limited impact.
- **Medium** – Moderate issue that should be reviewed.
- **High** – Significant issue requiring attention before model training.

---

## Supported Data Types

Depending on the detector, the following column types are supported:

| Detector             | Supported Columns   |
| -------------------- | ------------------- |
| MissingValueDetector | All columns         |
| OutlierDetector      | Numerical columns   |
| CorrelationDetector  | Numerical columns   |
| CardinalityDetector  | Categorical columns |
| DuplicateDetector    | Entire dataset      |

---

## Design Principles

The detectors package follows several design principles:

- Single Responsibility Principle (one detector per file)
- Consistent public API
- Read-only analysis (no data modification)
- Reusable and modular components
- Easy integration into preprocessing pipelines

---

## Extending the Package

To implement a new detector:

1. Create a new module inside the `detectors` package.
2. Inherit from `BaseDetector`.
3. Implement the required `_fit()` and `_transform()` methods.
4. Export the new detector in `__init__.py`.

---

## Testing

Each detector should have its own dedicated unit tests.

Example structure:

```text
tests/
└── detectors/
    ├── test_missing_value.py
    ├── test_outlier.py
    ├── test_correlation.py
    ├── test_cardinality.py
    └── test_duplicate.py
```

---

## Future Improvements

Potential future detectors include:

- ConstantFeatureDetector
- DriftDetector
- LeakageDetector
- DatetimeDetector
- DistributionDetector
- ClassImbalanceDetector
- MulticollinearityDetector

---

## Related Modules

The detectors package is typically used alongside:

- `transformers`
- `encoders`
- `imputers`
- `validators`
- `feature_selection`

Together, these modules provide a complete preprocessing workflow for tabular data.
