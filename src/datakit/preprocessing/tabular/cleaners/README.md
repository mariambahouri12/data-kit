# Cleaners

## Overview

The `cleaners` package provides reusable data cleaning components for tabular datasets. Each cleaner is responsible for handling a specific type of data quality issue while preserving a consistent API across the library.

These components are designed to transform datasets into a cleaner and more suitable format for downstream preprocessing and machine learning tasks.

---

## Purpose

The main objectives of this package are to:

- Handle common data quality issues.
- Provide configurable cleaning strategies.
- Preserve consistency across preprocessing pipelines.
- Offer reusable and modular cleaning components.
- Integrate seamlessly with the library's preprocessing workflow.

---

## Available Cleaners

| Cleaner               | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `MissingValueCleaner` | Handles missing values using configurable imputation strategies or row removal. |
| `OutlierCleaner`      | Detects and treats outliers using configurable detection and cleaning methods.  |
| `DuplicateCleaner`    | Removes duplicated rows from a dataset.                                         |

---

## Package Structure

```text
cleaners/
│
├── __init__.py
├── missing_value.py
├── outlier.py
└── duplicate.py
```

Each cleaner is implemented in its own module to improve readability, maintainability, and extensibility.

---

## Common Interface

All cleaners inherit from `BasePreprocessor` and expose a consistent interface.

Typical workflow:

```python
from preprocessing.tabular.cleaners import MissingValueCleaner

cleaner = MissingValueCleaner(strategy="median")

cleaner.fit(df)

clean_df = cleaner.transform(df)
```

---

## Supported Cleaning Operations

### MissingValueCleaner

Handles missing values separately for numerical and categorical columns.

Supported numerical imputation strategies include:

- Mean
- Median
- Most Frequent
- Constant
- K-Nearest Neighbors (KNN)
- Drop rows containing missing values

Categorical columns are imputed using the **most frequent** category.

---

### OutlierCleaner

Detects and handles numerical outliers.

Supported detection methods:

- Interquartile Range (IQR)
- Z-Score

Supported cleaning actions:

- Winsorization (clipping)
- Row removal

---

### DuplicateCleaner

Removes duplicated observations from a dataset.

Users can configure:

- Duplicate subset columns
- Duplicate retention strategy (`first`, `last`, or `False`)

---

## Design Principles

The package follows several software engineering principles:

- Single Responsibility Principle (one cleaner per module)
- Modular architecture
- Consistent public API
- Reusable components
- Safe and configurable preprocessing
- Compatibility with pandas DataFrames

---

## Output

Each cleaner returns a cleaned `pandas.DataFrame`.

Depending on the selected strategy, cleaning operations may include:

- Imputing missing values
- Removing rows
- Clipping outliers
- Removing duplicated observations

---

## Extending the Package

To implement a new cleaner:

1. Create a new module inside the package.
2. Inherit from `BasePreprocessor`.
3. Implement the required `_fit()` and `_transform()` methods.
4. Export the new cleaner in `__init__.py`.

---

## Testing

Each cleaner should have its own dedicated unit tests.

Example structure:

```text
tests/
└── cleaners/
    ├── test_missing_value.py
    ├── test_outlier.py
    └── test_duplicate.py
```

---

## Future Improvements

Potential future cleaners include:

- ConstantFeatureCleaner
- LowVarianceCleaner
- InvalidValueCleaner
- StringCleaner
- DateCleaner
- CategoryCleaner
- InfiniteValueCleaner
- WhitespaceCleaner

---

## Related Modules

The cleaners package is typically used together with:

- `detectors`
- `feature_engineering`
- `encoders`
- `scalers`
- `feature_selection`

Together, these modules provide a complete preprocessing workflow for tabular datasets.
