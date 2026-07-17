# Encoders

## Overview

The `encoders` package provides reusable categorical encoding components for tabular datasets.

Its primary component, `CategoricalEncoder`, supports multiple encoding strategies through a single, consistent interface while automatically handling high-cardinality features.

The package is designed to simplify categorical preprocessing while remaining modular, extensible, and compatible with machine learning workflows.

---

## Purpose

The main objectives of this package are to:

- Encode categorical features using different strategies.
- Automatically handle high-cardinality variables.
- Provide a unified API for all encoding methods.
- Reduce feature explosion during one-hot encoding.
- Integrate seamlessly into preprocessing pipelines.

---

## Package Structure

```text
encoders/
│
├── __init__.py
├── README.md
├── categorical.py
└── _encoding_utils.py
```

### Files

| File                 | Description                                                                    |
| -------------------- | ------------------------------------------------------------------------------ |
| `categorical.py`     | Contains the public `CategoricalEncoder` class.                                |
| `_encoding_utils.py` | Internal helper functions implementing encoding and transformation algorithms. |

---

## Available Encoder

| Encoder              | Description                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| `CategoricalEncoder` | Flexible encoder supporting multiple categorical encoding strategies through a single interface. |

---

## Supported Encoding Methods

The encoder currently supports:

| Method    | Description                                                 |
| --------- | ----------------------------------------------------------- |
| One-Hot   | Creates one binary feature per category.                    |
| Label     | Assigns an integer to each category.                        |
| Ordinal   | Encodes categories according to learned ordinal values.     |
| Frequency | Replaces categories by their occurrence frequency.          |
| Target    | Encodes categories using target statistics with smoothing.  |
| CatBoost  | Uses target-based encoding inspired by CatBoost.            |
| Binary    | Encodes categories into binary digits.                      |
| Hash      | Generates deterministic hash-based integer representations. |
| None      | Leaves categorical features unchanged.                      |

---

## High Cardinality Handling

To prevent excessive feature expansion, columns exceeding the configured `max_categories` threshold are automatically encoded using **Frequency Encoding**, regardless of the selected encoding strategy.

This behavior helps:

- Reduce memory usage.
- Improve training efficiency.
- Prevent extremely wide feature matrices.

---

## Common Interface

All encoders inherit from `BasePreprocessor`.

Typical usage:

```python
from preprocessing.tabular.encoders import CategoricalEncoder

encoder = CategoricalEncoder(method="one_hot")

encoder.fit(X_train)

X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
```

---

## Internal Design

The package separates the public API from implementation details.

### `categorical.py`

Contains:

- `CategoricalEncoder`
- Public preprocessing API
- High-level workflow

### `_encoding_utils.py`

Contains internal helper functions responsible for:

- Fitting encoding models
- Performing transformations
- Target validation
- Frequency fallback logic
- Binary encoding utilities
- Hash encoding
- Cardinality management

These helper functions are private to the package and are **not intended for direct use**.

---

## Design Principles

The package follows several software engineering principles:

- Single Responsibility Principle
- Modular implementation
- Clean separation between API and algorithms
- Consistent preprocessing interface
- Reusable internal utilities
- Easy extensibility

---

## Output

Depending on the selected method, the encoder returns a transformed `pandas.DataFrame` containing:

- Numerical encoded features
- One-hot expanded features
- Binary encoded columns
- Frequency or target encoded values

---

## Extending the Package

To add a new encoding strategy:

1. Implement the encoding logic inside `_encoding_utils.py`.
2. Register the new method in `CategoricalEncoder`.
3. Update the corresponding configuration enum.
4. Add unit tests.
5. Update this documentation.

---

## Testing

Recommended test structure:

```text
tests/
└── encoders/
    └── test_categorical_encoder.py
```

Tests should validate:

- Fit/transform behavior
- Unknown category handling
- High-cardinality fallback
- Output feature names
- Each supported encoding strategy

---

## Future Improvements

Potential future encoding methods include:

- Leave-One-Out Encoding
- James-Stein Encoding
- Helmert Encoding
- Sum Encoding
- Weight of Evidence (WOE)
- Count Encoding
- Bayesian Target Encoding

---

## Related Modules

The encoders package is commonly used together with:

- `cleaners`
- `detectors`
- `feature_engineering`
- `scalers`
- `feature_selection`

Together, these modules provide a complete preprocessing workflow for tabular machine learning.
