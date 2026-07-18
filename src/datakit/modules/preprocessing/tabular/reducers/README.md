# Reducers

## Overview

The `reducers` package provides reusable dimensionality reduction and feature selection components for tabular datasets.

It contains tools for reducing the number of features while preserving the most informative characteristics of the data. These components help improve model performance, reduce computational cost, and mitigate the curse of dimensionality.

The package follows a modular design where each reduction technique is implemented independently while sharing a consistent preprocessing interface.

---

## Purpose

The main objectives of this package are to:

- Reduce feature dimensionality.
- Select the most informative features.
- Improve model efficiency and generalization.
- Provide multiple feature reduction strategies.
- Integrate seamlessly into preprocessing pipelines.

---

## Package Structure

```text
reducers/
│
├── __init__.py
├── README.md
├── feature_selector.py
├── pca.py
├── lda.py
└── _selection_utils.py
```

### Files

| File                  | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| `feature_selector.py` | Feature selection methods based on statistical tests and model importance. |
| `pca.py`              | Principal Component Analysis (PCA) dimensionality reduction.               |
| `lda.py`              | Linear Discriminant Analysis (LDA) supervised dimensionality reduction.    |
| `_selection_utils.py` | Internal helper functions used by feature selection algorithms.            |

---

## Available Reducers

| Component         | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `FeatureSelector` | Selects the most relevant input features using multiple selection strategies.      |
| `PCAReducer`      | Performs unsupervised dimensionality reduction using Principal Component Analysis. |
| `LDAReducer`      | Performs supervised dimensionality reduction using Linear Discriminant Analysis.   |

---

## Feature Selection Methods

`FeatureSelector` currently supports:

| Method      | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| Variance    | Removes numerical features with low variance.                               |
| Correlation | Selects features based on statistical correlation with the target variable. |
| Importance  | Uses Random Forest feature importance scores.                               |
| RFE         | Recursive Feature Elimination using a Random Forest estimator.              |
| None        | Keeps all features unchanged.                                               |

---

## Dimensionality Reduction Methods

### PCA

Principal Component Analysis is an unsupervised technique that:

- Projects data onto orthogonal components.
- Maximizes explained variance.
- Reduces redundancy between numerical variables.
- Can automatically retain a desired amount of explained variance.

---

### LDA

Linear Discriminant Analysis is a supervised technique that:

- Uses class labels during training.
- Maximizes class separability.
- Produces discriminant components.
- Is available only for classification tasks.

---

## Common Interface

All reducers inherit from `BasePreprocessor`.

Typical usage:

```python
from preprocessing.tabular.reducers import PCAReducer

reducer = PCAReducer(variance_ratio=0.95)

reducer.fit(X_train)

X_train_reduced = reducer.transform(X_train)
X_test_reduced = reducer.transform(X_test)
```

---

## Internal Design

The package separates public classes from internal implementation utilities.

### Public modules

- `feature_selector.py`
- `pca.py`
- `lda.py`

Each file contains one public preprocessing component.

### Private utilities

`_selection_utils.py` contains helper functions used internally by `FeatureSelector`, including:

- Random Forest model creation
- Categorical feature encoding
- Shared helper utilities

These functions are private implementation details and are **not intended for direct use**.

---

## Design Principles

The package follows several software engineering principles:

- Single Responsibility Principle
- One public class per module
- Modular architecture
- Reusable internal utilities
- Consistent preprocessing API
- Easy extensibility
- Separation between public API and implementation details

---

## Output

Depending on the selected reducer, the output may contain:

- A subset of the original features
- Principal components (PC1, PC2, ...)
- Linear discriminant components (LD1, LD2, ...)
- Feature importance information

---

## Extending the Package

To add a new reduction technique:

1. Create a new module inside the package.
2. Inherit from `BasePreprocessor`.
3. Implement the required `_fit()` and `_transform()` methods.
4. Export the new reducer in `__init__.py`.
5. Add unit tests.
6. Update this documentation.

---

## Testing

Recommended test structure:

```text
tests/
└── reducers/
    ├── test_feature_selector.py
    ├── test_pca.py
    └── test_lda.py
```

Tests should validate:

- Fit/transform behavior
- Selected feature correctness
- PCA explained variance
- LDA component generation
- Statistical feature selection
- Model-based feature selection
- Edge cases and invalid inputs

---

## Future Improvements

Potential future reducers include:

- Kernel PCA
- Incremental PCA
- Sparse PCA
- Truncated SVD
- ICA (Independent Component Analysis)
- UMAP
- t-SNE
- Autoencoder-based reduction

---

## Related Modules

The reducers package is commonly used together with:

- `cleaners`
- `detectors`
- `encoders`
- `feature_engineering`
- `scalers`

Together, these modules provide a complete preprocessing pipeline for tabular machine learning.
