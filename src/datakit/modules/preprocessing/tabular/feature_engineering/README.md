# Feature Engineering

## Overview

The `feature_engineering` package provides reusable feature generation techniques for tabular datasets. Each feature creator is responsible for generating a specific type of engineered feature while preserving a consistent API across the library.

These components are designed to enrich datasets with meaningful features that can improve the performance of machine learning models.

---

## Purpose

The main objectives of this package are to:

- Generate informative features from existing data.
- Simplify feature engineering workflows.
- Provide reusable and modular feature generators.
- Prevent feature explosion through configurable safeguards.
- Integrate seamlessly with the preprocessing pipeline.

---

## Available Feature Creators

| Feature Creator             | Description                                                             |
| --------------------------- | ----------------------------------------------------------------------- |
| `PolynomialFeatureCreator`  | Generates polynomial and interaction features from numerical variables. |
| `InteractionFeatureCreator` | Creates interaction features by multiplying numerical columns together. |
| `RatioFeatureCreator`       | Creates ratio features between pairs of numerical columns.              |
| `AggregationFeatureCreator` | Generates group-based statistical aggregation features.                 |
| `DateFeatureCreator`        | Extracts temporal features from datetime columns.                       |

---

## Package Structure

```text
feature_engineering/
│
├── __init__.py
├── polynomial.py
├── interaction.py
├── ratio.py
├── aggregation.py
└── date.py
```

Each feature creator is implemented in its own module to improve readability, maintainability, and extensibility.

---

## Common Interface

All feature creators inherit from `BasePreprocessor` and expose a consistent interface.

Typical workflow:

```python
from preprocessing.tabular.feature_engineering import PolynomialFeatureCreator

creator = PolynomialFeatureCreator(degree=2)

creator.fit(df)

transformed_df = creator.transform(df)
```

---

## Supported Feature Generators

### PolynomialFeatureCreator

Creates polynomial combinations of numerical features using scikit-learn's `PolynomialFeatures`.

Main options include:

- polynomial degree
- interaction-only features
- optional bias term
- maximum number of input features
- maximum number of generated features

---

### InteractionFeatureCreator

Creates multiplicative interaction features between numerical columns.

Example:

```
A × B
A × C
B × C
```

The maximum interaction order can be configured.

---

### RatioFeatureCreator

Creates ratio features between numerical columns while preventing division-by-zero using a configurable epsilon.

Example:

```
salary / age
age / salary
```

---

### AggregationFeatureCreator

Creates aggregated statistics based on a grouping column.

Supported aggregations include:

- mean
- sum
- std
- min
- max
- count

Example:

```
Average salary per department
Maximum sales per region
```

---

### DateFeatureCreator

Extracts useful temporal information from datetime columns.

Supported features include:

- Year
- Month
- Day
- Day of week
- Quarter
- Weekend indicator

Date columns can be specified manually or automatically detected.

---

## Design Principles

The package follows several software engineering principles:

- Single Responsibility Principle (one feature creator per module)
- Modular architecture
- Consistent public API
- Reusable components
- Safe feature generation with configurable limits
- Compatibility with pandas DataFrames

---

## Output

Each feature creator returns a transformed `pandas.DataFrame` containing the newly generated features.

Original columns are preserved unless explicitly replaced by the corresponding feature creator.

---

## Extending the Package

To implement a new feature creator:

1. Create a new module inside the package.
2. Inherit from `BasePreprocessor`.
3. Implement the required `_fit()` and `_transform()` methods.
4. Export the new class in `__init__.py`.

---

## Testing

Each feature creator should have its own dedicated unit tests.

Example structure:

```text
tests/
└── feature_engineering/
    ├── test_polynomial.py
    ├── test_interaction.py
    ├── test_ratio.py
    ├── test_aggregation.py
    └── test_date.py
```

---

## Future Improvements

Potential future feature creators include:

- LogFeatureCreator
- PowerFeatureCreator
- TrigonometricFeatureCreator
- RollingWindowFeatureCreator
- CyclicalDateFeatureCreator
- LagFeatureCreator
- TargetStatisticFeatureCreator
- TextFeatureCreator

---

## Related Modules

The feature engineering package is typically used together with:

- `detectors`
- `encoders`
- `imputers`
- `scalers`
- `feature_selection`

Together, these modules provide a complete preprocessing workflow for tabular datasets.
