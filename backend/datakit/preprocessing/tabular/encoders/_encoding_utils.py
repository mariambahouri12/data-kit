
# preprocessing/tabular/_encoding_utils.py

import hashlib
import warnings
from typing import Optional, List

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

from ..config import EncodingMethod


def split_by_cardinality(
    encoder,
    X: pd.DataFrame,
    columns: List[str]
) -> tuple:
    """Splits columns into (normally encodable, too high cardinality)."""
    normal, fallback = [], []

    for col in columns:
        n_unique = X[col].nunique()

        if n_unique <= encoder.max_categories:
            normal.append(col)
        else:
            fallback.append(col)

            warnings.warn(
                f"Column {col} has {n_unique} categories "
                f"(> max_categories={encoder.max_categories}), "
                f"falling back to frequency encoding.",
                RuntimeWarning,
            )

    return normal, fallback


def fit_onehot(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    encoder.encoder = OneHotEncoder(
        handle_unknown=encoder.handle_unknown,
        sparse_output=encoder.sparse,
        drop="if_binary",
        min_frequency=encoder.min_frequency,
    )

    encoder.encoder.fit(X[encoder.columns_to_encode])
    encoder.column_names = list(
        encoder.encoder.get_feature_names_out(
            encoder.columns_to_encode
        )
    )


def fit_label(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    """
    Note: LabelEncoder has no notion of unknown categories;
    unseen values are handled manually during transform (fallback -1).
    """
    encoder.encoder = {}

    for col in encoder.columns_to_encode:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        encoder.encoder[col] = le


def fit_target(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    """
    WARNING: data leakage if used on the entire dataset without CV.
    Use KFold target encoding in production.
    """
    target = resolve_target(encoder, y)
    global_mean = target.mean()
    tmp = X.assign(_target_=target.values)

    for col in encoder.columns_to_encode:
        group_sizes = tmp.groupby(col).size()
        group_means = tmp.groupby(col)["_target_"].mean()

        smoothing = group_sizes / (group_sizes + 10)
        smoothed = (
            smoothing * group_means
            + (1 - smoothing) * global_mean
        )

        encoder.mapping[col] = {
            "encoding": smoothed.to_dict()
        }


def fit_frequency(
    encoder,
    X: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> None:
    for col in (
        columns
        if columns is not None
        else encoder.columns_to_encode
    ):
        encoder.mapping[col] = {
            "encoding": X[col].value_counts(
                normalize=True
            ).to_dict()
        }


def fit_binary(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    for col in encoder.columns_to_encode:
        categories = X[col].unique()
        n_bits = max(1, len(categories).bit_length())

        binary_mapping = {
            cat: [
                int(b)
                for b in format(i, f"0{n_bits}b")
            ]
            for i, cat in enumerate(categories)
        }

        encoder.mapping[col] = {
            "encoding": binary_mapping,
            "n_bits": n_bits
        }


def fit_catboost(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    target = resolve_target(encoder, y)
    global_mean = target.mean()

    for col in encoder.columns_to_encode:
        group_sizes = X.groupby(col).size()
        group_sums = target.groupby(X[col]).sum()

        cat_means = (
            (group_sums + global_mean)
            / (group_sizes + 1)
        ).to_dict()

        encoder.mapping[col] = {
            "encoding": cat_means
        }


def fit_hash(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    """Truncated MD5 hash for cross-run reproducibility (unlike hash())."""
    for col in encoder.columns_to_encode:
        hash_mapping = {
            cat: int(
                hashlib.md5(
                    str(cat).encode()
                ).hexdigest(),
                16
            ) % 1_000_000
            for cat in X[col].unique()
        }

        encoder.mapping[col] = {
            "encoding": hash_mapping
        }


def fit_ordinal(
    encoder,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> None:
    encoder.encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    encoder.encoder.fit(
        X[encoder.columns_to_encode]
    )


def resolve_target(
    encoder,
    y: Optional[pd.Series]
) -> pd.Series:
    if y is None:
        raise ValueError(
            f"{encoder.method} encoder requires a target variable (y)."
        )

    return y


# -- Transform -------------------------------------------------------------


def apply_frequency_fallback(
    encoder,
    X: pd.DataFrame
) -> pd.DataFrame:
    """Applies frequency encoding to columns with too high cardinality for the selected method."""
    X_copy = X.copy()

    for col in encoder.fallback_columns:
        encoding_map = encoder.mapping[col]["encoding"]
        X_copy[col] = X_copy[col].map(
            encoding_map
        ).fillna(0)

    return X_copy


def transform_onehot(
    encoder,
    X: pd.DataFrame
) -> pd.DataFrame:
    if not encoder.columns_to_encode:
        return X.copy()

    encoded = encoder.encoder.transform(
        X[encoder.columns_to_encode]
    )

    if encoder.sparse:
        encoded = encoded.toarray()

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.column_names,
        index=X.index
    )

    return pd.concat(
        [
            X.drop(columns=encoder.columns_to_encode),
            encoded_df
        ],
        axis=1
    )


def transform_ordinal(
    encoder,
    X: pd.DataFrame
) -> pd.DataFrame:
    X_copy = X.copy()

    if (
        encoder.columns_to_encode
        and encoder.encoder is not None
    ):
        X_copy[encoder.columns_to_encode] = (
            encoder.encoder.transform(
                X_copy[encoder.columns_to_encode]
            )
        )

    return X_copy


def transform_other(
    encoder,
    X: pd.DataFrame
) -> pd.DataFrame:
    X_copy = X.copy()

    for col in encoder.columns_to_encode:
        if col not in X_copy.columns:
            continue

        if encoder.method == EncodingMethod.LABEL:
            X_copy[col] = transform_label_column(
                encoder,
                X_copy[col]
            )

        elif encoder.method == EncodingMethod.BINARY:
            X_copy = transform_binary_column(
                encoder,
                X_copy,
                col
            )

        else:
            # TARGET, FREQUENCY, CATBOOST, HASH
            # share the same mapping logic
            X_copy[col] = transform_mapped_column(
                encoder,
                X_copy[col],
                col
            )

    return X_copy


def transform_label_column(
    encoder,
    column: pd.Series
) -> pd.Series:
    le: LabelEncoder = encoder.encoder[column.name]
    known_classes = set(le.classes_)

    return column.astype(str).apply(
        lambda x: (
            le.transform([x])[0]
            if x in known_classes
            else -1
        )
    )


def transform_mapped_column(
    encoder,
    column: pd.Series,
    col_name: str
) -> pd.Series:
    encoding_map = encoder.mapping[col_name]["encoding"]
    result = column.map(encoding_map)

    if result.isna().any():
        if encoder.handle_unknown == "ignore":
            result = result.fillna(0)
        else:
            raise ValueError(
                f"Unknown categories found in column {col_name}"
            )

    return result


def transform_binary_column(
    encoder,
    X: pd.DataFrame,
    col_name: str
) -> pd.DataFrame:
    n_bits = encoder.mapping[col_name]["n_bits"]
    encoding_map = encoder.mapping[col_name]["encoding"]
    default_bits = [0] * n_bits

    for i in range(n_bits):
        X[f"{col_name}_bit_{i}"] = X[col_name].apply(
            lambda x, i=i: encoding_map.get(
                x,
                default_bits
            )[i]
        )

    return X.drop(columns=[col_name])
