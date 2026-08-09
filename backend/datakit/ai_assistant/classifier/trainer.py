"""
Training utilities for the query scope classifier.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..embeddings.embeddings import EmbeddingModel


class QueryClassifierTrainer:
    """Train and persist the private/shared query classifier."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.embedding_model = embedding_model

    def train(
        self,
        dataset_path: str,
        model_path: str,
    ) -> None:
        """
        Train the classifier.

        Expected CSV columns:
        - question
        - label
        """

        dataframe = pd.read_csv(dataset_path)

        required_columns = {"question", "label"}

        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if dataframe.empty:
            raise ValueError("Training dataset is empty.")

        labels = set(dataframe["label"].unique())

        if labels != {"private", "shared"}:
            raise ValueError(
                "Labels must contain exactly: "
                "'private' and 'shared'."
            )

        embeddings = self.embedding_model.encode_documents(
            dataframe["question"].astype(str).tolist()
        )

        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            dataframe["label"],
            test_size=0.2,
            random_state=42,
            stratify=dataframe["label"],
        )

        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        pipeline.fit(
            x_train,
            y_train,
        )

        score = pipeline.score(
            x_test,
            y_test,
        )

        print(f"Classifier validation accuracy: {score:.4f}")

        output_path = Path(model_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            output_path,
        )