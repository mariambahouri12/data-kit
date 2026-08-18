"""
Dataset and preprocessing context management.

NOTE: this class holds mutable per-dataset state on the instance
(dataset_context, current_dataset_name, _dataset_fingerprint). If a
single ContextManager instance is shared across multiple users or
sessions (as it currently is via factory.create_assistant()), that
state will leak between them. Scope one ContextManager per
session/user rather than sharing a single instance app-wide.
"""

import hashlib
import json
from typing import Any, Optional

import pandas as pd

from .dataset_context import DatasetContextBuilder
from .preprocessing_context import PreprocessingContext


class ContextManager:
    """Manage dataset and preprocessing context."""

    def __init__(
        self,
        dataset_builder: DatasetContextBuilder,
        preprocessing_context: PreprocessingContext,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.preprocessing_context = preprocessing_context

        self.dataset_context: Optional[dict[str, Any]] = None
        self.current_dataset_name: Optional[str] = None
        self._dataset_fingerprint: Optional[str] = None

    def update_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:

        if dataframe is None or dataframe.empty:
            self.clear_dataset()
            return

        self.dataset_context = self.dataset_builder.build(
            dataframe,
            dataset_name,
        )

        self.current_dataset_name = dataset_name

        self._dataset_fingerprint = self._compute_fingerprint(
            dataframe,
            dataset_name,
        )

    def clear_dataset(self) -> None:
        """Clear dataset-related context."""

        self.dataset_context = None
        self.current_dataset_name = None
        self._dataset_fingerprint = None

    def update_preprocessing(
        self,
        operation_name: str,
        columns: list[str],
        parameters: dict | None = None,
    ) -> None:

        self.preprocessing_context.add_operation(
            operation_name,
            columns,
            parameters,
        )

    def get_dataset_fingerprint(self) -> Optional[str]:
        """Return current dataset fingerprint."""
        return self._dataset_fingerprint

    def get_full_context(self) -> dict[str, Any]:
        """Return complete context."""

        return {
            "dataset": self.dataset_context,
            "preprocessing": self.preprocessing_context.get_context(),
        }

    def to_prompt_format(self) -> str:
        """Serialize context as JSON."""

        return json.dumps(
            self.get_full_context(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    def to_markdown_context(self) -> str:
        """Create a compact Markdown context for the LLM."""

        if not self.dataset_context:
            return "No dataset loaded."

        dataset = self.dataset_context

        lines = [
            f"## Dataset: {dataset.get('dataset_name', 'unknown')}",
            "",
            f"- Rows: {dataset.get('shape', {}).get('rows', 0)}",
            f"- Columns: {dataset.get('shape', {}).get('columns', 0)}",
        ]

        problems = dataset.get("detected_problems", [])

        lines.extend(["", "## Detected issues"])

        if problems:
            for problem in problems:
                lines.append(f"- {problem}")
        else:
            lines.append("- No issues detected.")

        preprocessing = self.preprocessing_context.get_context().get(
            "operations", []
        )

        lines.extend(["", "## Preprocessing"])

        if preprocessing:
            for operation in preprocessing:
                name = operation.get("operation", "unknown")

                columns = ", ".join(operation.get("columns", []))

                lines.append(
                    f"- {name}" + (f" on {columns}" if columns else "")
                )
        else:
            lines.append("- No preprocessing applied.")

        return "\n".join(lines)

    @staticmethod
    def _compute_fingerprint(
        dataframe: pd.DataFrame,
        dataset_name: str,
    ) -> str:
        """
        Compute a fingerprint identifying this dataset's schema AND
        content. Content is included (via a vectorized row hash) so
        that replacing the dataset with another one sharing the same
        shape/dtypes still invalidates the private cache.
        """

        columns_signature = ",".join(
            f"{column}:{dtype}"
            for column, dtype in zip(
                dataframe.columns,
                dataframe.dtypes.astype(str),
            )
        )

        content_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                dataframe,
                index=False,
            ).values.tobytes()
        ).hexdigest()[:16]

        signature = (
            f"{dataset_name}|"
            f"{dataframe.shape[0]}x{dataframe.shape[1]}|"
            f"{columns_signature}|"
            f"{content_hash}"
        )

        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]