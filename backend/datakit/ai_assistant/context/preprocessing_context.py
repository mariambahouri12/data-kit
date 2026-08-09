"""
Preprocessing context tracking.
"""

from typing import Any


class PreprocessingContext:
    """Track preprocessing operations."""

    def __init__(self) -> None:
        self._operations: list[dict[str, Any]] = []

    def add_operation(
        self,
        operation_name: str,
        columns: list[str],
        parameters: dict | None = None,
    ) -> None:
        self._operations.append(
            {
                "operation": operation_name,
                "columns": columns,
                "parameters": parameters or {},
            }
        )

    def get_context(self) -> dict[str, Any]:
        """Return preprocessing history."""
        return {
            "operations": list(
                self._operations
            )
        }

    def clear(self) -> None:
        """Clear preprocessing history."""
        self._operations.clear()