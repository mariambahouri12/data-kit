"""
Preprocessing context module.

Tracks preprocessing operations
performed during the workflow.
"""

from typing import List, Dict, Any, Optional

from ..models import PreprocessingOperation


class PreprocessingContext:

    def __init__(self):
        self._operations: List[PreprocessingOperation] = []

    def add_operation(
        self,
        operation_name: str,
        columns: List[str],
        parameters: Optional[dict] = None
    ) -> None:
        """
        Register preprocessing action.

        FIX (#5, duplication éliminée) : construit désormais un
        PreprocessingOperation (models.py) plutôt qu'un dict à la main —
        c'était la même structure de données dupliquée à deux endroits.
        """
        operation = PreprocessingOperation(
            operation=operation_name,
            columns=columns,
            parameters=parameters or {}
        )
        self._operations.append(operation)

    def get_context(self) -> Dict[str, Any]:
        """Return preprocessing history."""
        return {
            "operations": [op.to_dict() for op in self._operations]
        }

    def clear(self) -> None:
        self._operations = []