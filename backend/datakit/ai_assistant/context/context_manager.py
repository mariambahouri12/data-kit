# datakit/ai_assistant/context/context_manager.py
"""
Context manager for DataKit AI Assistant.
"""

import json
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages dataset and preprocessing context for the AI assistant."""

    def __init__(self, dataset_builder, preprocessing_context):
        self.dataset_builder = dataset_builder
        self.preprocessing_context = preprocessing_context
        self.dataset_context: Optional[Dict[str, Any]] = None
        self.current_dataset_name: Optional[str] = None

    def update_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> None:
        """Generate and store dataset context."""
        if dataframe is None or dataframe.empty:
            self.dataset_context = None
            logger.info("Dataset context cleared")
            return

        self.dataset_context = self.dataset_builder.build(
            dataframe,
            dataset_name
        )
        self.current_dataset_name = dataset_name
        logger.info(f"Dataset context updated: {dataset_name} ({len(dataframe)} rows)")

    def update_preprocessing(
        self,
        operation_name: str,
        columns: list,
        parameters: dict = None
    ) -> None:
        """Add preprocessing operation to context."""
        self.preprocessing_context.add_operation(
            operation_name,
            columns,
            parameters
        )
        logger.debug(f"Preprocessing operation added: {operation_name}")

    def get_full_context(self) -> Dict[str, Any]:
        """Return complete AI context."""
        return {
            "dataset": self.dataset_context,
            "preprocessing": self.preprocessing_context.get_context()
        }

    def to_prompt_format(self) -> str:
        """Convert context into readable text for PromptManager."""
        context = self.get_full_context()
        return json.dumps(context, indent=2, ensure_ascii=False)

    def to_markdown_context(self) -> str:
        """Convert context to markdown format for better LLM readability."""
        if not self.dataset_context:
            return "Aucune donnée chargée."

        lines = []
        
        # Dataset overview
        lines.append(f"## 📊 Dataset: {self.dataset_context.get('dataset_name', 'unknown')}")
        lines.append("")
        lines.append(f"- **Lignes:** {self.dataset_context.get('shape', {}).get('rows', 0)}")
        lines.append(f"- **Colonnes:** {self.dataset_context.get('shape', {}).get('columns', 0)}")
        
        # Quality
        quality = self.dataset_context.get('quality', {})
        lines.append("")
        lines.append("### Qualité des données")
        lines.append(f"- **Doublons:** {quality.get('duplicates', 0)}")
        lines.append(f"- **Valeurs manquantes totales:** {quality.get('total_missing_values', 0)}")
        
        # Columns
        lines.append("")
        lines.append("### Colonnes")
        
        for col in self.dataset_context.get('columns', []):
            lines.append("")
            lines.append(f"#### `{col.get('name', 'unknown')}`")
            lines.append(f"- **Type:** {col.get('dtype', 'unknown')}")
            lines.append(f"- **Valeurs manquantes:** {col.get('missing_count', 0)} ({col.get('missing_percentage', 0):.1f}%)")
            lines.append(f"- **Valeurs uniques:** {col.get('unique_values', 0)}")
            
            if 'statistics' in col:
                stats = col['statistics']
                lines.append(f"- **Moyenne:** {stats.get('mean', 'N/A')}")
                lines.append(f"- **Médiane:** {stats.get('median', 'N/A')}")
                lines.append(f"- **Écart-type:** {stats.get('std', 'N/A')}")
            
            if 'top_values' in col:
                top_vals = ", ".join([f"{k}: {v}" for k, v in col['top_values'].items()])
                lines.append(f"- **Valeurs fréquentes:** {top_vals}")
        
        # Preprocessing history
        preprocessing = self.preprocessing_context.get_context()
        if preprocessing and preprocessing.get('operations'):
            lines.append("")
            lines.append("### 🔧 Opérations de prétraitement effectuées")
            for op in preprocessing['operations']:
                cols_str = ", ".join(op.get('columns', []))
                lines.append(f"- **{op.get('operation')}** sur: {cols_str}")
                if op.get('parameters'):
                    lines.append(f"  - Paramètres: {op['parameters']}")
        else:
            lines.append("")
            lines.append("### 🔧 Opérations de prétraitement")
            lines.append("- Aucune opération effectuée pour l'instant")

        return "\n".join(lines)