# datakit/ai_assistant/llm/prompt_manager.py

"""
Prompt construction module for DataKit AI Assistant.
"""

import os
from typing import Optional

class PromptManager:
    """Manages prompt construction with templates."""

    # Système prompt fixe
    SYSTEM_PROMPT = """You are DataKit AI, an AI assistant specialized in Data Science.

Your role:
- Analyze dataset context
- Explain preprocessing choices
- Recommend ML practices
- Provide justified suggestions
"""

    DATASET_TEMPLATE = """
DATASET CONTEXT:
----------------
{dataset_context}
"""

    KNOWLEDGE_TEMPLATE = """
KNOWLEDGE BASE:
---------------
{knowledge}
"""

    QUESTION_TEMPLATE = """
USER QUESTION:
--------------
{question}

Answer requirements:
- Explain your reasoning
- Mention advantages and limitations
- Give practical recommendation
"""

    def __init__(self, assistant_name: str = "DataKit AI"):
        self.assistant_name = assistant_name
        self._system_prompt = self.SYSTEM_PROMPT.replace("DataKit AI", assistant_name)

    def build_prompt(
        self,
        user_question: str,
        dataset_context: Optional[str] = None,
        retrieved_documents: Optional[list] = None
    ) -> str:
        """Build final prompt sent to LLM."""
        
        prompt_parts = [self._system_prompt]
        
        if dataset_context:
            prompt_parts.append(
                self.DATASET_TEMPLATE.format(dataset_context=dataset_context)
            )
        
        if retrieved_documents:
            knowledge_text = "\n\n".join([
                doc.get('content', str(doc)) if isinstance(doc, dict) else str(doc)
                for doc in retrieved_documents
            ])
            prompt_parts.append(
                self.KNOWLEDGE_TEMPLATE.format(knowledge=knowledge_text)
            )
        
        prompt_parts.append(
            self.QUESTION_TEMPLATE.format(question=user_question)
        )
        
        return "\n".join(prompt_parts)

    def build_system_prompt(self) -> str:
        """Return only the system prompt."""
        return self._system_prompt