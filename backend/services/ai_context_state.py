
"""
Shared AI context state (dataset + preprocessing history).

Follows the same pattern as services/state.py (data_state): a single
lightweight instance (no LLM/RAG dependency), instantiated immediately
when the module is loaded — unlike AssistantService, which
lazily and expensively initializes the LangGraph agent (Ollama, embeddings, FAISS).

Separating this singleton from create_assistant() allows UploadService and
PreprocessingService to populate the context as soon as a dataset is
loaded or transformed, without depending on whether a conversation
with the assistant has already taken place.
"""

from datakit.ai_assistant.context.context_manager import ContextManager
from datakit.ai_assistant.context.dataset_context import DatasetContextBuilder
from datakit.ai_assistant.context.preprocessing_context import PreprocessingContext


context_manager = ContextManager(
    dataset_builder=DatasetContextBuilder(),
    preprocessing_context=PreprocessingContext()
)
