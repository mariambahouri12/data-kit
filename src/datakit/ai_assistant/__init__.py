# datakit/ai_assistant/__init__.py

"""
AI Assistant module for DataKit.
"""

import logging
from typing import Optional

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import PreprocessingContext

from .llm.ollama_client import OllamaClient
from .llm.prompt_manager import PromptManager
from .llm.response_formatter import ResponseFormatter

from .rag.embeddings import EmbeddingModel
from .rag.rag_pipeline import RAGPipeline
from .rag.retriever import Retriever
from .rag.vector_store import VectorStore


__all__ = [
    'ContextManager',
    'DatasetContextBuilder',
    'PreprocessingContext',
    'OllamaClient',
    'PromptManager',
    'ResponseFormatter',
    'EmbeddingModel',
    'RAGPipeline',
    'Retriever',
    'VectorStore',
    'logger'
]


def create_assistant(
    model_name: str = "mistral",
    knowledge_base_path: Optional[str] = None
) -> dict:
    """Factory function to create a fully configured AI assistant."""
    
    logger.info("Creating AI Assistant...")
    
    # Components
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()
    retriever = Retriever(embedding_model, vector_store)
    prompt_manager = PromptManager()
    llm_client = OllamaClient(model_name=model_name)
    
    # Check availability
    status = llm_client.check_connection()
    if status.get("status"):
        logger.info(f"✅ Ollama ready with model: {model_name}")
    else:
        logger.warning(f"⚠️ {status.get('message')}")
    
    # Pipeline
    pipeline = RAGPipeline(retriever, prompt_manager, llm_client)
    
    # Context
    dataset_builder = DatasetContextBuilder()
    preprocessing_context = PreprocessingContext()
    context_manager = ContextManager(dataset_builder, preprocessing_context)
    
    # Try to load existing vector store
    try:
        if vector_store.load():
            retriever._is_initialized = True
            logger.info("✅ Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not load vector store: {e}")
    
    logger.info("✅ Assistant created successfully")
    
    return {
        'pipeline': pipeline,
        'context_manager': context_manager,
        'llm_client': llm_client,
        'retriever': retriever,
        'vector_store': vector_store
    }