"""
AI Assistant factory.
"""

from pathlib import Path

from .assistant_service import AssistantService
from .cache.redis_client import RedisClient
from .cache.redis_index import RedisVectorIndex
from .cache.semantic_cache import SemanticCache
from .classifier.classifier_ml import QueryClassifier
from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import (
    PreprocessingContext,
)
from .embeddings.embeddings import EmbeddingModel
from .llm.ollama_client import OllamaClient
from .llm.prompt_builder import PromptBuilder
from .rag.qdrant_client import QdrantVectorStore
from .rag.retriever import QdrantRetriever


def create_assistant(
    classifier_model_path: str = (
        "datakit/ai_assistant/"
        "classifier/model/"
        "query_classifier.joblib"
    ),
    embedding_model_name: str = (
        "BAAI/bge-small-en-v1.5"
    ),
    redis_url: str = (
        "redis://localhost:6379"
    ),
    qdrant_url: str = (
        "http://localhost:6333"
    ),
    qdrant_collection: str = (
        "datakit_knowledge"
    ),
    ollama_model: str = "mistral",
) -> AssistantService:
    """Create a fully configured DataKit assistant."""

    embedding_model = EmbeddingModel(
        model_name=embedding_model_name
    )

    classifier = QueryClassifier(
        model_path=classifier_model_path
    )

    redis = RedisClient(
        url=redis_url
    )

    redis_index = RedisVectorIndex(
        client=redis.client,
        dimension=embedding_model.dimension,
    )

    redis_index.create_indexes()

    semantic_cache = SemanticCache(
        client=redis.client,
        dimension=embedding_model.dimension,
    )

    vector_store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=qdrant_collection,
    )

    retriever = QdrantRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    dataset_builder = DatasetContextBuilder()

    preprocessing_context = (
        PreprocessingContext()
    )

    context_manager = ContextManager(
        dataset_builder=dataset_builder,
        preprocessing_context=preprocessing_context,
    )

    prompt_builder = PromptBuilder()

    llm = OllamaClient(
        model=ollama_model,
    )

    return AssistantService(
        embedding_model=embedding_model,
        classifier=classifier,
        semantic_cache=semantic_cache,
        retriever=retriever,
        context_manager=context_manager,
        prompt_builder=prompt_builder,
        llm=llm,
    )