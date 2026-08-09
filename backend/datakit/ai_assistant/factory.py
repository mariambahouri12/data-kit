"""
AI Assistant factory.

"""

from dataclasses import dataclass
from typing import Optional

from .cache.redis_client import RedisClient
from .cache.redis_index import RedisVectorIndex
from .cache.semantic_cache import SemanticCache
from .classifier.classifier_ml import QueryClassifier
from .context.context_manager import ContextManager
from .context.dataset_context import DatasetContextBuilder
from .context.preprocessing_context import PreprocessingContext
from .embeddings.embeddings import EmbeddingModel
from .llm.ollama_client import OllamaClient
from .llm.prompt_builder import PromptBuilder
from .orchestrator import AssistantOrchestrator
from .rag.chunker import TextChunker
from .rag.document_loader import DocumentLoader
from .rag.document_processor import DocumentProcessor
from .rag.qdrant_client import QdrantVectorStore
from .rag.qdrant_indexer import QdrantIndexer
from .rag.retriever import QdrantRetriever


@dataclass
class AssistantRuntime:
    """Orchestrateur + indexeur + context_manager, prêts à l'emploi."""

    orchestrator: AssistantOrchestrator
    indexer: QdrantIndexer
    context_manager: ContextManager


def _default_context_manager() -> ContextManager:
    """Fallback si aucun context_manager n'est injecté (tests, scripts)."""

    return ContextManager(
        dataset_builder=DatasetContextBuilder(),
        preprocessing_context=PreprocessingContext(),
    )


def _build_orchestrator(
    embedding_model: EmbeddingModel,
    classifier_model_path: str,
    redis_url: str,
    vector_store: QdrantVectorStore,
    context_manager: ContextManager,
    ollama_model: str,
) -> AssistantOrchestrator:

    classifier = QueryClassifier(model_path=classifier_model_path)

    redis = RedisClient(url=redis_url)
    RedisVectorIndex(
        client=redis.client,
        dimension=embedding_model.dimension,
    ).create_indexes()

    semantic_cache = SemanticCache(
        client=redis.client,
        dimension=embedding_model.dimension,
    )

    retriever = QdrantRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return AssistantOrchestrator(
        embedding_model=embedding_model,
        classifier=classifier,
        semantic_cache=semantic_cache,
        retriever=retriever,
        context_manager=context_manager,
        prompt_builder=PromptBuilder(),
        llm=OllamaClient(model=ollama_model),
    )


def _build_indexer(
    embedding_model: EmbeddingModel,
    vector_store: QdrantVectorStore,
    knowledge_base_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> QdrantIndexer:

    return QdrantIndexer(
        loader=DocumentLoader(knowledge_base_path),
        processor=DocumentProcessor(),
        chunker=TextChunker(chunk_size=chunk_size, overlap=chunk_overlap),
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def create_assistant(
    context_manager: Optional[ContextManager] = None,
    classifier_model_path: str = (
        "datakit/ai_assistant/classifier/model/query_classifier.joblib"
    ),
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    redis_url: str = "redis://localhost:6379",
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "datakit_knowledge",
    knowledge_base_path: str = "knowledge_base",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    ollama_model: str = "mistral",
) -> AssistantRuntime:
    """
    Create a fully configured DataKit assistant runtime.


    """

    embedding_model = EmbeddingModel(model_name=embedding_model_name)

    vector_store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=qdrant_collection,
    )

    context_manager = context_manager or _default_context_manager()

    orchestrator = _build_orchestrator(
        embedding_model=embedding_model,
        classifier_model_path=classifier_model_path,
        redis_url=redis_url,
        vector_store=vector_store,
        context_manager=context_manager,
        ollama_model=ollama_model,
    )

    indexer = _build_indexer(
        embedding_model=embedding_model,
        vector_store=vector_store,
        knowledge_base_path=knowledge_base_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return AssistantRuntime(
        orchestrator=orchestrator,
        indexer=indexer,
        context_manager=context_manager,
    )